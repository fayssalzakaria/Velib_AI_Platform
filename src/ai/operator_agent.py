"""
operator_agent.py — Agent opérateur (LangGraph, architecture superviseur) : produit un plan de rééquilibrage vérifié.

    superviseur ─► spécialistes en parallèle (temps réel, prévision, historique)
                ─► planificateur ─► vérificateur ─┬─► réponse
                         ▲                         │
                         └──── corrections ◄───────┘   (2 corrections au maximum)

- superviseur : Claude Haiku 5.5 (choisir les spécialistes est une décision simple)
- planificateur : Claude Opus 5.5 (raisonnement sur les stocks et les distances)
  Les deux passent par la gateway et repassent sur Groq si Claude est indisponible.
- spécialistes : outils métier partagés (toolkit) et modèles de prévision, sans LLM sauf le RAG
- vérificateur : règles déterministes (stations connues, stocks, bornes libres, distances).
  Un LLM vérifié par un autre LLM peut se tromper deux fois ; une règle, non.
"""
import json
import math
import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from llm_client import CLAUDE_JUDGE_MODEL, call_complex_json, track_usage, verify_relevance
from src.ai.toolkit import build_toolkit, run_tool

MAX_CORRECTIONS = 2
MAX_DISTANCE_KM = 2.0   # au-delà, un déplacement de vélos n'est plus réaliste pour une tournée
MAX_CANDIDATES = 25     # par liste, pour garder un prompt du planificateur de taille raisonnable
SPECIALISTS = ("temps_reel", "prevision", "historique")


def _merge(left: dict | None, right: dict | None) -> dict:
    """Réducteur de `facts` : les spécialistes tournent en parallèle, chacun ajoute sa clé."""
    return {**(left or {}), **(right or {})}


class OperatorState(TypedDict, total=False):
    question: str
    horizon: int
    specialists: list[str]
    facts: Annotated[dict, _merge]              # chaque spécialiste ajoute sa partie
    steps: Annotated[list, operator.add]        # journal des étapes (affiché dans l'app)
    plan: dict
    issues: list[str]
    corrections: int
    models: Annotated[list, operator.add]


@dataclass
class OperatorTrace:
    question: str
    steps: list = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    dropped: list = field(default_factory=list)  # actions écartées par le vérificateur
    issues: list = field(default_factory=list)
    corrections: int = 0
    models: list = field(default_factory=list)
    candidates: dict = field(default_factory=dict)
    relevance_score: int | None = None
    relevance_explanation: str = ""
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)
    tools_called: list = field(default_factory=list)


# ---------------------------------------------------------------- données des candidats

def _distance_km(a: dict, b: dict) -> float:
    """Distance à vol d'oiseau (haversine) entre deux stations."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def _station(row, **extra) -> dict:
    return {"station": row["name"], "velos": int(row["numbikesavailable"]),
            "bornes_libres": int(row["numdocksavailable"]),
            "lat": float(row["lat"]), "lon": float(row["lon"]), **extra}


def _candidates(df: pd.DataFrame, forecast: pd.DataFrame | None, horizon: int) -> dict:
    """
    Stations à approvisionner (vides ou à risque d'être vides) et stations où prendre des vélos
    (pleines, presque pleines ou à risque d'être pleines), avec les paires proches.
    Les clés préfixées par « _ » servent au vérificateur et ne sont pas envoyées au LLM.
    """
    snapshot = df.dropna(subset=["lat", "lon"])
    risk = {}
    if forecast is not None and not forecast.empty:
        at_h = forecast[forecast["horizon"] == horizon]
        risk = {r.name: (r.risk_empty, r.risk_full) for r in at_h.itertuples()}

    # Sans prévision, les risques valent NaN : `x == x` est faux pour NaN, d'où les tests ci-dessous
    needs, offers = [], []
    for _, row in snapshot.iterrows():
        r_empty, r_full = risk.get(row["name"], (float("nan"), float("nan")))
        r_empty = 1.0 if row["is_empty"] else r_empty
        if row["is_empty"] or (r_empty == r_empty and r_empty >= 0.5) or row["numbikesavailable"] <= 1:
            needs.append(_station(row, risque_vide=None if r_empty != r_empty else round(float(r_empty), 2)))
        if row["is_full"] or (r_full == r_full and r_full >= 0.5) or row["bike_ratio"] >= 0.85:
            if row["numbikesavailable"] >= 3:  # sinon pas assez de vélos pour justifier un passage
                offers.append(_station(row, risque_pleine=None if r_full != r_full else round(float(r_full), 2)))

    needs = sorted(needs, key=lambda s: (-(s["risque_vide"] or 0), s["velos"]))[:MAX_CANDIDATES]
    offers = sorted(offers, key=lambda s: -s["velos"])[:MAX_CANDIDATES]
    pairs = sorted(
        ({"depuis": o["station"], "vers": n["station"], "distance_km": round(_distance_km(o, n), 2)}
         for o in offers for n in needs if o["station"] != n["station"]),
        key=lambda p: p["distance_km"],
    )
    pairs = [p for p in pairs if p["distance_km"] <= MAX_DISTANCE_KM][:40]
    strip = lambda s: {k: v for k, v in s.items() if k not in ("lat", "lon")}
    return {"a_approvisionner": [strip(s) for s in needs], "ou_prendre": [strip(s) for s in offers],
            "paires_proches": pairs, "_coords": {s["station"]: s for s in needs + offers},
            "_sources": {s["station"] for s in offers}, "_destinations": {s["station"] for s in needs}}


# ---------------------------------------------------------------- vérificateur (déterministe)

def verify_plan(plan: dict, candidates: dict) -> list[str]:
    """
    Contrôle chaque action du plan contre les données : stations connues et du bon côté,
    nombre de vélos valide, distance, puis stocks et bornes cumulés sur tout le plan.
    Renvoie la liste des problèmes (vide si le plan est valide).
    """
    stations = candidates["_coords"]
    issues, taken, given = [], {}, {}
    for i, action in enumerate(plan.get("actions", []), 1):
        src, dst, bikes = action.get("depuis"), action.get("vers"), action.get("velos")
        if src not in candidates["_sources"]:
            issues.append(f"Action {i} : « {src} » n'est pas une station où prendre des vélos.")
            continue
        if dst not in candidates["_destinations"]:
            issues.append(f"Action {i} : « {dst} » n'est pas une station à approvisionner.")
            continue
        if not isinstance(bikes, int) or bikes < 1:
            issues.append(f"Action {i} : nombre de vélos invalide ({bikes}).")
            continue
        taken[src] = taken.get(src, 0) + bikes
        given[dst] = given.get(dst, 0) + bikes
        distance = _distance_km(stations[src], stations[dst])
        if distance > MAX_DISTANCE_KM:
            issues.append(f"Action {i} : {src} → {dst} fait {distance:.1f} km (maximum {MAX_DISTANCE_KM} km).")
    for src, n in taken.items():
        if n > stations[src]["velos"]:
            issues.append(f"{src} : {n} vélos retirés pour {stations[src]['velos']} disponibles.")
    for dst, n in given.items():
        if n > stations[dst]["bornes_libres"]:
            issues.append(f"{dst} : {n} vélos déposés pour {stations[dst]['bornes_libres']} bornes libres.")
    return issues


# ---------------------------------------------------------------- graphe

_SUPERVISOR_SCHEMA = {
    "type": "object",
    "properties": {
        "specialists": {"type": "array", "items": {"type": "string", "enum": list(SPECIALISTS)}},
        "horizon": {"type": "integer"},
        "raison": {"type": "string"},
    },
    "required": ["specialists", "horizon", "raison"],
    "additionalProperties": False,
}

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "resume": {"type": "string"},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "depuis": {"type": "string"},
                    "vers": {"type": "string"},
                    "velos": {"type": "integer"},
                    "raison": {"type": "string"},
                },
                "required": ["depuis", "vers", "velos", "raison"],
                "additionalProperties": False,
            },
        },
        "points_de_vigilance": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["resume", "actions", "points_de_vigilance"],
    "additionalProperties": False,
}

_SUPERVISOR_SYSTEM = (
    "Tu coordonnes une équipe d'analystes du réseau Vélib' de Paris pour un opérateur qui rééquilibre "
    "les stations. Choisis les spécialistes utiles à la demande : temps_reel (état actuel, anomalies), "
    "prevision (stations qui seront vides ou pleines dans 1 à 6 h), historique (habitudes des stations). "
    "Choisis aussi l'horizon en heures (1 à 6) : celui de la demande, sinon 1."
)

_PLANNER_SYSTEM = (
    "Tu es le planificateur d'un opérateur Vélib'. À partir des données des spécialistes, propose des "
    "déplacements de vélos réalistes : prendre des vélos dans les stations de « ou_prendre » et les déposer "
    "dans celles de « a_approvisionner », en privilégiant les « paires_proches ». Utilise les noms de "
    "stations EXACTEMENT comme ils apparaissent. Ne retire jamais plus de vélos qu'une station n'en a, ne "
    "dépose jamais plus de vélos qu'il n'y a de bornes libres. Au plus 8 actions, les plus utiles d'abord. "
    "S'il n'y a rien d'utile à faire, renvoie une liste d'actions vide et explique pourquoi. "
    "Réponds en français."
)


def build_graph(df: pd.DataFrame, rag_documents=None, qdrant_client=None):
    """Graphe superviseur → spécialistes → planificateur ⇄ vérificateur (voir le schéma en tête)."""
    tools = build_toolkit(df, rag_documents, qdrant_client)

    def supervisor(state: OperatorState) -> dict:
        from src.ai.nlp_utils import parse_time_target

        decision, model = call_complex_json(
            system=_SUPERVISOR_SYSTEM, prompt=f"Demande de l'opérateur : {state['question']}",
            schema=_SUPERVISOR_SCHEMA, effort="low", max_tokens=4000, model=CLAUDE_JUDGE_MODEL,
        )
        # Un horizon écrit dans la question (« dans 3 h ») prime sur celui choisi par le LLM
        target = parse_time_target(state["question"])
        horizon = target.get("hours_ahead") or decision.get("horizon") or 1
        specialists = [s for s in decision.get("specialists", []) if s in SPECIALISTS] or ["temps_reel"]
        if "temps_reel" not in specialists:  # le plan a toujours besoin des stocks actuels
            specialists.insert(0, "temps_reel")
        return {"specialists": specialists, "horizon": max(1, min(6, int(horizon))), "models": [model],
                "steps": [f"Superviseur : {', '.join(specialists)} — horizon {horizon} h ({decision.get('raison', '')})"]}

    def temps_reel(state: OperatorState) -> dict:
        anomalies = json.loads(run_tool(tools, "detect_anomalies", {}))
        network = json.loads(run_tool(tools, "get_network_stats", {}))
        return {"facts": {"temps_reel": {"anomalies": anomalies, "reseau": network}},
                "steps": ["Spécialiste temps réel : anomalies et état du réseau"]}

    def prevision(state: OperatorState) -> dict:
        from src.ml.forecast import predict

        predictions, info = predict()
        if predictions is None:
            return {"facts": {"prevision": {"disponible": False, "raison": info.get("reason")}},
                    "steps": [f"Spécialiste prévision : indisponible ({info.get('reason')})"]}
        return {"facts": {"prevision": {"disponible": True, "frame": predictions}},
                "steps": [f"Spécialiste prévision : risques à {state['horizon']} h"]}

    def historique(state: OperatorState) -> dict:
        result = json.loads(run_tool(tools, "search_history", {"query": state["question"]}))
        return {"facts": {"historique": result},
                "steps": ["Spécialiste historique : habitudes des stations (RAG)"]}

    def planner(state: OperatorState) -> dict:
        forecast = state["facts"].get("prevision", {}).get("frame")
        # Candidats calculés une seule fois : les corrections travaillent sur les mêmes données
        candidates = state["facts"].get("candidats") or _candidates(df, forecast, state["horizon"])
        context = {
            "demande": state["question"],
            "horizon_heures": state["horizon"],
            **{k: v for k, v in candidates.items() if not k.startswith("_")},
            "anomalies": state["facts"].get("temps_reel", {}).get("anomalies"),
            "historique": state["facts"].get("historique"),
            "prevision_disponible": state["facts"].get("prevision", {}).get("disponible", False),
        }
        prompt = json.dumps(context, ensure_ascii=False, default=str)
        if state.get("issues"):
            prompt += ("\n\nTon plan précédent a été refusé par le vérificateur. Corrige ces problèmes :\n- "
                       + "\n- ".join(state["issues"]))
        plan, model = call_complex_json(system=_PLANNER_SYSTEM, prompt=prompt, schema=_PLAN_SCHEMA, effort="high")
        label = "Planificateur" if not state.get("issues") else f"Planificateur (correction {state.get('corrections', 0)})"
        return {"plan": plan, "facts": {"candidats": candidates}, "models": [model],
                "steps": [f"{label} : {len(plan.get('actions', []))} déplacements proposés"]}

    def verifier(state: OperatorState) -> dict:
        issues = verify_plan(state["plan"], state["facts"]["candidats"])
        corrections = state.get("corrections", 0) + (1 if issues else 0)
        verdict = "plan validé" if not issues else f"{len(issues)} problème(s) → correction"
        return {"issues": issues, "corrections": corrections, "steps": [f"Vérificateur : {verdict}"]}

    def after_verification(state: OperatorState) -> str:
        return "planner" if state["issues"] and state["corrections"] <= MAX_CORRECTIONS else END

    graph = StateGraph(OperatorState)
    for name, node in (("supervisor", supervisor), ("temps_reel", temps_reel), ("prevision", prevision),
                       ("historique", historique), ("planner", planner), ("verifier", verifier)):
        graph.add_node(name, node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", lambda s: s["specialists"], list(SPECIALISTS))
    for name in SPECIALISTS:
        graph.add_edge(name, "planner")
    graph.add_edge("planner", "verifier")
    graph.add_conditional_edges("verifier", after_verification, ["planner", END])
    return graph.compile()


# ---------------------------------------------------------------- point d'entrée

def keep_verified_actions(plan: dict, candidates: dict) -> tuple[dict, list[dict]]:
    """
    Après les corrections, ne garde que les actions qui passent le vérificateur (en respectant
    les stocks cumulés) : l'opérateur ne voit jamais une action invalide.
    """
    kept, dropped = [], []
    for action in plan.get("actions", []):
        if verify_plan({"actions": kept + [action]}, candidates):
            dropped.append(action)
        else:
            kept.append(action)
    return {**plan, "actions": kept}, dropped


def format_plan(plan: dict, dropped: list[dict]) -> str:
    """Plan vérifié en Markdown pour l'opérateur."""
    lines = [plan.get("resume", "")]
    for i, a in enumerate(plan.get("actions", []), 1):
        lines.append(f"{i}. **{a['velos']} vélos** : {a['depuis']} → {a['vers']} — {a['raison']}")
    if not plan.get("actions"):
        lines.append("Aucun déplacement vérifié à proposer pour le moment.")
    if plan.get("points_de_vigilance"):
        lines.append("\n**Points de vigilance** : " + " ; ".join(plan["points_de_vigilance"]))
    if dropped:
        lines.append(f"\n{len(dropped)} action(s) écartée(s) par le vérificateur (données incohérentes).")
    return "\n".join(lines)


def run_operator(question: str, df: pd.DataFrame, rag_documents=None, qdrant_client=None,
                 on_step=None) -> tuple[str, OperatorTrace]:
    """Exécute le graphe ; `on_step(texte)` reçoit chaque étape au fil de l'eau (affichage en direct)."""
    trace = OperatorTrace(question=question)
    started = time.perf_counter()
    state: dict[str, Any] = {"question": question}
    with track_usage() as usage:
        try:
            for update in build_graph(df, rag_documents, qdrant_client).stream(
                    {"question": question, "corrections": 0}, stream_mode="updates"):
                for node, delta in update.items():
                    # stream_mode="updates" ne donne que les deltas : on rejoue les réducteurs du graphe
                    for key, value in (delta or {}).items():
                        if key in ("facts",):
                            state[key] = _merge(state.get(key), value)
                        elif key in ("steps", "models"):
                            state[key] = state.get(key, []) + value
                        else:
                            state[key] = value
                    for step in (delta or {}).get("steps", []):
                        trace.steps.append(step)
                        if on_step:
                            on_step(step)
            candidates = state.get("facts", {}).get("candidats", {})
            trace.issues = state.get("issues", [])
            trace.plan, dropped = keep_verified_actions(state.get("plan", {}), candidates)
            trace.dropped = dropped
            trace.corrections, trace.models = state.get("corrections", 0), sorted(set(state.get("models", [])))
            trace.candidates = {k: v for k, v in candidates.items() if not k.startswith("_")}
            trace.tools_called = state.get("specialists", [])
            answer = format_plan(trace.plan, dropped)
            trace.relevance_score, trace.relevance_explanation = verify_relevance(
                question=question, answer=answer, evaluator_name="agent opérateur Vélib",
                context=json.dumps(trace.candidates, ensure_ascii=False))
        except Exception as e:
            answer = f"Erreur agent opérateur : {e}"
    trace.latency_s = round(time.perf_counter() - started, 2)
    trace.usage = usage.as_dict()
    return answer, trace
