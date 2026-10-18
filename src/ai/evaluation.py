"""
evaluation.py — Banc d'évaluation : Assistant (routage par règles) vs Agent (le LLM choisit ses outils).

Chaque question du jeu annoté est posée aux deux systèmes. On mesure :
- qualité : pertinence (LLM-as-a-judge, 0-100), bons outils appelés, bonne station citée, erreurs ;
- consommation : appels LLM, tokens, latence, coût estimé.

    python -m src.ai.evaluation          # nécessite GROQ_API_KEY (et LAKE_ROOT pour l'historique)
"""
import time
from dataclasses import dataclass

import pandas as pd

from src.data.filters import normalize_name

# Les deux systèmes nomment différemment l'outil d'historique
_TOOL_ALIASES = {"search_history_rag": "search_history"}


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_tools: frozenset
    expected_station: str | None = None


EVAL_SET = [
    EvalCase("Combien de vélos à Bastille - Beaumarchais en ce moment ?",
             frozenset({"get_station_info"}), "BASTILLE"),
    EvalCase("Y a-t-il des anomalies sur le réseau ?",
             frozenset({"detect_anomalies"})),
    EvalCase("Combien de vélos sont disponibles sur tout le réseau ?",
             frozenset({"get_network_stats"})),
    EvalCase("Bastille est-elle souvent vide le matin ?",
             frozenset({"search_history"}), "BASTILLE"),
    EvalCase("Pigalle a-t-elle des vélos le soir en général ?",
             frozenset({"search_history"}), "PIGALLE"),
    EvalCase("Est-ce que la station Nation risque d'être vide dans une heure ?",
             frozenset({"forecast_station"}), "NATION"),
    # Question composée : il faut l'état actuel ET l'historique
    EvalCase("La station Nation est vide là, c'est habituel à cette heure-ci ?",
             frozenset({"get_station_info", "search_history"}), "NATION"),
]


# Questions de rééquilibrage pour l'agent opérateur (plan d'actions, pas de réponse factuelle unique)
OPERATOR_QUESTIONS = [
    "Que faut-il rééquilibrer en ce moment ?",
    "Quelles stations approvisionner en priorité dans les 2 prochaines heures ?",
]


def _normalize_tools(tools: list[str]) -> set[str]:
    return {_TOOL_ALIASES.get(t, t) for t in tools}


def _score(case: EvalCase, system: str, answer: str, tools: list[str], trace) -> dict:
    used = _normalize_tools(tools)
    return {
        "system": system,
        "question": case.question,
        "relevance": trace.relevance_score,
        "tools_ok": case.expected_tools <= used,
        "station_ok": (
            None if case.expected_station is None
            else case.expected_station in normalize_name(answer)
        ),
        "error": answer.startswith("Erreur") or answer.startswith("Cle Groq"),
        "tools_used": ", ".join(sorted(used)) or "aucun",
        "latency_s": trace.latency_s,
        **trace.usage,
    }


def run_benchmark(df, rag_documents=None, qdrant_client=None, cases=EVAL_SET, pause_s: float = 2.0,
                  on_progress=None) -> pd.DataFrame:
    """Pose chaque question aux deux systèmes. `pause_s` évite les limites de débit de l'offre gratuite."""
    from src.ai.agent import run_agent
    from src.ai.assistant import run_unified_assistant

    rows, total = [], 2 * len(cases)
    for i, case in enumerate(cases):
        answer, trace = run_unified_assistant(case.question, df, qdrant_client=qdrant_client,
                                              rag_documents=rag_documents)
        rows.append(_score(case, "Assistant", answer, trace.tools_used, trace))
        if on_progress:
            on_progress(2 * i + 1, total)
        time.sleep(pause_s)

        answer, trace = run_agent(case.question, df, qdrant_client, rag_documents)
        rows.append(_score(case, "Agent", answer, trace.tools_called, trace))
        if on_progress:
            on_progress(2 * i + 2, total)
        time.sleep(pause_s)

    return pd.DataFrame(rows)


def run_operator_benchmark(df, rag_documents=None, qdrant_client=None, questions=OPERATOR_QUESTIONS,
                           pause_s: float = 2.0) -> pd.DataFrame:
    """
    Agent opérateur : pertinence (juge), part des actions validées par le vérificateur,
    nombre de corrections, consommation (dont Claude) et latence.
    """
    from src.ai.operator_agent import run_operator

    rows = []
    for question in questions:
        answer, trace = run_operator(question, df, rag_documents, qdrant_client)
        proposed = len(trace.plan.get("actions", [])) + answer.count("écartée")
        rows.append({
            "system": "Opérateur",
            "question": question,
            "relevance": trace.relevance_score,
            "actions_validees": len(trace.plan.get("actions", [])),
            "corrections": trace.corrections,
            "taux_actions_valides": (len(trace.plan.get("actions", [])) / proposed) if proposed else None,
            "error": answer.startswith("Erreur"),
            "latency_s": trace.latency_s,
            "modeles": ", ".join(trace.models),
            **trace.usage,
        })
        time.sleep(pause_s)
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par système : moyennes de qualité et de consommation par question."""
    if results.empty:
        return results
    return (
        results.groupby("system")
        .agg(
            questions=("question", "count"),
            pertinence_moy=("relevance", "mean"),
            bons_outils=("tools_ok", "mean"),
            bonne_station=("station_ok", lambda s: s.dropna().astype(float).mean()),
            erreurs=("error", "mean"),
            appels_llm_moy=("llm_calls", "mean"),
            tokens_moy=("total_tokens", "mean"),
            latence_moy_s=("latency_s", "mean"),
            cout_moy_usd=("cost_usd", "mean"),
        )
        .round(3)
    )


if __name__ == "__main__":
    from src.ai.rag import build_rag_index
    from src.data.data_loader import fetch_api_snapshot

    snapshot = fetch_api_snapshot()
    documents, _ = build_rag_index()
    results = run_benchmark(snapshot, documents,
                            on_progress=lambda done, total: print(f"{done}/{total}", end="\r"))
    pd.set_option("display.width", 200)
    print(results[["system", "question", "relevance", "tools_ok", "station_ok",
                   "llm_calls", "total_tokens", "latency_s", "cost_usd"]])
    print(summarize(results).T)
