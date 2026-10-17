"""Agent opérateur (LangGraph) : spécialistes, plan, vérification déterministe et boucle de correction."""
import pandas as pd

import src.ai.operator_agent as op

# Deux stations pleines (où prendre) et deux vides (à approvisionner), à quelques centaines de mètres
SNAPSHOT = pd.DataFrame([
    {"station_id": "1", "name": "REPUBLIQUE", "numbikesavailable": 30, "numdocksavailable": 0, "ebike": 5, "mechanical": 25,
     "is_empty": False, "is_full": True, "bike_ratio": 1.0, "lat": 48.8675, "lon": 2.3637},
    {"station_id": "2", "name": "OBERKAMPF", "numbikesavailable": 18, "numdocksavailable": 2, "ebike": 3, "mechanical": 15,
     "is_empty": False, "is_full": False, "bike_ratio": 0.9, "lat": 48.8645, "lon": 2.3680},
    {"station_id": "3", "name": "BASTILLE", "numbikesavailable": 0, "numdocksavailable": 20, "ebike": 0, "mechanical": 0,
     "is_empty": True, "is_full": False, "bike_ratio": 0.0, "lat": 48.8530, "lon": 2.3690},
    {"station_id": "4", "name": "RICHARD LENOIR", "numbikesavailable": 1, "numdocksavailable": 14, "ebike": 0, "mechanical": 1,
     "is_empty": False, "is_full": False, "bike_ratio": 0.07, "lat": 48.8600, "lon": 2.3710},
    {"station_id": "5", "name": "ORLY", "numbikesavailable": 25, "numdocksavailable": 0, "ebike": 2, "mechanical": 23,
     "is_empty": False, "is_full": True, "bike_ratio": 1.0, "lat": 48.7262, "lon": 2.3652},
])


def test_candidates_and_nearby_pairs():
    c = op._candidates(SNAPSHOT, None, horizon=1)
    assert c["_sources"] == {"REPUBLIQUE", "OBERKAMPF", "ORLY"}
    assert c["_destinations"] == {"BASTILLE", "RICHARD LENOIR"}
    assert all(p["distance_km"] <= op.MAX_DISTANCE_KM for p in c["paires_proches"])
    assert not any(p["depuis"] == "ORLY" for p in c["paires_proches"])  # 15 km : jamais proposée


def test_verifier_rules():
    c = op._candidates(SNAPSHOT, None, horizon=1)
    ok = {"actions": [{"depuis": "REPUBLIQUE", "vers": "BASTILLE", "velos": 10, "raison": "r"}]}
    assert op.verify_plan(ok, c) == []
    bad = {"actions": [
        {"depuis": "REPUBLIQUE", "vers": "BASTILLE", "velos": 25, "raison": "r"},   # 25 > 20 bornes libres
        {"depuis": "ORLY", "vers": "RICHARD LENOIR", "velos": 5, "raison": "r"},    # trop loin
        {"depuis": "BASTILLE", "vers": "REPUBLIQUE", "velos": 5, "raison": "r"},    # sens inversé
        {"depuis": "GARE INCONNUE", "vers": "BASTILLE", "velos": 5, "raison": "r"},
    ]}
    issues = " | ".join(op.verify_plan(bad, c))
    assert "bornes libres" in issues and "km" in issues
    assert "BASTILLE » n'est pas une station où prendre" in issues and "GARE INCONNUE" in issues


def test_graph_runs_specialists_and_corrects_the_plan(monkeypatch):
    """Le planificateur se trompe une fois : le vérificateur le renvoie corriger, puis valide."""
    calls = []

    def fake_complex(*, system, prompt, schema, effort="high", max_tokens=16000):
        calls.append(effort)
        if "specialists" in schema["properties"]:
            return {"specialists": ["temps_reel", "historique"], "horizon": 2, "raison": "test"}, "fake-opus"
        if "refusé par le vérificateur" not in prompt:
            return {"resume": "v1", "actions": [{"depuis": "REPUBLIQUE", "vers": "BASTILLE", "velos": 40, "raison": "r"}],
                    "points_de_vigilance": []}, "fake-opus"
        return {"resume": "v2", "actions": [{"depuis": "REPUBLIQUE", "vers": "BASTILLE", "velos": 15, "raison": "r"}],
                "points_de_vigilance": ["trafic"]}, "fake-opus"

    monkeypatch.setattr(op, "call_complex_json", fake_complex)
    monkeypatch.setattr(op, "verify_relevance", lambda **k: (88, "ok"))
    streamed = []
    answer, trace = op.run_operator("Que faut-il faire dans 2 h ?", SNAPSHOT, on_step=streamed.append)

    assert trace.corrections == 1 and trace.issues == []
    assert trace.plan["resume"] == "v2" and "15 vélos" in answer
    assert streamed == trace.steps and streamed[0].startswith("Superviseur")
    assert any("Spécialiste historique" in s for s in streamed)
    assert not any("Spécialiste prévision" in s for s in streamed)   # non choisi par le superviseur
    assert calls == ["low", "high", "high"]                           # superviseur puis 2 plans
    assert trace.relevance_score == 88 and trace.models == ["fake-opus"]


def test_only_verified_actions_reach_the_operator():
    c = op._candidates(SNAPSHOT, None, horizon=1)
    plan = {"resume": "r", "points_de_vigilance": [], "actions": [
        {"depuis": "REPUBLIQUE", "vers": "BASTILLE", "velos": 15, "raison": "ok"},
        {"depuis": "REPUBLIQUE", "vers": "RICHARD LENOIR", "velos": 14, "raison": "ok : 29 vélos retirés sur 30, 14 bornes libres"},
        {"depuis": "REPUBLIQUE", "vers": "RICHARD LENOIR", "velos": 5, "raison": "refusée : 34 vélos retirés sur 30, 19 déposés sur 14 bornes"},
        {"depuis": "ORLY", "vers": "BASTILLE", "velos": 2, "raison": "trop loin"},
    ]}
    kept, dropped = op.keep_verified_actions(plan, c)
    assert len(kept["actions"]) == 2 and len(dropped) == 2
    text = op.format_plan(kept, dropped)
    assert "2 action(s) écartée(s)" in text and "ORLY" not in text
