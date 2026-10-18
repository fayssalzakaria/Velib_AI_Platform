"""Outils partagés (LangChain) et retriever RAG."""
import json

import pandas as pd

from src.ai.toolkit import anthropic_schemas, build_toolkit, openai_schemas, run_tool

SNAPSHOT = pd.DataFrame([{
    "station_id": "1", "name": "BASTILLE - BISCORNET", "numbikesavailable": 0, "mechanical": 0, "ebike": 0,
    "numdocksavailable": 20, "is_empty": True, "is_full": False, "bike_ratio": 0.0,
}])
NAMES = {"get_station_info", "get_network_stats", "detect_anomalies", "search_history", "forecast_station"}


def test_one_definition_two_formats():
    tools = build_toolkit(SNAPSHOT)
    openai = {s["function"]["name"]: s["function"] for s in openai_schemas(tools)}
    anthropic = {s["name"]: s for s in anthropic_schemas(tools)}
    assert set(openai) == set(anthropic) == NAMES
    assert openai["get_station_info"]["parameters"]["required"] == ["station_name"]
    assert anthropic["forecast_station"]["input_schema"]["required"] == ["station_name"]
    assert all(s["description"] for s in anthropic.values())


def test_run_tool_returns_json_even_on_errors():
    tools = build_toolkit(SNAPSHOT)
    info = json.loads(run_tool(tools, "get_station_info", {"station_name": "bastille"}))
    assert info["etat"] == "vide"
    assert "error" in json.loads(run_tool(tools, "inconnu", {}))
    assert json.loads(run_tool(tools, "search_history", {"query": "x"})) == {"error": "Historique non disponible"}


def test_profile_retriever_wraps_hybrid_search(monkeypatch):
    import src.ai.rag as rag
    from src.ai.retriever import ProfileRetriever

    docs = [{"text": "Station BASTILLE — vide 42 % le matin", "metadata": {"station": "BASTILLE", "run_at": "2026-10-17 09:00"}}]
    monkeypatch.setattr(rag, "hybrid_search", lambda q, d, **k: (d, {"techniques_used": ["BM25", "RRF"]}))
    found = ProfileRetriever(documents=docs, k=3).invoke("Bastille le matin ?")
    assert found[0].page_content.startswith("Station BASTILLE")
    assert found[0].metadata["techniques"] == ["BM25", "RRF"]


def test_assistant_finds_the_station_in_the_question():
    from src.ai.assistant import _extract_possible_station

    names = ["STATION PARIS BASKETBALL - NEY", "BROCHANT - CLICHY", "BERTHIER - PORTE DE CLICHY",
             "PLACE D'ITALIE - AURIOL", "RÉPUBLIQUE - TEMPLE"] + [f"PLACE {i} - RUE {i}" for i in range(60)]
    df = pd.DataFrame({"name": names})
    assert _extract_possible_station("état de la station clichy", df) == "BROCHANT - CLICHY"
    assert _extract_possible_station("station porte de clichy", df) == "BERTHIER - PORTE DE CLICHY"
    assert _extract_possible_station("La station republique sera vide à 18h ?", df) == "RÉPUBLIQUE - TEMPLE"
    assert _extract_possible_station("Combien de vélos dans Paris ?", df) is None
    assert _extract_possible_station("Quelles stations place ?", df) is None   # mot trop courant
