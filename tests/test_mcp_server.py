"""Serveur MCP : mêmes outils que l'application, en lecture seule."""
import asyncio
import json

import pandas as pd

import src.ai.mcp_server as mcp_server

SNAPSHOT = pd.DataFrame([{
    "station_id": "1", "name": "BASTILLE - BISCORNET", "numbikesavailable": 0, "mechanical": 0, "ebike": 0,
    "numdocksavailable": 20, "is_empty": True, "is_full": False, "bike_ratio": 0.0,
}])


def test_tools_are_listed_read_only():
    tools = asyncio.run(mcp_server.server.list_tools())
    assert {t.name for t in tools} == {"get_station_info", "get_network_stats", "detect_anomalies",
                                      "search_history", "forecast_station"}
    assert all(t.annotations.read_only_hint for t in tools)
    forecast = next(t for t in tools if t.name == "forecast_station")
    assert forecast.input_schema["required"] == ["station_name"]


def test_call_tool_uses_the_shared_toolkit(monkeypatch):
    monkeypatch.setattr(mcp_server, "_snapshot", lambda: SNAPSHOT)
    monkeypatch.setattr(mcp_server, "_rag_documents", lambda: None)
    result = asyncio.run(mcp_server.server.call_tool("get_station_info", {"station_name": "bastille"}))
    payload = json.loads(result.content[0].text)
    assert payload["station"] == "BASTILLE - BISCORNET" and payload["etat"] == "vide"
