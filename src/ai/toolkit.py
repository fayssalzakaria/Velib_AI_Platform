"""
toolkit.py — Les outils métier, définis une seule fois (outils LangChain).

Utilisés par l'agent simple (Groq, format OpenAI), l'agent opérateur (Claude, format Anthropic)
et le serveur MCP. Chaque outil renvoie du JSON ; la logique métier vit dans tools.py.
"""
import json

from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from src.ai.tools import (
    detect_anomalies_tool,
    forecast_tool,
    get_network_stats_tool,
    get_station_info_tool,
    search_history_tool,
)


def _json(data) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def build_toolkit(df, rag_documents=None, qdrant_client=None) -> list[BaseTool]:
    """Outils liés aux données de la requête (snapshot temps réel, index RAG, Qdrant)."""

    def get_station_info(station_name: str) -> str:
        """État temps réel d'une station Vélib' : vélos (mécaniques, électriques), bornes libres, état."""
        return _json(get_station_info_tool(station_name, df))

    def get_network_stats() -> str:
        """Statistiques globales du réseau Vélib' en temps réel."""
        return _json(get_network_stats_tool(df))

    def detect_anomalies() -> str:
        """Anomalies actuelles du réseau : stations vides, presque pleines, manque de vélos électriques."""
        return _json(detect_anomalies_tool(df))

    def search_history(query: str) -> str:
        """Habitudes et tendances historiques des stations (RAG sur les profils), par exemple « souvent vide le matin »."""
        return _json(search_history_tool(query, rag_documents, qdrant_client))

    def forecast_station(station_name: str, hours_ahead: int | None = None, at_hour: int | None = None) -> str:
        """Prévision dans 1 à 6 heures : risque d'être vide, risque d'être pleine, vélos attendus avec intervalle.
        hours_ahead : dans combien d'heures (1 à 6) ; at_hour : heure précise demandée (0-23), par exemple 18."""
        return _json(forecast_tool(station_name, hours_ahead=hours_ahead, at_hour=at_hour))

    return [StructuredTool.from_function(f) for f in
            (get_station_info, get_network_stats, detect_anomalies, search_history, forecast_station)]


def openai_schemas(tools: list[BaseTool]) -> list[dict]:
    """Schémas au format OpenAI (Groq)."""
    return [convert_to_openai_tool(t) for t in tools]


def anthropic_schemas(tools: list[BaseTool]) -> list[dict]:
    """Schémas au format Anthropic (Claude)."""
    schemas = []
    for t in tools:
        function = convert_to_openai_tool(t)["function"]
        schemas.append({
            "name": function["name"],
            "description": function["description"],
            "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
        })
    return schemas


def run_tool(tools: list[BaseTool], name: str, args: dict) -> str:
    """Exécute un outil par son nom ; une erreur devient un résultat JSON lisible par le LLM."""
    by_name = {t.name: t for t in tools}
    if name not in by_name:
        return _json({"error": f"Outil {name} inconnu"})
    try:
        return by_name[name].invoke(args or {})
    except Exception as e:
        return _json({"error": f"{type(e).__name__} : {e}"})
