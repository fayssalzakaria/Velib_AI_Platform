"""
mcp_server.py — Serveur MCP : les outils Vélib' utilisables par n'importe quel client IA
(Claude Desktop, Claude Code, Cursor, autres agents).

Mêmes outils que l'application (toolkit.py), en lecture seule : état temps réel, anomalies,
habitudes historiques (RAG) et prévisions de 1 à 6 heures.

    python -m src.ai.mcp_server                         # transport stdio (client local)
    claude mcp add velib -- python -m src.ai.mcp_server # exemple d'ajout dans Claude Code
Variables utiles : LAKE_ROOT (historique et prévisions), AWS_* (lecture S3), GROQ_API_KEY (HyDE du RAG).
"""
import time

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from src.ai.toolkit import build_toolkit, run_tool

SNAPSHOT_TTL_SECONDS = 60
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=True)

server = MCPServer(
    name="velib",
    title="Vélib' Paris — temps réel, historique et prévisions",
    instructions=(
        "Outils sur le réseau Vélib' de Paris. État actuel : get_station_info, get_network_stats, "
        "detect_anomalies. Habitudes : search_history. Prévision à 1-6 h (vide, pleine, vélos attendus) : "
        "forecast_station. Les noms de stations peuvent être partiels et sans accents."
    ),
)

_cache: dict = {}


def _snapshot():
    """Snapshot de l'API Vélib', réutilisé pendant une minute."""
    from src.data.data_loader import fetch_api_snapshot

    if "snapshot" not in _cache or time.time() - _cache["snapshot"][0] > SNAPSHOT_TTL_SECONDS:
        _cache["snapshot"] = (time.time(), fetch_api_snapshot())
    return _cache["snapshot"][1]


def _rag_documents():
    """Profils des stations (vecteurs précalculés par le pipeline), chargés une fois."""
    if "rag" not in _cache:
        from src.ai.rag import load_precomputed_documents
        _cache["rag"] = load_precomputed_documents() or None
    return _cache["rag"]


def _run(name: str, **args) -> str:
    return run_tool(build_toolkit(_snapshot(), _rag_documents()), name, args)


@server.tool(annotations=READ_ONLY)
def get_station_info(station_name: str) -> str:
    """État temps réel d'une station Vélib' : vélos (mécaniques, électriques), bornes libres, état."""
    return _run("get_station_info", station_name=station_name)


@server.tool(annotations=READ_ONLY)
def get_network_stats() -> str:
    """Statistiques globales du réseau Vélib' en temps réel."""
    return _run("get_network_stats")


@server.tool(annotations=READ_ONLY)
def detect_anomalies() -> str:
    """Anomalies actuelles : stations vides, presque pleines, manque de vélos électriques."""
    return _run("detect_anomalies")


@server.tool(annotations=READ_ONLY)
def search_history(query: str) -> str:
    """Habitudes historiques des stations (par exemple « Bastille est-elle souvent vide le matin ? »)."""
    return _run("search_history", query=query)


@server.tool(annotations=READ_ONLY)
def forecast_station(station_name: str, hours_ahead: int | None = None, at_hour: int | None = None) -> str:
    """Prévision dans 1 à 6 h : risque d'être vide, d'être pleine, vélos attendus avec intervalle.
    hours_ahead : dans combien d'heures (1 à 6) ; at_hour : heure précise (0-23), par exemple 18."""
    return _run("forecast_station", station_name=station_name, hours_ahead=hours_ahead, at_hour=at_hour)


if __name__ == "__main__":
    server.run("stdio")
