"""
agent.py — Agent simple : le LLM (Groq) choisit lui-même ses outils par function calling,
en plusieurs tours si besoin (par exemple l'état actuel d'une station, puis son historique).

Les outils viennent de toolkit.py (définis une seule fois, partagés avec l'agent opérateur
et le serveur MCP).
"""
import json
import os
import time
from dataclasses import dataclass, field

import pandas as pd

from llm_client import call_llm, track_usage, verify_relevance
from src.ai.toolkit import build_toolkit, openai_schemas, run_tool


@dataclass
class AgentTrace:
    question: str = ""
    tools_called: list = field(default_factory=list)
    tool_results: dict = field(default_factory=dict)
    prompt_sent: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    relevance_score: int = 0
    relevance_explanation: str = ""
    final_answer: str = ""
    tool_rounds: int = 0
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MAX_TOOL_ROUNDS = 3

AGENT_SYSTEM_PROMPT = (
    "Tu es un agent expert du réseau Vélib' à Paris. Tu disposes d'outils et tu choisis toi-même "
    "lesquels appeler. Tu peux procéder en plusieurs étapes : par exemple consulter l'état actuel "
    "d'une station, puis son historique pour dire si la situation est habituelle. "
    "État actuel et anomalies : get_station_info, get_network_stats, detect_anomalies. "
    "Habitudes et tendances : search_history. "
    "État d'une station dans 1 à 6 heures (vide, pleine, vélos attendus) : forecast_station. "
    "N'invente aucune donnée. Réponds en français, de façon concise ; "
    "quand tu listes des stations, cite les plus importantes (10 au maximum) et le nombre total."
)


def _format_messages(messages: list[dict]) -> str:
    """Conversation envoyée au LLM, lisible dans la trace."""
    return "\n".join(f"[{m['role'].upper()}] {m.get('content') or ''}" for m in messages if m.get("content"))


def _agent_loop(question: str, tools, trace: AgentTrace) -> tuple[str, str]:
    """
    Boucle agentique : le LLM appelle des outils, lit les résultats et peut en rappeler
    d'autres, jusqu'à MAX_TOOL_ROUNDS tours. Renvoie (réponse, données des outils).
    """
    schemas = openai_schemas(tools)
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    all_tool_data = []

    for _ in range(MAX_TOOL_ROUNDS):
        message = call_llm(messages, tools=schemas, max_tokens=1000)["choices"][0]["message"]
        if not message.get("tool_calls"):
            trace.prompt_sent = _format_messages(messages)
            return message.get("content") or "Aucune réponse.", "\n".join(all_tool_data)

        trace.tool_rounds += 1
        messages.append({"role": "assistant", "content": message.get("content") or "",
                         "tool_calls": message["tool_calls"]})
        for tool_call in message["tool_calls"]:
            name = tool_call["function"]["name"]
            try:
                args = json.loads(tool_call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}

            result = run_tool(tools, name, args)
            trace.tools_called.append(name)
            key = name if name not in trace.tool_results else f"{name} #{trace.tools_called.count(name)}"
            trace.tool_results[key] = json.loads(result)
            all_tool_data.append(f"[{name}] {result}")
            messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result})

    # Nombre de tours atteint : réponse finale sans nouvel outil
    trace.prompt_sent = _format_messages(messages)
    message = call_llm(messages, max_tokens=1000)["choices"][0]["message"]
    return message.get("content") or "Aucune réponse.", "\n".join(all_tool_data)


def run_agent(question: str, df: pd.DataFrame, qdrant_client=None, rag_documents=None) -> tuple[str, AgentTrace]:
    trace = AgentTrace(question=question)

    if not GROQ_API_KEY:
        return "Cle Groq non configuree.", trace

    started = time.perf_counter()
    with track_usage() as usage:
        try:
            tools = build_toolkit(df, rag_documents, qdrant_client)
            answer, tool_data = _agent_loop(question, tools, trace)
            trace.final_answer = answer
            trace.relevance_score, trace.relevance_explanation = verify_relevance(
                question=question, answer=answer, context=tool_data, evaluator_name="agent IA Vélib")
        except Exception as e:
            answer = trace.final_answer = f"Erreur agent : {e}"

    trace.latency_s = round(time.perf_counter() - started, 2)
    trace.usage = usage.as_dict()
    trace.prompt_tokens = usage.prompt_tokens
    trace.completion_tokens = usage.completion_tokens
    trace.total_tokens = usage.total_tokens
    return answer, trace
