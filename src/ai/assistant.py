"""
assistant.py — Assistant IA unifié Vélib

L'assistant route automatiquement les questions vers :
- données temps réel
- anomalies
- RAG historique
- recherche sémantique
- rapport
"""

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from llm_client import (
    CLAUDE_COMPLEX_MODEL, call_claude, call_llm_text, claude_available, claude_text, track_usage, verify_relevance,
)
from src.ai.prompts import FINAL_SYNTHESIS_PROMPT
from src.ai.router import classify_question
from src.ai.tools import (
    detect_anomalies_tool,
    forecast_tool,
    generate_report_tool,
    get_network_stats_tool,
    get_station_info_tool,
    search_history_rag_tool,
    semantic_search_tool,
    tool_result_to_text,
)


@dataclass
class AssistantTrace:
    question: str
    intent: str = ""
    confidence: float | None = None
    rule: str = ""
    matched_keywords: list[str] = field(default_factory=list)
    routing_reason: str = ""
    tools_used: list[str] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    relevance_score: int | None = None
    relevance_explanation: str = ""
    final_answer: str = ""
    delegated_to_agent: bool = False
    escalated: bool = False
    escalation_note: str = ""
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)


def _extract_possible_station(question: str, df: pd.DataFrame) -> str | None:
    """
    Détection simple d'une station mentionnée dans la question.
    On évite un appel LLM supplémentaire.
    """
    if df is None or df.empty or "name" not in df.columns:
        return None

    q = question.upper()

    stations = sorted(
        df["name"].dropna().unique().tolist(),
        key=len,
        reverse=True,
    )

    for station in stations:
        if station and station in q:
            return station

    # fallback par mots distinctifs
    words = [w for w in q.split() if len(w) >= 4]
    best_station = None
    best_score = 0

    for station in stations:
        score = sum(1 for w in words if w in station)
        if score > best_score:
            best_score = score
            best_station = station

    return best_station if best_score >= 1 else None


def _synthesize_answer(
    question: str,
    intent: str,
    tool_results_text: str,
) -> tuple[str, dict[str, int]]:
    prompt = f"""
Question utilisateur :
{question}

Intention détectée :
{intent}

Résultats des outils :
{tool_results_text}

Produis la réponse finale.
"""

    return call_llm_text(
        prompt,
        system_prompt=FINAL_SYNTHESIS_PROMPT,
        max_tokens=700,
        temperature=0.2,
        timeout=30,
    )


# En dessous de cette note du juge, la réponse de Groq est reprise par Claude Opus (cascade)
ESCALATION_THRESHOLD = 60


def _escalate(question: str, intent: str, tool_results_text: str) -> str | None:
    """Même question, mêmes résultats d'outils, rédigée par Claude Opus 5.5. None en cas d'échec."""
    try:
        response = call_claude(
            model=CLAUDE_COMPLEX_MODEL, system=FINAL_SYNTHESIS_PROMPT, effort="medium", max_tokens=8000,
            messages=[{"role": "user", "content": (
                f"Question utilisateur :\n{question}\n\nIntention détectée :\n{intent}\n\n"
                f"Résultats des outils :\n{tool_results_text}\n\nProduis la réponse finale."
            )}],
        )
        return claude_text(response) or None
    except Exception:
        return None


def run_unified_assistant(
    question: str,
    df: pd.DataFrame,
    *,
    qdrant_client=None,
    rag_documents=None,
    qdrant_docs=None,
    use_llm_router: bool = False,
) -> tuple[str, AssistantTrace]:
    """
    Point d'entrée de l'assistant : routage décidé par le code, puis outils, puis synthèse LLM.
    Mesure la latence et la consommation de tous les appels LLM de la question.
    """
    started = time.perf_counter()
    with track_usage() as usage:
        answer, trace = _run_assistant(
            question, df,
            qdrant_client=qdrant_client,
            rag_documents=rag_documents,
            qdrant_docs=qdrant_docs,
            use_llm_router=use_llm_router,
        )
    trace.latency_s = round(time.perf_counter() - started, 2)
    trace.usage = usage.as_dict()
    trace.tokens = {
        "prompt": usage.prompt_tokens,
        "completion": usage.completion_tokens,
        "total": usage.total_tokens,
    }
    return answer, trace


def _run_assistant(
    question: str,
    df: pd.DataFrame,
    *,
    qdrant_client=None,
    rag_documents=None,
    qdrant_docs=None,
    use_llm_router: bool = False,
) -> tuple[str, AssistantTrace]:
    trace = AssistantTrace(question=question)

    routing = classify_question(question, use_llm=use_llm_router)
    intent = routing.get("intent", "general")

    trace.intent = intent
    trace.confidence = routing.get("confidence")
    trace.rule = routing.get("rule", "")
    trace.matched_keywords = routing.get("matched", [])
    trace.routing_reason = routing.get("reason", "")

    tool_outputs = {}

    try:
        if intent == "realtime":
            station = _extract_possible_station(question, df)

            if station:
                result = get_station_info_tool(station, df)
                trace.tools_used.append("get_station_info")
                tool_outputs["get_station_info"] = result
            else:
                result = get_network_stats_tool(df)
                trace.tools_used.append("get_network_stats")
                tool_outputs["get_network_stats"] = result

        elif intent == "anomaly":
            anomalies = detect_anomalies_tool(df)
            stats = get_network_stats_tool(df)

            trace.tools_used.extend(["detect_anomalies", "get_network_stats"])
            tool_outputs["detect_anomalies"] = anomalies
            tool_outputs["get_network_stats"] = stats

        elif intent == "historical":
            answer, rag_trace = search_history_rag_tool(
                question,
                rag_documents,
                qdrant_client=qdrant_client,
                qdrant_docs=qdrant_docs,
            )

            trace.tools_used.append("search_history_rag")
            tool_outputs["search_history_rag"] = {
                "answer": answer,
                "trace": rag_trace,
            }

            # Si le RAG donne déjà une réponse bien formée, on la retourne directement,
            # mais on récupère ou recalcule aussi la pertinence.
            trace.tool_results = tool_outputs
            trace.final_answer = answer

            if isinstance(rag_trace, dict):
                trace.tokens = rag_trace.get("tokens", {})
                trace.relevance_score = rag_trace.get("relevance_score")
                trace.relevance_explanation = rag_trace.get("relevance_explanation", "")

            if trace.relevance_score is None:
                rag_context = tool_result_to_text(tool_outputs["search_history_rag"])

                score, explanation = _evaluate_unified_answer(
                    question=question,
                    answer=answer,
                    tool_results_text=rag_context,
                    intent=intent,
                )

                trace.relevance_score = score
                trace.relevance_explanation = explanation

            return answer, trace

        elif intent == "semantic":
            semantic_result = semantic_search_tool(
                question,
                qdrant_client,
                n_results=8,
            )

            trace.tools_used.append("semantic_search")
            tool_outputs["semantic_search"] = semantic_result

        elif intent == "forecast":
            from src.ai.nlp_utils import parse_time_target

            station = _extract_possible_station(question, df)
            target = parse_time_target(question) or {"hours_ahead": 1}
            trace.tools_used.append("forecast_station")
            tool_outputs["forecast_station"] = forecast_tool(station, **target)

        elif intent == "report":
            stats = get_network_stats_tool(df)
            anomalies = detect_anomalies_tool(df)
            report = generate_report_tool(df)

            trace.tools_used.extend(
                ["get_network_stats", "detect_anomalies", "generate_report"]
            )
            tool_outputs["get_network_stats"] = stats
            tool_outputs["detect_anomalies"] = anomalies
            tool_outputs["generate_report"] = {
                k: v for k, v in report.items() if k != "pdf_bytes"
            }

        else:
            # Question que les règles ne savent pas classer : on laisse l'agent choisir ses outils
            from src.ai.agent import run_agent

            answer, agent_trace = run_agent(question, df, qdrant_client, rag_documents)
            trace.delegated_to_agent = True
            trace.routing_reason += " Délégation à l'agent, qui choisit lui-même ses outils."
            trace.tools_used = ["agent"] + agent_trace.tools_called
            trace.tool_results = agent_trace.tool_results
            trace.relevance_score = agent_trace.relevance_score
            trace.relevance_explanation = agent_trace.relevance_explanation
            trace.final_answer = answer
            return answer, trace

        trace.tool_results = tool_outputs

        tool_results_text = "\n\n".join(
            f"[{name}]\n{tool_result_to_text(result)}"
            for name, result in tool_outputs.items()
        )

        answer, tokens = _synthesize_answer(
            question,
            intent,
            tool_results_text,
        )

        trace.tokens = tokens
        trace.final_answer = answer

        score, explanation = _evaluate_unified_answer(
            question=question,
            answer=answer,
            tool_results_text=tool_results_text,
            intent=intent,
        )

        trace.relevance_score = score
        trace.relevance_explanation = explanation

        # Cascade : réponse jugée faible → Claude Opus reprend ; on garde la meilleure des deux
        if score < ESCALATION_THRESHOLD and claude_available():
            better = _escalate(question, intent, tool_results_text)
            if better:
                new_score, new_explanation = _evaluate_unified_answer(
                    question=question, answer=better, tool_results_text=tool_results_text, intent=intent)
                trace.escalated = True
                trace.escalation_note = f"Réponse Groq notée {score}/100 → Claude Opus : {new_score}/100"
                if new_score >= score:
                    answer, trace.final_answer = better, better
                    trace.relevance_score, trace.relevance_explanation = new_score, new_explanation

        return answer, trace

    except Exception as e:
        trace.final_answer = f"Erreur assistant unifie : {e}"
        return trace.final_answer, trace


def _evaluate_unified_answer(
    question: str,
    answer: str,
    tool_results_text: str,
    intent: str,
) -> tuple[int, str]:
    """
    Évalue la pertinence de la réponse finale de l'assistant unifié
    par rapport aux résultats des outils utilisés.
    """
    context = f"""
Intention détectée :
{intent}

Résultats des outils :
{tool_results_text}
"""

    return verify_relevance(
        question=question,
        answer=answer,
        context=context,
        evaluator_name="assistant IA unifié Vélib",
    )