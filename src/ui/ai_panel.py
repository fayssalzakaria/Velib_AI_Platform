"""
ai_panel.py — Section Intelligence Artificielle de l'application.

- 💬 Assistant : routage décidé par le code (règles), puis outils, puis synthèse LLM
- 🤖 Agent : le LLM choisit lui-même ses outils, en plusieurs tours si besoin
- 📊 Qualité & consommation : métriques de la session et banc d'évaluation comparatif
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from src.ai.agent import run_agent
from src.ai.assistant import run_unified_assistant
from src.ai.nlp_utils import detect_time_expression, extract_keywords, normalize_text
from src.ai.vector_store import build_qdrant_index, semantic_search

ASSISTANT, AGENT, OPERATOR = "Assistant", "Agent", "Opérateur"
# Protège les crédits Claude d'une app publique : nombre de plans opérateur par session
OPERATOR_SESSION_QUOTA = 5


# ---------------------------------------------------------------- ressources partagées

def _get_qdrant_client_cached():
    if "qdrant_client" not in st.session_state:
        with st.spinner("Connexion Qdrant Cloud..."):
            client, n = build_qdrant_index()
            st.session_state.qdrant_client = client
            st.session_state.qdrant_docs = n
    return st.session_state.get("qdrant_client")


@st.cache_resource(ttl=3600, show_spinner=False)
def _shared_rag_index():
    """Index RAG (documents + embeddings) partagé entre toutes les sessions, reconstruit au plus 1x/h."""
    from src.ai.rag import build_rag_index
    return build_rag_index()


def _rag_documents():
    if "rag_documents" not in st.session_state:
        try:
            with st.spinner("Chargement des profils de stations (gold)..."):
                docs, n_docs = _shared_rag_index()
        except Exception:
            docs, n_docs = None, 0
        st.session_state.rag_documents = docs
        st.session_state.rag_docs = n_docs
    return st.session_state.rag_documents


# ---------------------------------------------------------------- métriques

def _record(system: str, question: str, answer: str, trace) -> None:
    """Garde les métriques de chaque réponse pour l'onglet Qualité & consommation."""
    st.session_state.setdefault("ai_metrics", []).append({
        "system": system,
        "question": question,
        "relevance": trace.relevance_score,
        "error": answer.startswith("Erreur") or answer.startswith("Cle Groq"),
        "latency_s": trace.latency_s,
        **trace.usage,
    })


def _metrics_line(trace) -> None:
    usage = trace.usage or {}
    score = trace.relevance_score
    color = "green" if (score or 0) >= 80 else "orange" if (score or 0) >= 50 else "red"
    score_txt = f"<span style='color:{color}'>**{score}/100**</span>" if score is not None else "n/a"
    short = {"openai/gpt-oss-120b": "Groq", "claude-opus-5-5": "Claude Opus", "claude-haiku-5-5": "Claude Haiku"}
    providers = " · ".join(f"{short.get(m, m)} ×{d['calls']}" for m, d in (usage.get("by_model") or {}).items())
    st.markdown(
        f"Pertinence {score_txt} · {usage.get('llm_calls', 0)} appels LLM"
        f"{f' ({providers})' if providers else ''} · "
        f"{usage.get('total_tokens', 0):,} tokens · {trace.latency_s:.1f} s · "
        f"{usage.get('cost_usd', 0):.4f} $",
        unsafe_allow_html=True,
    )


def _tool_results(results: dict) -> None:
    for name, result in results.items():
        with st.expander(f"Résultat de l'outil : {name}"):
            st.json(result)


# ---------------------------------------------------------------- 💬 Assistant

def _assistant_details(trace) -> None:
    with st.expander("Comment j'ai répondu"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Intention", trace.intent)
        c2.metric("Règle appliquée", trace.rule or "—")
        c3.metric("Outils", len(trace.tools_used))
        st.caption(
            "Mots qui ont déclenché la règle : "
            + (", ".join(f"« {kw} »" for kw in trace.matched_keywords) or "aucun")
        )
        st.caption(f"Routage : {trace.routing_reason}")
        if trace.delegated_to_agent:
            st.info("Question non reconnue par les règles : l'agent a choisi ses outils lui-même.")

        st.caption(
            f"NLP — texte normalisé : `{normalize_text(trace.question)}` · "
            f"mots-clés : {', '.join(extract_keywords(trace.question)) or 'aucun'} · "
            f"temps : {', '.join(detect_time_expression(trace.question)) or 'aucun'}"
        )
        if trace.tools_used:
            st.caption(f"Outils utilisés : {', '.join(trace.tools_used)}")

        rag = (trace.tool_results.get("search_history_rag") or {}).get("trace") or {}
        if rag.get("techniques_used"):
            st.caption(f"RAG : {' → '.join(rag['techniques_used'])}")
        if rag.get("final_docs"):
            st.caption("Sources : " + " · ".join(rag["final_docs"]))
        if trace.relevance_explanation:
            st.caption(f"Évaluation : {trace.relevance_explanation}")

        _tool_results({k: v for k, v in trace.tool_results.items() if k != "search_history_rag"})


def _render_assistant(df) -> None:
    st.caption(
        "Le **code** choisit le traitement selon la question (règles du routeur) : temps réel, "
        "anomalies, historique (RAG), rapport. Si la question n'est pas reconnue, l'agent prend le relais."
    )
    qdrant_client = _get_qdrant_client_cached()
    rag_documents = _rag_documents()

    history = st.session_state.setdefault("assistant_history", [])
    for question, answer, trace in history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
            _metrics_line(trace)
            _assistant_details(trace)

    question = st.chat_input(
        "Ex : Y a-t-il des anomalies ? Bastille est-elle souvent vide le matin ?",
        key="assistant_input",
    )
    if question:
        with st.spinner("Assistant en cours..."):
            qdrant_docs = semantic_search(question, qdrant_client, n_results=10) if qdrant_client else []
            answer, trace = run_unified_assistant(
                question, df,
                qdrant_client=qdrant_client,
                rag_documents=rag_documents,
                qdrant_docs=qdrant_docs,
            )
        history.append((question, answer, trace))
        _record(ASSISTANT, question, answer, trace)
        st.rerun()


# ---------------------------------------------------------------- 🤖 Agent

def _agent_details(trace) -> None:
    with st.expander("Comment j'ai répondu"):
        c1, c2 = st.columns(2)
        c1.metric("Tours d'outils", trace.tool_rounds)
        c2.metric("Outils appelés", len(trace.tools_called))
        st.caption(f"Outils, dans l'ordre : {' → '.join(trace.tools_called) or 'aucun'}")
        if trace.relevance_explanation:
            st.caption(f"Évaluation : {trace.relevance_explanation}")
        _tool_results(trace.tool_results)
        if trace.prompt_sent:
            with st.expander("Conversation envoyée au LLM"):
                st.text(trace.prompt_sent[:3000])


def _render_agent(df) -> None:
    st.caption(
        "Le **LLM** choisit lui-même ses outils (function calling) et peut enchaîner jusqu'à "
        "3 tours : par exemple l'état actuel d'une station, puis son historique pour comparer."
    )
    qdrant_client = _get_qdrant_client_cached()
    rag_documents = _rag_documents()

    history = st.session_state.setdefault("agent_history", [])
    for question, answer, trace in history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
            _metrics_line(trace)
            _agent_details(trace)

    question = st.chat_input(
        "Ex : La station Nation est vide là, c'est habituel à cette heure-ci ?",
        key="agent_input",
    )
    if question:
        with st.spinner("Agent en cours..."):
            answer, trace = run_agent(question, df, qdrant_client, rag_documents)
        history.append((question, answer, trace))
        _record(AGENT, question, answer, trace)
        st.rerun()


# ---------------------------------------------------------------- 🧭 Opérateur (LangGraph)

def _operator_details(trace) -> None:
    with st.expander("Comment le plan a été construit"):
        for step in trace.steps:
            st.caption(f"• {step}")
        st.caption(f"Modèles : {', '.join(trace.models) or '—'} · corrections : {trace.corrections}")
        if trace.relevance_explanation:
            st.caption(f"Évaluation : {trace.relevance_explanation}")
        if trace.candidates:
            col_need, col_offer = st.columns(2)
            col_need.markdown("**À approvisionner**")
            col_need.dataframe(pd.DataFrame(trace.candidates.get("a_approvisionner", [])), hide_index=True)
            col_offer.markdown("**Où prendre des vélos**")
            col_offer.dataframe(pd.DataFrame(trace.candidates.get("ou_prendre", [])), hide_index=True)


def _render_operator(df) -> None:
    from llm_client import CLAUDE_MONTHLY_BUDGET_USD, claude_available, claude_spend_this_month
    from src.ai.operator_agent import run_operator

    engine = (f"Claude Opus 5.5 (dépense Claude ce mois-ci : {claude_spend_this_month():.2f} $ / "
              f"{CLAUDE_MONTHLY_BUDGET_USD:.0f} $)" if claude_available() else "Groq (Claude non configuré ou budget atteint)")
    st.caption(
        "Système **multi-agents (LangGraph)** : un superviseur choisit les spécialistes (temps réel, prévision, "
        "historique), un planificateur propose des déplacements de vélos, un vérificateur contrôle chaque chiffre "
        f"et renvoie corriger si besoin. Moteur : {engine}."
    )
    history = st.session_state.setdefault("operator_history", [])
    for question, answer, trace in history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.markdown(answer)
            _metrics_line(trace)
            _operator_details(trace)

    used = st.session_state.get("operator_runs", 0)
    if claude_available() and used >= OPERATOR_SESSION_QUOTA:
        st.info(f"Limite de {OPERATOR_SESSION_QUOTA} plans par session atteinte (protection des crédits Claude).")
        return
    question = st.chat_input("Ex : Que faut-il rééquilibrer dans les 2 prochaines heures ?", key="operator_input")
    if question:
        with st.status("Agents au travail…", expanded=True) as status:
            answer, trace = run_operator(question, df, _rag_documents(), _get_qdrant_client_cached(),
                                         on_step=lambda step: status.write(step))
            status.update(label="Plan prêt", state="complete")
        st.session_state.operator_runs = used + 1
        history.append((question, answer, trace))
        _record(OPERATOR, question, answer, trace)
        st.rerun()


# ---------------------------------------------------------------- 📊 Qualité & consommation

def _summary_table(results: pd.DataFrame) -> pd.DataFrame:
    table = results.groupby("system").agg(
        Questions=("question", "count"),
        **{
            "Pertinence moy. (/100)": ("relevance", "mean"),
            "Erreurs": ("error", "mean"),
            "Appels LLM moy.": ("llm_calls", "mean"),
            "Tokens moy.": ("total_tokens", "mean"),
            "Latence moy. (s)": ("latency_s", "mean"),
            "Coût moy. ($)": ("cost_usd", "mean"),
        },
    )
    table["Erreurs"] = (table["Erreurs"] * 100).round(0).astype(int).astype(str) + " %"
    return table.round({"Pertinence moy. (/100)": 0, "Appels LLM moy.": 1, "Tokens moy.": 0,
                        "Latence moy. (s)": 1, "Coût moy. ($)": 5})


def _render_benchmark(df) -> None:
    from src.ai.evaluation import EVAL_SET, run_benchmark, summarize

    st.subheader("Banc d'évaluation : Assistant vs Agent")
    st.caption(
        f"Les mêmes {len(EVAL_SET)} questions annotées sont posées aux deux systèmes "
        f"({2 * len(EVAL_SET)} exécutions, 1 à 3 minutes). On vérifie les outils appelés, "
        "la station citée, et la pertinence notée par un LLM juge."
    )
    with st.expander("Questions et réponses attendues"):
        st.dataframe(pd.DataFrame([
            {"Question": c.question, "Outils attendus": ", ".join(sorted(c.expected_tools)),
             "Station attendue": c.expected_station or "—"}
            for c in EVAL_SET
        ]), hide_index=True, use_container_width=True)

    if st.button("▶️ Lancer le banc d'évaluation"):
        bar = st.progress(0.0, text="Évaluation en cours...")
        results = run_benchmark(
            df, _rag_documents(), _get_qdrant_client_cached(),
            on_progress=lambda done, total: bar.progress(done / total, text=f"{done}/{total} exécutions"),
        )
        bar.empty()
        st.session_state.benchmark = results

    results = st.session_state.get("benchmark")
    if results is None:
        return

    summary = summarize(results)
    view = pd.DataFrame({
        "Pertinence moy. (/100)": summary["pertinence_moy"].round(0),
        "Bons outils": (summary["bons_outils"] * 100).round(0).astype(int).astype(str) + " %",
        "Bonne station": (summary["bonne_station"] * 100).round(0).astype(int).astype(str) + " %",
        "Erreurs": (summary["erreurs"] * 100).round(0).astype(int).astype(str) + " %",
        "Appels LLM moy.": summary["appels_llm_moy"].round(1),
        "Tokens moy.": summary["tokens_moy"].round(0),
        "Latence moy. (s)": summary["latence_moy_s"].round(1),
        "Coût moy. ($)": summary["cout_moy_usd"].round(5),
    })
    st.dataframe(view, use_container_width=True)

    melted = results.melt(id_vars=["system", "question"], value_vars=["relevance", "total_tokens", "latency_s"])
    labels = {"relevance": "Pertinence (/100)", "total_tokens": "Tokens", "latency_s": "Latence (s)"}
    cols = st.columns(3)
    for col, metric in zip(cols, labels):
        fig = px.bar(
            melted[melted["variable"] == metric], x="question", y="value", color="system",
            barmode="group", labels={"value": labels[metric], "question": ""},
            color_discrete_map={ASSISTANT: "#185FA5", AGENT: "#EF9F27"},
        )
        fig.update_layout(title=labels[metric], height=320, margin=dict(t=40, b=10), xaxis_showticklabels=False)
        col.plotly_chart(fig, use_container_width=True)

    with st.expander("Détail par question"):
        st.dataframe(results[["system", "question", "relevance", "tools_ok", "station_ok", "tools_used",
                              "llm_calls", "total_tokens", "latency_s", "cost_usd"]],
                     hide_index=True, use_container_width=True)


def _render_metrics(df) -> None:
    st.caption(
        "Consommation mesurée sur tous les appels LLM de chaque réponse (routage, HyDE, synthèse, "
        "juge…). Coût estimé au tarif payant de Groq : rien n'est facturé sur l'offre gratuite."
    )
    st.subheader("Cette session")
    metrics = pd.DataFrame(st.session_state.get("ai_metrics", []))
    if metrics.empty:
        st.info("Posez des questions à l'Assistant ou à l'Agent pour voir leurs métriques ici.")
    else:
        st.dataframe(_summary_table(metrics), use_container_width=True)
        st.caption(
            f"Total de la session : {int(metrics['total_tokens'].sum()):,} tokens, "
            f"{metrics['cost_usd'].sum():.4f} $"
        )
    st.divider()
    _render_benchmark(df)


# ---------------------------------------------------------------- point d'entrée

def render_ai_tabs() -> None:
    """L'IA travaille toujours sur le snapshot complet de la session (les filtres d'affichage ne s'appliquent pas)."""
    from src.ui.ui import render_snapshot_bar

    st.subheader("Intelligence Artificielle")
    df = render_snapshot_bar(key="ai", label="Données temps réel de l'IA")
    if df is None:
        return
    tab_assistant, tab_agent, tab_operator, tab_metrics = st.tabs([
        "💬 Assistant (routage par règles)",
        "🤖 Agent (le LLM choisit ses outils)",
        "🧭 Opérateur (multi-agents)",
        "📊 Qualité & consommation",
    ])
    with tab_assistant:
        _render_assistant(df)
    with tab_agent:
        _render_agent(df)
    with tab_operator:
        _render_operator(df)
    with tab_metrics:
        _render_metrics(df)
