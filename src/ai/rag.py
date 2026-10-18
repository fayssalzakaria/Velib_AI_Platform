"""
rag.py — RAG avancé avec Hybrid Search + Reranking + MMR + HyDE + Citations
"""
import os
import numpy as np
import pandas as pd
from llm_client import call_llm_text, call_llm_json, verify_relevance
import pytz

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
PARIS_TZ = pytz.timezone("Europe/Paris")


def _load_profiles() -> pd.DataFrame:
    """Profils gold du lakehouse (dbt) : une ligne par station / type de jour / période."""
    from src.data.lake import station_period_profiles
    return station_period_profiles()


def _hyde_expand_query(question: str) -> str:
    try:
        prompt = f"""Tu dois generer un exemple de document Velib qui pourrait repondre a cette question.
Le document doit ressembler exactement a ce format :
"Station NOM_STATION — profil historique (N observations horaires).
En semaine : matin (6h-12h) : vide XX% du temps, pleine XX%, X vélos en moyenne dont Y électriques, Z bornes libres ; soir (18h-22h) : ...
Le week-end : ...
Moment le plus critique : en semaine, matin (6h-12h), vide XX% du temps."

Genere 1 ou 2 exemples de ce format pour repondre a la question.
Ne invente pas de faits historiques, reste dans le format des donnees.

Question : {question}

Exemples de documents :"""

        answer, _ = call_llm_text(
            prompt,
            max_tokens=200,
            temperature=0.1,
            timeout=15,
        )

        return f"{question}\n{answer}"

    except Exception:
        return question

def _bm25_search(query: str, documents: list[dict], top_k: int = 20) -> list[tuple[int, float]]:
    """
    BM25 — recherche lexicale sur les documents.
    Retourne liste de (index, score).
    """
    try:
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [
            doc["text"].lower().split()
            for doc in documents
        ]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
    except Exception:
        return []


def _cosine_search(query_expanded, documents, top_k=20):
    try:
        from src.ai.embeddings import embed_passages, embed_query
        query_embedding = embed_query(query_expanded)

        # Utilise les embeddings precalcules si disponibles
        if documents and "embedding" in documents[0]:
            corpus_embeddings = np.array([d["embedding"] for d in documents])
        else:
            corpus_embeddings = embed_passages([doc["text"] for doc in documents])

        norms = np.linalg.norm(corpus_embeddings, axis=1)
        norm_query = np.linalg.norm(query_embedding)
        similarities = np.dot(corpus_embeddings, query_embedding) / (norms * norm_query + 1e-10)

        ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
    except Exception:
        return []

def _reciprocal_rank_fusion(
    bm25_results: list[tuple[int, float]],
    cosine_results: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    RRF — fusionne les rankings BM25 et cosine.
    Score RRF = 1/(k + rank_bm25) + 1/(k + rank_cosine)
    """
    rrf_scores: dict[int, float] = {}

    for rank, (idx, _) in enumerate(bm25_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

    for rank, (idx, _) in enumerate(cosine_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def _mmr_rerank(
    query_embedding: np.ndarray,
    candidates: list[tuple[int, float]],
    documents: list[dict],
    top_k: int = 8,
    lambda_param: float = 0.7,
) -> list[int]:
    # Déduplique par station pour éviter les doublons
    seen_stations = set()
    candidates_deduped = []
    for idx, score in candidates:
        station = documents[idx]["metadata"].get("station", "")
        if station not in seen_stations:
            seen_stations.add(station)
            candidates_deduped.append((idx, score))
    candidates = candidates_deduped
    """
    MMR — Maximal Marginal Relevance.
    Sélectionne les documents pertinents ET diversifiés.
    lambda_param : 1.0 = pertinence pure, 0.0 = diversité pure
    """
    try:
        candidate_indices = [idx for idx, _ in candidates[:30]]
        if all("embedding" in documents[i] for i in candidate_indices):
            candidate_embeddings = np.array([documents[i]["embedding"] for i in candidate_indices])
        else:
            from src.ai.embeddings import embed_passages
            candidate_embeddings = embed_passages([documents[i]["text"] for i in candidate_indices])

        selected = []
        remaining = list(range(len(candidate_indices)))

        while len(selected) < top_k and remaining:
            if not selected:
                # Premier doc — le plus pertinent
                cosine_scores = np.dot(candidate_embeddings, query_embedding) / (
                    np.linalg.norm(candidate_embeddings, axis=1)
                    * np.linalg.norm(query_embedding)
                    + 1e-10
                )
                best = remaining[np.argmax([cosine_scores[i] for i in remaining])]
            else:
                best_score = -float("inf")
                best = None

                for i in remaining:
                    # Pertinence avec la query
                    relevance = np.dot(candidate_embeddings[i], query_embedding) / (
                        np.linalg.norm(candidate_embeddings[i])
                        * np.linalg.norm(query_embedding)
                        + 1e-10
                    )

                    # Similarité max avec les docs déjà sélectionnés
                    max_sim = max(
                        np.dot(candidate_embeddings[i], candidate_embeddings[s]) / (
                            np.linalg.norm(candidate_embeddings[i])
                            * np.linalg.norm(candidate_embeddings[s])
                            + 1e-10
                        )
                        for s in selected
                    )

                    # Score MMR
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                    if mmr_score > best_score:
                        best_score = mmr_score
                        best = i

            selected.append(best)
            remaining.remove(best)

        return [candidate_indices[i] for i in selected]

    except Exception:
        return [idx for idx, _ in candidates[:top_k]]


def _rerank_with_llm(
    question: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Reranking léger par LLM — évalue la pertinence de chaque doc.
    Utilisé sur les top candidats après MMR.
    """
    if len(candidates) <= top_k:
        return candidates

    try:
        docs_str = "\n\n".join([
            f"[Doc {i+1}] {doc['text'][:300]}"
            for i, doc in enumerate(candidates)
        ])

        prompt = f"""Tu dois classer ces documents par pertinence pour répondre à la question.
Réponds UNIQUEMENT avec un JSON : {{"ordre": [1, 3, 2, ...]}}

Question : {question}

Documents :
{docs_str}

Classe les {len(candidates)} documents du plus au moins pertinent."""

        data, _ = call_llm_json(
            prompt,
            max_tokens=100,
            temperature=0.1,
            timeout=15,
        )

        order = data.get("ordre", [])
        reranked = [candidates[i - 1] for i in order if 0 < i <= len(candidates)]

        return reranked[:top_k]

    except Exception:
        return candidates[:top_k]

def hybrid_search(
    question: str,
    documents: list[dict],
    top_k: int = 8,
    use_hyde: bool = True,
    use_mmr: bool = True,
    use_rerank: bool = False,
) -> tuple[list[dict], dict]:
    """
    Pipeline complet :
    HyDE → BM25 + Cosine → RRF → MMR → Reranking LLM
    Retourne (documents_selectionnés, trace)
    """
    trace = {
        "hyde_query": "",
        "bm25_top5": [],
        "cosine_top5": [],
        "rrf_top5": [],
        "mmr_selected": [],
        "final_docs": [],
        "techniques_used": [],
    }

    if not documents:
        return [], trace
    # Station(s) citée(s) dans la question (« Bastille » → les 5 stations Bastille) : leurs profils
    # sont exactement les documents utiles. La recherche hybride (et MMR, qui écarterait des
    # profils trop semblables) sert aux questions sans station précise.
    from src.data.filters import stations_in_question

    by_station = {d["metadata"].get("station", ""): d for d in documents}
    cited = stations_in_question(question, by_station)
    if cited:
        station_docs = [by_station[name] for name in cited[:top_k]]
        trace["techniques_used"] = ["Station citée"]
        trace["final_docs"] = [f"{d['metadata']['station']} ({d['metadata']['run_at']})" for d in station_docs]
        return station_docs, trace

    # Étape 1 — HyDE
    if use_hyde and GROQ_API_KEY:
        expanded_query = _hyde_expand_query(question)
        trace["hyde_query"] = expanded_query
        trace["techniques_used"].append("HyDE")
    else:
        expanded_query = question

    # Étape 2 — BM25
    bm25_results = _bm25_search(question, documents, top_k=30)
    trace["bm25_top5"] = [
        documents[idx]["metadata"]["station"]
        for idx, _ in bm25_results[:5]
    ]
    trace["techniques_used"].append("BM25")

    # Étape 3 — Cosine similarity
    cosine_results = _cosine_search(expanded_query, documents, top_k=30)
    trace["cosine_top5"] = [
        documents[idx]["metadata"]["station"]
        for idx, _ in cosine_results[:5]
    ]
    trace["techniques_used"].append("Cosine Similarity")

    # Étape 4 — RRF fusion
    fused = _reciprocal_rank_fusion(bm25_results, cosine_results)
    trace["rrf_top5"] = [
        documents[idx]["metadata"]["station"]
        for idx, _ in fused[:5]
    ]
    trace["techniques_used"].append("RRF Fusion")

    # Étape 5 — MMR
    if use_mmr:
        try:
            from src.ai.embeddings import embed_query
            query_embedding = embed_query(expanded_query)
            mmr_indices = _mmr_rerank(query_embedding, fused, documents, top_k=min(15, len(fused)))
            candidates = [documents[i] for i in mmr_indices]
            trace["mmr_selected"] = [d["metadata"]["station"] for d in candidates[:5]]
            trace["techniques_used"].append("MMR")
        except Exception:
            candidates = [documents[idx] for idx, _ in fused[:15]]
    else:
        candidates = [documents[idx] for idx, _ in fused[:15]]

    # Étape 6 — Reranking LLM
    if use_rerank and GROQ_API_KEY and len(candidates) > top_k:
        final_docs = _rerank_with_llm(question, candidates, top_k=top_k)
        trace["techniques_used"].append("LLM Reranking")
    else:
        final_docs = candidates[:top_k]

    trace["final_docs"] = [
        f"{d['metadata']['station']} ({d['metadata']['run_at']})"
        for d in final_docs
    ]

    return final_docs, trace


def load_precomputed_documents() -> list[dict]:
    """Documents + vecteurs calculés chaque jour par le pipeline (gold/station_profile_embeddings)."""
    from src.ai.embeddings import EMBEDDING_MODEL
    from src.ai.profile_docs import documents_from_frame
    from src.data.lake import station_embeddings

    return documents_from_frame(station_embeddings(), EMBEDDING_MODEL)


def build_rag_index(profiles: pd.DataFrame = None):
    """
    Index RAG : un document par station. Les vecteurs viennent du pipeline (aucun calcul lourd
    dans l'application) ; en secours seulement (gold pas encore calculé), on encode localement.
    """
    documents = load_precomputed_documents()
    if documents:
        return documents, len(documents)

    from src.ai.profile_docs import station_documents

    if profiles is None or profiles.empty:
        profiles = _load_profiles()
    documents = station_documents(profiles)
    if not documents:
        return None, 0

    try:
        from src.ai.embeddings import embed_passages
        embeddings = embed_passages([d["text"] for d in documents])
        for doc, embedding in zip(documents, embeddings):
            doc["embedding"] = embedding
    except Exception:
        pass

    return documents, len(documents)


def ask_rag(question: str, documents) -> tuple[str, dict]:
    """
    RAG avancé avec pipeline complet.
    Retourne (réponse, trace).
    """
    if not documents:
        return "Index RAG non disponible : aucun profil de station dans le lakehouse.", {}

    # Pipeline hybrid search
    final_docs, search_trace = hybrid_search(question, documents)

    if not final_docs:
        return "Aucune donnee pertinente trouvee.", search_trace

    # Construction du contexte avec citations
    context_parts = []
    for i, doc in enumerate(final_docs):
        meta = doc["metadata"]
        citation = (
            f"[Source {i+1}: {meta['station']} — {meta['run_at']} — "
            f"snapshot {meta['snapshot_id']}]"
        )
        context_parts.append(f"{citation}\n{doc['text']}")

    context = "\n\n".join(context_parts)

    prompt = f"""Tu es un expert en mobilite urbaine specialise dans le reseau Velib Paris.
Reponds en francais de maniere concise et precise.

INSTRUCTIONS :
- Utilise UNIQUEMENT les donnees ci-dessous
- Pour chaque affirmation, cite la source entre crochets [Source N]
- Si une information n'est pas dans les donnees, dis-le clairement
- Ne fais pas d'hypotheses sur des donnees non fournies
- Commence par une reponse directe et coherente avec la suite : si les donnees ne couvrent pas
  la periode demandee, dis-le des la premiere phrase (ni « oui » ni « non »), puis donne ce que
  les donnees montrent sur les periodes disponibles
- Si plusieurs stations correspondent (par exemple plusieurs stations « Bastille »), reponds pour chacune

Donnees historiques :
{context}

Question : {question}
"""

    try:
        answer, tokens = call_llm_text(
            prompt,
            max_tokens=600,
            temperature=0.2,
            timeout=30,
        )

        search_trace["tokens"] = tokens
        search_trace["context"] = context

        search_trace["sources_used"] = [
            f"{d['metadata']['station']} — {d['metadata']['run_at']}"
            for d in final_docs
        ]

        # Verification pertinence
        sources_str = "\n".join([
            f"- {d['metadata']['station']} ({d['metadata']['run_at']}): {d['text'][:200]}"
            for d in final_docs
        ])

        relevance_score, relevance_explanation = verify_relevance(
            question=question,
            answer=answer,
            context=sources_str,
            evaluator_name="système RAG Vélib",
        )

        search_trace["relevance_score"] = relevance_score
        search_trace["relevance_explanation"] = relevance_explanation

        return answer, search_trace

    except Exception as e:
        return f"Erreur LLM : {e}", search_trace


