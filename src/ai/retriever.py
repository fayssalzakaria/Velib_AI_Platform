"""
retriever.py — Le RAG hybride du projet sous forme de retriever LangChain standard.

La logique reste celle de rag.hybrid_search (HyDE, BM25 + cosinus, RRF, MMR, reranking LLM
optionnel) : LangChain ne fait qu'exposer une interface commune (invoke, chaînes LCEL, agents).
"""
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict


class ProfileRetriever(BaseRetriever):
    """Profils historiques des stations les plus pertinents pour une question."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    documents: list[Any]
    k: int = 5
    use_hyde: bool = True
    use_rerank: bool = False

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        from src.ai.rag import hybrid_search

        docs, trace = hybrid_search(query, self.documents, top_k=self.k, use_hyde=self.use_hyde,
                                    use_mmr=True, use_rerank=self.use_rerank)
        techniques = trace.get("techniques_used", [])
        return [
            Document(page_content=d["text"], metadata={**d["metadata"], "techniques": techniques})
            for d in docs
        ]
