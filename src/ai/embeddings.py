"""
embeddings.py — Modèle d'embeddings unique pour le RAG et Qdrant.

Par défaut : intfloat/multilingual-e5-small (gratuit, CPU, 384 dimensions,
entraîné pour la recherche de documents, bon en français).
Le modèle est chargé une seule fois par processus.

Les modèles E5 attendent un préfixe : « query: » pour la question,
« passage: » pour les documents indexés.
"""
import os
import threading
from functools import lru_cache

import numpy as np

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")


def _uses_e5_prefixes() -> bool:
    return "e5" in EMBEDDING_MODEL.lower()


_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def _model():
    with _lock:  # un seul chargement, même si le préchargement est en cours
        return _load()


def warm_up() -> None:
    """
    Charge le modèle en arrière-plan (10 à 40 s à froid : import + poids) pour que la première
    question de l'utilisateur ne paie pas ce temps. Sans effet si c'est déjà fait.
    """
    if _load.cache_info().currsize or getattr(warm_up, "started", False):
        return
    warm_up.started = True
    threading.Thread(target=_model, daemon=True, name="embeddings-warm-up").start()


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embeddings normalisés des documents (le produit scalaire vaut la similarité cosinus)."""
    if _uses_e5_prefixes():
        texts = [f"passage: {t}" for t in texts]
    return _model().encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)


def embed_query(text: str) -> np.ndarray:
    """Embedding normalisé d'une question."""
    if _uses_e5_prefixes():
        text = f"query: {text}"
    return _model().encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
