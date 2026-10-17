"""
tools.py — Outils métier utilisables par l'assistant IA Vélib
"""

import json
from datetime import datetime

import pandas as pd
import pytz

from src.ai.rag import ask_rag_with_qdrant_context
from src.ai.vector_store import semantic_search
from src.reports.report_generator import generate_pdf_report


PARIS_TZ = pytz.timezone("Europe/Paris")
# Le nombre total est transmis ; seul un échantillon de noms part au LLM
MAX_LISTED_STATIONS = 15


def get_station_info_tool(station_name: str, df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"error": "Données temps réel non disponibles."}

    from src.data.filters import search_stations

    # Recherche insensible aux accents et à l'ordre des mots (« republique » → « RÉPUBLIQUE »)
    results = search_stations(df, station_name)

    if results.empty:
        return {"error": f"Station '{station_name}' non trouvée."}

    row = results.iloc[0]

    return {
        "station": row["name"],
        "velos_disponibles": int(row["numbikesavailable"]),
        "velos_electriques": int(row["ebike"]),
        "velos_mecaniques": int(row["mechanical"]),
        "bornes_disponibles": int(row["numdocksavailable"]),
        "taux_remplissage": f"{float(row['bike_ratio']):.0%}",
        "etat": (
            "vide"
            if row["is_empty"]
            else "pleine"
            if row["is_full"]
            else "disponible"
        ),
    }


def get_network_stats_tool(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"error": "Données temps réel non disponibles."}

    return {
        "source": "temps_reel",
        "stations_actives": len(df),
        "velos_disponibles": int(df["numbikesavailable"].sum()),
        "velos_electriques": int(df["ebike"].sum()),
        "velos_mecaniques": int(df["mechanical"].sum()),
        "bornes_disponibles": int(df["numdocksavailable"].sum()),
        "taux_remplissage_moyen": f"{df['bike_ratio'].mean():.0%}",
        "stations_vides": int(df["is_empty"].sum()),
        "stations_pleines": int(df["is_full"].sum()),
        "top3_mieux_fournies": df.nlargest(3, "numbikesavailable")["name"].tolist(),
        "top3_plus_vides": df.nsmallest(3, "numbikesavailable")["name"].tolist(),
    }


def detect_anomalies_tool(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"error": "Données temps réel non disponibles."}

    now = datetime.now(PARIS_TZ)
    heure = now.hour
    anomalies = []

    if 7 <= heure <= 9 or 17 <= heure <= 19:
        vides = df[df["is_empty"]]["name"].tolist()
        if vides:
            anomalies.append(
                {
                    "type": "stations_vides_heure_pointe",
                    "heure": heure,
                    "stations": vides[:MAX_LISTED_STATIONS],
                    "count": len(vides),
                }
            )

    presque_pleines = df[df["numdocksavailable"] <= 2]
    if not presque_pleines.empty:
        anomalies.append(
            {
                "type": "stations_presque_pleines",
                "stations": presque_pleines["name"].tolist()[:MAX_LISTED_STATIONS],
                "count": len(presque_pleines),
            }
        )

    vides = df[df["is_empty"]]
    if not vides.empty:
        anomalies.append(
            {
                "type": "stations_vides",
                "stations": vides["name"].tolist()[:MAX_LISTED_STATIONS],
                "count": len(vides),
            }
        )

    total_ebike = int(df["ebike"].sum())
    total_mechanical = int(df["mechanical"].sum())
    total = total_ebike + total_mechanical
    ratio_ebike = total_ebike / max(total, 1)

    if ratio_ebike < 0.2:
        anomalies.append(
            {
                "type": "manque_velos_electriques",
                "ratio_actuel": f"{ratio_ebike:.0%}",
                "seuil": "20%",
            }
        )

    return {
        "source": "temps_reel",
        "heure_analyse": f"{heure}h",
        "total_stations": len(df),
        "anomalies": anomalies if anomalies else ["Aucune anomalie détectée"],
    }


def search_history_rag_tool(
    question: str,
    documents,
    qdrant_client=None,
    qdrant_docs=None,
) -> tuple[str, dict]:
    """
    Utilise le moteur RAG historique existant.
    """
    qdrant_docs = qdrant_docs or []

    return ask_rag_with_qdrant_context(
        question,
        documents,
        qdrant_docs,
    )


def search_history_tool(query: str, rag_documents=None, qdrant_client=None, rerank: bool = True) -> dict:
    """
    Recherche dans l'historique (profils des stations) : RAG hybride via le retriever LangChain,
    ou recherche sémantique Qdrant si l'index RAG n'est pas chargé.
    """
    if rag_documents:
        from src.ai.retriever import ProfileRetriever

        docs = ProfileRetriever(documents=rag_documents, k=5, use_rerank=rerank).invoke(query)
        return {
            "resultats": [d.page_content for d in docs],
            "sources": [f"{d.metadata['station']} — {d.metadata['run_at']}" for d in docs],
            "techniques_utilisees": docs[0].metadata.get("techniques", []) if docs else [],
        }
    if qdrant_client is not None:
        return {"resultats": semantic_search(query, qdrant_client, n_results=5), "techniques_utilisees": ["Qdrant"]}
    return {"error": "Historique non disponible"}


def semantic_search_tool(
    question: str,
    qdrant_client,
    n_results: int = 8,
) -> dict:
    if qdrant_client is None:
        return {
            "error": "Qdrant non configuré.",
            "documents": [],
        }

    docs = semantic_search(question, qdrant_client, n_results=n_results)

    return {
        "source": "qdrant",
        "documents": docs,
        "count": len(docs),
    }


def forecast_tool(station_name: str | None, hours_ahead: int | None = None, at_hour: int | None = None) -> dict:
    """Prévision de 1 à 6 h : risque vide / pleine, vélos attendus avec intervalle (modèles LightGBM)."""
    from src.ml.forecast import forecast_for_station
    return forecast_for_station(station_name or "", hours_ahead=hours_ahead, at_hour=at_hour)


def generate_report_tool(df: pd.DataFrame) -> dict:
    """
    Génère un rapport PDF en mémoire.
    Le rendu Streamlit gèrera le téléchargement si besoin.
    """
    if df is None or df.empty:
        return {"error": "Impossible de générer un rapport sans données."}

    pdf_bytes = generate_pdf_report(df)

    return {
        "status": "success",
        "message": "Rapport PDF généré.",
        "size_bytes": len(pdf_bytes),
        "pdf_bytes": pdf_bytes,
    }


def tool_result_to_text(data) -> str:
    """
    Convertit un résultat tool en texte lisible par le LLM.
    """
    try:
        safe_data = data.copy() if isinstance(data, dict) else data

        if isinstance(safe_data, dict) and "pdf_bytes" in safe_data:
            safe_data = safe_data.copy()
            safe_data["pdf_bytes"] = f"<{len(data['pdf_bytes'])} bytes>"

        return json.dumps(safe_data, ensure_ascii=False, indent=2)

    except Exception:
        return str(data)