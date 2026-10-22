"""
tools.py — Logique métier des outils de l'IA : état d'une station, du réseau, anomalies,
historique, prévision et rapport PDF. Chaque fonction renvoie un dict prêt à passer au LLM.
"""

import json
from datetime import datetime

import pandas as pd
import pytz

from src.ai.rag import ask_rag
from src.ai.vector_store import semantic_search
from src.reports.report_generator import generate_pdf_report


PARIS_TZ = pytz.timezone("Europe/Paris")
# Le nombre total est transmis ; seul un échantillon de noms part au LLM
MAX_LISTED_STATIONS = 15


def current_moment() -> dict:
    """Heure de Paris et créneau, avec les mêmes périodes que les profils historiques (dbt)."""
    now = datetime.now(PARIS_TZ)
    period = ("matin" if 6 <= now.hour < 12 else "après-midi" if now.hour < 18
              else "soir" if now.hour < 22 else "nuit") if now.hour >= 6 else "nuit"
    days = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    return {"heure_paris": now.strftime("%H:%M"), "jour": days[now.weekday()],
            "type_de_jour": "week-end" if now.weekday() >= 5 else "semaine", "periode": period}


def get_station_info_tool(station_name: str, df: pd.DataFrame) -> dict:
    """État temps réel de la station la plus proche du nom demandé, et les autres candidates."""
    if df is None or df.empty:
        return {"error": "Données temps réel non disponibles."}

    from src.data.filters import normalize_name, search_stations, stations_in_question

    # Recherche insensible aux accents et à l'ordre des mots (« republique » → « RÉPUBLIQUE »)
    results = search_stations(df, station_name)
    # Repli : la requête peut être une phrase entière où le nom de station est noyé
    if results.empty:
        cited = stations_in_question(station_name, df["name"].dropna().unique())
        results = df[df["name"].isin(cited)]
    if results.empty:
        return {"error": f"Station '{station_name}' non trouvée."}

    # Nom exact d'abord, sinon le nom le plus court (le plus proche de la recherche)
    results = results.iloc[results["name"].str.len().argsort()]
    exact = results[results["name"].map(normalize_name) == normalize_name(station_name)]
    row = (exact if not exact.empty else results).iloc[0]
    others = [n for n in results["name"].tolist() if n != row["name"]]

    return {
        "moment": current_moment(),
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
        **({"autres_stations_correspondantes": others[:MAX_LISTED_STATIONS], "nombre_correspondances": len(results),
            "remarque": "Plusieurs stations correspondent : préciser le nom pour en choisir une autre."}
           if others else {}),
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
    """Règles simples sur le snapshot : stations vides ou presque pleines, part d'électriques trop basse."""
    if df is None or df.empty:
        return {"error": "Données temps réel non disponibles."}

    now = datetime.now(PARIS_TZ)
    heure = now.hour
    anomalies = []

    # Une station vide pèse plus aux heures de pointe : on la signale à part
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


def search_history_rag_tool(question: str, documents) -> tuple[str, dict]:
    """Réponse rédigée par le RAG sur les profils des stations (utilisée par l'assistant à routage)."""
    if documents:
        return ask_rag(question, documents)
    return (
        "Historique indisponible : le lakehouse n'a pas encore de profils de stations "
        "(le job quotidien dbt n'est peut-être pas encore passé).",
        {"techniques_used": [], "relevance_score": None, "relevance_explanation": ""},
    )


def search_history_tool(query: str, rag_documents=None, qdrant_client=None, rerank: bool = True) -> dict:
    """
    Extraits bruts des profils historiques, sans rédaction : le LLM appelant s'en charge.
    RAG hybride en priorité ; Qdrant ne sert que de repli si l'index RAG n'est pas chargé.
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


def forecast_tool(station_name: str | None, hours_ahead: int | None = None, at_hour: int | None = None) -> dict:
    """Prévision de 1 à 6 h : risque vide / pleine, vélos attendus avec intervalle (modèles LightGBM)."""
    from src.ml.forecast import forecast_for_station
    return forecast_for_station(station_name or "", hours_ahead=hours_ahead, at_hour=at_hour)


def generate_report_tool(df: pd.DataFrame) -> dict:
    """Rapport PDF en mémoire ; l'interface Streamlit propose ensuite le téléchargement."""
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
    """Résultat d'outil en JSON pour le LLM ; le PDF est remplacé par sa taille."""
    try:
        safe_data = data.copy() if isinstance(data, dict) else data

        if isinstance(safe_data, dict) and "pdf_bytes" in safe_data:
            safe_data = safe_data.copy()
            safe_data["pdf_bytes"] = f"<{len(data['pdf_bytes'])} bytes>"

        return json.dumps(safe_data, ensure_ascii=False, indent=2)

    except Exception:
        return str(data)