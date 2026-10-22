"""
data_loader.py — Snapshot temps réel depuis l'API Open Data Vélib' (mode Temps réel et onglets IA).
"""
import time

import pandas as pd
import requests


API_V21_EXPORT = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/"
    "velib-disponibilite-en-temps-reel/exports/json"
)


def _get_with_retries(url: str, attempts: int = 3) -> list[dict]:
    """GET JSON avec quelques tentatives : l'API coupe parfois la connexion ou renvoie un 5xx passager."""
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == attempts:
                raise
            time.sleep(1.5 * attempt)
    return []


def fetch_api_snapshot() -> pd.DataFrame:
    """
    État de toutes les stations en un seul appel (export API v2.1).
    Volontairement sans cache : l'utilisateur choisit quand rafraîchir.
    """
    rows = []
    for rec in _get_with_retries(API_V21_EXPORT):
        bikes = int(rec.get("numbikesavailable") or 0)
        docks = int(rec.get("numdocksavailable") or 0)
        geo = rec.get("coordonnees_geo") or {}
        rows.append({
            "station_id":        rec.get("stationcode"),
            "name":              " ".join(str(rec.get("name", "")).split()).upper(),
            "numbikesavailable": bikes,
            "mechanical":        int(rec.get("mechanical") or 0),
            "ebike":             int(rec.get("ebike") or 0),
            "numdocksavailable": docks,
            "is_empty":          bikes == 0,
            "is_full":           docks == 0,
            "bike_ratio":        round(bikes / max(bikes + docks, 1), 2),
            "lat":               geo.get("lat"),
            "lon":               geo.get("lon"),
        })
    return pd.DataFrame(rows)
