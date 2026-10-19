"""
extract.py — Récupération brute de l'API Open Data Vélib' (v2.1, endpoint d'export).
"""
import json
import logging
import time

import requests

from pipeline.settings import API_URL

logger = logging.getLogger(__name__)

# En dessous, la réponse est jugée incomplète (l'API renvoie parfois des stations sans code
# ni nom pendant sa mise à jour) : on réessaie au lieu d'écrire un snapshot vide
MIN_VALID_SHARE = 0.9
# Identifie le projet auprès de l'API (certaines plateformes filtrent les requêtes anonymes des serveurs cloud)
HEADERS = {"User-Agent": "velib-ai-platform/1.0 (+https://github.com/fayssalzakaria/Velib_AI_Platform)",
           "Accept": "application/json"}


def _valid_share(records: list) -> float:
    valid = sum(1 for r in records if isinstance(r, dict) and r.get("stationcode") and r.get("name"))
    return valid / len(records)


def fetch_raw(timeout: int = 60, retries: int = 3, wait_seconds: int = 20) -> list[dict]:
    """Retourne la liste brute des stations, telle que renvoyée par l'API."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(API_URL, timeout=timeout, headers=HEADERS)
            response.raise_for_status()
            records = response.json()
            if not isinstance(records, list) or not records:
                raise ValueError("Réponse API vide ou inattendue.")
            share = _valid_share(records)
            if share < MIN_VALID_SHARE:
                sample = json.dumps(records[:2], ensure_ascii=False)[:400]
                raise ValueError(f"Réponse API incomplète : {share:.0%} des stations ont un code et un nom "
                                 f"({len(records)} éléments, {len(response.content):,} octets). Extrait : {sample}")
            logger.info(f"{len(records)} stations récupérées (tentative {attempt})")
            return records
        except (requests.RequestException, ValueError) as e:
            last_error = e
            logger.warning(f"Échec API Vélib' (tentative {attempt}/{retries}) : {e}")
            if attempt < retries:
                time.sleep(wait_seconds)
    raise RuntimeError(f"API Vélib' indisponible : {last_error}")
