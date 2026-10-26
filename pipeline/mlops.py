"""
mlops.py — Faut-il réentraîner le modèle de prévision ? Décision prise chaque jour après le monitoring.

Réentraînement si l'une de ces règles est vraie (la première trouvée est la raison tracée) :
- aucun modèle en production ;
- le modèle en production a plus de MAX_MODEL_AGE_DAYS jours ;
- le monitoring signale une dérive (PSI > 0,25) sur au moins une feature ;
- l'AUC « vide » observée en production à 1 h a chuté de plus de AUC_DROP sous celle du test.

Le nouveau modèle passe ensuite par la même porte que l'entraînement manuel (baselines, puis
champion / challenger) : un réentraînement automatique ne peut pas dégrader la production.

    python -m pipeline.run retrain
"""
import logging
import os
from datetime import datetime, timezone

import pandas as pd
import pyarrow.parquet as pq
from pyarrow import fs as pa_fs

from pipeline.lake import _filesystem
from pipeline.settings import gold_uri

logger = logging.getLogger(__name__)

MAX_MODEL_AGE_DAYS = int(os.environ.get("FORECAST_MAX_AGE_DAYS", 7))
AUC_DROP = 0.05


def _read_gold(name: str) -> pd.DataFrame:
    """Table de monitoring écrite par pipeline/monitoring.py ; vide si elle n'existe pas encore."""
    filesystem, path = _filesystem(gold_uri(name))
    if filesystem.get_file_info(path).type == pa_fs.FileType.NotFound:
        return pd.DataFrame()
    with filesystem.open_input_file(path) as source:
        return pq.read_table(source).to_pandas()


def retrain_reason(now: datetime | None = None) -> str | None:
    """Raison de réentraîner, ou None si le modèle en production est encore bon."""
    from pipeline.forecast import MODEL_NAME
    from src.ml.storage import model_uri, read_json

    champion = read_json(model_uri(MODEL_NAME, "latest.json"))
    if not champion:
        return "aucun modèle en production"

    now = now or datetime.now(timezone.utc)
    age = now - pd.Timestamp(champion["trained_at"]).to_pydatetime()
    if age.days >= MAX_MODEL_AGE_DAYS:
        return f"modèle en production depuis {age.days} jours (limite {MAX_MODEL_AGE_DAYS})"

    drift = _read_gold("model_drift")
    if not drift.empty and (drift["status"] == "dérive").any():
        return "dérive des données : " + ", ".join(drift.loc[drift["status"] == "dérive", "feature"])

    daily = _read_gold("model_monitoring_daily")
    if not daily.empty and "empty" in champion.get("promoted", []):
        live = daily[(daily["model_version"] == champion["version"]) & (daily["horizon"] == 1)].dropna(subset=["auc_empty"])
        test_auc = champion["families"]["empty"]["evaluation"]["model"].get("1h", {}).get("auc")
        if not live.empty and test_auc is not None:
            last = live.sort_values("date").iloc[-1]
            if last["auc_empty"] < test_auc - AUC_DROP:
                return (f"performance en baisse : AUC vide à 1 h de {last['auc_empty']:.2f} le {last['date']} "
                        f"contre {test_auc:.2f} au test")
    return None


def retrain_if_needed(min_days: int | None = None) -> dict:
    """Réentraîne seulement si une règle le demande ; sinon ne touche à rien."""
    from pipeline.forecast import train_forecast

    reason = retrain_reason()
    if reason is None:
        logger.info("Réentraînement inutile : le modèle en production est récent et stable.")
        return {"status": "up_to_date"}
    logger.info(f"Réentraînement automatique : {reason}")
    return {"reason": reason, **train_forecast(min_days=min_days, trigger="automatique", reason=reason)}
