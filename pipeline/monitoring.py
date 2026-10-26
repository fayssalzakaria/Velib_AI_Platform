"""
monitoring.py — Suivi du modèle de prévision en production.

Chaque heure (après l'ingestion) : la prévision de l'heure suivante est enregistrée avec ses
features dans lake/monitoring/predictions (Delta).

Chaque jour : les prévisions sont comparées à ce qui s'est réellement passé, puis écrites dans le gold :
- model_monitoring_daily : AUC, Brier, risque moyen prédit vs taux de vide observé, par jour et par version
- model_calibration      : quand le modèle annonce 30 %, la station est-elle vide 30 % du temps ?
- model_drift            : dérive des features (PSI) entre la production récente et l'entraînement
"""
import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from deltalake import DeltaTable, write_deltalake

from pipeline.lake import _delta_uri, _filesystem
from pipeline.settings import delta_storage_options, gold_uri, lake_root

logger = logging.getLogger(__name__)

PSI_WARNING, PSI_ALERT = 0.1, 0.25
# Exclues du calcul de dérive : sur une fenêtre de 24 h, le calendrier ne couvre qu'un ou deux jours
# (différent de l'entraînement par construction) et past_n augmente mécaniquement avec le temps.
DRIFT_EXCLUDED = {"horizon", "hour_target", "weekday_target", "is_weekend_target",
                  "is_public_holiday", "is_school_holiday", "past_n"}


def predictions_uri() -> str:
    return f"{lake_root()}/monitoring/predictions"


# ---------------------------------------------------------------- chaque heure

PREDICTION_COLUMNS = ["risk_empty", "risk_full", "bikes_q10", "bikes_q50", "bikes_q90"]


def log_predictions() -> int:
    """Enregistre les prévisions de tous les horizons (si des modèles sont publiés). Renvoie le nombre de lignes."""
    import src.ml.forecast as forecast
    from src.ml.features import FEATURES

    forecast._cache.clear()
    predictions, info = forecast._compute_predictions(use_stored=False)
    if predictions is None:
        logger.info(f"Pas de prévision à enregistrer : {info.get('reason')}")
        return 0

    frame = predictions[["station_id", *PREDICTION_COLUMNS, *FEATURES]].copy()
    frame["snapshot_at"] = info["snapshot_at"].tz_convert("UTC")
    frame["predicted_for"] = predictions["ts_target"].dt.tz_convert("UTC")
    frame["model_version"] = info["model"]["version"]
    frame["date"] = info["snapshot_at"].strftime("%Y-%m-%d")

    uri, options = _delta_uri(predictions_uri()), delta_storage_options()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if not DeltaTable.is_deltatable(uri, storage_options=options):
        write_deltalake(uri, table, partition_by=["date"], storage_options=options)
        return len(frame)
    metrics = (
        DeltaTable(uri, storage_options=options)
        .merge(source=table, source_alias="s", target_alias="t",
               predicate=f"t.date = '{frame['date'].iloc[0]}' AND t.snapshot_at = s.snapshot_at "
                         "AND t.station_id = s.station_id AND t.horizon = s.horizon")
        .when_not_matched_insert_all()
        .execute()
    )
    inserted = int(metrics.get("num_target_rows_inserted", 0))
    logger.info(f"Prévisions enregistrées à partir du snapshot de {info['snapshot_at']:%d/%m %H:%M} : {inserted} lignes")
    return inserted


# ---------------------------------------------------------------- chaque jour

def psi(reference: dict, values: np.ndarray) -> float:
    """Population Stability Index entre la distribution d'entraînement et les valeurs récentes."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan")
    edges = np.asarray(reference["edges"], dtype=float)
    expected = np.asarray(reference["proportions"], dtype=float)
    counts, _ = np.histogram(np.clip(values, edges[0], edges[-1]), bins=edges)
    actual = counts / counts.sum()
    eps = 1e-4
    expected, actual = np.clip(expected, eps, None), np.clip(actual, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _daily_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    """Par jour et par horizon : AUC vide/pleine, erreur sur les vélos, couverture de l'intervalle."""
    from sklearn.metrics import roc_auc_score

    def auc(y, p):
        mask = ~np.isnan(p)
        y, p = y[mask], p[mask]
        return roc_auc_score(y, p) if len(y) and 0 < y.sum() < len(y) else np.nan

    rows = []
    for (date, horizon, version), group in scored.groupby(["day", "horizon", "model_version"]):
        bikes = group["actual_bikes"].to_numpy()
        rows.append({
            "date": date, "horizon": int(horizon), "model_version": version, "predictions": len(group),
            "auc_empty": auc(group["actual_empty"].to_numpy(), group["risk_empty"].to_numpy(dtype=float)),
            "auc_full": auc(group["actual_full"].to_numpy(), group["risk_full"].to_numpy(dtype=float)),
            "mae_bikes": np.nanmean(np.abs(bikes - group["bikes_q50"].to_numpy(dtype=float))),
            "coverage_80": np.nanmean((bikes >= group["bikes_q10"]) & (bikes <= group["bikes_q90"])),
            "observed_empty_rate": group["actual_empty"].mean(),
        })
    return pd.DataFrame(rows)


def _calibration(scored: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Risque annoncé vs fréquence observée (vide et pleine), sur les 7 derniers jours."""
    recent = scored[scored["predicted_for"] >= scored["predicted_for"].max() - pd.Timedelta(days=7)]
    frames = []
    for family, risk, actual in (("vide", "risk_empty", "actual_empty"), ("pleine", "risk_full", "actual_full")):
        part = recent.dropna(subset=[risk])
        if part.empty:
            continue
        bucket = np.minimum((part[risk] * bins).astype(int), bins - 1)
        frames.append(
            part.assign(bucket=bucket).groupby("bucket")
            .agg(mean_predicted=(risk, "mean"), observed_rate=(actual, "mean"), predictions=(risk, "size"))
            .reset_index().assign(question=family)
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _drift(predictions: pd.DataFrame, reference: dict) -> pd.DataFrame:
    recent = predictions[predictions["predicted_for"] >= predictions["predicted_for"].max() - pd.Timedelta(hours=24)]
    rows = []
    for feature, ref in reference.items():
        if feature not in recent or feature in DRIFT_EXCLUDED:
            continue
        value = psi(ref, recent[feature].to_numpy(dtype=float))
        status = "stable" if value < PSI_WARNING else "à surveiller" if value < PSI_ALERT else "dérive"
        rows.append({"feature": feature, "psi": round(value, 4), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def _write_gold(name: str, frame: pd.DataFrame) -> None:
    import os

    filesystem, path = _filesystem(gold_uri(name))
    if not gold_uri(name).startswith("s3://"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with filesystem.open_output_stream(path, compression=None) as out:
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), out)


def evaluate_production(silver: pd.DataFrame | None = None) -> dict:
    """Compare les prévisions enregistrées à ce qui s'est passé, puis écrit les tables de monitoring."""
    from pipeline.forecast import MODEL_NAME, _read_silver
    from src.ml.storage import model_uri, read_json

    uri, options = _delta_uri(predictions_uri()), delta_storage_options()
    if not DeltaTable.is_deltatable(uri, storage_options=options):
        return {"status": "skipped", "reason": "aucune prévision enregistrée"}
    predictions = DeltaTable(uri, storage_options=options).to_pandas()
    predictions["predicted_for"] = pd.to_datetime(predictions["predicted_for"], utc=True)

    silver = _read_silver() if silver is None else silver
    actual = pd.DataFrame({
        "station_id": silver["station_id"].astype(str),
        "predicted_for": pd.to_datetime(silver["ingested_at"], utc=True),
        "actual_empty": (silver["numbikesavailable"] == 0).astype(float),
        "actual_full": (silver["numdocksavailable"] == 0).astype(float),
        "actual_bikes": silver["numbikesavailable"].astype(float),
    })
    scored = predictions.merge(actual, on=["station_id", "predicted_for"], how="inner")
    scored["day"] = scored["predicted_for"].dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
    if scored.empty:
        return {"status": "skipped", "reason": "aucune prévision dont l'heure est passée"}

    daily = _daily_metrics(scored)
    _write_gold("model_monitoring_daily", daily)
    _write_gold("model_calibration", _calibration(scored))

    published = read_json(model_uri(MODEL_NAME, "latest.json")) or {}
    drift = _drift(predictions, published.get("reference_distribution", {}))
    if not drift.empty:
        _write_gold("model_drift", drift)

    result = {
        "status": "evaluated",
        "scored_predictions": len(scored),
        "last_day_auc_empty_1h": None if daily.empty else round(float(
            daily[daily["horizon"] == 1]["auc_empty"].iloc[-1]), 4) if (daily["horizon"] == 1).any() else None,
        "features_in_drift": drift[drift["status"] == "dérive"]["feature"].tolist() if not drift.empty else [],
    }
    logger.info(f"Monitoring : {result}")
    return result
