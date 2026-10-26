"""
forecast.py — Prévisions de 1 à 6 heures, côté application et agents : risque d'être vide,
risque d'être pleine, nombre de vélos avec un intervalle de 80 % (quantiles 10 et 90).

Les modèles publiés (pipeline/forecast.py) sont lus dans le registre du lakehouse.
Serving en batch d'abord : à chaque collecte horaire, le pipeline calcule et enregistre les
prévisions (pipeline/monitoring.py) ; l'app sert ces mêmes chiffres, ceux que le monitoring juge.
Si elles manquent (modèle tout juste publié, collecte en échec), calcul à la demande avec les
features de src/ml/features.py, comme à l'entraînement.
"""
import time

import numpy as np
import pandas as pd

from src.data import lake
from src.ml.features import FEATURES, HORIZONS, PARIS_TZ, SILVER_COLUMNS, serving_frame
from src.ml.storage import model_uri, read_json, read_text

MODEL_NAME = "forecast"
MODEL_FILES = {"empty": ["model"], "full": ["model"], "bikes": ["q10", "q50", "q90"]}
# Le silver n'avance qu'une fois par heure : inutile de recalculer à chaque question
_CACHE_SECONDS = 600
_cache: dict = {}  # prédictions et boosters, partagés par toutes les sessions du processus


def model_status() -> dict:
    """Dernier entraînement (status.json) et modèles publiés (latest.json), s'ils existent."""
    return {
        "status": read_json(model_uri(MODEL_NAME, "status.json")) or {},
        "published": read_json(model_uri(MODEL_NAME, "latest.json")),
    }


def _booster(version: str, family: str, key: str):
    """Modèle LightGBM lu une fois par version ; une nouvelle publication change la clé de cache."""
    import lightgbm as lgb

    cache_key = ("booster", version, family, key)
    if cache_key not in _cache:
        _cache[cache_key] = lgb.Booster(model_str=read_text(model_uri(MODEL_NAME, version, f"{family}_{key}.txt")))
    return _cache[cache_key]


def _snapshot(ts) -> pd.DataFrame:
    """Snapshot silver à l'instant exact `ts` ; le filtre sur date limite les partitions lues."""
    columns = ", ".join(SILVER_COLUMNS)
    return lake.query(
        f"SELECT {columns} FROM {lake._silver()} WHERE date >= ? AND ingested_at = ?",
        [(ts - pd.Timedelta(days=1)).tz_convert(PARIS_TZ).strftime("%Y-%m-%d"), ts.to_pydatetime()],
    )


def _stored_predictions(ts, version: str, current: pd.DataFrame) -> pd.DataFrame | None:
    """
    Prévisions enregistrées à la collecte pour ce snapshot et cette version, complétées du nom et
    de la position des stations. None si elles manquent ou sont incomplètes.
    """
    uri = f"{lake.lake_root()}/monitoring/predictions"
    try:
        rows = lake.query(
            f"SELECT * FROM delta_scan('{uri}') WHERE date = ? AND snapshot_at = ? AND model_version = ?",
            [ts.tz_convert(PARIS_TZ).strftime("%Y-%m-%d"), ts.to_pydatetime(), version],
        )
    except Exception:  # table pas encore créée (aucun modèle n'a encore servi)
        return None
    if len(rows) < 0.9 * len(current) * len(HORIZONS):
        return None
    rows = rows.rename(columns={"predicted_for": "ts_target"})
    rows["ts_target"] = pd.to_datetime(rows["ts_target"], utc=True)
    rows["station_id"] = rows["station_id"].astype(str)
    meta = current[["station_id", "name", "lat", "lon"]].assign(station_id=lambda d: d["station_id"].astype(str))
    return rows.merge(meta, on="station_id", how="left").reset_index(drop=True)


def _compute_predictions(use_stored: bool = True) -> tuple[pd.DataFrame | None, dict]:
    """
    Toutes les stations × tous les horizons, à partir du dernier snapshot du silver.
    use_stored=False force le calcul (c'est le cas du pipeline, qui produit les prévisions batch).
    """
    published = read_json(model_uri(MODEL_NAME, "latest.json"))
    if not published:
        return None, {"reason": "Aucun modèle publié pour l'instant."}

    last = lake.query(f"SELECT max(ingested_at) AS ts FROM {lake._silver()}")
    if last.empty or pd.isna(last.loc[0, "ts"]):
        return None, {"reason": "Lakehouse vide ou inaccessible."}
    ts = pd.Timestamp(last.loc[0, "ts"]).tz_convert("UTC")
    info = {"snapshot_at": ts.tz_convert(PARIS_TZ), "model": published, "horizons": list(HORIZONS)}

    current = _snapshot(ts)
    if use_stored:
        stored = _stored_predictions(ts, published["version"], current)
        if stored is not None:
            return stored, {**info, "source": "batch (calculées à la collecte)"}

    lag1, lag2 = _snapshot(ts - pd.Timedelta(hours=1)), _snapshot(ts - pd.Timedelta(hours=2))
    # Habitudes : seulement les heures de la journée visées par les horizons, pour alléger la requête
    target_hours = sorted({int((ts + pd.Timedelta(hours=h)).tz_convert(PARIS_TZ).hour) for h in HORIZONS})
    hour_stats = lake.query(
        f"""SELECT station_id, hour, count(*) AS past_n, sum(is_empty::int) AS past_empty,
                   sum(is_full::int) AS past_full, sum(numbikesavailable) AS past_bikes_sum
            FROM {lake._silver()} WHERE hour IN ({", ".join("?" * len(target_hours))})
            GROUP BY station_id, hour""",
        target_hours,
    )
    weather = lake.query(
        f"SELECT * FROM delta_scan('{lake.lake_root()}/silver/weather') WHERE ts > ? AND ts <= ?",
        [ts.to_pydatetime(), (ts + pd.Timedelta(hours=max(HORIZONS))).to_pydatetime()],
    )
    school = lake.query(f"SELECT * FROM read_parquet('{lake.lake_root()}/silver/school_holidays.parquet')")
    frame = serving_frame(current, lag1, lag2, hour_stats,
                          weather if not weather.empty else None, school if not school.empty else None)

    # Seules les familles qui ont battu leurs baselines sont publiées ; les autres restent NaN
    version, promoted = published["version"], set(published["promoted"])
    X = frame[FEATURES]
    frame["risk_empty"] = _booster(version, "empty", "model").predict(X) if "empty" in promoted else np.nan
    frame["risk_full"] = _booster(version, "full", "model").predict(X) if "full" in promoted else np.nan
    if "bikes" in promoted:
        # Modèles récents : variation à ajouter aux vélos actuels ; anciens : niveau directement
        delta = published.get("bikes_target") == "delta"
        q10, q50, q90 = (
            np.clip(frame["bikes"].to_numpy() + raw if delta else raw, 0, frame["capacity"].to_numpy())
            for raw in (_booster(version, "bikes", q).predict(X) for q in ("q10", "q50", "q90"))
        )
        # Les quantiles sont appris séparément et peuvent se croiser : on remet q10 <= q50 <= q90 <= capacité
        frame["bikes_q50"] = q50
        frame["bikes_q10"] = np.minimum(q10, q50)
        frame["bikes_q90"] = np.minimum(np.maximum(q90, q50), frame["capacity"])
    else:
        frame[["bikes_q10", "bikes_q50", "bikes_q90"]] = np.nan

    return frame.reset_index(drop=True), {**info, "source": "à la demande"}


def predict() -> tuple[pd.DataFrame | None, dict]:
    """
    Prévisions de tous les horizons pour le dernier snapshot, en cache 10 minutes.
    Les échecs sont aussi mis en cache, pour ne pas réinterroger un lakehouse en panne à chaque question.
    """
    cached = _cache.get("predictions")
    if cached and time.time() - cached[0] < _CACHE_SECONDS:
        return cached[1]
    try:
        result = _compute_predictions()
    except Exception as e:
        result = (None, {"reason": f"Prévision indisponible : {e}"})
    _cache["predictions"] = (time.time(), result)
    return result


def horizon_for(target_hour: int, snapshot_at: pd.Timestamp) -> int:
    """
    « À 18h » → nombre d'heures après le dernier snapshot (le lendemain si l'heure est passée,
    ce qui sort alors des horizons et sera refusé par l'appelant).
    """
    return (target_hour - snapshot_at.hour) % 24 or 24


def forecast_for_station(station_query: str, hours_ahead: int | None = None, at_hour: int | None = None) -> dict:
    """
    Outil de prévision des agents : pour les stations qui correspondent à la recherche, risque
    d'être vide ou pleine et vélos attendus, à un horizon donné ou à tous. Sans nom de station,
    renvoie les stations les plus à risque d'être vides.
    """
    from src.data.filters import search_stations

    predictions, info = predict()
    if predictions is None:
        return {"error": info["reason"]}
    if at_hour is not None:
        hours_ahead = horizon_for(at_hour, info["snapshot_at"])
    if hours_ahead is not None and hours_ahead not in info["horizons"]:
        return {"error": f"Prévision possible de 1 à {max(info['horizons'])} h après le dernier snapshot "
                         f"({info['snapshot_at']:%H:%M}) ; demandé : {hours_ahead} h."}

    rows = predictions if hours_ahead is None else predictions[predictions["horizon"] == hours_ahead]
    matches = search_stations(rows, station_query) if station_query else rows.nlargest(10, "risk_empty")
    if matches.empty:
        return {"error": f"Station '{station_query}' introuvable dans le dernier snapshot."}

    pct = lambda v: None if pd.isna(v) else f"{v:.0%}"
    return {
        "source": "modèles LightGBM (prévision)",
        "a_partir_du_snapshot_de": f"{info['snapshot_at']:%d/%m %H:%M}",
        "previsions": [
            {
                "station": row.name,
                "heure_visee": f"{row.ts_target.tz_convert(PARIS_TZ):%H:%M}",
                "dans": f"{int(row.horizon)} h",
                "velos_au_dernier_snapshot": int(row.bikes),
                "risque_vide": pct(row.risk_empty),
                "risque_pleine": pct(row.risk_full),
                "velos_attendus": None if pd.isna(row.bikes_q50) else
                f"{row.bikes_q50:.0f} (entre {row.bikes_q10:.0f} et {row.bikes_q90:.0f})",
            }
            for row in matches.sort_values(["name", "horizon"]).head(12).itertuples()
        ],
    }
