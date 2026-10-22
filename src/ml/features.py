"""
features.py — Features des modèles de prévision, pour un horizon de 1 à 6 heures.

Trois questions, mêmes features :
- la station sera-t-elle vide à t + h ?     (empty_target)
- la station sera-t-elle pleine à t + h ?   (full_target)
- combien de vélos à t + h ?                (bikes_target, en quantiles)

Une ligne = (station, instant t, horizon h). L'horizon est une feature : un seul modèle par
question apprend que l'état actuel compte à court terme et les habitudes à plus long terme.

Une seule définition, utilisée à l'entraînement (pipeline/forecast.py) et à la prédiction
(src/ml/forecast.py) : c'est ce qui évite l'écart entre ce que le modèle a appris et ce qu'on
lui donne en production (training-serving skew).

Aucune feature ne regarde le futur : les habitudes d'une station (taux de vide, de plein,
vélos moyens à l'heure visée) n'utilisent que les observations antérieures. Seule entorse
assumée : la météo de l'heure visée est observée à l'entraînement, prévue (Open-Meteo) à la
prédiction.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

PARIS_TZ = "Europe/Paris"
HORIZONS = (1, 2, 3, 4, 5, 6)

FEATURES = [
    "horizon",
    "bikes", "docks", "ebike", "bike_ratio", "capacity", "is_empty_now", "is_full_now",
    "delta_1h", "delta_2h",
    "hour_target", "weekday_target", "is_weekend_target",
    "past_rate_empty", "past_rate_full", "past_mean_bikes", "past_n",
    # Données externes : une valeur manquante reste NaN, LightGBM sait la traiter
    "temperature", "precipitation", "wind_speed",
    "is_public_holiday", "is_school_holiday",
]
TARGETS = {"empty": "empty_target", "full": "full_target", "bikes": "bikes_target"}
WEATHER_COLUMNS = ["temperature", "precipitation", "wind_speed"]

SILVER_COLUMNS = [
    "station_id", "name", "ingested_at", "numbikesavailable", "numdocksavailable",
    "ebike", "bike_ratio", "is_empty", "is_full", "lat", "lon",
]
# Habitudes par (station, heure de la journée) : colonne d'état -> somme cumulée correspondante.
# Divisées par past_n dans _assemble pour donner past_rate_empty, past_rate_full, past_mean_bikes.
_PAST_STATS = {"is_empty_now": "past_empty", "is_full_now": "past_full", "bikes": "past_bikes_sum"}


def _state(df: pd.DataFrame) -> pd.DataFrame:
    """État de chaque station à l'instant du snapshot, recalculé depuis les comptes bruts."""
    out = pd.DataFrame({
        "station_id": df["station_id"].astype(str),
        "ts": pd.to_datetime(df["ingested_at"], utc=True),
        "bikes": df["numbikesavailable"].astype(float),
        "docks": df["numdocksavailable"].astype(float),
        "ebike": df["ebike"].astype(float),
    })
    out["is_empty_now"] = (out["bikes"] == 0).astype(float)
    out["is_full_now"] = (out["docks"] == 0).astype(float)
    out["capacity"] = out["bikes"] + out["docks"]
    out["bike_ratio"] = (out["bikes"] / out["capacity"].replace(0, np.nan)).fillna(0.0)
    for column in ("name", "lat", "lon"):
        if column in df.columns:
            out[column] = df[column].values
    return out


def _lag(state: pd.DataFrame, hours: int) -> pd.DataFrame:
    """
    Nombre de vélos `hours` heures plus tôt, décalé pour se joindre sur l'instant courant.
    Jointure exacte sur ts : si le snapshot correspondant manque, le lag reste NaN.
    """
    lag = state[["station_id", "ts", "bikes"]].copy()
    lag["ts"] = lag["ts"] + pd.Timedelta(hours=hours)
    return lag.rename(columns={"bikes": f"bikes_lag{hours}"})


def _calendar(ts_target: pd.Series) -> pd.DataFrame:
    """Heure et jour de l'heure visée, en heure de Paris (c'est elle qui rythme les usages)."""
    local = ts_target.dt.tz_convert(PARIS_TZ)
    return pd.DataFrame({
        "hour_target": local.dt.hour.astype(float),
        "weekday_target": local.dt.dayofweek.astype(float),
        "is_weekend_target": (local.dt.dayofweek >= 5).astype(float),
    }, index=ts_target.index)


@lru_cache(maxsize=8)
def _public_holidays(year: int) -> frozenset:
    import holidays
    return frozenset(holidays.France(years=year))


def _external(ts_target: pd.Series, weather: pd.DataFrame | None, school: pd.DataFrame | None) -> pd.DataFrame:
    """Météo et calendrier (jours fériés, vacances scolaires) de l'heure visée."""
    out = pd.DataFrame({"ts_target": ts_target}, index=ts_target.index)
    if weather is not None and not weather.empty:
        w = weather.assign(ts_target=pd.to_datetime(weather["ts"], utc=True))[["ts_target", *WEATHER_COLUMNS]]
        out = (out.rename_axis("row").reset_index()
               .merge(w.drop_duplicates("ts_target"), on="ts_target", how="left").set_index("row"))
    for column in WEATHER_COLUMNS:
        if column not in out:
            out[column] = np.nan

    # Calcul par heure distincte (quelques milliers) puis propagation aux lignes (millions)
    hours = pd.Series(ts_target.unique())
    local_dates = hours.dt.tz_convert(PARIS_TZ).dt.date
    index = pd.DatetimeIndex(hours)  # garde le fuseau UTC pour que la correspondance fonctionne
    public = pd.Series([float(d in _public_holidays(d.year)) for d in local_dates], index=index)
    out["is_public_holiday"] = ts_target.map(public).to_numpy()
    if school is not None and not school.empty:
        starts, ends = school["start"].to_numpy(), school["end"].to_numpy()
        vacation = pd.Series([float(((starts <= v) & (v < ends)).any()) for v in hours.to_numpy()],
                             index=index)
        out["is_school_holiday"] = ts_target.map(vacation).to_numpy()
    else:
        out["is_school_holiday"] = np.nan
    return out[[*WEATHER_COLUMNS, "is_public_holiday", "is_school_holiday"]]


def _assemble(frame: pd.DataFrame, weather: pd.DataFrame | None = None,
              school: pd.DataFrame | None = None) -> pd.DataFrame:
    """Dernière étape commune aux deux usages : variations, calendrier, habitudes et données externes."""
    frame = frame.reset_index(drop=True)
    frame["delta_1h"] = frame["bikes"] - frame["bikes_lag1"]
    frame["delta_2h"] = frame["bikes"] - frame["bikes_lag2"]
    frame["ts_target"] = frame["ts"] + pd.to_timedelta(frame["horizon"], unit="h")
    frame = frame.join(_calendar(frame["ts_target"])).join(_external(frame["ts_target"], weather, school))
    n = frame["past_n"].replace(0, np.nan)  # sans historique, les taux restent NaN plutôt que 0
    frame["past_rate_empty"] = frame["past_empty"] / n
    frame["past_rate_full"] = frame["past_full"] / n
    frame["past_mean_bikes"] = frame["past_bikes_sum"] / n
    return frame


def training_frame(silver: pd.DataFrame, weather: pd.DataFrame | None = None,
                   school: pd.DataFrame | None = None, horizons=HORIZONS,
                   sample: float | None = None, seed: int = 0) -> pd.DataFrame:
    """
    Une ligne par (station, instant t, horizon h) : features à t, cibles à t + h.
    `sample` garde une fraction des instants par horizon (limite la mémoire sur des mois de données).
    Les lignes sans observation à t + h (trou de collecte, fin de l'historique) ont des cibles vides.
    """
    state = _state(silver).sort_values(["station_id", "ts"])
    state["hour_local"] = state["ts"].dt.tz_convert(PARIS_TZ).dt.hour

    # Habitudes « même heure de la journée » : cumul moins la ligne courante, donc uniquement
    # les observations antérieures (pas de fuite de la cible dans les features)
    group = state.groupby(["station_id", "hour_local"], sort=False)
    state["past_n"] = group.cumcount()
    for column, stat in _PAST_STATS.items():
        state[stat] = group[column].cumsum() - state[column]

    base = (
        state.drop(columns=["hour_local", "past_n", *_PAST_STATS.values()])
        .merge(_lag(state, 1), on=["station_id", "ts"], how="left")
        .merge(_lag(state, 2), on=["station_id", "ts"], how="left")
    )
    # Cibles et habitudes de l'heure visée : on lit la ligne de t + h. Ses habitudes ne couvrent
    # que les jours précédents (h < 24), donc rien d'inconnu à l'instant t.
    target_cols = ["station_id", "ts", "is_empty_now", "is_full_now", "bikes", "past_n", *_PAST_STATS.values()]
    renamed = {"is_empty_now": TARGETS["empty"], "is_full_now": TARGETS["full"], "bikes": TARGETS["bikes"]}

    frames = []
    for h in horizons:
        rows = base.sample(frac=sample, random_state=seed + h) if sample and sample < 1 else base
        target = state[target_cols].rename(columns=renamed)
        target["ts"] = target["ts"] - pd.Timedelta(hours=h)
        frames.append(rows.merge(target, on=["station_id", "ts"], how="left").assign(horizon=float(h)))
    return _assemble(pd.concat(frames, ignore_index=True), weather, school)


def serving_frame(current: pd.DataFrame, lag1: pd.DataFrame, lag2: pd.DataFrame,
                  hour_stats: pd.DataFrame, weather: pd.DataFrame | None = None,
                  school: pd.DataFrame | None = None, horizons=HORIZONS) -> pd.DataFrame:
    """
    Mêmes features pour le dernier snapshot connu, pour chaque horizon.
    hour_stats : agrégats par (station, heure de la journée) sur tout l'historique, qui est
    entièrement passé ici : past_n, past_empty, past_full, past_bikes_sum.
    """
    state = _state(current)
    base = state.merge(_lag(_state(lag1), 1) if not lag1.empty else _empty_lag(1), on=["station_id", "ts"], how="left")
    base = base.merge(_lag(_state(lag2), 2) if not lag2.empty else _empty_lag(2), on=["station_id", "ts"], how="left")
    stats = hour_stats.assign(station_id=hour_stats["station_id"].astype(str), hour=hour_stats["hour"].astype(int))
    stat_cols = ["past_n", *_PAST_STATS.values()]

    frames = []
    for h in horizons:
        rows = base.assign(horizon=float(h))
        rows["hour"] = (rows["ts"] + pd.Timedelta(hours=h)).dt.tz_convert(PARIS_TZ).dt.hour
        rows = rows.merge(stats[["station_id", "hour", *stat_cols]], on=["station_id", "hour"], how="left")
        rows[stat_cols] = rows[stat_cols].fillna(0)  # station nouvelle : past_n = 0, comme à l'entraînement
        frames.append(rows.drop(columns="hour"))
    return _assemble(pd.concat(frames, ignore_index=True), weather, school)


def _empty_lag(hours: int) -> pd.DataFrame:
    """Lag vide mais typé, quand le snapshot d'il y a `hours` heures manque."""
    return pd.DataFrame({"station_id": pd.Series(dtype=str), "ts": pd.Series(dtype="datetime64[ns, UTC]"),
                         f"bikes_lag{hours}": pd.Series(dtype=float)})
