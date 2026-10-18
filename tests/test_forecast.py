"""Tests des modèles de prévision : fuite de données, cohérence entraînement/prédiction, bout en bout."""
import numpy as np
import pandas as pd
import pytest

from src.ml.features import FEATURES, HORIZONS, PARIS_TZ, TARGETS, serving_frame, training_frame
from tests.synthetic import synthetic_silver


@pytest.fixture(scope="module")
def silver():
    return synthetic_silver(n_stations=9, days=16, seed=1)


def _with_ts(silver):
    return silver.assign(ts=pd.to_datetime(silver["ingested_at"], utc=True))


def test_targets_are_the_state_h_hours_later(silver):
    frame = training_frame(silver).dropna(subset=list(TARGETS.values()))
    observed = _with_ts(silver).set_index(["station_id", "ts"])
    sample = frame.sample(300, random_state=0)
    for r in sample.itertuples():
        later = observed.loc[(r.station_id, r.ts + pd.Timedelta(hours=r.horizon))]
        assert r.empty_target == float(later["numbikesavailable"] == 0)
        assert r.full_target == float(later["numdocksavailable"] == 0)
        assert r.bikes_target == later["numbikesavailable"]
    assert set(frame["horizon"]) == set(map(float, HORIZONS))


def test_station_habits_use_only_the_past(silver):
    frame = training_frame(silver).dropna(subset=list(TARGETS.values()))
    history = _with_ts(silver)
    for row in frame[frame["horizon"] == 4].sample(20, random_state=1).itertuples():
        target_ts = row.ts + pd.Timedelta(hours=4)
        station = history[history["station_id"] == row.station_id]
        before = station[(station["ts"].dt.tz_convert(PARIS_TZ).dt.hour == target_ts.tz_convert(PARIS_TZ).hour)
                         & (station["ts"] < target_ts)]
        assert row.past_n == len(before)
        assert row.past_empty == (before["numbikesavailable"] == 0).sum()
        assert row.past_bikes_sum == before["numbikesavailable"].sum()


def test_serving_features_match_training_features(silver):
    """Même station, même instant, même horizon : la prédiction voit exactement les features de l'entraînement."""
    frame = training_frame(silver).dropna(subset=list(TARGETS.values()))
    ts = frame["ts"].sort_values().unique()[-40]
    data = _with_ts(silver)
    snap = lambda t: data[data["ts"] == t].drop(columns="ts")
    past = data[data["ts"] <= ts]
    hour_stats = (
        past.assign(hour=past["ts"].dt.tz_convert(PARIS_TZ).dt.hour,
                    e=past["numbikesavailable"] == 0, f=past["numdocksavailable"] == 0)
        .groupby(["station_id", "hour"])
        .agg(past_n=("e", "size"), past_empty=("e", "sum"), past_full=("f", "sum"),
             past_bikes_sum=("numbikesavailable", "sum"))
        .reset_index()
    )
    served = serving_frame(snap(ts), snap(ts - pd.Timedelta(hours=1)), snap(ts - pd.Timedelta(hours=2)), hour_stats)
    trained = frame[frame["ts"] == ts]
    key = ["station_id", "horizon"]
    compared = [f for f in FEATURES if f != "horizon"]  # l'horizon sert de clé
    left = served.set_index(key)[compared].sort_index().astype(float)
    right = trained.set_index(key)[compared].sort_index().astype(float)
    assert len(left) == len(right) == 9 * len(HORIZONS)
    np.testing.assert_allclose(left.to_numpy(), right.to_numpy(), equal_nan=True)


def test_training_is_skipped_without_enough_history(tmp_path, monkeypatch, silver):
    monkeypatch.setenv("LAKE_ROOT", str(tmp_path))
    from pipeline.forecast import train_forecast

    short = silver[silver["date"] <= silver["date"].min()]  # un seul jour
    assert train_forecast(short)["status"] == "skipped"


def test_external_features_weather_and_calendar():
    """Météo de l'heure visée, jour férié (11 novembre) et vacances scolaires."""
    from src.ml.features import _external

    ts = pd.Series(pd.to_datetime(["2026-11-11 10:00", "2026-10-20 10:00", "2026-09-15 10:00"], utc=True))
    weather = pd.DataFrame({"ts": ts, "temperature": [8.0, 12.5, 20.0],
                            "precipitation": [0.0, 1.2, 0.0], "wind_speed": [10.0, 5.0, 3.0]})
    school = pd.DataFrame({"start": [pd.Timestamp("2026-10-17", tz="UTC")],
                           "end": [pd.Timestamp("2026-11-02", tz="UTC")]})

    out = _external(ts, weather, school)
    assert out["temperature"].tolist() == [8.0, 12.5, 20.0]
    assert out["is_public_holiday"].tolist() == [1.0, 0.0, 0.0]
    assert out["is_school_holiday"].tolist() == [0.0, 1.0, 0.0]

    without = _external(ts, None, None)
    assert without["temperature"].isna().all() and without["is_school_holiday"].isna().all()


def _simulated_lake(tmp_path, monkeypatch, n_stations, days, seed, live_hours=0):
    """Lakehouse simulé : silver Delta, avec éventuellement des heures « live » ajoutées ensuite."""
    from deltalake import write_deltalake

    monkeypatch.setenv("LAKE_ROOT", str(tmp_path).replace("\\", "/"))
    silver = synthetic_silver(n_stations=n_stations, days=days, seed=seed)
    ts = pd.to_datetime(silver["ingested_at"], utc=True)
    delta = silver.assign(
        ingested_at=ts, hour=ts.dt.tz_convert(PARIS_TZ).dt.hour.astype("int32"),
        snapshot_id=ts.dt.strftime("%Y%m%dT%H00Z"), is_full=silver["numdocksavailable"] == 0, ingested_by="test",
    )
    instants = sorted(ts.unique())
    cut = instants[-live_hours] if live_hours else instants[-1] + pd.Timedelta(hours=1)
    # reset_index : sinon pandas transmet l'index filtré comme une colonne de plus
    write_deltalake(str(tmp_path / "silver" / "station_status"), delta[ts < cut].reset_index(drop=True),
                    partition_by=["date"])
    live = [delta[ts == t].reset_index(drop=True) for t in instants[-live_hours:]] if live_hours else []
    return silver, silver[ts < cut], live


def test_end_to_end_train_publish_and_predict(tmp_path, monkeypatch):
    """Entraînement des 3 modèles, publication, puis prédiction avec le code de l'application."""
    silver, history, _ = _simulated_lake(tmp_path, monkeypatch, n_stations=30, days=16, seed=2)

    from pipeline.forecast import train_forecast
    result = train_forecast(history)
    assert result["status"] == "trained" and set(result["promoted"]) == {"empty", "full", "bikes"}

    import src.data.lake as lake
    import src.ml.forecast as forecast
    monkeypatch.setattr(lake, "_connection", None)
    forecast._cache.clear()

    predictions, info = forecast.predict()
    assert predictions is not None, info
    assert len(predictions) == 30 * len(HORIZONS)
    assert predictions[["risk_empty", "risk_full"]].stack().between(0, 1).all()
    assert (predictions["bikes_q10"] <= predictions["bikes_q50"]).all()
    assert (predictions["bikes_q50"] <= predictions["bikes_q90"]).all()

    answer = forecast.forecast_for_station("STATION 001", hours_ahead=3)
    assert [p["dans"] for p in answer["previsions"]] == ["3 h"]
    assert answer["previsions"][0]["station"] == "STATION 001"
    assert "error" in forecast.forecast_for_station("STATION 001", hours_ahead=9)


def test_production_monitoring_end_to_end(tmp_path, monkeypatch):
    """12 heures de « production » simulées : prévisions des 6 horizons enregistrées à chaque collecte, puis évaluées."""
    from deltalake import write_deltalake

    silver, history, live = _simulated_lake(tmp_path, monkeypatch, n_stations=30, days=16, seed=4, live_hours=12)
    from pipeline.forecast import train_forecast
    assert train_forecast(history)["promoted"]

    import src.data.lake as lake
    from pipeline.monitoring import evaluate_production, log_predictions
    logged = 0
    for snapshot in live:
        write_deltalake(str(tmp_path / "silver" / "station_status"), snapshot, partition_by=["date"], mode="append")
        monkeypatch.setattr(lake, "_connection", None)
        logged += log_predictions()
    assert logged == 12 * 30 * len(HORIZONS)

    result = evaluate_production(silver)
    # prévisions dont l'heure visée est déjà observée : 6+6+6+6+6+6+5+4+3+2+1+0 = 51 par station
    assert result["status"] == "evaluated" and result["scored_predictions"] == 51 * 30

    import pyarrow.parquet as pq
    daily = pq.read_table(str(tmp_path / "gold" / "model_monitoring_daily.parquet")).to_pandas()
    calibration = pq.read_table(str(tmp_path / "gold" / "model_calibration.parquet")).to_pandas()
    drift = pq.read_table(str(tmp_path / "gold" / "model_drift.parquet")).to_pandas()
    assert daily["predictions"].sum() == 51 * 30 and set(daily["horizon"]) == set(HORIZONS)
    assert set(calibration["question"]) == {"vide", "pleine"}
    assert set(drift["status"]) <= {"stable", "à surveiller", "dérive"} and len(drift) >= 8
    assert not {"horizon", "weekday_target", "past_n"} & set(drift["feature"])  # exclues par construction


def test_psi_detects_a_shift():
    from pipeline.forecast import reference_distribution
    from pipeline.monitoring import psi

    rng = np.random.default_rng(0)
    reference = reference_distribution(pd.DataFrame({"x": rng.normal(0, 1, 5000)}), ["x"])["x"]
    assert psi(reference, rng.normal(0, 1, 2000)) < 0.1      # même distribution : stable
    assert psi(reference, rng.normal(1.5, 1, 2000)) > 0.25   # moyenne décalée : dérive


def test_test_training_is_never_served(tmp_path, monkeypatch):
    """Entraînement de test (publish=False) : métriques enregistrées, mais l'app ne sert aucun modèle."""
    _, history, _ = _simulated_lake(tmp_path, monkeypatch, n_stations=30, days=16, seed=2)
    from pipeline.forecast import train_forecast
    from src.ml.storage import model_uri, read_json

    result = train_forecast(history, publish=False)
    assert result["status"] == "test" and result["promoted"] == []
    assert read_json(model_uri("forecast", "latest.json")) is None
    metrics = read_json(model_uri("forecast", result["version"], "metrics.json"))
    assert metrics["published"] is False and metrics["beats_baselines"]


def test_min_days_can_be_lowered(tmp_path, monkeypatch, silver):
    monkeypatch.setenv("LAKE_ROOT", str(tmp_path))
    from pipeline.forecast import train_forecast

    short = silver[silver["date"] <= sorted(silver["date"].unique())[1]]  # deux jours
    assert train_forecast(short)["status"] == "skipped"
    assert train_forecast(short, min_days=2, publish=False)["status"] != "skipped"
