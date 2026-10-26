"""Cycle MLOps : champion / challenger, règles de réentraînement, serving batch."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from tests.test_forecast import _simulated_lake


def _results(**scores):
    """Résultats d'entraînement minimaux : {famille: (bat les baselines, score)}."""
    out = {}
    for family, (beats, score) in scores.items():
        metric = "mae" if family == "bikes" else "auc"
        out[family] = {"promoted": beats, "evaluation": {"model": {"all": {metric: score}}}}
    return out


def test_champion_challenger_rules():
    from pipeline.forecast import decide_promotion

    champion = {"version": "v1", "promoted": ["empty", "bikes"],
                "scores": {"empty": {"auc": 0.80}, "bikes": {"mae": 3.0}}}

    published, reason = decide_promotion(_results(empty=(True, 0.82), bikes=(True, 2.9), full=(False, 0.6)), None)
    assert published == ["empty", "bikes"] and "aucun modèle" in reason

    published, reason = decide_promotion(_results(empty=(True, 0.82), bikes=(True, 2.9)), champion)
    assert published == ["empty", "bikes"] and "bat le champion v1" in reason

    published, reason = decide_promotion(_results(empty=(True, 0.85), bikes=(True, 3.4)), champion)
    assert published == [] and "moins bon sur bikes" in reason          # gagner ici ne rachète pas une perte là

    published, reason = decide_promotion(_results(empty=(True, 0.85), bikes=(False, 2.0)), champion)
    assert published == [] and "bikes" in reason                         # ne sert plus une question du champion

    published, reason = decide_promotion(_results(empty=(True, 0.80), bikes=(True, 3.0), full=(True, 0.7)), champion)
    assert published == ["empty", "bikes", "full"] and "nouvelle question" in reason

    published, reason = decide_promotion(_results(empty=(False, 0.9)), champion)
    assert published == [] and "baselines" in reason


def test_identical_challenger_keeps_the_champion(tmp_path, monkeypatch):
    """Même données, même modèle : aucune amélioration, le champion reste en production ; tout est tracé."""
    _, history, _ = _simulated_lake(tmp_path, monkeypatch, n_stations=30, days=16, seed=2)
    from pipeline.forecast import train_forecast
    from src.ml.storage import model_uri, read_json

    first = train_forecast(history, trigger="test", reason="premier")
    second = train_forecast(history, trigger="test", reason="second")
    assert first["promoted"] and second["promoted"] == []
    assert "conservé" in second["decision"]
    assert read_json(model_uri("forecast", "latest.json"))["version"] == first["version"]

    runs = read_json(model_uri("forecast", "history.json"))["runs"]
    assert [r["reason"] for r in runs] == ["second", "premier"] and runs[0]["published"] is False


def test_retrain_rules(tmp_path, monkeypatch):
    _, history, _ = _simulated_lake(tmp_path, monkeypatch, n_stations=30, days=16, seed=2)
    from pipeline.forecast import train_forecast
    from pipeline.mlops import retrain_reason
    from pipeline.monitoring import _write_gold

    assert retrain_reason() == "aucun modèle en production"
    train_forecast(history)
    assert retrain_reason() is None                                           # modèle neuf, pas de monitoring

    assert "depuis 8 jours" in retrain_reason(now=datetime.now(timezone.utc) + timedelta(days=8))

    _write_gold("model_drift", pd.DataFrame({"feature": ["temperature", "bikes"], "psi": [0.4, 0.02],
                                             "status": ["dérive", "stable"]}))
    assert retrain_reason() == "dérive des données : temperature"


def test_app_serves_the_batch_predictions(tmp_path, monkeypatch):
    """Prévisions calculées à la collecte puis servies telles quelles : l'app montre ce que le monitoring juge."""
    _, history, _ = _simulated_lake(tmp_path, monkeypatch, n_stations=30, days=16, seed=2)
    from pipeline.forecast import train_forecast
    from pipeline.monitoring import log_predictions
    import src.data.lake as lake
    import src.ml.forecast as forecast

    train_forecast(history)
    monkeypatch.setattr(lake, "_connection", None)
    forecast._cache.clear()
    on_demand, info = forecast._compute_predictions(use_stored=False)
    assert info["source"] == "à la demande"

    assert log_predictions() == len(on_demand)
    monkeypatch.setattr(lake, "_connection", None)
    forecast._cache.clear()
    served, info = forecast.predict()
    assert info["source"].startswith("batch") and len(served) == len(on_demand)

    key = ["station_id", "horizon"]
    a = on_demand.sort_values(key).reset_index(drop=True)
    b = served.assign(station_id=served["station_id"].astype(str)).sort_values(key).reset_index(drop=True)
    np.testing.assert_allclose(a["risk_empty"], b["risk_empty"], rtol=1e-6)
    assert b["name"].notna().all() and {"lat", "lon", "ts_target"} <= set(b.columns)
