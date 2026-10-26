"""
forecast.py — Entraînement des modèles de prévision (LightGBM), horizons de 1 à 6 heures.

Trois questions, un modèle par question (l'horizon est une feature) :
- vide  : la station sera-t-elle vide à t + h ?          classification
- plein : la station sera-t-elle pleine à t + h ?        classification
- vélos : combien de vélos à t + h ? (quantiles 10/50/90) régression quantile → valeur + intervalle
          Le modèle apprend la VARIATION (vélos à t + h − vélos à t) : il part de la persistance
          (« rien ne change ») et la corrige, au lieu de deviner le niveau à partir de zéro.

1. features à partir de tout le silver (src/ml/features.py, partagé avec la prédiction)
2. découpage dans le temps : entraînement < validation < test
3. LightGBM avec early stopping sur la validation
4. évaluation par horizon face à deux baselines : persistance et habitudes de la station
5. chaque question est publiée seulement si son modèle bat les baselines
6. champion / challenger : le nouveau modèle ne remplace celui en production que s'il fait au
   moins aussi bien sur la même période de test (et mieux sur au moins une question)

Chaque entraînement est tracé dans models/forecast/history.json (déclencheur, raison, décision).

    python -m pipeline.run forecast            # à la main
    python -m pipeline.run retrain             # automatique : seulement si pipeline/mlops.py le juge utile
"""
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from deltalake import DeltaTable

from pipeline.lake import _delta_uri
from pipeline.settings import delta_storage_options, silver_uri

logger = logging.getLogger(__name__)

MODEL_NAME = "forecast"
MIN_DAYS = int(os.environ.get("FORECAST_MIN_DAYS", 7))  # une semaine : chaque jour vu au moins une fois
HISTORY_LENGTH = 50  # entraînements gardés dans history.json
TEST_SHARE, VALID_SHARE = 0.2, 0.15
TOP_K = 20
QUANTILES = (0.1, 0.5, 0.9)
# Au-delà, on échantillonne les instants par horizon pour garder la mémoire raisonnable
MAX_ROWS_PER_HORIZON = 600_000

BASE_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
    "verbose": -1,
}


def _read_silver() -> pd.DataFrame:
    from src.ml.features import SILVER_COLUMNS

    table = DeltaTable(_delta_uri(silver_uri()), storage_options=delta_storage_options())
    return table.to_pandas(columns=SILVER_COLUMNS + ["date"])


def _time_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpage par instants : les derniers snapshots servent au test, ceux d'avant à la validation."""
    instants = np.sort(data["ts"].unique())
    test_start = instants[int(len(instants) * (1 - TEST_SHARE))]
    valid_start = instants[int(len(instants) * (1 - TEST_SHARE - VALID_SHARE))]
    return (
        data[data["ts"] < valid_start],
        data[(data["ts"] >= valid_start) & (data["ts"] < test_start)],
        data[data["ts"] >= test_start],
    )


def reference_distribution(frame: pd.DataFrame, features: list[str], bins: int = 10) -> dict:
    """
    Distribution de chaque feature sur le jeu d'entraînement (déciles) : sert de référence
    pour mesurer la dérive des données en production (PSI).
    """
    reference = {}
    for feature in features:
        values = frame[feature].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:  # variable constante : un seul intervalle
            edges = np.array([values.min() - 0.5, values.max() + 0.5])
        counts, _ = np.histogram(np.clip(values, edges[0], edges[-1]), bins=edges)
        reference[feature] = {"edges": edges.round(6).tolist(), "proportions": (counts / counts.sum()).round(6).tolist()}
    return reference


# ---------------------------------------------------------------- métriques

def _precision_at_k(test: pd.DataFrame, target: str, now: str, scores: np.ndarray, k: int = TOP_K) -> float:
    """
    Usage opérateur : à chaque snapshot et horizon, parmi les stations pas encore dans l'état visé,
    les k jugées les plus à risque y sont-elles vraiment ? Moyenne sur les snapshots du test.
    """
    frame = test.assign(score=scores)
    frame = frame[frame[now] == 0]
    hits = [group.nlargest(k, "score")[target].mean()
            for _, group in frame.groupby(["ts", "horizon"]) if len(group) >= k]
    return float(np.mean(hits)) if hits else float("nan")


def _classification_metrics(test: pd.DataFrame, target: str, now: str, scores: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y = test[target].to_numpy()
    brier = round(float(brier_score_loss(y, np.clip(scores, 0, 1))), 4)
    if y.min() == y.max():  # une seule classe sur la période : AUC non définie
        return {"auc": float("nan"), "average_precision": float("nan"), "brier": brier,
                f"precision_at_{TOP_K}": float("nan")}
    return {
        "auc": round(float(roc_auc_score(y, scores)), 4),
        "average_precision": round(float(average_precision_score(y, scores)), 4),
        "brier": brier,
        f"precision_at_{TOP_K}": round(_precision_at_k(test, target, now, scores), 4),
    }


def _regression_metrics(y: np.ndarray, median: np.ndarray, low=None, high=None) -> dict:
    metrics = {"mae": round(float(np.mean(np.abs(y - median))), 3)}
    if low is not None:
        metrics["coverage_80"] = round(float(np.mean((y >= low) & (y <= high))), 3)
        metrics["interval_width"] = round(float(np.mean(high - low)), 2)
    return metrics


def _by_horizon(test: pd.DataFrame, compute) -> dict:
    """Une métrique globale + la même métrique pour chaque horizon."""
    result = {"all": compute(test)}
    for h, group in test.groupby("horizon"):
        result[f"{int(h)}h"] = compute(group)
    return result


# ---------------------------------------------------------------- entraînement

def _fit(model, train, valid, target, features):
    import lightgbm as lgb

    model.fit(train[features], train[target], eval_X=valid[features], eval_y=valid[target],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def _train_classifier(name, train, valid, test, features, profile_col):
    """Vide ou plein : modèle + baselines (persistance, habitudes de la station), par horizon."""
    import lightgbm as lgb

    from src.ml.features import TARGETS

    target, now = TARGETS[name], f"is_{name}_now"
    model = _fit(lgb.LGBMClassifier(objective="binary", **BASE_PARAMS), train, valid, target, features)
    base_rate = float(train[target].mean())
    scorers = {
        "model": lambda d: model.predict_proba(d[features])[:, 1],
        "persistence": lambda d: d[now].to_numpy(),
        "profile": lambda d: d[profile_col].fillna(base_rate).to_numpy(),
    }
    evaluation = {key: _by_horizon(test, lambda d, f=f: _classification_metrics(d, target, now, f(d)))
                  for key, f in scorers.items()}
    best_baseline = max(evaluation["persistence"]["all"]["auc"], evaluation["profile"]["all"]["auc"])
    promoted = (evaluation["model"]["all"]["auc"] > best_baseline
                and evaluation["model"]["all"]["brier"] < evaluation["profile"]["all"]["brier"])
    return {"models": {"model": model}, "evaluation": evaluation, "promoted": bool(promoted),
            "base_rate": round(base_rate, 4), "best_iteration": int(model.best_iteration_ or 0)}


BIKES_DELTA = "bikes_delta_target"


def bikes_from_delta(frame: pd.DataFrame, delta: np.ndarray) -> np.ndarray:
    """Vélos prévus = vélos maintenant + variation prévue, bornés entre 0 et la capacité."""
    return np.clip(frame["bikes"].to_numpy() + delta, 0, frame["capacity"].to_numpy())


def _train_bikes(train, valid, test, features):
    """
    Nombre de vélos : trois quantiles de la variation (intervalle de 80 %), remis en niveau
    pour l'évaluation. Prédire une variation nulle redonne exactement la persistance.
    """
    import lightgbm as lgb

    from src.ml.features import TARGETS

    target = TARGETS["bikes"]
    train, valid = (d.assign(**{BIKES_DELTA: d[target] - d["bikes"]}) for d in (train, valid))
    models = {f"q{int(q * 100)}": _fit(lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS),
                                       train, valid, BIKES_DELTA, features) for q in QUANTILES}

    def model_metrics(d):
        low, med, high = (bikes_from_delta(d, models[k].predict(d[features])) for k in ("q10", "q50", "q90"))
        return _regression_metrics(d[target].to_numpy(), med, np.minimum(low, med), np.maximum(high, med))

    evaluation = {
        "model": _by_horizon(test, model_metrics),
        "persistence": _by_horizon(test, lambda d: _regression_metrics(d[target].to_numpy(), d["bikes"].to_numpy())),
        "profile": _by_horizon(test, lambda d: _regression_metrics(
            d[target].to_numpy(), d["past_mean_bikes"].fillna(d["bikes"]).to_numpy())),
    }
    best_baseline = min(evaluation["persistence"]["all"]["mae"], evaluation["profile"]["all"]["mae"])
    return {"models": models, "evaluation": evaluation,
            "promoted": bool(evaluation["model"]["all"]["mae"] < best_baseline)}


# ---------------------------------------------------------------- champion / challenger

def _champion_scores(test: pd.DataFrame, features: list[str]) -> dict | None:
    """
    Le modèle en production (champion) rejoue la période de test du nouveau modèle (challenger) :
    les deux sont jugés sur exactement les mêmes lignes. None s'il n'y a pas de champion ou s'il
    utilise d'autres features (comparaison impossible : on repart de zéro).
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    from src.ml.features import TARGETS
    from src.ml.storage import model_uri, read_json, read_text

    champion = read_json(model_uri(MODEL_NAME, "latest.json"))
    if not champion or champion.get("features") != features:
        return None

    def booster(family, key):
        return lgb.Booster(model_str=read_text(model_uri(MODEL_NAME, champion["version"], f"{family}_{key}.txt")))

    scores = {}
    for family in champion.get("promoted", []):
        y = test[TARGETS[family]].to_numpy()
        if family == "bikes":
            raw = booster("bikes", "q50").predict(test[features])
            median = bikes_from_delta(test, raw) if champion.get("bikes_target") == "delta" else np.clip(raw, 0, None)
            scores[family] = {"mae": round(float(np.mean(np.abs(y - median))), 3)}
        elif y.min() != y.max():  # AUC indéfinie avec une seule classe : pas de comparaison
            scores[family] = {"auc": round(float(roc_auc_score(y, booster(family, "model").predict(test[features]))), 4)}
    return {"version": champion["version"], "promoted": champion.get("promoted", []), "scores": scores}


def decide_promotion(results: dict, champion: dict | None) -> tuple[list[str], str]:
    """
    Familles à publier et raison de la décision.
    Le challenger doit battre les baselines ; s'il y a un champion, il ne doit perdre sur aucune
    des questions que le champion sert déjà, et gagner sur au moins une (ou en servir une de plus).
    """
    challenger = [family for family, result in results.items() if result["promoted"]]
    if not challenger:
        return [], "le nouveau modèle ne bat pas les baselines"
    if champion is None:
        return challenger, "aucun modèle comparable en production : publication"

    better, worse = [], []
    for family in champion["promoted"]:
        if family not in challenger:
            worse.append(family)
            continue
        score = champion["scores"].get(family)
        if score is None:
            continue
        mine = results[family]["evaluation"]["model"]["all"]
        if "mae" in score:
            better += [family] if mine["mae"] < score["mae"] else []
            worse += [family] if mine["mae"] > score["mae"] else []
        else:
            better += [family] if mine["auc"] > score["auc"] else []
            worse += [family] if mine["auc"] < score["auc"] else []
    new_families = sorted(set(challenger) - set(champion["promoted"]))

    if worse:
        return [], f"champion {champion['version']} conservé : le challenger est moins bon sur {', '.join(worse)}"
    if not better and not new_families:
        return [], f"champion {champion['version']} conservé : aucune amélioration"
    gains = better + [f"{family} (nouvelle question servie)" for family in new_families]
    return challenger, f"bat le champion {champion['version']} sur {', '.join(gains)}"


def _record_history(entry: dict) -> None:
    """Ajoute l'entraînement à l'historique du registre (les plus récents en premier)."""
    from src.ml.storage import model_uri, read_json, write_json

    runs = (read_json(model_uri(MODEL_NAME, "history.json")) or {}).get("runs", [])
    write_json(model_uri(MODEL_NAME, "history.json"), {"runs": [entry, *runs][:HISTORY_LENGTH]})


def train_forecast(silver: pd.DataFrame | None = None, weather: pd.DataFrame | None = None,
                   school: pd.DataFrame | None = None, min_days: int | None = None, publish: bool = True,
                   trigger: str = "manuel", reason: str = "") -> dict:
    """
    Entraîne les 3 familles de modèles et décide de leur publication (baselines, puis champion).
    min_days : historique minimal exigé (MIN_DAYS par défaut).
    publish=False : entraînement de test, les métriques sont enregistrées mais l'app ne sert pas ces modèles.
    trigger / reason : qui a lancé l'entraînement et pourquoi (tracés dans l'historique).
    """
    from pipeline.external import read_school_holidays, read_weather
    from src.ml.features import FEATURES, HORIZONS, TARGETS, training_frame
    from src.ml.storage import model_uri, write_json, write_text

    silver = _read_silver() if silver is None else silver
    weather = read_weather() if weather is None else weather
    school = read_school_holidays() if school is None else school
    min_days = MIN_DAYS if min_days is None else min_days
    days = silver["date"].nunique() if "date" in silver else silver["ingested_at"].dt.date.nunique()
    if days < min_days:
        result = {"status": "skipped", "days": int(days),
                  "reason": f"{days} jour{'s' if days > 1 else ''} d'historique sur {min_days} requis"}
        logger.info(f"Modèles non entraînés : {result['reason']}")
        write_json(model_uri(MODEL_NAME, "status.json"), {**result, "trigger": trigger,
                                                          "checked_at": datetime.now(timezone.utc)})
        return result

    sample = min(1.0, MAX_ROWS_PER_HORIZON / len(silver))
    data = training_frame(silver, weather, school, sample=sample).dropna(subset=list(TARGETS.values()))
    train, valid, test = _time_split(data)
    logger.info(f"{len(HORIZONS)} horizons — entraînement {len(train):,} / validation {len(valid):,} / test {len(test):,} lignes")

    results = {
        "empty": _train_classifier("empty", train, valid, test, FEATURES, "past_rate_empty"),
        "full": _train_classifier("full", train, valid, test, FEATURES, "past_rate_full"),
        "bikes": _train_bikes(train, valid, test, FEATURES),
    }

    champion = _champion_scores(test, FEATURES)
    promoted, decision = decide_promotion(results, champion)
    if not publish:
        decision = f"test, non publié ({decision})"
        promoted = []
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for family, result in results.items():
        for key, model in result["models"].items():
            write_text(model_uri(MODEL_NAME, version, f"{family}_{key}.txt"), model.booster_.model_to_string())

    importance = results["empty"]["models"]["model"].booster_.feature_importance(importance_type="gain")
    metrics = {
        "model_name": MODEL_NAME,
        "version": version,
        "trained_at": datetime.now(timezone.utc),
        "horizons": list(HORIZONS),
        "days_of_history": int(days),
        "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
        "test_period": [test["ts"].min(), test["ts"].max()],
        "features": FEATURES,
        "bikes_target": "delta",  # le modèle vélos prédit une variation (voir _train_bikes)
        "families": {
            family: {k: v for k, v in result.items() if k != "models"} | {"files": sorted(result["models"])}
            for family, result in results.items()
        },
        "promoted": promoted,
        "beats_baselines": [family for family, result in results.items() if result["promoted"]],
        "published": publish,
        "trigger": trigger,
        "trigger_reason": reason,
        "champion": champion,
        "decision": decision,
        "feature_importance_gain": dict(sorted(zip(FEATURES, importance.round(1).tolist()), key=lambda i: -i[1])),
        "external_data": {"weather": weather is not None, "school_holidays": school is not None},
        "reference_distribution": reference_distribution(train, FEATURES),
    }
    write_json(model_uri(MODEL_NAME, version, "metrics.json"), metrics)
    if metrics["promoted"]:
        write_json(model_uri(MODEL_NAME, "latest.json"), metrics)
    status = "trained" if publish else "test"
    write_json(model_uri(MODEL_NAME, "status.json"), {"status": status, "version": version,
                                                      "promoted": promoted, "decision": decision,
                                                      "trigger": trigger, "checked_at": datetime.now(timezone.utc)})

    summary = {family: {"score": r["evaluation"]["model"]["all"].get("auc", r["evaluation"]["model"]["all"].get("mae")),
                        "bat les baselines": r["promoted"]} for family, r in results.items()}
    _record_history({
        "version": version, "trained_at": metrics["trained_at"], "trigger": trigger, "reason": reason,
        "days_of_history": int(days), "published": bool(promoted), "promoted": promoted, "decision": decision,
        "scores": {family: s["score"] for family, s in summary.items()},
    })
    logger.info(f"Modèles {version} : {decision} — {summary}")
    return {"status": status, "version": version, "promoted": promoted, "decision": decision, "summary": summary}
