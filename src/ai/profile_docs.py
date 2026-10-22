"""
profile_docs.py — Documents RAG / Qdrant construits à partir du gold dbt.

Un document par station résume son comportement habituel (semaine / week-end ×
matin, après-midi, soir, nuit). C'est ce qui permet de répondre à
« Bastille est-elle souvent vide le matin ? » : une tendance, pas une ligne brute.
"""
import json

import numpy as np
import pandas as pd
import pytz

PARIS_TZ = pytz.timezone("Europe/Paris")
# Valeurs telles qu'écrites par dbt (sans accent) ; les libellés affichés sont dans PERIOD_LABELS
PERIODS = ["matin", "apres-midi", "soir", "nuit"]
PERIOD_LABELS = {"matin": "matin (6h-12h)", "apres-midi": "après-midi (12h-18h)",
                 "soir": "soir (18h-22h)", "nuit": "nuit (22h-6h)"}
DAY_LABELS = {"semaine": "En semaine", "weekend": "Le week-end"}


def _fmt_date(value) -> str:
    return pd.Timestamp(value).tz_convert(PARIS_TZ).strftime("%d/%m/%Y")


def _period_sentence(row) -> str:
    sentence = (
        f"{PERIOD_LABELS[row['period']]} : vide {row['pct_empty']:.0%} du temps, "
        f"pleine {row['pct_full']:.0%}, {row['avg_bikes']:.1f} vélos en moyenne "
        f"dont {row['avg_ebike']:.1f} électriques, {row['avg_docks']:.1f} bornes libres"
    )
    return f"{sentence} ({int(row['n_obs'])} obs.)"


def station_documents(profiles: pd.DataFrame) -> list[dict]:
    """Transforme gold_station_period_profile en documents {text, metadata}."""
    if profiles is None or profiles.empty:
        return []

    documents = []
    for station_id, group in profiles.groupby("station_id"):
        # Une station peut être renommée : on garde son nom le plus récent
        name = group.sort_values("last_seen")["name"].iloc[-1]
        lines = [
            f"Station {name} — profil historique "
            f"({int(group['n_obs'].sum())} observations horaires, "
            f"du {_fmt_date(group['first_seen'].min())} au {_fmt_date(group['last_seen'].max())})."
        ]

        for day_type in ["semaine", "weekend"]:
            day = group[group["day_type"] == day_type].set_index("period")
            parts = [_period_sentence(day.loc[p].to_dict() | {"period": p}) for p in PERIODS if p in day.index]
            if parts:
                lines.append(f"{DAY_LABELS[day_type]} : " + " ; ".join(parts) + ".")

        worst = group.sort_values("pct_empty", ascending=False).iloc[0]
        if worst["pct_empty"] > 0:
            lines.append(
                f"Moment le plus critique : {DAY_LABELS[worst['day_type']].lower()}, "
                f"{PERIOD_LABELS[worst['period']]}, vide {worst['pct_empty']:.0%} du temps."
            )
        else:
            lines.append("La station n'a jamais été observée vide sur la période.")

        last_seen = pd.Timestamp(group["last_seen"].max()).tz_convert(PARIS_TZ)
        documents.append({
            "text": "\n".join(lines),
            "metadata": {
                "station": str(name),
                "station_id": str(station_id),
                "snapshot_id": "profil gold",
                "run_at": last_seen.strftime("%Y-%m-%d %H:%M"),
                "n_obs": int(group["n_obs"].sum()),
                "max_pct_empty": float(worst["pct_empty"]),
                "worst_period": f"{worst['day_type']} {worst['period']}",
            },
        })
    return documents


def profiles_version(profiles: pd.DataFrame) -> str:
    """Empreinte du gold (dernière observation + volume) : change dès que dbt ajoute des données."""
    if profiles is None or profiles.empty:
        return ""
    return f"{pd.Timestamp(profiles['last_seen'].max()).isoformat()}|{int(profiles['n_obs'].sum())}"


def documents_to_frame(documents: list[dict], vectors, version: str, model: str) -> pd.DataFrame:
    """Documents + vecteurs sous forme de table (écrite dans le gold par le pipeline)."""
    return pd.DataFrame({
        "station_id": [d["metadata"]["station_id"] for d in documents],
        "text": [d["text"] for d in documents],
        "metadata_json": [json.dumps(d["metadata"], ensure_ascii=False) for d in documents],
        "embedding": [np.asarray(v, dtype="float32").tolist() for v in vectors],
        "embedding_model": model,
        "profiles_version": version,
    })


def documents_from_frame(frame: pd.DataFrame, model: str) -> list[dict]:
    """
    Relit les documents précalculés. Si la table vient d'un autre modèle d'embeddings, on l'ignore :
    ses vecteurs ne seraient pas comparables à ceux des questions.
    """
    if frame is None or frame.empty or (frame["embedding_model"] != model).any():
        return []
    return [
        {
            "text": row.text,
            "metadata": json.loads(row.metadata_json),
            "embedding": np.asarray(row.embedding, dtype="float32"),
            "profiles_version": row.profiles_version,
        }
        for row in frame.itertuples(index=False)
    ]
