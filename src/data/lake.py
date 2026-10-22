"""
lake.py — Lecture du lakehouse Vélib' depuis Streamlit (DuckDB, lecture seule).

- silver/station_status (Delta)               : snapshots bruts, base du mode Historique et du secours API
- gold/gold_station_period_profile.parquet    : profils des stations (RAG)
- gold/station_profile_embeddings.parquet     : vecteurs de ces profils, calculés par le pipeline

LAKE_ROOT vaut s3://<bucket>/lake (secrets Streamlit) ou un chemin local.
Si le lakehouse est inaccessible, tout renvoie un DataFrame vide : l'application reste en temps réel seul.
"""
import logging
import os
import threading
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytz

logger = logging.getLogger(__name__)

PARIS_TZ = pytz.timezone("Europe/Paris")
_lock = threading.Lock()
_connection = None


def lake_root() -> str:
    return os.environ.get("LAKE_ROOT", "").rstrip("/")


def is_configured() -> bool:
    return bool(lake_root())


def _silver() -> str:
    return f"delta_scan('{lake_root()}/silver/station_status')"


def _gold(table: str) -> str:
    return f"read_parquet('{lake_root()}/gold/{table}.parquet')"


def _connect():
    """
    Connexion DuckDB unique, ouverte à la première requête.
    Le secret S3 inclut le jeton de session : il est requis avec des identifiants temporaires (rôle IAM).
    """
    global _connection
    if _connection is not None:
        return _connection

    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL delta; LOAD delta;")
    if lake_root().startswith("s3://"):
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute(
            "CREATE OR REPLACE SECRET lake (TYPE s3, KEY_ID ?, SECRET ?, SESSION_TOKEN ?, REGION ?)",
            [
                os.environ.get("AWS_ACCESS_KEY_ID", ""),
                os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                os.environ.get("AWS_SESSION_TOKEN", ""),
                os.environ.get("AWS_REGION", "eu-north-1"),
            ],
        )
    _connection = con
    return con


def query(sql: str, params: list | None = None) -> pd.DataFrame:
    """Exécute une requête ; DataFrame vide (et log) si le lakehouse est inaccessible."""
    if not is_configured():
        return pd.DataFrame()
    try:
        with _lock:  # connexion partagée entre les sessions Streamlit : une requête à la fois
            return _connect().execute(sql, params or []).df()
    except Exception as e:
        logger.warning(f"Lakehouse inaccessible : {e}")
        return pd.DataFrame()


def lake_status() -> dict:
    """Résumé du silver : nombre de snapshots, dernier snapshot et qui l'a collecté."""
    df = query(f"""
        SELECT count(DISTINCT snapshot_id)  AS snapshots,
               min(ingested_at)             AS first_snapshot,
               max(ingested_at)             AS last_snapshot,
               arg_max(ingested_by, ingested_at) AS last_ingested_by
        FROM {_silver()}
    """)
    if df.empty or not df.loc[0, "snapshots"]:
        return {"available": False}
    row = df.iloc[0]
    return {
        "available": True,
        "snapshots": int(row["snapshots"]),
        "first_snapshot": pd.Timestamp(row["first_snapshot"]).tz_convert(PARIS_TZ),
        "last_snapshot": pd.Timestamp(row["last_snapshot"]).tz_convert(PARIS_TZ),
        "last_ingested_by": row["last_ingested_by"],
    }


def station_history(station_name: str, hours: int) -> pd.DataFrame:
    """
    Évolution d'une station ; ne lit que les partitions des jours concernés.
    Nom exact d'abord (plusieurs stations contiennent « BASTILLE »), sinon recherche partielle.
    """
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    since_date = since.astimezone(PARIS_TZ).strftime("%Y-%m-%d")
    sql = f"""
        SELECT ingested_at AS run_at, name, numbikesavailable, ebike, mechanical,
               numdocksavailable, bike_ratio, ingested_by
        FROM {_silver()}
        WHERE date >= ? AND ingested_at >= ? AND {{name_filter}}
        ORDER BY ingested_at
    """
    df = query(sql.format(name_filter="name = ?"), [since_date, since, station_name.upper()])
    if df.empty:
        df = query(sql.format(name_filter="name ILIKE ?"), [since_date, since, f"%{station_name.upper()}%"])
    if not df.empty:
        df["run_at"] = pd.to_datetime(df["run_at"], utc=True).dt.tz_convert(PARIS_TZ)
    return df


def latest_snapshot() -> pd.DataFrame:
    """Dernier snapshot du silver : secours quand l'API temps réel est injoignable (seuls les 2 derniers jours sont lus)."""
    df = query(f"""
        WITH s AS (SELECT * FROM {_silver()} WHERE date >= ?)
        SELECT station_id, name, numbikesavailable, mechanical, ebike, numdocksavailable,
               is_empty, is_full, coalesce(bike_ratio, 0) AS bike_ratio, lat, lon,
               ingested_at, ingested_by
        FROM s
        WHERE snapshot_id = (SELECT max(snapshot_id) FROM s)
    """, [(datetime.now(timezone.utc) - timedelta(days=2)).astimezone(PARIS_TZ).strftime("%Y-%m-%d")])
    if not df.empty:
        df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True).dt.tz_convert(PARIS_TZ)
    return df


def station_period_profiles() -> pd.DataFrame:
    """Profils gold par station, type de jour et période (reconstruits chaque jour par dbt)."""
    return query(f"SELECT * FROM {_gold('gold_station_period_profile')}")


def station_embeddings() -> pd.DataFrame:
    """Vecteurs des profils de stations, calculés par le pipeline (vide si absents)."""
    return query(f"SELECT * FROM {_gold('station_profile_embeddings')}")


def _since(days: int | None) -> tuple[str, list]:
    """
    Clause WHERE de période. Le filtre sur `date` en plus de `ingested_at` permet à DuckDB
    d'ignorer les partitions hors période. days=None : tout l'historique.
    """
    if days is None:
        return "TRUE", []
    since = datetime.now(timezone.utc) - timedelta(days=days)
    return "date >= ? AND ingested_at >= ?", [since.astimezone(PARIS_TZ).strftime("%Y-%m-%d"), since]


def network_history(days: int | None = 7) -> pd.DataFrame:
    """Indicateurs du réseau à chaque snapshot du silver : une ligne = un snapshot."""
    where, params = _since(days)
    df = query(
        f"""
        SELECT snapshot_id,
               any_value(ingested_at)            AS ingested_at,
               any_value(ingested_by)            AS ingested_by,
               count(*)                          AS n_stations,
               sum(numbikesavailable)            AS total_bikes,
               sum(ebike)                        AS total_ebike,
               sum(numdocksavailable)            AS total_docks,
               sum(is_empty::int)                AS empty_stations,
               sum(is_full::int)                 AS full_stations,
               avg(bike_ratio)                   AS avg_bike_ratio
        FROM {_silver()}
        WHERE {where}
        GROUP BY snapshot_id
        ORDER BY ingested_at
        """,
        params,
    )
    if not df.empty:
        df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True).dt.tz_convert(PARIS_TZ)
    return df


def station_stats(days: int | None = 7) -> pd.DataFrame:
    """Comportement de chaque station sur la période : une ligne = une station."""
    where, params = _since(days)
    return query(
        f"""
        SELECT station_id,
               arg_max(name, ingested_at)        AS name,
               max(lat)                          AS lat,
               max(lon)                          AS lon,
               count(*)                          AS n_snapshots,
               avg(numbikesavailable)            AS avg_bikes,
               avg(ebike)                        AS avg_ebike,
               avg(numdocksavailable)            AS avg_docks,
               avg(is_empty::int)                AS pct_empty,
               avg(is_full::int)                 AS pct_full
        FROM {_silver()}
        WHERE {where}
        GROUP BY station_id
        """,
        params,
    )
