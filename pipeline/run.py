"""
run.py — Point d'entrée du pipeline (exécuté par GitHub Actions).

    python -m pipeline.run ingest     # API -> bronze -> silver
    python -m pipeline.run maintain   # OPTIMIZE + VACUUM de la table silver
    python -m pipeline.run backfill   # importe les anciens CSV S3 dans silver
    python -m pipeline.run embed      # embeddings des profils gold (+ Qdrant)
    python -m pipeline.run forecast   # entraîne les modèles de prévision (lancé à la main)
        [--min-days 3] [--no-publish]  # historique minimal ; test sans servir les modèles dans l'app
    python -m pipeline.run external   # météo et vacances scolaires (features du modèle)
    python -m pipeline.run monitor    # compare les prévisions passées à la réalité (monitoring)
"""
import argparse
import json
import logging

from pipeline.extract import fetch_raw
from pipeline.lake import ensure_local_layout, maintain_silver, merge_silver, snapshot_exists, write_bronze
from pipeline.settings import lake_root, orchestrator
from pipeline.transform import partition_date, snapshot_id_for, snapshot_time, to_silver

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("pipeline")


def ingest() -> dict:
    """
    Collecte le snapshot de l'heure courante, sauf s'il existe déjà.

    Airflow (PC local) est le collecteur principal ; GitHub Actions passe plus tard
    dans l'heure et ne fait rien si Airflow a déjà écrit le snapshot.
    """
    ts = snapshot_time()
    snapshot_id = snapshot_id_for(ts)
    logger.info(f"=== Ingestion {snapshot_id} par {orchestrator()} vers {lake_root()} ===")

    already_by = snapshot_exists(snapshot_id, partition_date(ts))
    if already_by:
        logger.info(f"Snapshot {snapshot_id} déjà collecté par {already_by} : rien à faire.")
        return {"snapshot_id": snapshot_id, "skipped": True, "already_ingested_by": already_by}

    ensure_local_layout()
    records = fetch_raw()
    bronze_file = write_bronze(records, ts, snapshot_id)
    silver = to_silver(records, ts)
    inserted = merge_silver(silver)

    # Monitoring : on garde la prévision de l'heure suivante pour la comparer à la réalité
    try:
        from pipeline.monitoring import log_predictions
        predictions_logged = log_predictions()
    except Exception as e:  # le suivi du modèle ne doit jamais bloquer la collecte
        logger.warning(f"Prévisions non enregistrées : {e}")
        predictions_logged = 0

    return {
        "snapshot_id": snapshot_id,
        "skipped": False,
        "ingested_by": orchestrator(),
        "stations": silver.num_rows,
        "silver_rows_inserted": inserted,
        "predictions_logged": predictions_logged,
        "bronze_file": bronze_file,
    }


def main():
    parser = argparse.ArgumentParser(description="Pipeline lakehouse Vélib'")
    parser.add_argument("command", choices=["ingest", "maintain", "backfill", "embed", "forecast", "external", "monitor"])
    parser.add_argument("--min-days", type=int, default=None, help="forecast : historique minimal en jours")
    parser.add_argument("--no-publish", action="store_true", help="forecast : test, modèles non servis par l'app")
    args = parser.parse_args()

    if args.command == "ingest":
        result = ingest()
    elif args.command == "maintain":
        result = maintain_silver()
    elif args.command == "backfill":
        from pipeline.backfill import backfill
        result = backfill()
    elif args.command == "embed":
        from pipeline.embed import run_embed
        result = run_embed()
    elif args.command == "forecast":
        from pipeline.forecast import train_forecast
        result = train_forecast(min_days=args.min_days, publish=not args.no_publish)
    elif args.command == "external":
        from pipeline.external import run_external
        result = run_external()
    else:
        from pipeline.monitoring import evaluate_production
        result = evaluate_production()

    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
