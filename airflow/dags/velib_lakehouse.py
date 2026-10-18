"""
velib_lakehouse.py — DAGs Airflow 3 du lakehouse Vélib' (collecteur principal).

- velib_ingest_hourly : API Vélib' -> bronze -> silver, toutes les heures à :05
- velib_daily_gold    : OPTIMIZE + VACUUM silver, gold dbt (Cosmos), embeddings, météo et monitoring, à 13:15
- velib_train_forecast : entraînement des modèles de prévision, lancé à la main (bouton Trigger)

Le code métier est celui de pipeline/ (le même que GitHub Actions, collecteur de secours).
Il tourne dans /opt/velib-venv via @task.external_python pour isoler ses dépendances
de celles d'Airflow.
"""
import os
from datetime import timedelta

import pendulum
from airflow.sdk import Param, dag, task
from cosmos import DbtTaskGroup, ExecutionConfig, ProfileConfig, ProjectConfig, RenderConfig
from cosmos.constants import TestBehavior

PROJECT_DIR = os.environ.get("PROJECT_DIR", "/opt/project")
VENV_PYTHON = "/opt/velib-venv/bin/python"
DBT_EXECUTABLE = "/opt/velib-venv/bin/dbt"
DBT_TARGET = os.environ.get("DBT_TARGET", "prod")

START = pendulum.datetime(2026, 10, 1, tz="Europe/Paris")
DEFAULT_ARGS = {
    "owner": "velib",
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}


@dag(
    dag_id="velib_ingest_hourly",
    schedule="5 * * * *",
    start_date=START,
    # Pas de rattrapage : l'API ne donne que l'état courant, une heure manquée est perdue
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=20),
    default_args=DEFAULT_ARGS,
    tags=["velib", "bronze", "silver"],
    doc_md=__doc__,
)
def velib_ingest_hourly():
    @task.external_python(python=VENV_PYTHON, expect_airflow=False)
    def ingest() -> dict:
        import sys
        sys.path.insert(0, "/opt/project")
        from pipeline.run import ingest as run_ingest
        return run_ingest()

    ingest()


@dag(
    dag_id="velib_daily_gold",
    # En journée : le PC qui héberge Airflow est allumé (GitHub prend le relais le soir sinon)
    schedule="15 13 * * *",
    start_date=START,
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=1),
    default_args=DEFAULT_ARGS,
    tags=["velib", "silver", "gold", "dbt"],
    doc_md=__doc__,
)
def velib_daily_gold():
    @task.external_python(python=VENV_PYTHON, expect_airflow=False)
    def optimize_vacuum_silver() -> dict:
        import sys
        sys.path.insert(0, "/opt/project")
        from pipeline.lake import maintain_silver
        return maintain_silver()

    # Un modèle dbt = une tâche Airflow, testée juste après sa construction
    gold = DbtTaskGroup(
        group_id="dbt_gold",
        project_config=ProjectConfig(f"{PROJECT_DIR}/dbt"),
        profile_config=ProfileConfig(
            profile_name="velib",
            target_name=DBT_TARGET,
            profiles_yml_filepath=f"{PROJECT_DIR}/dbt/profiles.yml",
        ),
        execution_config=ExecutionConfig(dbt_executable_path=DBT_EXECUTABLE),
        render_config=RenderConfig(test_behavior=TestBehavior.AFTER_EACH),
        # DuckDB n'accepte qu'un écrivain : les tâches dbt passent une par une
        operator_args={"install_deps": False, "pool": "dbt_duckdb"},
        default_args=DEFAULT_ARGS,
    )

    @task.external_python(python=VENV_PYTHON, expect_airflow=False)
    def embed_profiles() -> dict:
        import sys
        sys.path.insert(0, "/opt/project")
        from pipeline.embed import run_embed
        # Vecteurs des profils écrits dans le gold + Qdrant réindexé : l'app n'encode plus que la question
        return run_embed()

    @task.external_python(python=VENV_PYTHON, expect_airflow=False)
    def fetch_external() -> dict:
        import sys
        sys.path.insert(0, "/opt/project")
        from pipeline.external import run_external
        # Météo (Open-Meteo) et vacances scolaires : features du modèle de prévision
        return run_external()

    @task.external_python(python=VENV_PYTHON, expect_airflow=False)
    def monitor_model() -> dict:
        import sys
        sys.path.insert(0, "/opt/project")
        from pipeline.monitoring import evaluate_production
        # Prévisions des dernières 24 h comparées à la réalité, calibration, dérive des features
        return evaluate_production()

    @task.external_python(python=VENV_PYTHON, expect_airflow=False)
    def mark_daily_done() -> str:
        import sys
        sys.path.insert(0, "/opt/project")
        from pipeline.control import mark_done
        # Signale à GitHub Actions (secours) que le job du jour est fait
        return mark_done("daily")

    external, monitor = fetch_external(), monitor_model()
    embed, done = embed_profiles(), mark_daily_done()

    optimize_vacuum_silver() >> gold >> [embed, external]
    external >> monitor
    [embed, monitor] >> done


@dag(
    dag_id="velib_train_forecast",
    # Pas de planification : l'entraînement se lance à la main, avec ses paramètres
    schedule=None,
    start_date=START,
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=1),
    default_args={**DEFAULT_ARGS, "retries": 0},
    render_template_as_native_obj=True,
    params={
        "min_days": Param(7, type="integer", minimum=1, description="Historique minimal exigé, en jours"),
        "publish": Param(True, type="boolean",
                         description="Publier les modèles (servis par l'app s'ils battent les baselines)"),
    },
    tags=["velib", "ml"],
    doc_md=__doc__,
)
def velib_train_forecast():
    @task.external_python(python=VENV_PYTHON, expect_airflow=False)
    def train_forecast(min_days: int, publish: bool) -> dict:
        import sys
        sys.path.insert(0, "/opt/project")
        from pipeline.forecast import train_forecast as train
        return train(min_days=int(min_days), publish=bool(publish))

    train_forecast("{{ params.min_days }}", "{{ params.publish }}")


velib_ingest_hourly()
velib_daily_gold()
velib_train_forecast()
