"""
pipeline/airflow_dag.py
=======================
Apache Airflow DAG for the FDA Drug Safety Signal Detection pipeline.

Schedule: @quarterly  — FDA publishes new FAERS data every quarter.
Each run:
    1. Detect the newest quarter available from openFDA
    2. Download it
    3. PySpark-parse to Parquet
    4. Push to HDFS
    5. Add the new Hive partition
    6. Run signal_detection.hql (PRR + ROR)
    7. Rebuild ML features
    8. Score predictions with XGBoost
    9. Push results to PostgreSQL
   10. Generate the PDF summary + alert on HIGH-confidence signals

Deploy: drop this file into ./pipeline/ — docker-compose mounts that into
/opt/airflow/dags inside the airflow-webserver container.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator   # noqa
from airflow.utils.dates import days_ago

ROOT = "/opt/airflow"                   # mounted from ./
sys.path.append(ROOT)

DEFAULT_ARGS = {
    "owner": "saif.mohammed",
    "depends_on_past": False,
    "email": ["smohammed8@seattleu.edu"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


# ---------------------------------------------------------------------------- #
# Helpers                                                                       #
# ---------------------------------------------------------------------------- #
def _current_quarter(execution_date: datetime) -> str:
    """Return 'YYYYqN' for the execution-date's quarter."""
    q = (execution_date.month - 1) // 3 + 1
    return f"{execution_date.year}q{q}"


def detect_new_quarter(**ctx) -> str:
    quarter = _current_quarter(ctx["execution_date"])
    ctx["ti"].xcom_push(key="quarter", value=quarter)
    print(f"Target quarter: {quarter}")
    return quarter


def alert_high_confidence(**ctx) -> None:
    """Print HIGH confidence signals — could send email / Slack here."""
    import psycopg2
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        dbname=os.getenv("POSTGRES_DB", "fda_signals"),
        user=os.getenv("POSTGRES_USER", "fda"),
        password=os.getenv("POSTGRES_PASSWORD", "fda"),
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT drug_name, reaction_term, case_count, prr, ror
            FROM   pharma.drug_signals
            WHERE  signal_status='STRONG_SIGNAL' AND confidence='HIGH'
            ORDER  BY prr DESC LIMIT 25
            """
        )
        rows = cur.fetchall()
    conn.close()
    print(f"==== {len(rows)} HIGH-confidence signals this run ====")
    for r in rows:
        print(r)


# ---------------------------------------------------------------------------- #
# DAG                                                                           #
# ---------------------------------------------------------------------------- #
with DAG(
    dag_id="fda_signal_detection",
    description="FDA FAERS Drug Safety Signal Detection — quarterly pipeline",
    default_args=DEFAULT_ARGS,
    start_date=days_ago(1),
    schedule_interval="@quarterly",
    catchup=False,
    tags=["pharmacovigilance", "fda", "hive", "spark", "xgboost"],
) as dag:

    detect = PythonOperator(
        task_id="detect_new_quarter",
        python_callable=detect_new_quarter,
        provide_context=True,
    )

    download = BashOperator(
        task_id="download_faers",
        bash_command=(
            "python {{ params.root }}/ingestion/download_faers.py "
            "--quarter {{ ti.xcom_pull(key='quarter') }}"
        ),
        params={"root": ROOT},
    )

    parse = BashOperator(
        task_id="parse_to_parquet",
        bash_command=(
            "spark-submit --master spark://spark-master:7077 "
            "{{ params.root }}/ingestion/parse_faers.py "
            "--quarter {{ ti.xcom_pull(key='quarter') }}"
        ),
        params={"root": ROOT},
    )

    load_hdfs = BashOperator(
        task_id="load_to_hdfs",
        bash_command="python {{ params.root }}/ingestion/load_to_hdfs.py",
        params={"root": ROOT},
    )

    add_partition = BashOperator(
        task_id="add_hive_partition",
        bash_command=(
            "beeline -u 'jdbc:hive2://hive-server:10000' "
            "-f {{ params.root }}/hive/partitioned_tables.hql"
        ),
        params={"root": ROOT},
    )

    run_signal_detection = BashOperator(
        task_id="run_signal_detection",
        bash_command=(
            "beeline -u 'jdbc:hive2://hive-server:10000' "
            "-f {{ params.root }}/hive/signal_detection.hql"
        ),
        params={"root": ROOT},
    )

    feature_eng = BashOperator(
        task_id="feature_engineering",
        bash_command=(
            "spark-submit --master spark://spark-master:7077 "
            "{{ params.root }}/spark/feature_engineering.py"
        ),
        params={"root": ROOT},
    )

    predict = BashOperator(
        task_id="run_ml_predictions",
        bash_command="python {{ params.root }}/ml/predictor.py",
        params={"root": ROOT},
    )

    update_pg = BashOperator(
        task_id="update_postgres",
        bash_command="python {{ params.root }}/ml/evaluate.py",
        params={"root": ROOT},
    )

    report = BashOperator(
        task_id="generate_report",
        bash_command=(
            "python -c 'from api.report import build_pdf_report; build_pdf_report()'"
        ),
    )

    alert = PythonOperator(
        task_id="alert_high_confidence",
        python_callable=alert_high_confidence,
        provide_context=True,
    )

    detect >> download >> parse >> load_hdfs >> add_partition \
        >> run_signal_detection >> feature_eng >> predict >> update_pg \
        >> report >> alert
