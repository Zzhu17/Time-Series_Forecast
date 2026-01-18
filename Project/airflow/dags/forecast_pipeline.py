from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import shlex

from airflow.models import DAG
from airflow.operators.bash import BashOperator


def _spark_submit() -> str:
    return os.getenv("SPARK_SUBMIT", "spark-submit")


def _py() -> str:
    return os.getenv("PYTHON", "python")


def _q(value: str) -> str:
    return shlex.quote(str(value))


def _dq(value: str) -> str:
    return f"\"{value}\""


PROJECT_ROOT = Path(os.getenv("TSF_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
PROJECT_DIR = Path(os.getenv("TSF_PROJECT_DIR", str(PROJECT_ROOT / "Project")))
SPARK_JOBS_DIR = Path(
    os.getenv("TSF_SPARK_JOBS_DIR", str(PROJECT_DIR / "spark_jobs"))
)

BASE_DIR = os.getenv("TSF_BASE_DIR", str(PROJECT_DIR / "Data"))
RAW_PATH = os.getenv("TSF_RAW_PATH", str(Path(BASE_DIR) / "sample_timeseries.csv"))
SOURCE = os.getenv("TSF_SOURCE", "demo")
TIME_COL = os.getenv("TSF_TIME_COL", "date")
VALUE_COL = os.getenv("TSF_VALUE_COL", "value")
SERIES_ID = os.getenv("TSF_SERIES_ID", "series_id")
MODEL_NAME = os.getenv("TSF_MODEL_NAME", "xgboost")
HORIZON = os.getenv("TSF_HORIZON", "24")
ARTIFACTS_DIR = os.getenv("TSF_ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts"))
OUTPUT_DIR = os.getenv("TSF_OUTPUT_DIR", str(PROJECT_DIR / "output"))
RUN_ID = "{{ data_interval_start.strftime('%Y-%m-%dT%H') }}"

BRONZE_PATH = os.path.join(BASE_DIR, "bronze", f"source={SOURCE}", f"dt={RUN_ID}")
SILVER_PATH = os.path.join(BASE_DIR, "silver", f"ds={RUN_ID}")
GOLD_PATH = os.path.join(BASE_DIR, "gold", f"ds={RUN_ID}")
PRED_PATH = os.path.join(BASE_DIR, "predictions", f"ds={RUN_ID}", "predictions.parquet")
DAG_DEFAULT_ARGS = {
    "owner": "tsf",
    "depends_on_past": False,
    "retries": 0,
}


with DAG(
    dag_id="forecast_platform_pipeline",
    default_args=DAG_DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["forecast", "spark"],
) as dag:
    extract_to_bronze = BashOperator(
        task_id="extract_to_bronze",
        bash_command=(
            f"{_spark_submit()} {SPARK_JOBS_DIR / 'extract_to_bronze.py'} "
            f"--input {_q(RAW_PATH)} --source {_q(SOURCE)} --dt {RUN_ID} --base-dir {_q(BASE_DIR)}"
        ),
    )

    dq_check_bronze = BashOperator(
        task_id="dq_check_bronze",
        bash_command=(
            f"{_spark_submit()} {SPARK_JOBS_DIR / 'dq_check_bronze.py'} "
            f"--input {_dq(BRONZE_PATH)} --time-col {_q(TIME_COL)} --output {_q(str(Path(OUTPUT_DIR) / 'dq_report.json'))}"
        ),
    )

    spark_clean_to_silver = BashOperator(
        task_id="spark_clean_to_silver",
        bash_command=(
            f"{_spark_submit()} {SPARK_JOBS_DIR / 'spark_clean_to_silver.py'} "
            f"--input {_dq(BRONZE_PATH)} --output {_dq(SILVER_PATH)} "
            f"--time-col {_q(TIME_COL)} --value-col {_q(VALUE_COL)}"
        ),
    )

    spark_features_to_gold = BashOperator(
        task_id="spark_features_to_gold",
        bash_command=(
            f"{_spark_submit()} {SPARK_JOBS_DIR / 'spark_features_to_gold.py'} "
            f"--input {_dq(SILVER_PATH)} --output {_dq(GOLD_PATH)} "
            f"--time-col {_q(TIME_COL)} --value-col {_q(VALUE_COL)} --series-id {_q(SERIES_ID)}"
        ),
    )

    train_and_register_model = BashOperator(
        task_id="train_and_register_model",
        bash_command=(
            f"{_py()} {SPARK_JOBS_DIR / 'train_and_register_model.py'} "
            f"--input {_dq(GOLD_PATH)} --model {_q(MODEL_NAME)} --time-col {_q(TIME_COL)} "
            f"--value-col {_q(VALUE_COL)} --horizon {HORIZON} --run-id {RUN_ID} "
            f"--output {_q(str(Path(OUTPUT_DIR) / 'run_registry.json'))}"
        ),
    )

    batch_predict_and_store = BashOperator(
        task_id="batch_predict_and_store",
        bash_command=(
            f"{_py()} {SPARK_JOBS_DIR / 'batch_predict_and_store.py'} "
            f"--input {_dq(GOLD_PATH)} --model {_q(MODEL_NAME)} --time-col {_q(TIME_COL)} "
            f"--value-col {_q(VALUE_COL)} --series-id {_q(SERIES_ID)} --horizon {HORIZON} "
            f"--output {_dq(PRED_PATH)} --run-id {RUN_ID}"
        ),
    )

    publish_leaderboard_report = BashOperator(
        task_id="publish_leaderboard_report",
        bash_command=(
            f"{_py()} {SPARK_JOBS_DIR / 'publish_leaderboard_report.py'} "
            f"--run-dir {_dq(str(Path(ARTIFACTS_DIR) / 'runs' / RUN_ID))} "
            f"--output-dir {_q(OUTPUT_DIR)}"
        ),
    )

    extract_to_bronze >> dq_check_bronze >> spark_clean_to_silver >> spark_features_to_gold
    spark_features_to_gold >> train_and_register_model >> batch_predict_and_store >> publish_leaderboard_report
