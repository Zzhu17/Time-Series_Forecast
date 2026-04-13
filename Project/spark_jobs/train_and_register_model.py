from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from common import read_table
from services.data_versioning import compute_dataset_id
from services.request_utils import resolve_feature_cols
from services.train_service import run_training_task


def _read_table(path: Path) -> pd.DataFrame:
    df = read_table(path, spark=None)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _write_registry(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                data.append(record)
            else:
                data = [data, record]
        except Exception:
            data = [record]
    else:
        data = [record]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and register model from gold table.")
    parser.add_argument("--input", required=True, help="Gold parquet path.")
    parser.add_argument("--model", default="xgboost", help="Model name.")
    parser.add_argument("--time-col", default="ds", help="Time column.")
    parser.add_argument("--value-col", default="y", help="Target column.")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon.")
    parser.add_argument("--feature-cols", default="auto", help="Comma-separated feature cols or auto.")
    parser.add_argument("--run-id", default=None, help="Optional run id (defaults to dataset_version).")
    parser.add_argument("--output", default="Project/output/run_registry.json", help="Run registry JSON path.")
    args = parser.parse_args()

    df = _read_table(Path(args.input))
    if df.empty:
        raise ValueError("Gold data empty")

    feature_cols: List[str]
    raw_feature_cols = None if args.feature_cols.strip().lower() == "auto" else [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    feature_cols = resolve_feature_cols(
        df,
        feature_cols=raw_feature_cols,
        time_col=args.time_col,
        value_col=args.value_col,
        auto_select_features=raw_feature_cols is None,
    )

    dataset_version = compute_dataset_id(df)
    run_id = args.run_id or dataset_version

    payload = {
        "model_name": args.model,
        "time_col": args.time_col,
        "value_col": args.value_col,
        "horizon": int(args.horizon),
        "rows": df.to_dict(orient="records"),
        "feature_cols": feature_cols,
        "allow_degrade": False,
        "run_id": run_id,
    }

    result = run_training_task(payload, task_id=run_id)

    record = {
        "run_id": run_id,
        "dataset_version": dataset_version,
        "model_version": result.get("model_record", {}).get("version") if isinstance(result, dict) else None,
        "model_name": args.model,
        "metrics": result.get("metrics") if isinstance(result, dict) else None,
        "artifacts": result.get("artifacts") if isinstance(result, dict) else None,
    }
    _write_registry(Path(args.output), record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
