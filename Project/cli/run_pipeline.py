from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
yaml = importlib.import_module("yaml")

from configs.config import load_yaml_config
from services.pipeline import run_pipeline_and_update_state
from services.request_utils import resolve_feature_cols
from services.snapshot import safe_jsonify
from services.train_service import build_training_config


DEFAULT_OUTPUT = PROJECT_DIR / "output" / "cli_last_run.json"


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg if isinstance(cfg, dict) else {}
    try:
        cfg = load_yaml_config()
    except Exception:
        cfg = {}
    return cfg if isinstance(cfg, dict) else {}


def _parse_feature_cols(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    if raw.strip().lower() == "auto":
        return None
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    return cols or None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(safe_jsonify(payload), f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run forecasting pipeline from CLI.")
    parser.add_argument("--data", required=True, help="Path to CSV/Parquet file.")
    parser.add_argument("--model", default="baseline", help="Model name (baseline/informer/lstm/xgboost/arima/prophet...).")
    parser.add_argument("--time-col", default="date", help="Time column name.")
    parser.add_argument("--value-col", default="value", help="Target column name.")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon.")
    parser.add_argument("--feature-cols", default="auto", help="Comma-separated feature cols or 'auto'.")
    parser.add_argument("--config", default=None, help="Optional YAML config path.")
    parser.add_argument("--device", default="cpu", help="cpu/cuda")
    parser.add_argument("--allow-degrade", action="store_true", help="Allow baseline fallback.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Write JSON summary to this path.")

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = _read_table(data_path)
    if df.empty:
        raise ValueError("Input data is empty.")

    config = _load_config(args.config)
    feature_cols = _parse_feature_cols(args.feature_cols)
    feature_cols = resolve_feature_cols(
        df,
        feature_cols=feature_cols,
        time_col=args.time_col,
        value_col=args.value_col,
        auto_select_features=not feature_cols,
    )

    run_id = str(uuid.uuid4())
    config = build_training_config(
        df=df,
        task_id=run_id,
        model_name=args.model,
        time_col=args.time_col,
        value_col=args.value_col,
        feature_cols=feature_cols,
        residual_modeling=config.get("residual_modeling"),
        device=args.device,
    )
    config.setdefault("prediction", {})["horizon"] = int(args.horizon)

    results = run_pipeline_and_update_state(
        df=df.copy(),
        config=config,
        feature_cols=feature_cols,
        uploaded_name=data_path.name,
        model_name=args.model,
        time_col=args.time_col,
        value_col=args.value_col,
        allow_degrade=bool(args.allow_degrade),
    )

    payload = {
        "run_id": run_id,
        "status": results.get("status"),
        "metrics": results.get("metrics"),
        "data": results.get("data"),
        "artifacts": results.get("artifacts"),
    }
    _write_json(Path(args.output), payload)

    print(json.dumps(safe_jsonify(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
