from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from common import read_table
from services.predict_service import run_prediction


def _read_table(path: Path) -> pd.DataFrame:
    df = read_table(path, spark=None)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _infer_freq(ts: pd.Series) -> str:
    ts = pd.to_datetime(ts, errors="coerce").dropna()
    if len(ts) < 3:
        return "D"
    diffs = ts.sort_values().diff().dropna()
    if diffs.empty:
        return "D"
    mode = diffs.mode().iloc[0]
    try:
        return pd.tseries.frequencies.to_offset(mode).freqstr
    except Exception:
        return "D"


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch predict and store outputs.")
    parser.add_argument("--input", required=True, help="Gold parquet path.")
    parser.add_argument("--model", required=True, help="Model name.")
    parser.add_argument("--time-col", default="ds", help="Time column.")
    parser.add_argument("--value-col", default="y", help="Target column.")
    parser.add_argument("--series-id", default="series_id", help="Series id column.")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon.")
    parser.add_argument("--model-id", default=None, help="Optional model registry id.")
    parser.add_argument("--model-version", default=None, help="Optional model registry version.")
    parser.add_argument("--run-id", default=None, help="Optional run id for traceability.")
    parser.add_argument("--output", required=True, help="Predictions output path.")
    args = parser.parse_args()

    df = _read_table(Path(args.input))
    if df.empty:
        raise ValueError("Gold data empty")

    if args.series_id not in df.columns:
        df[args.series_id] = "default"

    preds_rows: List[Dict[str, Any]] = []

    for sid, sdf in df.groupby(args.series_id):
        sdf = sdf.sort_values(args.time_col)
        freq = _infer_freq(sdf[args.time_col])
        payload = {
            "model_name": args.model,
            "time_col": args.time_col,
            "value_col": args.value_col,
            "rows": sdf.to_dict(orient="records"),
            "horizon": int(args.horizon),
            "model_id": args.model_id,
            "model_version": args.model_version,
        }
        result = run_prediction(payload)
        preds = np.asarray(result.get("predictions") or [], dtype=float).reshape(-1)
        if preds.size == 0:
            continue
        last_ts = pd.to_datetime(sdf[args.time_col].iloc[-1], errors="coerce")
        future_ts = pd.date_range(start=last_ts, periods=len(preds) + 1, freq=freq)[1:]
        for ts, yhat in zip(future_ts, preds):
            preds_rows.append(
                {
                    args.series_id: sid,
                    args.time_col: ts,
                    "yhat": float(yhat),
                    "model_name": args.model,
                    "run_id": args.run_id,
                }
            )

    out_df = pd.DataFrame(preds_rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_df.to_parquet(output_path, index=False)
    except Exception:
        out_df.to_csv(output_path.with_suffix(".csv"), index=False)

    print(json.dumps({"rows": len(out_df), "output": str(output_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
