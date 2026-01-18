from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import get_spark, read_table, write_json, now_utc


def _infer_freq(ts: pd.Series) -> str:
    ts = pd.to_datetime(ts, errors="coerce").dropna()
    if len(ts) < 3:
        return "unknown"
    diffs = ts.sort_values().diff().dropna()
    if diffs.empty:
        return "unknown"
    return str(diffs.mode().iloc[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="DQ check for bronze layer.")
    parser.add_argument("--input", required=True, help="Bronze path (parquet folder).")
    parser.add_argument("--time-col", default="ds", help="Time column name.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    spark = get_spark("dq_check_bronze")
    df = read_table(input_path, spark)

    if spark is not None and not isinstance(df, pd.DataFrame):
        sdf = df
        row_count = sdf.count()
        cols = sdf.columns
        missing_rate = {}
        for c in cols:
            miss = sdf.filter(sdf[c].isNull()).count()
            missing_rate[c] = float(miss) / float(row_count) if row_count else 0.0
        dup_count = sdf.count() - sdf.dropDuplicates().count()
        out = {
            "rows": int(row_count),
            "missing_rate_by_col": missing_rate,
            "duplicate_rows": int(dup_count),
            "generated_at": now_utc(),
        }
        if args.time_col in cols:
            ts = sdf.select(args.time_col).toPandas()[args.time_col]
            out["time_min"] = str(pd.to_datetime(ts, errors="coerce").min())
            out["time_max"] = str(pd.to_datetime(ts, errors="coerce").max())
            out["freq_mode"] = _infer_freq(ts)
        write_json(Path(args.output), out)
        spark.stop()
        return 0

    pdf = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    row_count = int(len(pdf))
    missing_rate = {str(c): float(pdf[c].isna().mean()) for c in pdf.columns}
    dup_count = int(pdf.duplicated().sum())
    out = {
        "rows": row_count,
        "missing_rate_by_col": missing_rate,
        "duplicate_rows": dup_count,
        "generated_at": now_utc(),
    }
    if args.time_col in pdf.columns:
        ts = pd.to_datetime(pdf[args.time_col], errors="coerce")
        out["time_min"] = str(ts.min())
        out["time_max"] = str(ts.max())
        out["freq_mode"] = _infer_freq(ts)
    write_json(Path(args.output), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
