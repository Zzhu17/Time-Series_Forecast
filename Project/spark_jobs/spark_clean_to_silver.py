from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import get_spark, read_table, write_parquet

try:
    from pyspark.sql import functions as F  
except Exception:
    F = None 


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean bronze -> silver.")
    parser.add_argument("--input", required=True, help="Bronze parquet path.")
    parser.add_argument("--output", required=True, help="Silver output path.")
    parser.add_argument("--time-col", default="ds", help="Time column name.")
    parser.add_argument("--value-col", default="y", help="Target column name.")
    parser.add_argument("--resample-freq", default="", help="Optional resample freq (e.g., 1 day).")
    parser.add_argument("--outlier-clip", action="store_true", help="Enable quantile clipping on value_col.")
    args = parser.parse_args()

    spark = get_spark("spark_clean_to_silver")
    df = read_table(Path(args.input), spark)

    if spark is not None and F is not None and not isinstance(df, pd.DataFrame):
        sdf = df
        sdf = sdf.withColumn(args.time_col, F.to_timestamp(F.col(args.time_col)))
        sdf = sdf.filter(F.col(args.time_col).isNotNull())
        sdf = sdf.withColumn(args.value_col, F.col(args.value_col).cast("double"))
        sdf = sdf.filter(F.col(args.value_col).isNotNull())
        if args.outlier_clip:
            quantiles = sdf.approxQuantile(args.value_col, [0.01, 0.99], 0.01)
            if len(quantiles) == 2:
                lo, hi = quantiles
                sdf = sdf.withColumn(
                    args.value_col,
                    F.when(F.col(args.value_col) < lo, lo)
                    .when(F.col(args.value_col) > hi, hi)
                    .otherwise(F.col(args.value_col)),
                )
        if args.resample_freq:
            sdf = (
                sdf.groupBy(F.window(F.col(args.time_col), args.resample_freq).alias("w"))
                .agg(F.avg(F.col(args.value_col)).alias(args.value_col))
                .withColumn(args.time_col, F.col("w.start"))
                .drop("w")
            )
        write_parquet(sdf, Path(args.output))
        spark.stop()
        return 0

    pdf = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    pdf[args.time_col] = pd.to_datetime(pdf[args.time_col], errors="coerce")
    pdf = pdf.dropna(subset=[args.time_col, args.value_col])
    pdf[args.value_col] = pd.to_numeric(pdf[args.value_col], errors="coerce")
    pdf = pdf.dropna(subset=[args.value_col])
    if args.outlier_clip and not pdf.empty:
        lo = pdf[args.value_col].quantile(0.01)
        hi = pdf[args.value_col].quantile(0.99)
        pdf[args.value_col] = pdf[args.value_col].clip(lo, hi)
    if args.resample_freq:
        pdf = pdf.set_index(args.time_col).sort_index()
        pdf = pdf.resample(args.resample_freq).mean().reset_index()
    write_parquet(pdf, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
