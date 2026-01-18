from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import get_spark, read_table, write_parquet

try:
    from pyspark.sql import functions as F 
    from pyspark.sql import Window 
except Exception:
    F = None 
    Window = None  


def _parse_int_list(raw: str) -> list[int]:
    if not raw:
        return []
    return [int(x) for x in raw.split(",") if x.strip().isdigit()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create gold features from silver.")
    parser.add_argument("--input", required=True, help="Silver parquet path.")
    parser.add_argument("--output", required=True, help="Gold output path.")
    parser.add_argument("--time-col", default="ds", help="Time column.")
    parser.add_argument("--value-col", default="y", help="Target column.")
    parser.add_argument("--series-id", default="series_id", help="Series id column.")
    parser.add_argument("--lags", default="1,2,3,6,12", help="Comma-separated lags.")
    parser.add_argument("--rolls", default="6,12,24", help="Comma-separated rolling windows.")
    args = parser.parse_args()

    lags = _parse_int_list(args.lags)
    rolls = _parse_int_list(args.rolls)

    spark = get_spark("spark_features_to_gold")
    df = read_table(Path(args.input), spark)

    if spark is not None and F is not None and Window is not None and not isinstance(df, pd.DataFrame):
        sdf = df
        if args.series_id not in sdf.columns:
            sdf = sdf.withColumn(args.series_id, F.lit("default"))
        sdf = sdf.withColumn(args.time_col, F.to_timestamp(F.col(args.time_col)))
        sdf = sdf.withColumn(args.value_col, F.col(args.value_col).cast("double"))
        sdf = sdf.filter(F.col(args.value_col).isNotNull())

        w = Window.partitionBy(args.series_id).orderBy(F.col(args.time_col))
        for k in lags:
            sdf = sdf.withColumn(f"lag_{k}", F.lag(F.col(args.value_col), k).over(w))
        for wlen in rolls:
            w_roll = w.rowsBetween(-wlen, -1)
            sdf = sdf.withColumn(f"rolling_mean_{wlen}", F.avg(F.col(args.value_col)).over(w_roll))
            sdf = sdf.withColumn(f"rolling_std_{wlen}", F.stddev(F.col(args.value_col)).over(w_roll))

        sdf = sdf.withColumn("month", F.month(F.col(args.time_col)))
        sdf = sdf.withColumn("day_of_month", F.dayofmonth(F.col(args.time_col)))
        sdf = sdf.withColumn("day_of_week", F.dayofweek(F.col(args.time_col)))
        sdf = sdf.withColumn("hour", F.hour(F.col(args.time_col)))
        sdf = sdf.withColumn("day_of_year", F.dayofyear(F.col(args.time_col)))

        write_parquet(sdf, Path(args.output))
        spark.stop()
        return 0

    pdf = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    if args.series_id not in pdf.columns:
        pdf[args.series_id] = "default"
    pdf[args.time_col] = pd.to_datetime(pdf[args.time_col], errors="coerce")
    pdf = pdf.sort_values([args.series_id, args.time_col])
    pdf[args.value_col] = pd.to_numeric(pdf[args.value_col], errors="coerce")
    pdf = pdf.dropna(subset=[args.value_col])

    for k in lags:
        pdf[f"lag_{k}"] = pdf.groupby(args.series_id)[args.value_col].shift(k)
    for wlen in rolls:
        pdf[f"rolling_mean_{wlen}"] = (
            pdf.groupby(args.series_id)[args.value_col].shift(1).rolling(wlen).mean().reset_index(level=0, drop=True)
        )
        pdf[f"rolling_std_{wlen}"] = (
            pdf.groupby(args.series_id)[args.value_col].shift(1).rolling(wlen).std().reset_index(level=0, drop=True)
        )

    pdf["month"] = pdf[args.time_col].dt.month
    pdf["day_of_month"] = pdf[args.time_col].dt.day
    pdf["day_of_week"] = pdf[args.time_col].dt.dayofweek
    pdf["hour"] = pdf[args.time_col].dt.hour
    pdf["day_of_year"] = pdf[args.time_col].dt.dayofyear

    write_parquet(pdf, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
