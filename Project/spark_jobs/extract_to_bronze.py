from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from common import get_spark, read_table, write_parquet


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract raw data into bronze layer.")
    parser.add_argument("--input", required=True, help="Input CSV/Parquet path.")
    parser.add_argument("--source", required=True, help="Data source name.")
    parser.add_argument("--dt", default=None, help="Partition date (YYYY-MM-DD).")
    parser.add_argument("--base-dir", default="Project/Data", help="Base data directory.")
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    dt = args.dt or datetime.utcnow().strftime("%Y-%m-%d")
    bronze_dir = Path(args.base_dir) / "bronze" / f"source={args.source}" / f"dt={dt}"
    out_path = bronze_dir / "data.parquet"

    spark = get_spark("extract_to_bronze")
    df = read_table(data_path, spark)
    write_parquet(df, out_path)

    if spark is not None:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
