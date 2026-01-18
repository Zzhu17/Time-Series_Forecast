from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


try:  # optional spark dependency
    from pyspark.sql import SparkSession  # type: ignore
    from pyspark.sql import functions as F  # type: ignore
    _HAS_SPARK = True
except Exception:  # pragma: no cover
    SparkSession = None  # type: ignore
    F = None  # type: ignore
    _HAS_SPARK = False


def get_spark(app_name: str) -> Optional["SparkSession"]:
    if not _HAS_SPARK:
        return None
    return SparkSession.builder.appName(app_name).getOrCreate()


def _resolve_input_path(path: Path) -> Path:
    if path.is_dir():
        parquet_files = sorted(path.glob("*.parquet"))
        if parquet_files:
            return parquet_files[0]
        csv_files = sorted(path.glob("*.csv"))
        if csv_files:
            return csv_files[0]
        raise FileNotFoundError(f"No .parquet or .csv files found in {path}")
    return path


def read_table(path: Path, spark: Optional["SparkSession"] = None):
    resolved = _resolve_input_path(path)
    if spark is not None:
        if resolved.suffix.lower() in (".parquet", ".pq"):
            return spark.read.parquet(str(resolved))
        if resolved.suffix.lower() in (".csv",):
            return spark.read.option("header", True).csv(str(resolved))
        try:
            return spark.read.parquet(str(resolved))
        except Exception:
            return spark.read.option("header", True).csv(str(resolved))
    if resolved.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(resolved)
    if resolved.suffix.lower() in (".csv",):
        return pd.read_csv(resolved)
    try:
        return pd.read_parquet(resolved)
    except Exception:
        return pd.read_csv(resolved)


def write_parquet(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_SPARK and not isinstance(df, pd.DataFrame):
        df.write.mode("overwrite").parquet(str(path))
        return
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_csv(path.with_suffix(".csv"), index=False)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def now_utc() -> str:
    return datetime.utcnow().isoformat() + "Z"
