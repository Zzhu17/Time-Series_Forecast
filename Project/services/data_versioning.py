from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from preprocessing.cleaning import check_columns, clean_data
from preprocessing.feature_engineering import generate_features

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "Data" / "processed"


def _hash_dataframe(df: pd.DataFrame) -> str:
    try:
        hashed = pd.util.hash_pandas_object(df, index=True).values
        payload = hashed.tobytes()
    except Exception:
        payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def compute_dataset_id(df: pd.DataFrame) -> str:
    return _hash_dataframe(df)


def _profile_dataframe(df: pd.DataFrame, *, time_col: str, value_col: str) -> Dict[str, Any]:
    profile: Dict[str, Any] = {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "columns": [str(c) for c in df.columns],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    try:
        ts = pd.to_datetime(df[time_col], errors="coerce") if time_col in df.columns else None
        if ts is not None and not ts.empty:
            profile["time_min"] = str(ts.min())
            profile["time_max"] = str(ts.max())
    except Exception:
        pass

    missing_rate_by_col = {}
    try:
        for col in df.columns:
            missing_rate_by_col[str(col)] = float(df[col].isna().mean())
    except Exception:
        missing_rate_by_col = {}
    profile["missing_rate_by_col"] = missing_rate_by_col
    try:
        profile["missing_rate_total"] = float(df.isna().mean().mean())
    except Exception:
        profile["missing_rate_total"] = None

    if value_col in df.columns:
        try:
            values = pd.to_numeric(df[value_col], errors="coerce")
            profile["value_stats"] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "p05": float(values.quantile(0.05)),
                "p95": float(values.quantile(0.95)),
            }
        except Exception:
            profile["value_stats"] = None

    return profile


def _apply_outlier_clip(df: pd.DataFrame, cols: list[str], lower_q: float, upper_q: float) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        if series.dropna().empty:
            continue
        lo = float(series.quantile(lower_q))
        hi = float(series.quantile(upper_q))
        out[col] = series.clip(lo, hi)
    return out


def preprocess_dataframe(
    df: pd.DataFrame,
    *,
    config: Dict[str, Any],
    time_col: str,
    value_col: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a DataFrame")

    work = df.copy()
    work = check_columns(work, time_col=time_col, value_col=value_col)

    prep_cfg = config.get("data_prep", {}) if isinstance(config, dict) else {}

    missing_cfg = prep_cfg.get("missing", {}) if isinstance(prep_cfg, dict) else {}
    strategy = str(missing_cfg.get("strategy", "drop")).lower()
    fill_value = missing_cfg.get("fill_value", 0)
    fill_method = str(missing_cfg.get("fill_method", "")).lower()

    if strategy == "fill":
        if fill_method in ("ffill", "bfill"):
            work = work.fillna(method=fill_method)
        else:
            work = work.fillna(fill_value)

    # Basic cleaning: drop NaN target, coerce numeric
    work = clean_data(work, value_col=value_col)

    # Optional resample
    resample_cfg = prep_cfg.get("resample", {}) if isinstance(prep_cfg, dict) else {}
    if bool(resample_cfg.get("enabled", False)):
        freq = str(resample_cfg.get("freq", "D"))
        agg = str(resample_cfg.get("agg", "mean"))
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.dropna(subset=[time_col])
        work = work.set_index(time_col).sort_index()
        numeric_cols = work.select_dtypes(include=[np.number]).columns
        if agg == "sum":
            work = work[numeric_cols].resample(freq).sum()
        else:
            work = work[numeric_cols].resample(freq).mean()
        work = work.reset_index()

    # Optional outlier clipping (winsorize)
    outlier_cfg = prep_cfg.get("outlier", {}) if isinstance(prep_cfg, dict) else {}
    if bool(outlier_cfg.get("enabled", True)):
        lower_q = float(outlier_cfg.get("lower_q", 0.01))
        upper_q = float(outlier_cfg.get("upper_q", 0.99))
        cols = outlier_cfg.get("columns")
        if not isinstance(cols, list) or not cols:
            cols = [value_col]
        work = _apply_outlier_clip(work, [str(c) for c in cols], lower_q, upper_q)

    # Feature engineering (time features)
    try:
        work, time_features = generate_features(work, config, manage_feature_cols=False)
    except Exception:
        time_features = []

    profile = _profile_dataframe(work, time_col=time_col, value_col=value_col)
    if time_features:
        profile["engineered_time_features"] = list(time_features)

    return work, profile


def save_processed_assets(
    df: pd.DataFrame,
    *,
    profile: Dict[str, Any],
    artifacts_dir: str,
) -> Dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset_id = compute_dataset_id(df)

    processed_path = DATA_DIR / "processed.parquet"
    profile_path = DATA_DIR / "data_profile.json"

    artifacts = {
        "dataset_id": dataset_id,
        "processed_data_path": str(processed_path),
        "data_profile_path": str(profile_path),
    }

    profile = dict(profile) if isinstance(profile, dict) else {}
    profile["dataset_id"] = dataset_id

    # Save parquet (fallback to csv if parquet engine missing)
    parquet_error = None
    try:
        df.to_parquet(processed_path, index=False)
    except Exception as exc:
        parquet_error = str(exc)
        try:
            df.to_csv(str(processed_path).replace(".parquet", ".csv"), index=False)
            artifacts["processed_data_path"] = str(processed_path).replace(".parquet", ".csv")
        except Exception:
            pass

    if parquet_error:
        profile["parquet_error"] = parquet_error

    with profile_path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    # Also save into run-specific artifacts dir
    try:
        run_dir = Path(artifacts_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_parquet = run_dir / "processed.parquet"
        df.to_parquet(run_parquet, index=False)
        run_profile = run_dir / "data_profile.json"
        with run_profile.open("w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        artifacts["run_processed_data_path"] = str(run_parquet)
        artifacts["run_data_profile_path"] = str(run_profile)
    except Exception:
        pass

    return artifacts
