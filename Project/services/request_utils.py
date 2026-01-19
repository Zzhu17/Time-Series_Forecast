from __future__ import annotations

import json
from typing import List, Optional

import pandas as pd
from fastapi import HTTPException, UploadFile


def read_csv_upload(file: UploadFile) -> pd.DataFrame:
    try:
        return pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to read csv: {e}") from e


def read_tabular_upload(file: UploadFile) -> pd.DataFrame:
    name = str(getattr(file, "filename", "") or "").lower()
    try:
        if name.endswith(".parquet") or name.endswith(".pq"):
            return pd.read_parquet(file.file)
        return pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to read data: {e}") from e


def ensure_required_columns(df: pd.DataFrame, *cols: str) -> None:
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"CSV missing columns: {missing_cols}")


def clean_dataframe_for_json(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.replace([pd.NA, pd.NaT, float("inf"), float("-inf")], pd.NA)
    cleaned = cleaned.dropna(axis=1, how="all")
    return cleaned.where(pd.notna(cleaned), None)


def parse_feature_cols(feature_cols: Optional[str]) -> Optional[List[str]]:
    if not feature_cols:
        return None
    try:
        parsed = json.loads(feature_cols)
        if isinstance(parsed, list):
            return [str(c).strip() for c in parsed if str(c).strip()]
    except Exception:
        pass
    return [c.strip() for c in feature_cols.split(",") if c.strip()]


def parse_residual_modeling(residual_modeling: Optional[str]) -> Optional[dict]:
    if not residual_modeling:
        return None
    try:
        parsed = json.loads(residual_modeling)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def ensure_target_in_features(feature_cols: Optional[List[str]], value_col: str) -> List[str]:
    cols = [c.strip() for c in (feature_cols or []) if isinstance(c, str) and c.strip()]
    if not value_col:
        return cols
    rest = [c for c in cols if c != value_col]
    return [value_col] + rest


def auto_feature_cols(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    *,
    miss_thresh: float = 0.4,
    corr_thresh: float = 0.05,
) -> List[str]:
    """
    Auto-select numeric-like feature columns:
    - drop time/target
    - drop missing rate above threshold
    - drop low correlation vs target
    Fallback: empty list (no exogenous features).
    """
    num_cols = []
    for c in df.columns:
        if c in (time_col, value_col):
            continue
        try:
            col = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            continue
        miss = float(col.isna().mean()) if len(col) else 1.0
        if miss > miss_thresh:
            continue
        num_cols.append((c, miss, col))

    tgt = pd.to_numeric(df[value_col], errors="coerce")
    feats = []
    for name, _miss, col in num_cols:
        try:
            corr = tgt.corr(col)
        except Exception:
            corr = None
        if corr is None or pd.isna(corr) or abs(float(corr)) < corr_thresh:
            continue
        feats.append(name)

    return feats
