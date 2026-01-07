from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.feature_contract import (
    ensure_calendar_features,
    is_recomputable_name,
    parse_recompute_name,
    recompute_feature_column,
    safe_time_features,
)

try:  # pragma: no cover - optional dependency
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


def load_json_file(path: str) -> Any:
    if not isinstance(path, str) or not path:
        return None
    if not Path(path).exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_feature_cols(artifacts: Optional[Dict[str, Any]]) -> List[str]:
    if isinstance(artifacts, dict):
        cols = artifacts.get("feature_cols")
        if isinstance(cols, (list, tuple)) and cols:
            return [str(c) for c in cols if str(c).strip()]
        path = artifacts.get("feature_cols_path")
    else:
        path = None
    if isinstance(path, str) and path:
        payload = load_json_file(path)
        if isinstance(payload, (list, tuple)):
            return [str(c) for c in payload if str(c).strip()]
        if isinstance(payload, dict):
            inner = payload.get("feature_cols")
            if isinstance(inner, (list, tuple)):
                return [str(c) for c in inner if str(c).strip()]
    return []


def load_pickle(path: str) -> Any:
    if not isinstance(path, str) or not path:
        raise FileNotFoundError("invalid path")
    if not Path(path).exists():
        raise FileNotFoundError(path)
    if joblib is not None:
        return joblib.load(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def _max_back_steps(feature_cols: List[str]) -> int:
    max_back = 0
    for c in feature_cols:
        spec = parse_recompute_name(c)
        if not spec:
            continue
        if "win" in spec:
            try:
                max_back = max(max_back, int(spec.get("win") or 0))
            except Exception:
                pass
        if "k" in spec:
            try:
                max_back = max(max_back, int(spec.get("k") or 0))
            except Exception:
                pass
    return max_back


def prepare_feature_frame(
    df: pd.DataFrame,
    *,
    feature_cols: List[str],
    time_col: str,
    value_col: str,
    tail_rows: Optional[int] = None,
    tail_only: bool = False,
    allow_nan: bool = False,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("empty dataframe")
    if not feature_cols:
        raise ValueError("feature_cols is empty")

    work = df.copy()
    if tail_only and tail_rows:
        max_back = _max_back_steps(feature_cols)
        keep = int(tail_rows) + int(max_back)
        if keep > 0 and len(work) > keep:
            work = work.tail(keep).reset_index(drop=True)
    if any(c in safe_time_features() for c in feature_cols):
        work = ensure_calendar_features(work, time_col=time_col)

    missing = []
    for c in feature_cols:
        if c == time_col:
            continue
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
            continue
        if is_recomputable_name(c) or c in safe_time_features():
            work[c] = recompute_feature_column(work, c, value_col=value_col, time_col=time_col)
            continue
        missing.append(c)
    if missing:
        raise KeyError(f"missing features: {sorted(set(missing))}")

    cols = [c for c in feature_cols if c != time_col]
    if not cols:
        raise ValueError("no usable feature columns")

    check = work[cols]
    if tail_rows is not None and int(tail_rows) > 0:
        check = check.tail(int(tail_rows))
    if not allow_nan and check.isna().any().any():
        raise ValueError("feature window contains NaN after recompute")

    return work
