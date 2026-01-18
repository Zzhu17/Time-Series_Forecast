from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from schemas.contract import FeatureContractReport
from utils.feature_contract import is_recomputable_name, parse_recompute_name, safe_time_features

_HYBRID_PRESETS = {
    "xgboost+lstm": {
        "base": "lstm",
        "residual_modeling": {
            "enabled": True,
            "model_type": "xgboost",
            "lags": [1, 2, 3, 6, 12, 24],
            "rolling_windows": [6, 12, 24, 48],
            "diffs": [1, 24],
        },
    },
    "xgboost+informer": {
        "base": "informer",
        "residual_modeling": {
            "enabled": True,
            "model_type": "xgboost",
            "lags": [1, 2, 3, 6, 12, 24],
            "rolling_windows": [6, 12, 24, 48],
            "diffs": [1, 24],
        },
    },
}


def normalize_model_name(model_name: Any) -> str:
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("请选择 model")
    cleaned = model_name.strip()
    if cleaned.lower() in ("none", "null"):
        raise ValueError("请选择 model")
    return cleaned


def apply_hybrid_preset(
    model_name: str,
    residual_modeling: Optional[dict],
) -> Tuple[str, Optional[dict], Optional[str]]:
    key = str(model_name or "").strip().lower()
    preset = _HYBRID_PRESETS.get(key)
    if not preset:
        return model_name, residual_modeling, None
    base = str(preset.get("base") or model_name)
    if residual_modeling is None:
        residual_modeling = preset.get("residual_modeling")
    return base, residual_modeling, key


def coerce_rows(rows: Any) -> List[Dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a non-empty list")
    coerced: List[Dict[str, Any]] = []
    for item in rows:
        if hasattr(item, "data"):
            data = getattr(item, "data")
            if isinstance(data, dict):
                coerced.append(data)
                continue
        if isinstance(item, dict):
            coerced.append(item)
            continue
        raise ValueError("rows must be a list of dict-like objects")
    return coerced


def dedupe_keep_order(items: List[str]) -> Tuple[List[str], List[str]]:
    seen = set()
    deduped: List[str] = []
    duplicates: List[str] = []
    for item in items:
        if item in seen:
            duplicates.append(item)
            continue
        seen.add(item)
        deduped.append(item)
    return deduped, duplicates


def normalize_feature_cols(
    feature_cols: Optional[List[str]],
    *,
    time_col: str,
    value_col: str,
) -> Tuple[List[str], Dict[str, List[str]]]:
    cleaned: List[str] = []
    invalid: List[str] = []
    for raw in feature_cols or []:
        if not isinstance(raw, str):
            invalid.append(str(raw))
            continue
        name = raw.strip()
        if not name:
            continue
        if name == time_col:
            continue
        cleaned.append(name)

    ordered = [value_col] + [c for c in cleaned if c != value_col]
    deduped, duplicates = dedupe_keep_order(ordered)
    return deduped, {"duplicate_features": duplicates, "invalid_features": invalid}


def validate_required_columns(df: pd.DataFrame, time_col: str, value_col: str) -> None:
    missing = [c for c in (time_col, value_col) if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    if ts.isna().all():
        raise ValueError(f"time_col '{time_col}' contains no valid timestamps")

    y = pd.to_numeric(df[value_col], errors="coerce")
    if y.isna().all():
        raise ValueError(f"value_col '{value_col}' contains no numeric values")

    if time_col == value_col:
        raise ValueError("time_col and value_col must be different")


def _is_recomputable_feature(
    col: str,
    *,
    df_columns: List[str],
    value_col: str,
) -> bool:
    if col in safe_time_features():
        return True
    if not is_recomputable_name(col):
        return False
    spec = parse_recompute_name(col) or {}
    base = spec.get("base") or value_col
    return base in df_columns


def build_feature_contract_report(
    df: pd.DataFrame,
    *,
    time_col: str,
    value_col: str,
    feature_cols: List[str],
    normalize_report: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    validate_required_columns(df, time_col, value_col)

    missing_feature_cols: List[str] = []
    recomputable_missing: List[str] = []
    df_cols = [str(c) for c in df.columns]

    for c in feature_cols:
        if c in df_cols:
            continue
        if _is_recomputable_feature(c, df_columns=df_cols, value_col=value_col):
            recomputable_missing.append(c)
        else:
            missing_feature_cols.append(c)

    if missing_feature_cols:
        raise ValueError(f"feature_cols missing in data: {sorted(set(missing_feature_cols))}")

    extra_columns = [c for c in df_cols if c not in feature_cols and c != time_col]
    payload = {
        "feature_cols": list(feature_cols),
        "missing_required_cols": [],
        "missing_feature_cols": [],
        "recomputable_missing_cols": sorted(set(recomputable_missing)),
        "duplicate_features": [],
        "invalid_features": [],
        "extra_columns": sorted(set(extra_columns)),
    }
    if isinstance(normalize_report, dict):
        for key in ("duplicate_features", "invalid_features"):
            if key in normalize_report:
                payload[key] = sorted(set(normalize_report.get(key) or []))

    return FeatureContractReport(**payload).dict()
