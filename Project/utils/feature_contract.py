from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def safe_time_features() -> List[str]:
    return ["month", "day_of_month", "day_of_week", "hour", "day_of_year"]


def ensure_calendar_features(df: pd.DataFrame, *, time_col: str) -> pd.DataFrame:
    if time_col not in df.columns:
        raise KeyError(f"Missing time_col '{time_col}' required to rebuild calendar features.")
    out = df.copy()
    # Normalize with utc=True to avoid mixed timezone warnings, then strip tz info.
    ts = pd.to_datetime(out[time_col], errors="coerce", utc=True)
    try:
        ts = ts.dt.tz_localize(None)
    except Exception:
        pass
    out["month"] = ts.dt.month
    out["day_of_month"] = ts.dt.day
    out["day_of_week"] = ts.dt.dayofweek
    out["hour"] = ts.dt.hour
    out["day_of_year"] = ts.dt.dayofyear
    return out


def parse_recompute_name(col: str) -> Optional[Dict[str, Any]]:
    """
    Supported patterns:
      - {base}_rolling_mean_7 / rolling_std_14 / rolling_min_7 / rolling_max_7 / rolling_median_7
      - rolling_mean_7 (base defaults to 'value')
      - {base}_lag_7 / lag_7
      - {base}_diff_1 / diff_1
    """
    s = str(col)
    patterns = [
        r"^(?:(?P<base>[A-Za-z0-9_]+)_)?rolling_(?P<stat>mean|std|min|max|median)_(?P<win>\d+)$",
        r"^(?:(?P<base>[A-Za-z0-9_]+)_)?lag_(?P<k>\d+)$",
        r"^(?:(?P<base>[A-Za-z0-9_]+)_)?diff_(?P<k>\d+)$",
    ]
    for p in patterns:
        m = re.match(p, s)
        if m:
            d = m.groupdict()
            return {k: v for k, v in d.items() if v is not None}
    return None


def is_recomputable_name(col: str) -> bool:
    return parse_recompute_name(col) is not None or col in safe_time_features()


def recompute_feature_column(
    df: pd.DataFrame,
    col: str,
    *,
    value_col: str,
    time_col: Optional[str] = None,
) -> pd.Series:
    if col in safe_time_features():
        if not time_col:
            raise KeyError("time_col is required to recompute calendar features.")
        tmp = ensure_calendar_features(df, time_col=time_col)
        if col not in tmp.columns:
            raise KeyError(f"Failed to recompute calendar feature '{col}'.")
        return pd.to_numeric(tmp[col], errors="coerce")

    spec = parse_recompute_name(col)
    if not spec:
        raise KeyError(f"Feature '{col}' is not recomputable by built-in rules.")
    base = spec.get("base") or value_col
    if base not in df.columns:
        raise KeyError(f"Cannot recompute '{col}': base column '{base}' not found.")
    base_s = pd.to_numeric(df[base], errors="coerce")

    if "stat" in spec:
        win = int(spec.get("win", 0) or 0)
        if win <= 0:
            raise ValueError(f"Invalid rolling window for '{col}'.")
        # leakage-safe: use only past values up to t-1
        past = base_s.shift(1)
        stat = spec["stat"]
        roll = past.rolling(window=win, min_periods=win)
        if stat == "mean":
            return roll.mean()
        if stat == "std":
            return roll.std(ddof=0)
        if stat == "min":
            return roll.min()
        if stat == "max":
            return roll.max()
        if stat == "median":
            return roll.median()
        raise ValueError(f"Unsupported rolling stat '{stat}' for '{col}'.")

    if "k" in spec:
        k = int(spec.get("k", 0) or 0)
        if k <= 0:
            raise ValueError(f"Invalid lag/diff k for '{col}'.")
        if "lag" in col:
            return base_s.shift(k)
        if "diff" in col:
            return base_s.diff(k)
    raise KeyError(f"Cannot recompute '{col}'.")


def align_df_to_feature_contract(
    df: pd.DataFrame,
    *,
    time_col: str,
    value_col: str,
    feature_cols: List[str],
    contract: Optional[Dict[str, Any]] = None,
    recompute_policy: str = "error",
    tail_rows: Optional[int] = None,
    allow_degrade: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """
    Enforce a frozen feature space at predict-time with 3-class missing strategy:
      1) Safe default calendar features: rebuild from time_col.
      2) Recomputable features (rolling/lag/diff): recompute from history if possible.
      3) Core signal features: missing => raise.

    Returns aligned df and a report.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    required_core = set([value_col])
    repairable_core = set()
    contract_order: List[str] = list(feature_cols)
    if isinstance(contract, dict):
        required_core |= set([c for c in (contract.get("required_core_cols") or contract.get("core_cols") or []) if isinstance(c, str)])
        repairable_core |= set([c for c in (contract.get("repairable_core_cols") or contract.get("recomputable_cols") or []) if isinstance(c, str)])
        frozen_order = contract.get("feature_order") or contract.get("feature_cols")
        if isinstance(frozen_order, list) and frozen_order:
            contract_order = [str(c) for c in frozen_order if str(c).strip()]

    out = df.copy()
    report: Dict[str, Any] = {
        "rebuilt": [],
        "recomputed": [],
        "dropped_optional": [],
        "missing_required": [],
        "contract_diff": {
            "missing_columns": [],
            "extra_columns": [],
            "order_mismatch": [],
            "type_cast_failed": [],
        },
    }

    expected_order = [c for c in contract_order if c != time_col]
    incoming_features = [str(c) for c in out.columns if str(c) != time_col]
    report["contract_diff"]["missing_columns"] = sorted([c for c in expected_order if c not in incoming_features])
    report["contract_diff"]["extra_columns"] = sorted([c for c in incoming_features if c not in expected_order])
    if incoming_features != expected_order:
        report["contract_diff"]["order_mismatch"] = [
            {"expected_index": idx, "expected": exp, "actual": incoming_features[idx] if idx < len(incoming_features) else None}
            for idx, exp in enumerate(expected_order)
            if idx >= len(incoming_features) or incoming_features[idx] != exp
        ]

    if any(c in safe_time_features() for c in feature_cols):
        out = ensure_calendar_features(out, time_col=time_col)

    for c in feature_cols:
        if c in out.columns and c != time_col:
            original = out[c]
            out[c] = pd.to_numeric(out[c], errors="coerce")
            bad_cast = bool(original.notna().any() and out[c].isna().all())
            if bad_cast:
                report["contract_diff"]["type_cast_failed"].append(c)

    def _has_nan(s: pd.Series) -> bool:
        if tail_rows is not None and tail_rows > 0:
            return bool(s.tail(int(tail_rows)).isna().any())
        return bool(s.isna().any())

    # Required core: fail-fast if missing/NaN
    for c in feature_cols:
        if c in required_core:
            if c not in out.columns or _has_nan(out[c]):
                report["missing_required"].append(c)
    if report["missing_required"]:
        if not allow_degrade:
            raise KeyError(f"feature contract validation failed: {report}")
        for c in report["missing_required"]:
            report["dropped_optional"].append(c)

    # Repairable core: recompute if needed
    for c in feature_cols:
        if c in required_core or c == time_col:
            continue
        if c in repairable_core or is_recomputable_name(c):
            if c not in out.columns or _has_nan(out[c]):
                out[c] = recompute_feature_column(out, c, value_col=value_col, time_col=time_col)
                report["recomputed"].append(c)
            if _has_nan(out[c]):
                if not allow_degrade:
                    raise ValueError(f"feature contract validation failed: {report}")
                report["dropped_optional"].append(c)

    # Optional: if missing/NaN -> drop from usable feature set (no silent fill)
    usable_cols: List[str] = []
    for c in feature_cols:
        if c in required_core or c in repairable_core:
            if allow_degrade and c in report["dropped_optional"]:
                continue
            usable_cols.append(c)
            continue
        if c not in out.columns or _has_nan(out[c]):
            report["dropped_optional"].append(c)
            continue
        usable_cols.append(c)

    has_diff = bool(
        report["contract_diff"]["missing_columns"]
        or report["contract_diff"]["order_mismatch"]
        or report["contract_diff"]["type_cast_failed"]
    )
    if has_diff and not allow_degrade:
        raise ValueError(f"feature contract validation failed: {report}")

    return out, report, usable_cols
