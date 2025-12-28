from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


def fit_target_transform(y: Any, method: str = "log1p") -> Dict[str, Any]:
    """
    Fit a target transform on 1D data and return a serializable params dict.

    Supported:
      - method='log1p': uses log1p for non-negative targets; falls back to signed_log1p if negatives exist.
    """
    method = str(method or "log1p").lower().strip()
    y_np = np.asarray(y, dtype=float).reshape(-1)
    y_np = y_np[np.isfinite(y_np)]
    y_min = float(np.min(y_np)) if y_np.size else 0.0

    if method == "log1p":
        mode = "plain" if y_min >= 0.0 else "signed"
        return {"method": "log1p", "mode": mode}

    raise ValueError(f"Unsupported target transform method: {method}")


def transform_array(x: Any, params: Optional[Dict[str, Any]]) -> np.ndarray:
    """Apply fitted transform to a 1D/ND array (elementwise)."""
    arr = np.asarray(x, dtype=float)
    if not params:
        return arr
    method = str(params.get("method", "")).lower()
    if method != "log1p":
        return arr
    mode = str(params.get("mode", "plain")).lower()
    if mode == "plain":
        return np.log1p(np.maximum(arr, 0.0))
    # signed log1p
    return np.sign(arr) * np.log1p(np.abs(arr))


def inverse_transform_array(x: Any, params: Optional[Dict[str, Any]]) -> np.ndarray:
    """Inverse the fitted transform for a 1D/ND array (elementwise)."""
    arr = np.asarray(x, dtype=float)
    if not params:
        return arr
    method = str(params.get("method", "")).lower()
    if method != "log1p":
        return arr
    mode = str(params.get("mode", "plain")).lower()
    if mode == "plain":
        return np.expm1(arr)
    # signed log1p inverse
    return np.sign(arr) * np.expm1(np.abs(arr))


def transform_df_target(df: pd.DataFrame, value_col: str, params: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Return a copy with target column transformed."""
    if not isinstance(df, pd.DataFrame) or df.empty or not params:
        return df
    if value_col not in df.columns:
        return df
    out = df.copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out[value_col] = transform_array(out[value_col].to_numpy(), params)
    return out

