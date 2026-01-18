from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from services.xgb_loader import XGBPredictor


def baseline_predict(df: pd.DataFrame, value_col: str, horizon: int) -> np.ndarray:
    """
    Simple persistence baseline: repeat the last observed value.
    Raises if the target column is missing or non-numeric.
    """
    if value_col not in df.columns:
        raise KeyError(f"Missing target column '{value_col}' in rows.")
    y = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if len(y) == 0:
        raise ValueError("No numeric values found for target column.")
    last = float(y.iloc[-1])
    return np.array([last for _ in range(horizon)], dtype=float)


def seasonal_naive_predict(
    df: pd.DataFrame,
    value_col: str,
    horizon: int,
    season_len: int,
) -> np.ndarray:
    """
    Seasonal naive baseline: repeat values from one season ago.
    """
    if season_len <= 0:
        return baseline_predict(df, value_col, horizon)
    if value_col not in df.columns:
        raise KeyError(f"Missing target column '{value_col}' in rows.")
    y = pd.to_numeric(df[value_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(y) < season_len:
        return baseline_predict(df, value_col, horizon)
    base = y[-season_len:]
    if len(base) >= horizon:
        return np.array(base[:horizon], dtype=float)
    pad = np.resize(base, horizon)
    return np.array(pad, dtype=float)


def predict_with_xgboost(
    df: pd.DataFrame,
    *,
    time_col: str,
    value_col: str,
    horizon: int,
    model_path: Optional[str | Path] = None,
    contract_path: Optional[str | Path] = None,
    baseline_fallback: bool = True,
) -> Tuple[np.ndarray, bool, str, str]:
    """
    Run XGBoost prediction if artifacts exist; optionally degrade to baseline.
    Returns (preds, degraded, used_model, reason).
    """
    model_path = model_path or "Project/artifacts/xgboost_model.json"
    contract_path = contract_path or "Project/artifacts/feature_cols.json"
    try:
        predictor = XGBPredictor(
            model_path=str(model_path),
            feature_contract_path=str(contract_path),
            target_transform=None,
            time_col=time_col,
            value_col=value_col,
        )
        preds, _meta, degraded, reason = predictor.predict(df, horizon=horizon)
        return preds, bool(degraded), "xgboost", reason or ""
    except Exception as e:
        if not baseline_fallback:
            raise
        preds = baseline_predict(df, value_col, horizon)
        return preds, True, "xgboost->baseline", f"xgboost failed: {e}"
