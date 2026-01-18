from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd


def rolling_backtest_naive(
    series: pd.Series,
    *,
    horizon: int = 1,
    step: int = 1,
    window: int = 24,
    seasonal_period: int | None = None,
) -> Dict[str, List[float]]:
    """
    Simple rolling backtest using naive or seasonal naive forecasts.
    Returns a dict with y_true and y_pred arrays aligned to rolling windows.
    """
    y = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    if y.size <= window + horizon:
        return {"y_true": [], "y_pred": []}

    y_true: List[float] = []
    y_pred: List[float] = []

    idx = window
    while idx + horizon <= len(y):
        hist = y[:idx]
        if seasonal_period and seasonal_period > 0 and len(hist) >= seasonal_period:
            base = hist[-seasonal_period : -seasonal_period + horizon]
            if len(base) < horizon:
                base = np.full(horizon, hist[-1])
        else:
            base = np.full(horizon, hist[-1])
        future = y[idx : idx + horizon]
        y_true.extend(future.tolist())
        y_pred.extend(base[: len(future)].tolist())
        idx += max(1, int(step))

    return {"y_true": y_true, "y_pred": y_pred}
