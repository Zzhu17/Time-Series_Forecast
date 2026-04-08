from __future__ import annotations

from typing import Dict, List, Tuple
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


def expanding_window_splits(
    n_samples: int,
    *,
    n_splits: int = 3,
    val_size: int = 7,
    min_train_size: int | None = None,
) -> List[Tuple[slice, slice]]:
    """
    Create time-ordered expanding-window train/validation slices.

    Example with n_splits=3:
      [train0|val0][+train growth|val1][+train growth|val2]
    """
    if n_samples <= 0 or n_splits <= 0 or val_size <= 0:
        return []

    if min_train_size is None:
        min_train_size = max(val_size * 2, 14)

    required = min_train_size + n_splits * val_size
    if n_samples < required:
        return []

    train_start = 0
    train_end = n_samples - n_splits * val_size
    if train_end < min_train_size:
        return []

    folds: List[Tuple[slice, slice]] = []
    for i in range(n_splits):
        val_start = train_end + i * val_size
        val_end = val_start + val_size
        folds.append((slice(train_start, val_start), slice(val_start, val_end)))

    return folds
