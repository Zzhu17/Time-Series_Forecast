from __future__ import annotations

from typing import Dict
import numpy as np


def compute_residual_drift(
    *,
    val_true: np.ndarray,
    val_pred: np.ndarray,
    test_true: np.ndarray,
    test_pred: np.ndarray,
    mean_shift_thresh: float = 1.0,
    std_ratio_bounds: tuple[float, float] = (0.5, 2.0),
) -> Dict[str, float | bool]:
    def _residuals(y_true, y_pred):
        yt = np.asarray(y_true, dtype=float).reshape(-1)
        yp = np.asarray(y_pred, dtype=float).reshape(-1)
        n = min(len(yt), len(yp))
        if n <= 0:
            return np.array([], dtype=float)
        return yt[:n] - yp[:n]

    res_val = _residuals(val_true, val_pred)
    res_test = _residuals(test_true, test_pred)

    if res_val.size == 0 or res_test.size == 0:
        return {"drifted": False, "mean_shift": 0.0, "std_ratio": 1.0, "score": 0.0}

    mean_val = float(np.mean(res_val))
    mean_test = float(np.mean(res_test))
    std_val = float(np.std(res_val) + 1e-8)
    std_test = float(np.std(res_test) + 1e-8)

    mean_shift = abs(mean_test - mean_val) / std_val
    std_ratio = std_test / std_val
    drifted = bool(mean_shift > mean_shift_thresh or std_ratio < std_ratio_bounds[0] or std_ratio > std_ratio_bounds[1])
    score = float(mean_shift + abs(1.0 - std_ratio))

    return {
        "drifted": drifted,
        "mean_shift": float(mean_shift),
        "std_ratio": float(std_ratio),
        "score": score,
    }
