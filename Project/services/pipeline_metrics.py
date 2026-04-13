from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    mean_abs = float(np.mean(np.abs(y_true))) if y_true.size else 0.0
    tau = max(eps, 0.01 * mean_abs) if np.isfinite(mean_abs) and mean_abs > 0 else eps
    mask = np.abs(y_true) > tau
    if int(mask.sum()) == 0:
        return float("nan")
    denom = np.abs(y_true[mask]) + eps
    return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / denom)))


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    n = min(len(yt), len(yp))
    yt = yt[:n]
    yp = yp[:n]
    if n == 0:
        return {"rmse": np.nan, "mape": np.nan, "nrmse": np.nan, "smape": np.nan}
    mask = np.isfinite(yt) & np.isfinite(yp)
    if int(mask.sum()) == 0:
        return {"rmse": np.nan, "mape": np.nan, "nrmse": np.nan, "smape": np.nan}
    yt = yt[mask]
    yp = yp[mask]
    diff = yp - yt
    rmse = float(np.sqrt(np.mean(diff * diff)))
    denom2 = np.abs(yt) + np.abs(yp) + 1e-8
    smape = float(np.mean(2.0 * np.abs(diff) / denom2))
    std = float(np.std(yt)) + 1e-8
    nrmse = float(rmse / std) if np.isfinite(std) and std > 1e-8 else np.nan
    mape = float(safe_mape(yt, yp))
    return {"rmse": rmse, "mape": mape, "nrmse": nrmse, "smape": smape}


def baseline_metrics(
    y_all: np.ndarray,
    train_len: int,
    val_len: int,
    test_len: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    y_all = np.asarray(y_all, dtype=float).reshape(-1)
    out: Dict[str, Any] = {"naive": {}, "seasonal": {}, "season_len": None}
    if train_len <= 0 or len(y_all) < train_len:
        return out
    last_train = y_all[train_len - 1]
    if val_len > 0:
        y_val = y_all[train_len : train_len + val_len]
        out["naive"]["val"] = calc_metrics(y_val, np.full(val_len, last_train, dtype=float))
    if test_len > 0:
        y_test = y_all[train_len + val_len : train_len + val_len + test_len]
        last_tv = y_all[train_len + val_len - 1] if train_len + val_len > 0 else last_train
        out["naive"]["test"] = calc_metrics(y_test, np.full(test_len, last_tv, dtype=float))

    season_len = int((config.get("baseline") or {}).get("season_len", 0) or 0)
    if season_len <= 0:
        season_len = int((config.get("prediction", {}) or {}).get("season_len", 0) or 0)
    if season_len > 0 and train_len >= season_len:
        out["season_len"] = season_len
        if val_len > 0:
            y_val = y_all[train_len : train_len + val_len]
            seasonal_val = y_all[train_len - season_len : train_len - season_len + val_len]
            out["seasonal"]["val"] = calc_metrics(y_val, seasonal_val)
        if test_len > 0:
            y_test = y_all[train_len + val_len : train_len + val_len + test_len]
            test_start = train_len + val_len - season_len
            seasonal_test = y_all[test_start : test_start + test_len] if test_start >= 0 else None
            if seasonal_test is not None and len(seasonal_test) == len(y_test):
                out["seasonal"]["test"] = calc_metrics(y_test, seasonal_test)
    return out


def compute_metrics_from_dense(df_dense: Optional[pd.DataFrame]) -> Optional[Dict[str, float]]:
    if not isinstance(df_dense, pd.DataFrame) or df_dense.empty:
        return None
    if not all(c in df_dense.columns for c in ["y_true", "yhat"]):
        return None
    dfm = df_dense[["y_true", "yhat"]].dropna()
    if dfm.empty:
        return None
    return calc_metrics(dfm["y_true"].values, dfm["yhat"].values)


def update_dense_metrics(
    data_block: Dict[str, Any],
    metrics_block: Dict[str, Any],
    val_dense_df: Optional[pd.DataFrame],
    test_dense_df: Optional[pd.DataFrame],
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    val_metrics_local = compute_metrics_from_dense(val_dense_df)
    test_metrics_local = compute_metrics_from_dense(test_dense_df)
    if isinstance(val_metrics_local, dict):
        metrics_block["val_rmse"] = val_metrics_local.get("rmse")
        metrics_block["val_mape"] = val_metrics_local.get("mape")
        metrics_block["val_nrmse"] = val_metrics_local.get("nrmse")
        metrics_block["val_smape"] = val_metrics_local.get("smape")
    if isinstance(test_metrics_local, dict):
        metrics_block["test_rmse"] = test_metrics_local.get("rmse")
        metrics_block["test_mape"] = test_metrics_local.get("mape")
        metrics_block["test_nrmse"] = test_metrics_local.get("nrmse")
        metrics_block["test_smape"] = test_metrics_local.get("smape")
    data_block["val_metrics"] = val_metrics_local
    data_block["test_metrics"] = test_metrics_local
    return val_metrics_local, test_metrics_local


def update_baseline_metrics(
    data_block: Dict[str, Any],
    metrics_block: Dict[str, Any],
    y_all_source: Any,
    val_len: int,
    test_len: int,
    config: Dict[str, Any],
) -> None:
    y_all = pd.to_numeric(y_all_source, errors="coerce").to_numpy(dtype=float)
    n_total = int(len(y_all))
    train_len_local = max(0, n_total - int(val_len) - int(test_len))
    base_metrics_local = baseline_metrics(y_all, train_len_local, int(val_len), int(test_len), config)
    data_block["baseline_metrics"] = base_metrics_local
    metrics_block["baseline"] = base_metrics_local
