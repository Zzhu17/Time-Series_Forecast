from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def compute_dense_metrics(df: Any) -> Optional[Dict[str, float]]:
    try:
        if isinstance(df, pd.DataFrame) and {"y_true", "yhat"}.issubset(df.columns) and len(df) > 0:
            diff = (df["yhat"].astype(float) - df["y_true"].astype(float)).to_numpy()
            true = df["y_true"].astype(float).to_numpy()
            rmse_f = float(np.sqrt(np.mean(diff ** 2)))
            mape_f = float(np.mean(np.abs(diff) / (np.abs(true) + 1e-8)))
            return {"rmse": rmse_f, "mape": mape_f}
    except Exception:
        return None
    return None


def _dense_metrics(df: Any) -> Optional[Tuple[float, float]]:
    if not (isinstance(df, pd.DataFrame) and {"y_true", "yhat"}.issubset(df.columns) and len(df) > 0):
        return None
    yt = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    yp = pd.to_numeric(df["yhat"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(yt) & np.isfinite(yp)
    if int(m.sum()) < 16:
        return None
    diff = yp[m] - yt[m]
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mape = float(np.mean(np.abs(diff) / (np.abs(yt[m]) + 1e-8)))
    return rmse, mape


def _fit_affine(df: pd.DataFrame, pcfg: Dict[str, Any]) -> Optional[Dict[str, float]]:
    yt = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    yp = pd.to_numeric(df["yhat"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[m]
    yp = yp[m]
    if yt.size < 16:
        return None
    mu_t = float(np.mean(yt))
    mu_p = float(np.mean(yp))
    x = yp - mu_p
    y = yt - mu_t
    ridge = float(pcfg.get("ridge", 1e-6))
    denom = float(np.dot(x, x) + ridge * yt.size)
    if not np.isfinite(denom) or denom <= 0:
        return None
    a = float(np.dot(x, y) / denom)
    b = float(mu_t - a * mu_p)
    a_min, a_max = pcfg.get("a_clip", [0.8, 1.2])
    try:
        a_min = float(a_min)
        a_max = float(a_max)
    except Exception:
        a_min, a_max = 0.8, 1.2
    if np.isfinite(a):
        a = float(np.clip(a, a_min, a_max))
    mean_abs = float(np.mean(np.abs(yt))) if yt.size else 0.0
    b_ratio = float(pcfg.get("b_clip_ratio", 0.1))
    b_lim = max(1e-6, mean_abs * b_ratio)
    if np.isfinite(b):
        b = float(np.clip(b, -b_lim, b_lim))
    return {"a": a, "b": b}


def apply_post_calibration(data_blk: Dict[str, Any], config: Dict[str, Any]) -> None:
    pcfg = (config.get("post_calibration") or {})
    if not bool(pcfg.get("enabled", True)):
        return

    val_df = data_blk.get("val_dense")
    test_df = data_blk.get("test_dense")
    if not (isinstance(val_df, pd.DataFrame) and not val_df.empty):
        return

    base = _dense_metrics(val_df)
    calib = _fit_affine(val_df, pcfg)
    if not (base and calib):
        return

    a = float(calib["a"])
    b = float(calib["b"])
    val_adj = val_df.copy()
    val_adj["yhat"] = pd.to_numeric(val_adj["yhat"], errors="coerce") * a + b
    newm = _dense_metrics(val_adj)
    if not newm:
        return

    rmse0, mape0 = base
    rmse1, mape1 = newm
    mape_guard_rel = float(pcfg.get("mape_guard_rel", 1.02))
    if (rmse1 < rmse0) and (mape1 <= mape0 * mape_guard_rel):
        data_blk["val_dense"] = val_adj
        if isinstance(test_df, pd.DataFrame) and not test_df.empty:
            test_adj = test_df.copy()
            test_adj["yhat"] = pd.to_numeric(test_adj["yhat"], errors="coerce") * a + b
            data_blk["test_dense"] = test_adj
        data_blk["val_calib"] = calib
        print(
            f"[post_calibration] applied: a={a:.6f}, b={b:.6f} | "
            f"val rmse {rmse0:.6f}->{rmse1:.6f}, mape {mape0:.6f}->{mape1:.6f}"
        )
    else:
        data_blk["val_calib"] = calib
        print(
            f"[post_calibration] skipped (guard): val rmse {rmse0:.6f}->{rmse1:.6f}, "
            f"mape {mape0:.6f}->{mape1:.6f}"
        )
