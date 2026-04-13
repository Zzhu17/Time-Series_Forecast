from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from services.result_contracts import (
    RUN_RESULT_DATA_KEYS,
    backfill_metric_slot,
    backfill_missing,
    ensure_metric_slot,
    ensure_run_result,
)


def pick_first_df(*candidates):
    for x in candidates:
        if isinstance(x, pd.DataFrame):
            return x
    return None


def pick_first_dict(*candidates):
    for x in candidates:
        if isinstance(x, dict):
            return x
    return None


def normalize_dense_for_plot(df_like: Optional[pd.DataFrame], which: str) -> Optional[pd.DataFrame]:
    if not isinstance(df_like, pd.DataFrame) or df_like.empty:
        return None
    df = df_like.copy()
    if {"y_true", "yhat"} <= set(df.columns):
        return df
    if which == "val":
        if {"validation_true", "validation_predict"} <= set(df.columns):
            df["y_true"] = df["validation_true"]
            df["yhat"] = df["validation_predict"]
            return df
    if which == "test":
        if {"test_true", "test_predict"} <= set(df.columns):
            df["y_true"] = df["test_true"]
            df["yhat"] = df["test_predict"]
            return df
    return None


def normalize_results_for_app(res, cfg: dict, src_df: pd.DataFrame) -> dict:
    out: dict = ensure_run_result(res if isinstance(res, dict) else {})
    out["artifacts"] = (cfg.get("artifacts") or {})
    data_blk = (cfg.get("data") or {}) if isinstance(cfg, dict) else {}
    if not isinstance(data_blk, dict):
        data_blk = {}

    if isinstance(res, dict):
        out.update(ensure_run_result(res))
        out["artifacts"] = out.get("artifacts") or (cfg.get("artifacts") or {})
    elif isinstance(res, (tuple, list)):
        out = ensure_run_result(out)
        out["artifacts"] = (cfg.get("artifacts") or {})
    else:
        out["status"] = "error"
        out["message"] = "Unknown pipeline return type"
        out = ensure_run_result(out)

    out_data = out["data"]
    out_metrics = out["metrics"]
    backfill_missing(out_data, data_blk, RUN_RESULT_DATA_KEYS)

    if "split" not in out_data or not isinstance(out_data.get("split"), dict):
        n = int(len(src_df)) if isinstance(src_df, pd.DataFrame) else 0
        t = int(n * 0.6)
        v = int(n * 0.2)
        out_data["split"] = {"train_len": t, "val_len": v, "test_len": n - t - v}

    backfill_metric_slot(out_metrics, "validation", pick_first_dict(data_blk.get("val_metrics"), data_blk.get("metrics_val"), data_blk.get("validation_metrics")))
    backfill_metric_slot(out_metrics, "test", pick_first_dict(data_blk.get("test_metrics"), data_blk.get("metrics_test"), data_blk.get("testing_metrics")))
    backfill_metric_slot(out_metrics, "baseline", data_blk.get("baseline_metrics"))
    backfill_metric_slot(out_metrics, "drift", data_blk.get("drift"))

    root_m = cfg.get("metrics") if isinstance(cfg.get("metrics"), dict) else {}
    if isinstance(root_m, dict):
        vm = ensure_metric_slot(out_metrics, "validation")
        tm = ensure_metric_slot(out_metrics, "test")
        if vm.get("rmse") is None and "val_rmse" in root_m:
            vm["rmse"] = root_m.get("val_rmse")
        if vm.get("mape") is None and "val_mape" in root_m:
            vm["mape"] = root_m.get("val_mape")
        if vm.get("nrmse") is None and "val_nrmse" in root_m:
            vm["nrmse"] = root_m.get("val_nrmse")
        if vm.get("smape") is None and "val_smape" in root_m:
            vm["smape"] = root_m.get("val_smape")
        if tm.get("rmse") is None and "test_rmse" in root_m:
            tm["rmse"] = root_m.get("test_rmse")
        if tm.get("mape") is None and "test_mape" in root_m:
            tm["mape"] = root_m.get("test_mape")
        if tm.get("nrmse") is None and "test_nrmse" in root_m:
            tm["nrmse"] = root_m.get("test_nrmse")
        if tm.get("smape") is None and "test_smape" in root_m:
            tm["smape"] = root_m.get("test_smape")

    return ensure_run_result(out)


def looks_like_required_core_error(err: Exception) -> bool:
    msg = str(err)
    keys = [
        "Required core feature",
        "Missing required core",
        "核心特征存在缺失值",
        "缺少必要列",
    ]
    return any(k in msg for k in keys)


def baseline_degraded_results(src_df: pd.DataFrame, cfg: dict, *, error: Exception) -> dict:
    tcol = (cfg.get("default", {}) or {}).get("time_col", "date")
    vcol = (cfg.get("default", {}) or {}).get("value_col", "value")

    df2 = src_df.copy()
    if tcol in df2.columns:
        ts = pd.to_datetime(df2[tcol], errors="coerce")
        if ts.isna().all():
            ts = pd.date_range(start=pd.Timestamp.today().normalize(), periods=len(df2), freq="D")
        df2["_ts_"] = ts
        df2 = df2.sort_values("_ts_")
        df2 = df2.set_index(pd.DatetimeIndex(df2["_ts_"], name=tcol))
        df2 = df2.drop(columns=["_ts_"], errors="ignore")
    else:
        df2.index = pd.date_range(start=pd.Timestamp.today().normalize(), periods=len(df2), freq="D", name=tcol)

    y = pd.to_numeric(df2.get(vcol), errors="coerce")
    if int(y.notna().sum()) == 0:
        raise ValueError(f"目标列 '{vcol}' 无可用数值（无法降级预测）。原始错误：{error}")

    y_ffill = y.ffill()
    yhat = y_ffill.shift(1)

    n = len(df2)
    t = int(n * 0.6)
    v = int(n * 0.2)
    te = n - t - v

    val_idx = slice(t, t + v)
    test_idx = slice(t + v, n)
    val_dense = pd.DataFrame({"y_true": y.iloc[val_idx].to_numpy(), "yhat": yhat.iloc[val_idx].to_numpy()}, index=df2.index[val_idx])
    test_dense = pd.DataFrame({"y_true": y.iloc[test_idx].to_numpy(), "yhat": yhat.iloc[test_idx].to_numpy()}, index=df2.index[test_idx])

    def _metrics(d: pd.DataFrame) -> dict:
        yt = d["y_true"].to_numpy(dtype=float)
        yp = d["yhat"].to_numpy(dtype=float)
        mask = np.isfinite(yt) & np.isfinite(yp)
        if int(mask.sum()) == 0:
            return {"rmse": np.nan, "mape": np.nan, "nrmse": np.nan, "smape": np.nan}
        rmse = float(np.sqrt(np.mean((yp[mask] - yt[mask]) ** 2)))
        diff = yp[mask] - yt[mask]
        denom = np.abs(yt[mask]) + np.abs(yp[mask]) + 1e-8
        smape = float(np.mean(2.0 * np.abs(diff) / denom))
        std = float(np.std(yt[mask])) + 1e-8
        nrmse = float(rmse / std) if np.isfinite(std) and std > 1e-8 else np.nan
        mean_abs = float(np.mean(np.abs(yt[mask]))) if int(mask.sum()) else 0.0
        tau = max(1e-8, 0.01 * mean_abs) if np.isfinite(mean_abs) and mean_abs > 0 else 1e-8
        mape_mask = np.abs(yt[mask]) > tau
        if int(mape_mask.sum()) == 0:
            mape = np.nan
        else:
            denom_m = np.abs(yt[mask][mape_mask]) + 1e-8
            mape = float(np.mean(np.abs(diff[mape_mask] / denom_m)))
        return {"rmse": rmse, "mape": mape, "nrmse": nrmse, "smape": smape}

    val_m = _metrics(val_dense)
    test_m = _metrics(test_dense)

    data_blk = cfg.setdefault("data", {})
    data_blk["degraded"] = True
    data_blk["degraded_mode"] = "naive_persistence"
    data_blk["degraded_reason"] = "required_core_missing"
    data_blk["degraded_error"] = str(error)
    data_blk["split"] = {"train_len": t, "val_len": v, "test_len": te}
    data_blk["val_dense"] = val_dense
    data_blk["test_dense"] = test_dense
    data_blk["val_metrics"] = val_m
    data_blk["test_metrics"] = test_m

    return {
        "status": "ok",
        "metrics": {"validation": val_m, "test": test_m},
        "data": {
            "split": data_blk["split"],
            "val_dense": val_dense,
            "test_dense": test_dense,
            "degraded": True,
            "degraded_mode": data_blk.get("degraded_mode"),
            "degraded_reason": data_blk.get("degraded_reason"),
            "degraded_error": data_blk.get("degraded_error"),
        },
        "artifacts": cfg.get("artifacts", {}),
    }
