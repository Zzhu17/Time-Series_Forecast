from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from services.pipeline_results import (
    normalize_dense_for_plot,
    pick_first_df,
    pick_first_dict,
)
from services.snapshot import as_int, pack_plot_series


def build_snapshot_meta(
    *,
    config: Dict[str, Any],
    uploaded_name: str | None,
    model_name: str,
    time_col: str,
    value_col: str,
) -> Dict[str, Any]:
    return {
        "uploaded_name": uploaded_name,
        "model_name": model_name,
        "time_col": time_col,
        "value_col": value_col,
        "run_id": (config.get("artifacts") or {}).get("run_id") or config.get("run_id"),
    }


def enrich_snapshot_payload(
    *,
    df: pd.DataFrame,
    config: Dict[str, Any],
    results: Dict[str, Any],
    snap_results: Dict[str, Any],
    time_col: str,
    value_col: str,
) -> None:
    def _ensure_data_block(obj: Dict[str, Any]) -> Dict[str, Any]:
        data = obj.get("data")
        if not isinstance(data, dict):
            data = {}
            obj["data"] = data
        return data

    def _sync_result_data_field(key: str, value: Any) -> None:
        _ensure_data_block(snap_results)[key] = value
        _ensure_data_block(results)[key] = value

    def _choose_plot_ts(dfr: pd.DataFrame, fallback_len: int):
        if isinstance(dfr.index, pd.DatetimeIndex):
            return dfr.index
        if time_col in dfr.columns:
            return dfr[time_col]
        if "timestamp" in dfr.columns:
            return dfr["timestamp"]
        return pd.date_range(start=pd.Timestamp.today().normalize(), periods=max(1, int(fallback_len)), freq="D")

    def _build_dense_plot(dense_df: pd.DataFrame | None, fallback_len: int):
        if not isinstance(dense_df, pd.DataFrame) or not {"y_true", "yhat"} <= set(dense_df.columns):
            return None
        return pack_plot_series(_choose_plot_ts(dense_df, fallback_len or len(dense_df)), dense_df["y_true"], dense_df["yhat"], max_n=4000)

    def _build_long_plot(long_payload: dict | None):
        if not isinstance(long_payload, dict):
            return None
        return pack_plot_series(long_payload.get("timestamps"), long_payload.get("y_true"), long_payload.get("yhat"), max_n=4000)

    split = (results.get("data", {}) or {}).get("split") or (config.get("data", {}) or {}).get("split") or {}
    t_len = as_int(split.get("train_len"), 0) or 0
    v_len = as_int(split.get("val_len"), 0) or 0
    te_len = as_int(split.get("test_len"), 0) or 0

    mean_abs_true_val = None
    mean_abs_true_test = None
    try:
        if v_len > 0 and value_col in df.columns:
            yv0 = pd.to_numeric(df.iloc[t_len : t_len + v_len][value_col], errors="coerce").to_numpy(dtype=float)
            mean_abs_true_val = float(np.nanmean(np.abs(yv0))) if yv0.size else None
        if te_len > 0 and value_col in df.columns:
            yt0 = pd.to_numeric(df.iloc[t_len + v_len : t_len + v_len + te_len][value_col], errors="coerce").to_numpy(dtype=float)
            mean_abs_true_test = float(np.nanmean(np.abs(yt0))) if yt0.size else None
    except Exception:
        mean_abs_true_val = None
        mean_abs_true_test = None

    train_plot = None
    val_plot = None
    test_plot = None
    try:
        dblk = (config.get("data", {}) or {})
        rdata = (results.get("data", {}) or {})
        vd0 = pick_first_df(dblk.get("val_dense"), rdata.get("val_dense"), dblk.get("val_result_df"), rdata.get("val_result_df"))
        td0 = pick_first_df(dblk.get("test_dense"), rdata.get("test_dense"), dblk.get("test_result_df"), rdata.get("test_result_df"))
        vd = normalize_dense_for_plot(vd0, "val")
        td = normalize_dense_for_plot(td0, "test")
        vlong = pick_first_dict(rdata.get("val_long"), dblk.get("val_long"), rdata.get("val_tail"), dblk.get("val_tail"))
        tlong = pick_first_dict(rdata.get("test_long"), dblk.get("test_long"), rdata.get("test_tail"), dblk.get("test_tail"))

        if t_len > 0 and value_col in df.columns:
            train_slice = df.iloc[:t_len]
            train_ts = train_slice[time_col] if time_col in train_slice.columns else None
            train_true = train_slice[value_col]
            train_plot = pack_plot_series(train_ts, train_true, train_true, max_n=4000)

        val_plot = _build_dense_plot(vd, v_len) or _build_long_plot(vlong)
        test_plot = _build_dense_plot(td, te_len) or _build_long_plot(tlong)
    except Exception as e:
        print(f"[services.pipeline] plot_data build failed: {e}", flush=True)
        train_plot = None
        val_plot = None
        test_plot = None

    if train_plot is None and val_plot is None and test_plot is None:
        try:
            dblk = (config.get("data", {}) or {})
            rdata = (results.get("data", {}) or {})
            vd_dbg = dblk.get("val_dense")
            td_dbg = dblk.get("test_dense")
            print(
                "[services.pipeline] plot_data missing | "
                f"val_dense={type(vd_dbg).__name__} cols={getattr(vd_dbg,'columns',None)} | "
                f"test_dense={type(td_dbg).__name__} cols={getattr(td_dbg,'columns',None)} | "
                f"rdata_keys={list(rdata.keys()) if isinstance(rdata,dict) else None}",
                flush=True,
            )
        except Exception:
            pass

    if train_plot or val_plot or test_plot:
        def _coerce_plot(p):
            if isinstance(p, str):
                try:
                    import ast
                    import json
                    try:
                        return json.loads(p)
                    except Exception:
                        return ast.literal_eval(p)
                except Exception:
                    return None
            return p

        plot_blob = {"train": _coerce_plot(train_plot), "val": _coerce_plot(val_plot), "test": _coerce_plot(test_plot)}
        _sync_result_data_field("plot_data", plot_blob)

    if isinstance(mean_abs_true_val, (int, float)) and np.isfinite(float(mean_abs_true_val)) and float(mean_abs_true_val) > 0:
        _sync_result_data_field("mean_abs_true_val", float(mean_abs_true_val))
    if isinstance(mean_abs_true_test, (int, float)) and np.isfinite(float(mean_abs_true_test)) and float(mean_abs_true_test) > 0:
        _sync_result_data_field("mean_abs_true_test", float(mean_abs_true_test))
