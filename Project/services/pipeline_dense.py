from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def normalize_dense(df_like: Any, time_col: str) -> Optional[pd.DataFrame]:
    if df_like is None:
        return None
    try:
        if isinstance(df_like, pd.DataFrame):
            base = df_like.copy()
            cols_keep = [c for c in ["y_true", "yhat"] if c in base.columns]
            df = base[cols_keep].copy() if cols_keep else pd.DataFrame(index=base.index)
            if not isinstance(df.index, pd.DatetimeIndex):
                ts_col = None
                for cand in ["timestamp", time_col, "date", "time", "ds"]:
                    if cand in base.columns:
                        ts_col = cand
                        break
                if ts_col is not None:
                    idx = pd.to_datetime(base[ts_col], errors="coerce", utc=True)
                    try:
                        idx = idx.dt.tz_localize(None)
                    except Exception:
                        pass
                    df = df.set_index(idx)
                else:
                    df.index = pd.to_datetime(base.index, errors="coerce", utc=True)
                    try:
                        df.index = df.index.tz_localize(None)
                    except Exception:
                        pass
            return df.sort_index()

        if isinstance(df_like, dict):
            ts = df_like.get("timestamps")
            if ts is None:
                return None
            idx = pd.to_datetime(ts, errors="coerce", utc=True)
            try:
                idx = idx.tz_localize(None)
            except Exception:
                pass
            cols = {}
            if "y_true" in df_like:
                cols["y_true"] = df_like["y_true"]
            if "yhat" in df_like:
                cols["yhat"] = df_like["yhat"]
            return pd.DataFrame(cols, index=idx).sort_index()

        if isinstance(df_like, (list, tuple)) and len(df_like) > 0 and isinstance(df_like[0], dict):
            df = pd.DataFrame(df_like)
            return normalize_dense(df, time_col)
    except Exception as e:
        print(f"[pipeline] normalize_dense failed: {e}")
    return None


def standardize_dense_df(df: Optional[pd.DataFrame], time_col: str) -> Optional[pd.DataFrame]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if time_col in out.columns:
            out[time_col] = pd.to_datetime(out[time_col], errors="coerce", utc=True)
            try:
                out[time_col] = out[time_col].dt.tz_localize(None)
            except Exception:
                pass
            out = out.set_index(time_col)
        else:
            try:
                out.index = pd.to_datetime(out.index, errors="coerce", utc=True)
                try:
                    out.index = out.index.tz_localize(None)
                except Exception:
                    pass
            except Exception:
                pass
    out.index.name = time_col
    if "y_true" not in out.columns:
        out["y_true"] = np.nan
    if "yhat" not in out.columns:
        out["yhat"] = np.nan
    out = out[["y_true", "yhat"]].copy()
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce").astype("float32")
    out["yhat"] = pd.to_numeric(out["yhat"], errors="coerce").astype("float32")
    return out.sort_index()


def attach_ts_and_rename(
    df_dense: Optional[pd.DataFrame],
    ts_list: Any,
    which: str,
    time_col: str,
) -> Optional[pd.DataFrame]:
    if not isinstance(df_dense, pd.DataFrame) or df_dense.empty:
        return None
    out = df_dense.copy()

    if ts_list is not None:
        idx = pd.to_datetime(ts_list, errors="coerce", utc=True)
        try:
            if hasattr(idx, "tz_localize"):
                idx = idx.tz_localize(None)
        except Exception:
            pass
        if isinstance(idx, pd.Series):
            idx = idx.values
        try:
            out.index = pd.DatetimeIndex(idx, name=time_col)
        except Exception:
            pass
    if not isinstance(out.index, pd.DatetimeIndex) or out.index.isna().all():
        out.index = pd.date_range(start=pd.Timestamp.today().normalize(), periods=len(out), freq="D", name=time_col)

    if which == "val":
        out["validation_true"] = out["y_true"]
        out["validation_predict"] = out["yhat"]
    else:
        out["test_true"] = out["y_true"]
        out["test_predict"] = out["yhat"]

    prefer = ["validation_true", "validation_predict"] if which == "val" else ["test_true", "test_predict"]
    cols = [c for c in prefer + ["y_true", "yhat"] if c in out.columns]
    return out[cols]


def to_dense_df(true_arr: Any, pred_arr: Any) -> Optional[pd.DataFrame]:
    true_arr = np.asarray(true_arr, dtype=float).ravel()
    pred_arr = np.asarray(pred_arr, dtype=float).ravel()
    length = min(len(true_arr), len(pred_arr))
    if length <= 0:
        return None
    return pd.DataFrame({"y_true": true_arr[:length], "yhat": pred_arr[:length]})
