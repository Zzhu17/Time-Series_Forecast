from __future__ import annotations

from typing import Any, Dict, Optional

import os
import numpy as np
import pandas as pd


def assign_missing_split_timestamps(
    data_blk: Dict[str, Any],
    df_input: pd.DataFrame,
    *,
    time_col: str,
    val_true: Any,
    test_true: Any,
) -> None:
    val_len = int(len(np.asarray(val_true).ravel()))
    test_len = int(len(np.asarray(test_true).ravel()))
    ts_series = None
    if isinstance(df_input, pd.DataFrame) and time_col in df_input.columns:
        ts_series = pd.to_datetime(df_input[time_col], errors="coerce", utc=True)
        try:
            ts_series = ts_series.dt.tz_localize(None)
        except Exception:
            pass
    if ts_series is None or (val_len + test_len) <= 0:
        return
    total = int(len(ts_series))
    train_len = max(0, total - val_len - test_len)
    if data_blk.get("val_timestamps") is None and val_len > 0:
        data_blk["val_timestamps"] = ts_series.iloc[train_len : train_len + val_len].tolist()
    if data_blk.get("test_timestamps") is None and test_len > 0:
        data_blk["test_timestamps"] = ts_series.iloc[train_len + val_len : train_len + val_len + test_len].tolist()


def inverse_series_1d_from_df_scaled(
    df_sc: pd.DataFrame,
    scaler: Any,
    config: Dict[str, Any],
    value_col: str,
) -> pd.Series:
    from utils.target_transform import inverse_transform_array as _inv_tt

    arr2d = df_sc[[value_col]].to_numpy().astype(np.float32)
    tt_params = (config.get("artifacts") or {}).get("target_transform")
    try:
        artifacts = (config.get("artifacts") or {})
        y_scaler_path = artifacts.get("y_scaler_path")
        if y_scaler_path and os.path.exists(y_scaler_path):
            import joblib

            y_scaler = joblib.load(y_scaler_path)
            inv = y_scaler.inverse_transform(arr2d)
            out = inv.reshape(-1)
            if tt_params:
                out = _inv_tt(out, tt_params)
            return pd.Series(out, index=df_sc.index)
    except Exception as e:
        print(f"[pipeline] y_scaler inverse failed: {e}")
    n_in = getattr(scaler, "n_features_in_", None)
    if n_in is None or not hasattr(scaler, "inverse_transform"):
        out = arr2d.reshape(-1)
        if tt_params:
            out = _inv_tt(out, tt_params)
        return pd.Series(out, index=df_sc.index)
    if arr2d.shape[1] == n_in:
        inv = scaler.inverse_transform(arr2d)
        out = inv.reshape(-1)
        if tt_params:
            out = _inv_tt(out, tt_params)
        return pd.Series(out, index=df_sc.index)
    all_cols = (
        (config.get("artifacts", {}) or {}).get("feature_cols")
        or (config.get("data", {}) or {}).get("all_feature_cols")
        or [value_col]
    )
    tmp = np.zeros((arr2d.shape[0], n_in), dtype=np.float32)
    try:
        idx = all_cols.index(value_col)
    except ValueError:
        idx = 0
    tmp[:, idx] = arr2d[:, 0]
    try:
        inv_wide = scaler.inverse_transform(tmp)
        out = inv_wide[:, idx]
    except Exception:
        out = arr2d[:, 0]
    if tt_params:
        out = _inv_tt(out, tt_params)
    return pd.Series(out, index=df_sc.index)


def build_training_true_series(
    data_block: Dict[str, Any],
    *,
    scaler: Any,
    config: Dict[str, Any],
    value_col: str,
    emit_warning: bool = False,
) -> Optional[pd.Series]:
    try:
        train_df_sc = data_block.get("train_df_sc")
        if isinstance(train_df_sc, pd.DataFrame) and len(train_df_sc) > 0 and scaler is not None:
            return inverse_series_1d_from_df_scaled(train_df_sc, scaler, config, value_col)
    except Exception as exc:
        if emit_warning:
            print(f"[pipeline] Warning: failed to build training_true series: {exc}")
    return None
