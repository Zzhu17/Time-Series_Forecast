from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from training.params_schema import build_training_params


def extract_split_predictions(
    result_df: object,
    *,
    split: Optional[dict] = None,
) -> Tuple[object, object, object, object]:
    if not isinstance(result_df, pd.DataFrame) or not {"y_true", "yhat"} <= set(result_df.columns):
        return None, None, None, None

    if "phase" in result_df.columns:
        is_val = result_df["phase"].astype(str).str.lower().eq("val")
        is_tst = result_df["phase"].astype(str).str.lower().eq("test")
        return (
            result_df.loc[is_val, "y_true"].to_numpy(),
            result_df.loc[is_val, "yhat"].to_numpy(),
            result_df.loc[is_tst, "y_true"].to_numpy(),
            result_df.loc[is_tst, "yhat"].to_numpy(),
        )

    split = split or {}
    val_len = int(split.get("val_len") or 0)
    test_len = int(split.get("test_len") or 0)
    if val_len > 0 and test_len > 0 and len(result_df) >= val_len + test_len:
        val_part = result_df.iloc[-(val_len + test_len): -test_len]
        test_part = result_df.iloc[-test_len:]
        return (
            val_part["y_true"].to_numpy(),
            val_part["yhat"].to_numpy(),
            test_part["y_true"].to_numpy(),
            test_part["yhat"].to_numpy(),
        )

    n = len(result_df)
    cut = int(n * 0.8)
    return (
        result_df.iloc[:cut]["y_true"].to_numpy(),
        result_df.iloc[:cut]["yhat"].to_numpy(),
        result_df.iloc[cut:]["y_true"].to_numpy(),
        result_df.iloc[cut:]["yhat"].to_numpy(),
    )


def extract_dense_predictions(df_like: object) -> Tuple[object, object]:
    if not isinstance(df_like, pd.DataFrame):
        return None, None
    if not {"y_true", "yhat"} <= set(df_like.columns):
        return None, None
    y_true = np.asarray(df_like["y_true"].to_numpy(), dtype=float).reshape(-1)
    y_pred = np.asarray(df_like["yhat"].to_numpy(), dtype=float).reshape(-1)
    length = min(len(y_true), len(y_pred))
    return y_true[:length], y_pred[:length]


def infer_split_lengths(df: pd.DataFrame, val_true: object, test_true: object) -> Dict[str, int]:
    val_len = int(len(val_true)) if val_true is not None else 0
    test_len = int(len(test_true)) if test_true is not None else 0
    train_len = max(0, int(len(df)) - val_len - test_len)
    return {"train_len": train_len, "val_len": val_len, "test_len": test_len}


def read_defaults(config: dict) -> Tuple[str, str]:
    dft = config.get("default", {}) or {}
    time_col = config.get("time_col", dft.get("time_col", "date"))
    value_col = config.get("value_col", dft.get("value_col", "value"))
    return str(time_col), str(value_col)


def build_adapter_training_params(
    *,
    model: str,
    df: pd.DataFrame,
    config: dict,
    split: Dict[str, int],
    core_hparams: Optional[Dict[str, Any]] = None,
    runtime: Optional[Dict[str, Any]] = None,
    legacy_fields: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    feature_cols: Optional[list] = None,
) -> Dict[str, Any]:
    data_signature: Dict[str, Any] = {
        "rows": int(len(df)),
        "time_col": config.get("time_col") or config.get("default", {}).get("time_col"),
        "value_col": config.get("value_col") or config.get("default", {}).get("value_col"),
    }
    if feature_cols is not None:
        data_signature["feature_cols"] = list(feature_cols)
    out = build_training_params(
        model=model,
        split=split,
        core_hparams=core_hparams or {},
        runtime=runtime or {"fit_status": "trained"},
        data_signature=data_signature,
        legacy_fields=legacy_fields or {},
    )
    if isinstance(artifacts, dict):
        artifacts["training_params"] = dict(out)
    return out
