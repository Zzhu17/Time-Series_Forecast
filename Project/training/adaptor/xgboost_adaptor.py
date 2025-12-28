from __future__ import annotations

import pandas as pd

from training.train_xgboost import train_xgboost_model
from utils.array_utils import clean_dataframe


def _read_defaults(config: dict) -> tuple[str, str]:
    dft = config.get("default", {}) or {}
    time_col = config.get("time_col", dft.get("time_col", "date"))
    value_col = config.get("value_col", dft.get("value_col", "value"))
    return str(time_col), str(value_col)


def train_xgboost_model_7tuple(df: pd.DataFrame, config: dict):
    """
    Adapter to match the unified 7-tuple trainer interface:
      (val_true, val_pred, test_true, test_pred, final_model, test_df, params)
    """
    time_col, value_col = _read_defaults(config)

    # Safety cleaning (do not change model semantics; strict feature policy happens in pipeline)
    _df_clean = clean_dataframe(df, value_col=value_col, time_col=time_col, feature_cols=None)
    if _df_clean is not None:
        df = _df_clean

    # Ensure consistent key locations for downstream plotting/caching
    config.setdefault("time_col", time_col)
    config.setdefault("value_col", value_col)
    config.setdefault("default", {}).setdefault("time_col", time_col)
    config.setdefault("default", {}).setdefault("value_col", value_col)
    config.setdefault("data", {})
    config["data"].setdefault("df", df.copy())
    config["data"].setdefault("dataframe", df.copy())

    return train_xgboost_model(df, config)

