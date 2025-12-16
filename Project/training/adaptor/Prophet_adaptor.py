# training/prophet_adapter.py
import pandas as pd
import numpy as np
from typing import Tuple, Any, cast

def train_prophet_model_7tuple(df, config):
    # 延迟导入避免循环依赖
    from training.train_prophet import train_prophet_model

    out = train_prophet_model(df, config)

    # If the original trainer already returns a 7-tuple, pass it through directly.
    if isinstance(out, tuple) and len(out) == 7:
        return out

    # Otherwise expect (model, result_df)
    if not (isinstance(out, tuple) and len(out) == 2):
        raise RuntimeError("train_prophet_model returned unexpected shape; expected 2-tuple or 7-tuple")

    final_model, result_df = cast(Tuple[Any, Any], out)

    val_true = val_forecast = test_true = test_forecast = None
    test_forecast_df = None
    best_params = None  # Prophet 通常不调参

    if isinstance(result_df, pd.DataFrame) and {"y_true", "yhat"} <= set(result_df.columns):
        if "phase" in result_df.columns:
            is_val = result_df["phase"].astype(str).str.lower().eq("val")
            is_tst = result_df["phase"].astype(str).str.lower().eq("test")
            val_true      = result_df.loc[is_val, "y_true"].to_numpy()
            val_forecast  = result_df.loc[is_val, "yhat"].to_numpy()
            test_true     = result_df.loc[is_tst, "y_true"].to_numpy()
            test_forecast = result_df.loc[is_tst, "yhat"].to_numpy()
        else:
            n = len(result_df)
            cut = int(n * 0.8)
            val_true      = result_df.iloc[:cut].loc[:, "y_true"].to_numpy()
            val_forecast  = result_df.iloc[:cut].loc[:, "yhat"].to_numpy()
            test_true     = result_df.iloc[cut:].loc[:, "y_true"].to_numpy()
            test_forecast = result_df.iloc[cut:].loc[:, "yhat"].to_numpy()

    return (val_true, val_forecast, test_true, test_forecast,
            final_model, test_forecast_df, best_params)