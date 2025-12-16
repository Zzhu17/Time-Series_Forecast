import pandas as pd
from typing import Tuple, Any, cast

def train_arima_model_7tuple(df, config):
    from training.train_arima import train_arima_model

    # 允许两种返回：7元组(直接透传) 或 (model, result_df)
    out = train_arima_model(df, config)

    if isinstance(out, tuple) and len(out) == 7:
        return out

    if not (isinstance(out, tuple) and len(out) == 2):
        raise RuntimeError("train_arima_model returned unexpected shape; expected 2-tuple or 7-tuple")

    final_model, result_df = cast(Tuple[Any, Any], out)

    val_true = val_forecast = test_true = test_forecast = None
    test_forecast_df = None
    # 如可读阶数信息，写入 best_params；否则 None
    best_params = getattr(final_model, "order_", None) or getattr(final_model, "order", None) or None

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