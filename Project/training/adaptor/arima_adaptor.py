import pandas as pd
from typing import Tuple, Any, cast

def train_arima_model_7tuple(df, config):
    from training.train_arima import train_arima_model
    artifacts = config.setdefault("artifacts", {})

    def _to_training_params(raw_params, train_len=0, val_len=0, test_len=0):
        out = {
            "model": "arima",
            "split": {"train_len": int(train_len), "val_len": int(val_len), "test_len": int(test_len)},
            "fit_status": "trained",
        }
        if isinstance(raw_params, dict):
            out.update(raw_params)
        elif raw_params is not None:
            out["order"] = raw_params
        artifacts["training_params"] = dict(out)
        return out

    # 允许两种返回：7元组(直接透传) 或 (model, result_df)
    out = train_arima_model(df, config)

    if isinstance(out, tuple) and len(out) == 7:
        val_true, val_forecast, test_true, test_forecast, final_model, test_forecast_df, raw_params = out
        v_len = int(len(val_true)) if val_true is not None else 0
        te_len = int(len(test_true)) if test_true is not None else 0
        tr_len = max(0, int(len(df)) - v_len - te_len)
        training_params = _to_training_params(raw_params, tr_len, v_len, te_len)
        return (val_true, val_forecast, test_true, test_forecast, final_model, test_forecast_df, training_params)

    if not (isinstance(out, tuple) and len(out) == 2):
        raise RuntimeError("train_arima_model returned unexpected shape; expected 2-tuple or 7-tuple")

    final_model, result_df = cast(Tuple[Any, Any], out)

    val_true = val_forecast = test_true = test_forecast = None
    test_forecast_df = None
    # 如可读阶数信息，写入 best_params；否则 None
    training_params = {"order": getattr(final_model, "order_", None) or getattr(final_model, "order", None) or None}

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

    v_len = int(len(val_true)) if val_true is not None else 0
    te_len = int(len(test_true)) if test_true is not None else 0
    tr_len = max(0, int(len(df)) - v_len - te_len)
    training_params = _to_training_params(training_params, tr_len, v_len, te_len)
    return (val_true, val_forecast, test_true, test_forecast,
            final_model, test_forecast_df, training_params)


def get_forecaster():
    from models.base import TrainFunctionForecaster

    return TrainFunctionForecaster("arima", train_arima_model_7tuple)
