# training/prophet_adapter.py
import pandas as pd
from typing import Tuple, Any, cast
from training.adaptor.common import build_adapter_training_params, extract_split_predictions, infer_split_lengths

def train_prophet_model_7tuple(df, config):
    # 延迟导入避免循环依赖
    from training.train_prophet import train_prophet_model

    out = train_prophet_model(df, config)

    artifacts = config.setdefault("artifacts", {})

    def _to_training_params(raw_params, train_len=0, val_len=0, test_len=0):
        params = raw_params if isinstance(raw_params, dict) else {}
        out = build_adapter_training_params(
            model="prophet",
            df=df,
            config=config,
            split={"train_len": int(train_len), "val_len": int(val_len), "test_len": int(test_len)},
            core_hparams=params,
            runtime={"fit_status": "trained"},
            legacy_fields={"fit_status": "trained", "model_name": "prophet", **params},
            artifacts=artifacts,
        )
        return out

    # If the original trainer already returns a 7-tuple, normalize the 7th slot to training_params(dict).
    if isinstance(out, tuple) and len(out) == 7:
        val_true, val_forecast, test_true, test_forecast, final_model, test_forecast_df, raw_params = out
        v_len = int(len(val_true)) if val_true is not None else 0
        te_len = int(len(test_true)) if test_true is not None else 0
        train_len = max(0, int(len(df)) - v_len - te_len)
        training_params = _to_training_params(raw_params, train_len, v_len, te_len)
        return (
            val_true, val_forecast, test_true, test_forecast,
            final_model, test_forecast_df, training_params
        )

    # Otherwise expect (model, result_df)
    if not (isinstance(out, tuple) and len(out) == 2):
        raise RuntimeError("train_prophet_model returned unexpected shape; expected 2-tuple or 7-tuple")

    final_model, result_df = cast(Tuple[Any, Any], out)

    val_true = val_forecast = test_true = test_forecast = None
    test_forecast_df = None
    training_params = {"model": "prophet"}  # Prophet 通常不调参

    val_true, val_forecast, test_true, test_forecast = extract_split_predictions(result_df)

    split = infer_split_lengths(df, val_true, test_true)
    training_params = _to_training_params(training_params, split["train_len"], split["val_len"], split["test_len"])
    return (val_true, val_forecast, test_true, test_forecast,
            final_model, test_forecast_df, training_params)


def get_forecaster():
    from models.base import TrainFunctionForecaster

    return TrainFunctionForecaster("prophet", train_prophet_model_7tuple)
