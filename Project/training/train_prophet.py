from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.array_utils import clean_and_unify_arrays
import numpy as np
import pandas as pd

def train_prophet_model(df, config):
    # 数据有效性检查
    if df.isnull().any().any():
        raise ValueError("❌ 输入数据包含缺失值，请先进行清洗处理。")
    if len(df) < 30:
        raise ValueError("❌ 数据行数不足，至少需要30行数据才能训练 Prophet 模型。")

    # Ensure Prophet-compatible column names
    time_col = config.get("time_col", "date")
    value_col = config.get("value_col", "value")
    df = df.rename(columns={
        time_col: "ds",
        value_col: "y"
    })

    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    model = Prophet(
        yearly_seasonality=config.get("yearly_seasonality", True),
        weekly_seasonality=config.get("weekly_seasonality", False),
        daily_seasonality=config.get("daily_seasonality", False),
        seasonality_mode=config.get("seasonality_mode", "additive"),
        changepoint_prior_scale=config.get("changepoint_prior_scale", 0.05)
    )

    model.fit(df_train)
    forecast = model.predict(df_test)
    y_true = df_test["y"].values
    y_pred = forecast["yhat"].values

    # 维度统一处理
    val_true, val_forecast, _ = clean_and_unify_arrays(y_true, y_pred)
    test_true, test_forecast, _ = clean_and_unify_arrays(y_true, y_pred)

    final_model = model
    test_forecast_df = forecast
    best_params = None

    return val_true, val_forecast, test_true, test_forecast, final_model, test_forecast_df, best_params
