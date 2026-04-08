import os
import pickle
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.array_utils import clean_and_unify_arrays
import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


def _save_model(model, path: str) -> None:
    if not isinstance(path, str) or not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if joblib is not None:
        joblib.dump(model, path)
        return
    with open(path, "wb") as f:
        pickle.dump(model, f)

def train_prophet_model(df, config):
    default_cfg = config.get("default", {}) if isinstance(config, dict) else {}
    time_col = config.get("time_col", default_cfg.get("time_col", "date"))
    value_col = config.get("value_col", default_cfg.get("value_col", "value"))

    df = df.loc[:, [time_col, value_col]].copy()

    # 数据有效性检查
    if df.isnull().any().any():
        raise ValueError("❌ 输入数据包含缺失值，请先进行清洗处理。")
    if len(df) < 30:
        raise ValueError("❌ 数据行数不足，至少需要30行数据才能训练 Prophet 模型。")

    # 限制超长序列，避免训练过慢
    max_train_rows = None
    try:
        pcfg = (config.get("prophet") or {}) if isinstance(config, dict) else {}
        max_train_rows = pcfg.get("max_train_rows")
        if max_train_rows is None and len(df) > 20000:
            max_train_rows = 20000
        if isinstance(max_train_rows, (int, float)) and int(max_train_rows) > 0 and len(df) > int(max_train_rows):
            df = df.tail(int(max_train_rows)).copy()
    except Exception:
        pass

    # Ensure Prophet-compatible column names
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
    cv_summary = None
    try:
        rolling_cfg = (((config.get("model_config") or {}).get("Prophet") or {}).get("rolling_cv") or {})
        if bool(rolling_cfg.get("enabled", False)):
            from prophet.diagnostics import cross_validation, performance_metrics

            cv_df = cross_validation(
                model,
                initial=str(rolling_cfg.get("initial", "180 days")),
                period=str(rolling_cfg.get("period", "30 days")),
                horizon=str(rolling_cfg.get("horizon", "30 days")),
                parallel=str(rolling_cfg.get("parallel", "processes")),
            )
            perf_df = performance_metrics(cv_df, rolling_window=float(rolling_cfg.get("rolling_window", 0.1)))
            if isinstance(perf_df, pd.DataFrame) and not perf_df.empty:
                last = perf_df.tail(1).iloc[0].to_dict()
                cv_summary = {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in last.items()}
    except Exception:
        cv_summary = None

    forecast = model.predict(df_test)
    y_true = df_test["y"].values
    y_pred = forecast["yhat"].values

    # 维度统一处理
    val_true, val_forecast, _ = clean_and_unify_arrays(y_true, y_pred)
    test_true, test_forecast, _ = clean_and_unify_arrays(y_true, y_pred)

    final_model = model
    test_forecast_df = forecast
    best_params = None

    # Persist model for registry-based inference
    arts = config.setdefault("artifacts", {})
    model_path = arts.get("model_path")
    if isinstance(model_path, str) and model_path:
        try:
            _save_model(final_model, model_path)
        except Exception:
            pass
    if cv_summary is not None:
        arts["prophet_rolling_cv"] = cv_summary

    return val_true, val_forecast, test_true, test_forecast, final_model, test_forecast_df, best_params
