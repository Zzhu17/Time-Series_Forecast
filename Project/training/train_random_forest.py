import os, json
import pickle
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.array_utils import clean_and_unify_arrays
from models.random_forest import build_random_forest

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

def train_random_forest_model(df: pd.DataFrame, config):
    """
    返回 7 元组：
    (val_true, val_forecast, test_true, test_forecast, final_model, test_forecast_df, best_params)
    """
    # === 基本配置 ===
    time_col  = config.get("default", {}).get("time_col",  "date")
    value_col = config.get("default", {}).get("value_col", "value")

    # 1) 读取 n_lags（若未给，默认 10）
    rf_cfg = (config.get("model_config", {}) or {}).get("RandomForest", {}) or {}
    n_lags = int(rf_cfg.get("n_lags", 10))

    # 2) 构造滞后特征
    work = df.copy()
    if time_col in work.columns:
        # 确保时间列为 datetime，用于后续对齐
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    for i in range(1, n_lags + 1):
        work[f"lag_{i}"] = work[value_col].shift(i)

    # 2.1) 额外季节性滞后/滚动特征已移除，避免过拟合

    # 2.1) 时间特征（有助于稳定泛化）
    if time_col in work.columns:
        work["hour"] = work[time_col].dt.hour
        work["dayofweek"] = work[time_col].dt.dayofweek
        work["month"] = work[time_col].dt.month
        work["day_of_month"] = work[time_col].dt.day
    work = work.dropna().reset_index(drop=True)

    # 3) 切分与缺失值填充（80/10/10 时序切分）
    n_total = len(work)
    train_end = int(n_total * 0.8)
    val_end = int(n_total * 0.9)

    # 使用所有数值特征（排除 time_col / value_col）
    drop_cols = [c for c in [time_col, value_col] if c in work.columns]
    X = work.drop(columns=drop_cols, errors="ignore")
    y = work[value_col].to_numpy().reshape(-1)

    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=work.index)

    X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    # 4) 训练（稳定参数，关闭 Optuna 调参以降低过拟合）
    model, feature_cols = build_random_forest(X_train, y_train, auto_tune=False)
    best_params = getattr(model, "best_params_", {}) or {}

    # 5) 预测
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    try:
        val_pred = np.maximum(val_pred, 0.0)
        test_pred = np.maximum(test_pred, 0.0)
        val_pred = pd.Series(val_pred).rolling(window=3, min_periods=1).mean().to_numpy()
        test_pred = pd.Series(test_pred).rolling(window=3, min_periods=1).mean().to_numpy()
    except Exception:
        # 后处理失败不影响主流程
        pass

    # 6) 统一数组（验证/测试分离）
    val_true, val_forecast, _ = clean_and_unify_arrays(y_val, val_pred)
    test_true, test_forecast, _ = clean_and_unify_arrays(y_test, test_pred)

    # 8) 落盘特征列（预测端严格对齐列顺序）
    arts = config.setdefault("artifacts", {})
    feat_path = arts.get("feature_cols_path", "artifacts/feature_cols.json")  # 路径来自 configs.yaml   [oai_citation:1‡configs.yaml](file-service://file-LvUz4wMVdpTQWJSmE8ievv)
    os.makedirs(os.path.dirname(feat_path), exist_ok=True)
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False)
    arts["feature_cols"] = list(feature_cols)

    # 8.1) 落盘模型（预测端加载）
    model_path = arts.get("model_path")
    if isinstance(model_path, str) and model_path:
        try:
            _save_model(model, model_path)
        except Exception:
            pass

    # 9) 将最优超参暴露到 artifacts，供 app 的“最佳超参”面板读取
    arts["randomforest_params"] = best_params

    # 10) 构造 test_forecast_df（带时间索引，兼容 pipeline 连续绘图）
    test_forecast_df = None
    try:
        if time_col in work.columns:
            ts_series = pd.to_datetime(work[time_col], errors="coerce")
            test_ts = ts_series.iloc[val_end:].reset_index(drop=True)
            # 对齐 y_true / yhat 的长度（以较短者为准）
            L = min(len(test_ts), len(test_true), len(test_forecast))
            if L > 0:
                test_forecast_df = pd.DataFrame({
                    "y_true": np.asarray(test_true, dtype=float)[:L].reshape(-1),
                    "yhat":   np.asarray(test_forecast, dtype=float)[:L].reshape(-1),
                }, index=pd.DatetimeIndex(test_ts[:L], name=time_col))
    except Exception as _e:
        # 构造失败不影响主流程
        test_forecast_df = None

    final_model = model
    return val_true, val_forecast, test_true, test_forecast, final_model, test_forecast_df, best_params


def get_forecaster():
    from models.base import TrainFunctionForecaster

    return TrainFunctionForecaster("randomforest", train_random_forest_model)
