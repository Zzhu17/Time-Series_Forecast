import os, json
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.array_utils import clean_and_unify_arrays
from models.random_forest import build_random_forest

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
    work = work.dropna().reset_index(drop=True)

    # 3) 切分与缺失值填充（简单 80/20；后续可替换为更严格的时序切法）
    split_idx = int(len(work) * 0.8)
    X = work.drop(columns=[c for c in [time_col, value_col] if c in work.columns])
    y = work[value_col].to_numpy().reshape(-1)

    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=work.index)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 4) 强制 Optuna 调参：把 configs.yaml 的 optimization.n_trials 写入环境变量
    n_trials = int((config.get("optimization", {}) or {}).get("n_trials", 50))  # 默认 50
    os.environ["OPTUNA_N_TRIALS"] = str(n_trials)  # models/random_forest.py 兼容读取
    os.environ["RF_N_TRIALS"] = str(n_trials)      # 同步一份，确保无论读哪个 key 都生效  [oai_citation:0‡configs.yaml](file-service://file-LvUz4wMVdpTQWJSmE8ievv)

    # 5) 训练（models/random_forest.py 会返回 (model, feature_cols) 且 model.best_params_ 已挂好）
    model, feature_cols = build_random_forest(X_train, y_train, auto_tune=True)
    best_params = getattr(model, "best_params_", {}) or {}

    # 6) 预测
    y_pred = model.predict(X_test)
    try:
        y_pred = np.maximum(y_pred, 0.0)
        y_pred = pd.Series(y_pred).rolling(window=3, min_periods=1).mean().to_numpy()
    except Exception:
        # 后处理失败不影响主流程
        pass

    # 7) 统一数组（验证/测试此处相同切分——若你以后引入专用验证集，可按需替换）
    val_true, val_forecast, _ = clean_and_unify_arrays(y_test, y_pred)
    test_true, test_forecast, _ = clean_and_unify_arrays(y_test, y_pred)

    # 8) 落盘特征列（预测端严格对齐列顺序）
    arts = config.setdefault("artifacts", {})
    feat_path = arts.get("feature_cols_path", "artifacts/feature_cols.json")  # 路径来自 configs.yaml   [oai_citation:1‡configs.yaml](file-service://file-LvUz4wMVdpTQWJSmE8ievv)
    os.makedirs(os.path.dirname(feat_path), exist_ok=True)
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False)

    # 9) 将最优超参暴露到 artifacts，供 app 的“最佳超参”面板读取
    arts["randomforest_params"] = best_params

    # 10) 构造 test_forecast_df（带时间索引，兼容 pipeline 连续绘图）
    test_forecast_df = None
    try:
        if time_col in work.columns:
            ts_series = pd.to_datetime(work[time_col], errors="coerce")
            test_ts = ts_series.iloc[split_idx:].reset_index(drop=True)
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