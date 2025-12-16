# models/random_forest.py
from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import optuna
from typing import List, Tuple


def _get_n_trials(default_val: int = 50) -> int:
    """
    从环境变量或默认值获取 n_trials。
    优先读取 RF_N_TRIALS，其次 OPTUNA_N_TRIALS，最后使用 default_val。
    """
    for key in ("RF_N_TRIALS", "OPTUNA_N_TRIALS"):
        v = os.environ.get(key, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return int(default_val)


def _suggest_params(trial: optuna.trial.Trial, random_state: int = 42) -> dict:
    """
    定义 RF 的搜索空间（避免使用已弃用的 'auto'）。
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 32, step=2),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.75, 1.0]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "random_state": random_state,
        "n_jobs": -1,
    }
    return params


def build_random_forest(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    auto_tune: bool = True,
) -> Tuple[RandomForestRegressor, List[str]]:
    """
    使用原生 RandomForestRegressor + Optuna 调参（默认开启），返回 (final_model, selected_features)。

    - 不自定义 sklearn 子类，规避 __init__ 规范报错。
    - 最优超参写入 final_model.best_params_ 供外部读取。
    - selected_features 返回训练时的特征列顺序。
    - n_trials 优先从环境变量 RF_N_TRIALS/OPTUNA_N_TRIALS 读取，没设则默认 50。
    """
    # 规范化 X/y，并拿到特征列名
    if isinstance(X_train, pd.DataFrame):
        X_df = X_train.copy()
        feature_cols = list(X_df.columns)
    else:
        X_arr = np.asarray(X_train)
        feature_cols = [f"f{i}" for i in range(X_arr.shape[1])]
        X_df = pd.DataFrame(X_arr, columns=feature_cols)

    y_arr = np.asarray(y_train).reshape(-1)

    random_state = 42
    tscv = TimeSeriesSplit(n_splits=5)

    if auto_tune:
        n_trials = _get_n_trials(default_val=50)

        def stable_mape(y_true, y_pred, eps: float = 1e-3) -> float:
            """Robust MAPE with epsilon in denominator to avoid explosion near zero."""
            y_true = np.asarray(y_true, dtype=float).ravel()
            y_pred = np.asarray(y_pred, dtype=float).ravel()
            denom = np.maximum(np.abs(y_true), eps)
            return float(np.mean(np.abs((y_pred - y_true) / denom)))

        def objective(trial: optuna.trial.Trial) -> float:
            params = _suggest_params(trial, random_state=random_state)
            model = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                bootstrap=params["bootstrap"],
                random_state=random_state,
                n_jobs=-1,
            )
            mape_scores = []
            for tr_idx, va_idx in tscv.split(X_df):
                X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
                y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]
                model.fit(X_tr, y_tr)
                pred = model.predict(X_va)
                mape = stable_mape(y_va, pred)
                mape_scores.append(mape)
            return float(np.mean(mape_scores))

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        # 补充稳定参数并显式指定全部参数
        best_params = {
            "n_estimators":      best_params.get("n_estimators"),
            "max_depth":         best_params.get("max_depth"),
            "min_samples_split": best_params.get("min_samples_split"),
            "min_samples_leaf":  best_params.get("min_samples_leaf"),
            "max_features":      best_params.get("max_features"),
            "bootstrap":         best_params.get("bootstrap", True),
            "random_state":      random_state,
            "n_jobs":            -1,
        }
    else:
        best_params = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "random_state": random_state,
            "n_jobs": -1,
        }

    # 用最优参数在全训练集拟合最终模型
    final_model = RandomForestRegressor(
        n_estimators=best_params.get("n_estimators", 200),
        max_depth=best_params.get("max_depth", 10),
        min_samples_split=best_params.get("min_samples_split", 2),
        min_samples_leaf=best_params.get("min_samples_leaf", 1),
        max_features=best_params.get("max_features", "sqrt"),
        bootstrap=best_params.get("bootstrap", True),
        random_state=best_params.get("random_state", 42),
        n_jobs=best_params.get("n_jobs", -1),
    )
    final_model.fit(X_df, y_arr)

    # 挂载最优超参（给外部读取）
    setattr(final_model, "best_params_", dict(best_params))

    selected_features = feature_cols
    return final_model, selected_features