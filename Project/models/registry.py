"""
Model/trainer registry.

Important: keep imports LAZY so the app can start even when optional dependencies
(torch/prophet/optuna/pmdarima/sklearn/...) are not installed.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict


def _lazy(module: str, attr: str, *, install_hint: str | None = None) -> Callable[..., Any]:
    """
    Return a callable proxy that imports `module.attr` only when invoked.
    This prevents Streamlit from crashing at import time due to missing optional deps.
    """

    def _call(*args, **kwargs):
        try:
            mod = importlib.import_module(module)
        except Exception as e:
            hint = f" Install: {install_hint}" if install_hint else ""
            raise RuntimeError(f"Failed to import `{module}` required for `{attr}`.{hint} Error: {e}") from e
        try:
            fn = getattr(mod, attr)
        except Exception as e:
            raise RuntimeError(f"Missing symbol `{attr}` in `{module}`: {e}") from e
        return fn(*args, **kwargs)

    _call.__name__ = f"lazy:{module}.{attr}"
    return _call

MODEL_REGISTRY = {
    "arima": _lazy("models.arima", "build_auto_arima", install_hint="pip install pmdarima"),
    "prophet": _lazy("models.prophet", "build_prophet", install_hint="pip install prophet"),
    "randomforest": _lazy("models.random_forest", "build_random_forest", install_hint="pip install scikit-learn optuna"),
    "informer": _lazy("models.informer.informer", "build_informer_model", install_hint="pip install torch"),
    "lstm": _lazy("models.lstm", "lstm_model", install_hint="pip install torch"),
}

TRAINER_REGISTRY = {
    "informer": _lazy("training.adaptor.informer_adaptor", "train_informer_model_7tuple", install_hint="pip install torch scikit-learn joblib pyyaml"),
    "prophet": _lazy("training.adaptor.Prophet_adaptor", "train_prophet_model_7tuple", install_hint="pip install prophet scikit-learn"),
    "arima": _lazy("training.adaptor.arima_adaptor", "train_arima_model_7tuple", install_hint="pip install pmdarima scikit-learn"),
    "randomforest": _lazy("training.train_random_forest", "train_random_forest_model", install_hint="pip install scikit-learn optuna"),
    "lstm": _lazy("training.adaptor.LSTM_adaptor", "train_lstm_model_7tuple", install_hint="pip install torch scikit-learn joblib"),
    "xgboost": _lazy("training.adaptor.xgboost_adaptor", "train_xgboost_model_7tuple", install_hint="pip install xgboost"),
}

FORECASTER_REGISTRY = {
    "baseline": _lazy("models.base", "BaselineForecaster"),
    "informer": _lazy("training.adaptor.informer_adaptor", "get_forecaster", install_hint="pip install torch scikit-learn joblib pyyaml"),
    "prophet": _lazy("training.adaptor.Prophet_adaptor", "get_forecaster", install_hint="pip install prophet scikit-learn"),
    "arima": _lazy("training.adaptor.arima_adaptor", "get_forecaster", install_hint="pip install pmdarima scikit-learn"),
    "randomforest": _lazy("training.train_random_forest", "get_forecaster", install_hint="pip install scikit-learn optuna"),
    "lstm": _lazy("training.adaptor.LSTM_adaptor", "get_forecaster", install_hint="pip install torch scikit-learn joblib"),
    "xgboost": _lazy("training.adaptor.xgboost_adaptor", "get_forecaster", install_hint="pip install xgboost"),
}

MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "baseline": {"description": "Naive last-value persistence.", "deps": []},
    "informer": {"description": "Transformer forecaster (requires torch).", "deps": ["torch"]},
    "lstm": {"description": "LSTM forecaster (requires torch).", "deps": ["torch"]},
    "xgboost": {"description": "Gradient boosting regressor (requires xgboost).", "deps": ["xgboost"]},
    "randomforest": {"description": "Random forest regressor (requires scikit-learn).", "deps": ["sklearn"]},
    "arima": {"description": "Auto ARIMA (requires pmdarima).", "deps": ["pmdarima"]},
    "prophet": {"description": "Prophet forecaster (requires prophet).", "deps": ["prophet"]},
    "xgboost+informer": {"description": "Informer forecast + XGBoost residual correction.", "deps": ["torch", "xgboost"]},
    "xgboost+lstm": {"description": "LSTM forecast + XGBoost residual correction.", "deps": ["torch", "xgboost"]},
}

__all__ = ("MODEL_REGISTRY", "TRAINER_REGISTRY", "FORECASTER_REGISTRY", "MODEL_CATALOG")
