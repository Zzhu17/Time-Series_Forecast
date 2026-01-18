from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import abc

import numpy as np
import pandas as pd


@dataclass
class FitResult:
    val_true: np.ndarray
    val_pred: np.ndarray
    test_true: np.ndarray
    test_pred: np.ndarray
    model: Any
    test_forecast_df: Optional[pd.DataFrame]
    params: Optional[Dict[str, Any]]


class BaseForecaster(abc.ABC):
    name: str = ""
    supports_predict: bool = False

    @abc.abstractmethod
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> FitResult:
        raise NotImplementedError

    def predict(self, context_df: pd.DataFrame, horizon: int, config: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError("predict not implemented for this forecaster")

    def save(self, artifacts_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
        return config.get("artifacts", {}) if isinstance(config, dict) else {}

    @classmethod
    def load(cls, artifacts_dir: str, config: Dict[str, Any]) -> "BaseForecaster":
        return cls()


class TrainFunctionForecaster(BaseForecaster):
    def __init__(self, name: str, trainer_fn):
        self.name = name
        self._trainer = trainer_fn
        self.supports_predict = False
        self._last_result: Optional[FitResult] = None

    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> FitResult:
        result = self._trainer(df, config)
        if not isinstance(result, (list, tuple)) or len(result) != 7:
            raise ValueError("trainer_fn must return 7-tuple")
        val_true, val_pred, test_true, test_pred, model, test_forecast_df, params = result
        fit = FitResult(
            val_true=np.asarray(val_true),
            val_pred=np.asarray(val_pred),
            test_true=np.asarray(test_true),
            test_pred=np.asarray(test_pred),
            model=model,
            test_forecast_df=test_forecast_df if isinstance(test_forecast_df, pd.DataFrame) else None,
            params=params if isinstance(params, dict) else None,
        )
        self._last_result = fit
        return fit


class BaselineForecaster(BaseForecaster):
    name = "baseline"
    supports_predict = True

    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> FitResult:
        time_col = (config.get("default", {}) or {}).get("time_col", "date")
        value_col = (config.get("default", {}) or {}).get("value_col", "value")
        y = pd.to_numeric(df[value_col], errors="coerce").dropna()
        if y.empty:
            raise ValueError("baseline fit requires numeric target values")
        # Split 6/2/2 as default
        n = len(y)
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)
        n_test = max(0, n - n_train - n_val)
        y_val = y.iloc[n_train : n_train + n_val].to_numpy(dtype=float)
        y_test = y.iloc[n_train + n_val : n_train + n_val + n_test].to_numpy(dtype=float)
        val_pred = self._naive_predict(y, len(y_val))
        test_pred = self._naive_predict(y.iloc[: n_train + n_val], len(y_test))
        return FitResult(
            val_true=y_val,
            val_pred=val_pred,
            test_true=y_test,
            test_pred=test_pred,
            model=None,
            test_forecast_df=None,
            params={"baseline": "naive_last"},
        )

    def predict(self, context_df: pd.DataFrame, horizon: int, config: Dict[str, Any]) -> np.ndarray:
        value_col = (config.get("default", {}) or {}).get("value_col", "value")
        y = pd.to_numeric(context_df[value_col], errors="coerce").dropna()
        if y.empty:
            raise ValueError("baseline predict requires numeric target values")
        return self._naive_predict(y, horizon)

    @staticmethod
    def _naive_predict(series: pd.Series, horizon: int) -> np.ndarray:
        last = float(series.iloc[-1])
        return np.full(int(horizon), last, dtype=float)
