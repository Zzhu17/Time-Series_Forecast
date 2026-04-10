import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.params_schema import validate_training_params_schema  # noqa: E402


def _base_df(n: int = 64) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "value": np.linspace(1.0, 10.0, n) + np.sin(np.arange(n) / 5.0),
            "feat_1": np.linspace(0.0, 1.0, n),
        }
    )


def test_arima_training_params_match_unified_schema():
    from training.adaptor.arima_adaptor import train_arima_model_7tuple

    cfg = {
        "default": {"time_col": "date", "value_col": "value"},
        "artifacts": {},
        "model_config": {"ARIMA": {"fixed_order": [1, 0, 0], "use_seasonal": False}},
    }
    *_, params = train_arima_model_7tuple(_base_df(), cfg)
    validate_training_params_schema(params)


def test_randomforest_training_params_match_unified_schema(tmp_path: Path):
    pytest.importorskip("optuna", reason="TEST_MATRIX_OPTIONAL_DEP_MISSING: optuna")
    from training.train_random_forest import train_random_forest_model

    cfg = {
        "default": {"time_col": "date", "value_col": "value"},
        "model_config": {"RandomForest": {"n_lags": 3}},
        "artifacts": {
            "feature_cols_path": str(tmp_path / "rf_features.json"),
            "model_path": str(tmp_path / "rf.pkl"),
        },
    }
    *_, params = train_random_forest_model(_base_df(120), cfg)
    validate_training_params_schema(params)


def test_xgboost_training_params_match_unified_schema():
    pytest.importorskip("xgboost", reason="TEST_MATRIX_OPTIONAL_DEP_MISSING: xgboost")
    from training.adaptor.xgboost_adaptor import train_xgboost_model_7tuple

    cfg = {
        "default": {"time_col": "date", "value_col": "value"},
        "model_config": {"XGBoost": {"n_estimators": 10, "max_depth": 2, "n_jobs": 1}},
        "data": {"all_feature_cols": ["value", "feat_1"]},
        "artifacts": {},
    }
    *_, params = train_xgboost_model_7tuple(_base_df(), cfg)
    validate_training_params_schema(params)


def test_lstm_training_params_match_unified_schema():
    pytest.importorskip("torch", reason="TEST_MATRIX_OPTIONAL_DEP_MISSING: torch")
    from training.adaptor.LSTM_adaptor import train_lstm_model_7tuple

    cfg = {
        "default": {"time_col": "date", "value_col": "value"},
        "model_config": {"LSTM": {"hidden_dim": 8, "num_layers": 1, "n_epochs": 1, "seq_len": 6}},
        "data": {"all_feature_cols": ["value", "feat_1"]},
        "artifacts": {},
    }
    *_, params = train_lstm_model_7tuple(_base_df(70), cfg)
    validate_training_params_schema(params)


def test_prophet_training_params_match_unified_schema():
    pytest.importorskip("prophet", reason="TEST_MATRIX_OPTIONAL_DEP_MISSING: prophet")
    from training.adaptor.Prophet_adaptor import train_prophet_model_7tuple

    cfg = {
        "default": {"time_col": "date", "value_col": "value"},
        "artifacts": {},
        "yearly_seasonality": False,
        "weekly_seasonality": False,
        "daily_seasonality": False,
    }
    *_, params = train_prophet_model_7tuple(_base_df(), cfg)
    validate_training_params_schema(params)


def test_informer_training_params_match_unified_schema(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("torch", reason="TEST_MATRIX_OPTIONAL_DEP_MISSING: torch")
    from training.adaptor.informer_adaptor import train_informer_model_7tuple

    def _fake_train_informer_model(_config):
        result_df = pd.DataFrame(
            {
                "phase": ["val", "val", "test", "test"],
                "y_true": [1.0, 2.0, 3.0, 4.0],
                "yhat": [1.1, 2.1, 3.1, 4.1],
            }
        )
        return object(), result_df

    monkeypatch.setattr("models.informer.train.train_informer_model", _fake_train_informer_model)
    cfg = {"default": {"time_col": "date", "value_col": "value"}, "data": {"split": {"train_len": 36, "val_len": 12, "test_len": 12}}}
    *_, params = train_informer_model_7tuple(_base_df(60), cfg)
    validate_training_params_schema(params)
