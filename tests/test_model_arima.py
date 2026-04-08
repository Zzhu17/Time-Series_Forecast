import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.adaptor.arima_adaptor import train_arima_model_7tuple  # noqa: E402


def _df(n: int = 48) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "value": np.linspace(2.0, 9.0, n) + np.sin(np.arange(n) / 5.0),
        }
    )


def _cfg() -> dict:
    return {
        "default": {"time_col": "date", "value_col": "value"},
        "value_col": "value",
        "split": {"train": 0.6, "val": 0.2, "test": 0.2},
        "model_config": {
            "ARIMA": {
                "rolling": {"enabled": False},
                "fixed_order": [1, 0, 0],
                "fixed_seasonal_order": [0, 0, 0, 0],
                "use_seasonal": False,
            }
        },
        "artifacts": {},
    }


def test_arima_minimal_training_returns_7tuple():
    out = train_arima_model_7tuple(_df(), _cfg())
    assert isinstance(out, tuple)
    assert len(out) == 7

    val_true, val_pred, test_true, test_pred, _model, _test_df, params = out
    assert isinstance(val_true, np.ndarray)
    assert isinstance(val_pred, np.ndarray)
    assert isinstance(test_true, np.ndarray)
    assert isinstance(test_pred, np.ndarray)
    assert len(val_true) == len(val_pred)
    assert len(test_true) == len(test_pred)
    assert isinstance(params, dict)
    assert params.get("model") == "arima"
    assert isinstance(params.get("split"), dict)


def test_arima_missing_target_column_raises():
    bad = _df().drop(columns=["value"])
    with pytest.raises((KeyError, ValueError)):
        train_arima_model_7tuple(bad, _cfg())


def test_arima_prediction_structure_stable():
    val_true, val_pred, test_true, test_pred, *_ = train_arima_model_7tuple(_df(), _cfg())
    assert len(val_true) == len(val_pred), "val y_true/yhat length mismatch"
    assert len(test_true) == len(test_pred), "test y_true/yhat length mismatch"
