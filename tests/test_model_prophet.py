import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("prophet")

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.adaptor.Prophet_adaptor import train_prophet_model_7tuple  # noqa: E402


def _df(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "value": np.linspace(5.0, 12.0, n) + np.sin(np.arange(n) / 6.0),
        }
    )


def _cfg() -> dict:
    return {
        "default": {"time_col": "date", "value_col": "value"},
        "artifacts": {},
        "yearly_seasonality": False,
        "weekly_seasonality": False,
        "daily_seasonality": False,
    }


def test_prophet_minimal_training_returns_7tuple():
    out = train_prophet_model_7tuple(_df(), _cfg())
    assert isinstance(out, tuple)
    assert len(out) == 7

    val_true, val_pred, test_true, test_pred, model, _test_df, params = out
    assert isinstance(val_true, np.ndarray)
    assert isinstance(val_pred, np.ndarray)
    assert isinstance(test_true, np.ndarray)
    assert isinstance(test_pred, np.ndarray)
    assert len(val_true) == len(val_pred)
    assert len(test_true) == len(test_pred)
    assert model is not None
    assert isinstance(params, dict)
    assert params.get("model_name") == "prophet"


def test_prophet_invalid_input_raises():
    with pytest.raises(ValueError, match="至少需要30行数据"):
        train_prophet_model_7tuple(_df(n=12), _cfg())


def test_prophet_prediction_structure_stable():
    val_true, val_pred, test_true, test_pred, *_ = train_prophet_model_7tuple(_df(), _cfg())
    assert len(val_true) == len(val_pred), "val y_true/yhat length mismatch"
    assert len(test_true) == len(test_pred), "test y_true/yhat length mismatch"
