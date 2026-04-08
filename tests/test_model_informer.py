import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.adaptor.informer_adaptor import train_informer_model_7tuple  # noqa: E402


def _df(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "value": np.linspace(1.0, 15.0, n),
        }
    )


def _cfg() -> dict:
    return {
        "default": {"time_col": "date", "value_col": "value"},
        "data": {"split": {"train_len": 36, "val_len": 12, "test_len": 12}},
    }


def test_informer_minimal_training_returns_7tuple(monkeypatch: pytest.MonkeyPatch):
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

    out = train_informer_model_7tuple(_df(), _cfg())
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
    assert params.get("model_name") == "informer"


def test_informer_missing_prediction_columns_degrades(monkeypatch: pytest.MonkeyPatch):
    def _fake_bad_train(_config):
        return object(), pd.DataFrame({"phase": ["val"], "foo": [1]})

    monkeypatch.setattr("models.informer.train.train_informer_model", _fake_bad_train)

    val_true, val_pred, test_true, test_pred, *_ = train_informer_model_7tuple(_df(), _cfg())
    assert val_true is None and val_pred is None
    assert test_true is None and test_pred is None


def test_informer_prediction_structure_stable(monkeypatch: pytest.MonkeyPatch):
    def _fake_train_informer_model(_config):
        result_df = pd.DataFrame(
            {
                "phase": ["val", "test", "test"],
                "y_true": [1.0, 2.0, 3.0],
                "yhat": [1.1, 2.2, 3.3],
            }
        )
        return object(), result_df

    monkeypatch.setattr("models.informer.train.train_informer_model", _fake_train_informer_model)

    val_true, val_pred, test_true, test_pred, *_ = train_informer_model_7tuple(_df(), _cfg())
    assert len(val_true) == len(val_pred), "val y_true/yhat length mismatch"
    assert len(test_true) == len(test_pred), "test y_true/yhat length mismatch"
