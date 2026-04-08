import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.adaptor.xgboost_adaptor import train_xgboost_model_7tuple  # noqa: E402


def _df(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "value": np.linspace(1.0, 10.0, n),
            "feat_1": np.linspace(0.0, 1.0, n),
        }
    )


def _cfg() -> dict:
    return {
        "default": {"time_col": "date", "value_col": "value"},
        "model_config": {"XGBoost": {"n_estimators": 10, "max_depth": 2, "n_jobs": 1}},
        "data": {"all_feature_cols": ["value", "feat_1"]},
        "artifacts": {},
    }


def test_xgboost_minimal_training_returns_7tuple():
    out = train_xgboost_model_7tuple(_df(), _cfg())
    assert isinstance(out, tuple)
    assert len(out) == 7

    val_true, val_pred, test_true, test_pred, model, test_df, params = out
    assert isinstance(val_true, np.ndarray)
    assert isinstance(val_pred, np.ndarray)
    assert isinstance(test_true, np.ndarray)
    assert isinstance(test_pred, np.ndarray)
    assert len(val_true) == len(val_pred)
    assert len(test_true) == len(test_pred)
    assert model is not None
    assert test_df is None
    assert isinstance(params, dict)
    assert params.get("model") == "xgboost"
    assert isinstance(params.get("split"), dict)


def test_xgboost_missing_target_column_raises():
    bad = _df().drop(columns=["value"])
    with pytest.raises((KeyError, ValueError)):
        train_xgboost_model_7tuple(bad, _cfg())


def test_xgboost_prediction_structure_stable():
    val_true, val_pred, test_true, test_pred, *_ = train_xgboost_model_7tuple(_df(), _cfg())
    assert len(val_true) == len(val_pred), "val y_true/yhat length mismatch"
    assert len(test_true) == len(test_pred), "test y_true/yhat length mismatch"
