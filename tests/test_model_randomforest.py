import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("optuna")

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_random_forest import train_random_forest_model  # noqa: E402


def _df(n: int = 120) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "value": np.linspace(10.0, 30.0, n) + np.sin(np.arange(n) / 4.0),
        }
    )


def _cfg(tmp_path: Path) -> dict:
    return {
        "default": {"time_col": "date", "value_col": "value"},
        "model_config": {"RandomForest": {"n_lags": 5}},
        "artifacts": {
            "feature_cols_path": str(tmp_path / "rf_features.json"),
            "model_path": str(tmp_path / "rf.pkl"),
        },
    }


def test_randomforest_minimal_training_returns_7tuple(tmp_path: Path):
    out = train_random_forest_model(_df(), _cfg(tmp_path))
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


def test_randomforest_missing_target_column_raises(tmp_path: Path):
    bad = _df().drop(columns=["value"])
    with pytest.raises((KeyError, ValueError)):
        train_random_forest_model(bad, _cfg(tmp_path))


def test_randomforest_prediction_structure_stable(tmp_path: Path):
    val_true, val_pred, test_true, test_pred, *_ = train_random_forest_model(_df(), _cfg(tmp_path))
    assert len(val_true) == len(val_pred), "val y_true/yhat length mismatch"
    assert len(test_true) == len(test_pred), "test y_true/yhat length mismatch"
