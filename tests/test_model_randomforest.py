from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import assert_7tuple_contract

pytest.importorskip("optuna", reason="TEST_MATRIX_OPTIONAL_DEP_MISSING: optuna")


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
    assert_7tuple_contract(out, "randomforest")

    *_, params = out
    assert params.get("model") == "randomforest"
    assert isinstance(params.get("split"), dict)



def test_randomforest_missing_target_column_raises(tmp_path: Path):
    bad = _df().drop(columns=["value"])
    with pytest.raises((KeyError, ValueError)):
        train_random_forest_model(bad, _cfg(tmp_path))


