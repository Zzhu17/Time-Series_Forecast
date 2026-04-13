import numpy as np
import pandas as pd
import pytest

from conftest import assert_7tuple_contract

pytest.importorskip("xgboost", reason="TEST_MATRIX_OPTIONAL_DEP_MISSING: xgboost")


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
    assert_7tuple_contract(out, "xgboost")

    *_, params = out
    assert out[5] is None
    assert params.get("model") == "xgboost"
    assert isinstance(params.get("split"), dict)



def test_xgboost_missing_target_column_raises():
    bad = _df().drop(columns=["value"])
    with pytest.raises((KeyError, ValueError)):
        train_xgboost_model_7tuple(bad, _cfg())


