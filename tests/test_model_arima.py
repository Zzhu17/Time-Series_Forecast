import numpy as np
import pandas as pd
import pytest

from conftest import assert_7tuple_contract


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
    assert_7tuple_contract(out, "arima")

    *_, params = out
    assert params.get("model") == "arima"
    assert isinstance(params.get("split"), dict)



def test_arima_missing_target_column_raises():
    bad = _df().drop(columns=["value"])
    with pytest.raises((KeyError, ValueError)):
        train_arima_model_7tuple(bad, _cfg())


