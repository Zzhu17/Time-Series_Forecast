import numpy as np
import pandas as pd
import pytest

from conftest import assert_7tuple_contract

pytest.importorskip("torch", reason="TEST_MATRIX_OPTIONAL_DEP_MISSING: torch")


from training.adaptor.LSTM_adaptor import train_lstm_model_7tuple  # noqa: E402


def _df(n: int = 70) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "value": np.linspace(1.0, 20.0, n) + np.cos(np.arange(n) / 8.0),
            "feat_1": np.linspace(0.0, 1.0, n),
        }
    )


def _cfg() -> dict:
    return {
        "default": {"time_col": "date", "value_col": "value", "dtype": "float32"},
        "model_config": {
            "LSTM": {
                "hidden_dim": 8,
                "num_layers": 1,
                "learning_rate": 1e-3,
                "n_epochs": 1,
                "seq_len": 6,
                "batch_size": 8,
                "patience": 1,
                "feature_cols": ["value", "feat_1"],
            }
        },
        "data": {"all_feature_cols": ["value", "feat_1"]},
        "artifacts": {},
    }


def test_lstm_minimal_training_returns_7tuple():
    out = train_lstm_model_7tuple(_df(), _cfg())
    assert_7tuple_contract(out, "lstm")

    *_, params = out
    assert params.get("model") == "lstm"
    assert isinstance(params.get("split"), dict)



def test_lstm_missing_target_column_raises():
    bad = _df().drop(columns=["value"])
    with pytest.raises((KeyError, ValueError)):
        train_lstm_model_7tuple(bad, _cfg())


