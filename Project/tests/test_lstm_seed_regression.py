import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.train_lstm import train_lstm_model


def _make_df(n: int = 240) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="D")
    x = np.linspace(0.0, 8.0 * np.pi, num=n)
    y = np.sin(x) + 0.05 * np.cos(2.0 * x)
    return pd.DataFrame({"date": ts, "value": y.astype(np.float32)})


def _run_once(seed: int):
    df = _make_df()
    cfg = {
        "seed": seed,
        "default": {"time_col": "date", "value_col": "value", "dtype": "float32"},
        "dtype": "float32",
        "device": "cpu",
        "training": {"seed": seed, "smoke": {"enabled": True, "batch_size": 8, "epochs": 2}},
        "model_config": {
            "LSTM": {
                "hidden_dim": 16,
                "num_layers": 1,
                "learning_rate": 1e-3,
                "n_epochs": 2,
                "seq_len": 12,
                "batch_size": 8,
                "dropout": 0.0,
            }
        },
        "data": {},
        "artifacts": {},
    }
    val_true, val_pred, *_ = train_lstm_model(df, cfg)
    rmse = float(np.sqrt(np.mean((np.asarray(val_pred) - np.asarray(val_true)) ** 2)))
    return rmse, cfg.get("data", {}).get("train_run_metadata", {})


def test_lstm_seed_regression_within_tolerance():
    rmse_1, meta_1 = _run_once(seed=123)
    rmse_2, meta_2 = _run_once(seed=123)

    assert meta_1.get("seed") == 123
    assert meta_1.get("device") == "cpu"
    assert meta_1.get("dtype") == "float32"
    assert abs(rmse_1 - rmse_2) <= 1e-6
    assert abs(rmse_1 - rmse_2) <= 1e-3
    assert meta_1 == meta_2
