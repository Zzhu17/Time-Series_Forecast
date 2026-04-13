from pathlib import Path
import json

import numpy as np

BASELINE_PATH = Path(__file__).resolve().parent / "fixtures" / "regression_baseline.json"


def _load_baseline():
    with BASELINE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compute_metrics(n: int = 60):
    x = np.arange(n, dtype=float)
    values = 10 + 0.2 * x + np.sin(x / 3.0)
    y_true = values[1:]
    y_pred = values[:-1]
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    std = float(np.std(y_true))
    nrmse = rmse / std if std > 1e-8 else float("nan")
    return nrmse, mae


def test_regression_metrics_not_degrade():
    baseline = _load_baseline()
    n = int(baseline.get("series", {}).get("length", 60))
    nrmse, mae = _compute_metrics(n=n)

    base_nrmse = float(baseline["metrics"]["nrmse"])
    base_mae = float(baseline["metrics"]["mae"])
    tol = 0.05

    assert nrmse <= base_nrmse * (1 + tol)
    assert mae <= base_mae * (1 + tol)
