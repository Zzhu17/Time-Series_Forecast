import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.adaptor.common import extract_dense_predictions, extract_split_predictions  # noqa: E402


def test_extract_dense_predictions_trims_to_common_length():
    df = pd.DataFrame({"y_true": [1.0, 2.0], "yhat": [1.1, 2.1]})
    y_true, y_pred = extract_dense_predictions(df)
    assert list(y_true) == [1.0, 2.0]
    assert list(y_pred) == [1.1, 2.1]


def test_extract_split_predictions_uses_phase_when_available():
    df = pd.DataFrame(
        {
            "phase": ["val", "val", "test"],
            "y_true": [1.0, 2.0, 3.0],
            "yhat": [1.1, 2.1, 3.1],
        }
    )
    val_true, val_pred, test_true, test_pred = extract_split_predictions(df)
    assert list(val_true) == [1.0, 2.0]
    assert list(test_true) == [3.0]
    assert list(test_pred) == [3.1]


def test_extract_split_predictions_uses_explicit_split_before_fallback_ratio():
    df = pd.DataFrame(
        {
            "y_true": [1.0, 2.0, 3.0, 4.0],
            "yhat": [1.1, 2.1, 3.1, 4.1],
        }
    )
    val_true, val_pred, test_true, test_pred = extract_split_predictions(df, split={"val_len": 1, "test_len": 1})
    assert list(val_true) == [3.0]
    assert list(val_pred) == [3.1]
    assert list(test_true) == [4.0]
    assert list(test_pred) == [4.1]
