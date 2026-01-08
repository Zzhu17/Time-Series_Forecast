import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.training_payloads import normalize_training_payload  # noqa: E402


def _base_rows():
    return [
        {"date": "2024-01-01", "value": 1.0},
        {"date": "2024-01-02", "value": 2.0},
        {"date": "2024-01-03", "value": 3.0},
    ]


def test_training_payload_missing_required_columns():
    payload = {
        "model_name": "informer",
        "time_col": "date",
        "value_col": "value",
        "rows": [{"date": "2024-01-01", "x": 1.0}],
    }
    with pytest.raises(ValueError, match="CSV missing columns"):
        normalize_training_payload(payload, auto_select_features=False)


def test_training_payload_missing_feature_cols():
    payload = {
        "model_name": "informer",
        "time_col": "date",
        "value_col": "value",
        "rows": _base_rows(),
        "feature_cols": ["value", "missing_feature"],
    }
    with pytest.raises(ValueError, match="feature_cols missing"):
        normalize_training_payload(payload, auto_select_features=False)


def test_training_payload_recomputable_feature_allowed():
    payload = {
        "model_name": "informer",
        "time_col": "date",
        "value_col": "value",
        "rows": _base_rows(),
        "feature_cols": ["value", "value_lag_1"],
    }
    df, normalized, report = normalize_training_payload(payload, auto_select_features=False)
    assert isinstance(df, pd.DataFrame)
    assert normalized["feature_cols"][0] == "value"
    assert "value_lag_1" in report.get("recomputable_missing_cols", [])
