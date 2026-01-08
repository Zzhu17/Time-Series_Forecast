import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.prediction_payloads import normalize_prediction_payload  # noqa: E402


def test_prediction_payload_missing_required_columns():
    payload = {
        "model_name": "baseline",
        "time_col": "date",
        "value_col": "value",
        "rows": [{"date": "2024-01-01", "x": 1.0}],
    }
    with pytest.raises(ValueError, match="CSV missing columns"):
        normalize_prediction_payload(payload)


def test_prediction_payload_missing_feature_cols():
    payload = {
        "model_name": "baseline",
        "time_col": "date",
        "value_col": "value",
        "rows": [{"date": "2024-01-01", "value": 1.0}],
        "feature_cols": ["value", "missing_feature"],
    }
    with pytest.raises(ValueError, match="feature_cols missing"):
        normalize_prediction_payload(payload)
