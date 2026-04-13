import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.request_utils import resolve_feature_cols  # noqa: E402


def test_resolve_feature_cols_keeps_explicit_feature_order():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "value": [1.0, 2.0],
            "feat_a": [3.0, 4.0],
        }
    )

    out = resolve_feature_cols(
        df,
        feature_cols=["value", "feat_a"],
        time_col="date",
        value_col="value",
        auto_select_features=True,
    )

    assert out == ["value", "feat_a"]


def test_resolve_feature_cols_auto_selects_when_requested():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "value": [1.0, 2.0, 3.0, 4.0],
            "feat_a": [1.0, 2.0, 3.0, 4.0],
        }
    )

    out = resolve_feature_cols(
        df,
        feature_cols=None,
        time_col="date",
        value_col="value",
        auto_select_features=True,
    )

    assert out == ["feat_a"]
