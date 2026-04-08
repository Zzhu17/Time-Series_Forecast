import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.predict_helpers import prepare_feature_frame
from utils.feature_pipeline import align_predict_df


def _base_df(n: int = 24) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "value": np.linspace(10.0, 20.0, n),
            "feat1": np.linspace(0.0, 1.0, n),
        }
    )


def test_align_predict_df_fail_fast_on_contract_mismatch():
    df = _base_df().drop(columns=["feat1"])
    contract = {
        "feature_cols": ["value", "feat1"],
        "feature_order": ["value", "feat1"],
        "required_core_cols": ["value"],
        "repairable_core_cols": [],
        "optional_cols": ["feat1"],
        "preprocess_version": "contract-v1",
    }

    with pytest.raises(ValueError, match="feature contract validation failed"):
        align_predict_df(df, contract=contract, time_col="date", value_col="value", tail_rows=1)


def test_align_predict_df_allow_degrade_drops_mismatch_feature():
    df = _base_df().drop(columns=["feat1"])
    contract = {
        "feature_cols": ["value", "feat1"],
        "feature_order": ["value", "feat1"],
        "required_core_cols": ["value"],
        "repairable_core_cols": [],
        "optional_cols": ["feat1"],
        "preprocess_version": "contract-v1",
    }

    _aligned, report, usable = align_predict_df(
        df,
        contract=contract,
        time_col="date",
        value_col="value",
        tail_rows=1,
        allow_degrade=True,
    )
    assert "feat1" in report["contract_diff"]["missing_columns"]
    assert usable == ["value"]


def test_prepare_feature_frame_allow_degrade_switch():
    df = _base_df()[["date", "value"]].copy()
    with pytest.raises(ValueError, match="feature contract validation failed"):
        prepare_feature_frame(
            df,
            feature_cols=["value", "feat1"],
            time_col="date",
            value_col="value",
            tail_rows=1,
        )

    out = prepare_feature_frame(
        df,
        feature_cols=["value", "feat1"],
        time_col="date",
        value_col="value",
        tail_rows=1,
        allow_degrade=True,
    )
    assert "value" in out.columns
