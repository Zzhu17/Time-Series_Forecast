import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.feature_pipeline import build_train_features, save_feature_contract_if_any  # noqa: E402


def test_feature_contract_saved(tmp_path: Path):
    n = 50
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "value": np.linspace(0, 1, n),
            "feat1": np.random.randn(n),
            "feat2": np.random.randn(n),
        }
    )

    config = {
        "model_config": {
            "Informer": {
                "feature_selection": {
                    "missing_rate_threshold": 0.4,
                    "low_variance_threshold": 0.0,
                    "redundant_corr_threshold": 0.99,
                    "max_features": None,
                }
            }
        }
    }

    cleaned, feat_cols, report = build_train_features(
        df,
        time_col="date",
        value_col="value",
        candidate_cols=["value", "feat1", "feat2"],
        config=config,
    )

    assert "feature_contract" in report and isinstance(report["feature_contract"], dict)
    assert feat_cols[0] == "value"
    assert not cleaned[feat_cols].isna().any().any()

    artifacts = {"feature_cols_path": str(tmp_path / "feature_cols.json")}
    save_feature_contract_if_any(report, artifacts)

    saved = json.loads(Path(artifacts["feature_cols_path"]).read_text(encoding="utf-8"))
    assert "feature_cols" in saved and "value" in saved["feature_cols"]
    assert saved.get("feature_order") == saved.get("feature_cols")
    assert isinstance(saved.get("preprocess_version"), str) and saved.get("preprocess_version")
