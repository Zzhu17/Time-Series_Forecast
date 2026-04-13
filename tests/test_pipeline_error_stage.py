import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services import pipeline  # noqa: E402


def test_run_pipeline_preserves_inferred_error_stage(monkeypatch):
    def _raise(_config):
        raise ValueError("feature missing column during preprocess")

    monkeypatch.setattr(pipeline, "run_train_predict_pipeline", _raise)

    df = pd.DataFrame({"date": ["2024-01-01"], "value": [1.0]})
    out = pipeline.run_pipeline_and_update_state(
        df=df,
        config={"artifacts": {}, "data": {}},
        feature_cols=["value"],
        uploaded_name="sample.csv",
        model_name="informer",
        time_col="date",
        value_col="value",
        allow_degrade=False,
    )

    assert out["status"] == "error"
    assert out["error_stage"] == "data_prep"
    assert out["action"] == "fail"
