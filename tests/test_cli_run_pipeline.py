import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.run_pipeline import _write_json  # noqa: E402


def test_write_json_serializes_pipeline_objects(tmp_path: Path):
    payload = {
        "run_id": "r1",
        "status": "ok",
        "metrics": {"test": {"rmse": 1.23}},
        "data": {"val_dense": pd.DataFrame({"y_true": [1.0], "yhat": [1.1]})},
        "artifacts": {"feature_cols": ["value"]},
    }

    out = tmp_path / "cli.json"
    _write_json(out, payload)

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["run_id"] == "r1"
    assert isinstance(saved["data"]["val_dense"], str)
    assert "DataFrame" in saved["data"]["val_dense"]
