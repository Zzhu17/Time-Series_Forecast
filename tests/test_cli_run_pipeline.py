from pathlib import Path
import json

import pandas as pd


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
