import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import app  # noqa: E402

client = TestClient(app)


def _csv_bytes(text: str) -> io.BytesIO:
    return io.BytesIO(text.encode("utf-8"))


def test_train_file_sync_missing_required_columns():
    csv_data = "date,foo\n2024-01-01,1\n"
    files = {"file": ("train.csv", _csv_bytes(csv_data), "text/csv")}
    data = {
        "model_name": "informer",
        "time_col": "date",
        "value_col": "value",
        "horizon": "1",
    }
    resp = client.post("/train_file_sync", files=files, data=data)
    assert resp.status_code == 400
    assert "CSV missing columns" in resp.json().get("detail", "")


def test_predict_online_file_missing_required_columns():
    csv_data = "date,foo\n2024-01-01,1\n"
    files = {"file": ("online.csv", _csv_bytes(csv_data), "text/csv")}
    data = {
        "model_name": "informer",
        "time_col": "date",
        "value_col": "value",
        "horizon_days": "1",
        "step_mode": "Block step (= horizon)",
    }
    resp = client.post("/predict_online_file", files=files, data=data)
    assert resp.status_code == 400
    assert "CSV missing columns" in resp.json().get("detail", "")
