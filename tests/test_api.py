import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"


def test_baseline_predict():
    payload = {
        "model_name": "baseline",
        "time_col": "date",
        "value_col": "value",
        "horizon": 3,
        "rows": [
            {"date": "2024-01-01", "value": 1.0},
            {"date": "2024-01-02", "value": 2.0},
        ],
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["predictions"] == [2.0, 2.0, 2.0]
