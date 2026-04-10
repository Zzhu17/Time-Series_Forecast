import sys
from pathlib import Path

import pytest

pytest.importorskip("httpx", reason="TEST_MATRIX_OPTIONAL_DEP_MISSING: httpx")
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import app  # noqa: E402

client = TestClient(app)


def test_models_register_and_list():
    payload = {
        "name": "baseline",
        "version": "test",
        "stage": "candidate",
        "params": {"note": "test"},
        "metrics": {"nrmse": 1.0},
        "artifacts": {"model_path": "artifacts/test.bin"},
    }
    resp = client.post("/models/register", json=payload)
    assert resp.status_code == 200
    rec = resp.json()
    assert rec.get("id")

    list_resp = client.get("/models/registry?limit=10&offset=0")
    assert list_resp.status_code == 200
    ids = [item.get("id") for item in list_resp.json()]
    assert rec.get("id") in ids


def test_tasks_list_ok():
    resp = client.get("/tasks")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_health_ready_ok_or_degraded():
    resp = client.get("/health/ready")
    assert resp.status_code in (200, 503)
    assert resp.json().get("status") in ("ok", "error")
