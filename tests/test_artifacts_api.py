import json

import pytest

pytest.importorskip("httpx", reason="TEST_MATRIX_OPTIONAL_DEP_MISSING: httpx")
from fastapi.testclient import TestClient


from api.app import app  # noqa: E402
from api.routes import artifacts as artifacts_route  # noqa: E402


def test_latest_artifacts_falls_back_to_run_registry(tmp_path, monkeypatch):
    project_dir = tmp_path / "Project"
    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True)
    run_dir = tmp_path / "artifacts" / "runs" / "run-123"
    run_dir.mkdir(parents=True)
    (run_dir / "leaderboard.csv").write_text("model_name,rmse\nbaseline,1.0\n", encoding="utf-8")
    (run_dir / "report.html").write_text("<html>ok</html>", encoding="utf-8")
    registry = [
        {
            "run_id": "run-123",
            "model_name": "baseline",
            "metrics": {"test": {"rmse": 1.0}},
            "artifacts": {
                "run_dir": str(run_dir),
                "model_path": str(run_dir / "model.bin"),
            },
        }
    ]
    (output_dir / "run_registry.json").write_text(json.dumps(registry), encoding="utf-8")

    monkeypatch.setattr(artifacts_route, "_project_dir", lambda: project_dir)
    monkeypatch.setattr(artifacts_route, "_repo_root", lambda: tmp_path)
    client = TestClient(app)

    resp = client.get("/artifacts/latest")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["run_id"] == "run-123"
    assert payload["model_name"] == "baseline"
    assert payload["data"]["leaderboard_path"].endswith("leaderboard.csv")
    assert payload["data"]["report_path"].endswith("report.html")
    assert payload["data"]["leaderboard"][0]["model_name"] == "baseline"


def test_latest_artifacts_falls_back_to_artifacts_dir(tmp_path, monkeypatch):
    project_dir = tmp_path / "Project"
    (project_dir / "output").mkdir(parents=True)
    run_dir = tmp_path / "artifacts" / "runs" / "run-456"
    run_dir.mkdir(parents=True)
    (run_dir / "leaderboard.csv").write_text("model_name,rmse\nbaseline,2.0\n", encoding="utf-8")
    (run_dir / "report.html").write_text("<html>ok</html>", encoding="utf-8")

    monkeypatch.setattr(artifacts_route, "_project_dir", lambda: project_dir)
    monkeypatch.setattr(artifacts_route, "_repo_root", lambda: tmp_path)
    client = TestClient(app)

    resp = client.get("/artifacts/latest")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["run_id"] == "run-456"
    assert payload["data"]["leaderboard_path"].endswith("leaderboard.csv")
    assert payload["data"]["report_path"].endswith("report.html")
