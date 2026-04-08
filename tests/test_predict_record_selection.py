import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.predict_service import _find_model_record  # noqa: E402


def test_find_model_record_prefers_candidate_then_production(monkeypatch):
    calls = []

    def _fake_latest(name, stage=None):
        calls.append((name, stage))
        if stage == "candidate":
            return None
        if stage == "production":
            return {"id": "prod-1", "name": name, "stage": "production"}
        return {"id": "archived-1", "name": name, "stage": "archived"}

    monkeypatch.setattr("services.predict_service.latest_model_for_name", _fake_latest)

    rec = _find_model_record(model_name="randomforest", model_id=None, model_version=None)
    assert rec is not None
    assert rec["stage"] == "production"
    assert calls == [("randomforest", "candidate"), ("randomforest", "production")]


def test_find_model_record_does_not_fallback_to_archived(monkeypatch):
    monkeypatch.setattr("services.predict_service.latest_model_for_name", lambda *args, **kwargs: None)
    rec = _find_model_record(model_name="xgboost", model_id=None, model_version=None)
    assert rec is None
