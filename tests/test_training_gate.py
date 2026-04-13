from pathlib import Path
import json

import pandas as pd


from services.train_service import run_training_task  # noqa: E402


def _normalized_payload(model_name: str = "randomforest") -> tuple[pd.DataFrame, dict, dict]:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    normalized = {
        "model_name": model_name,
        "time_col": "date",
        "value_col": "value",
        "feature_cols": ["x1"],
        "allow_degrade": False,
    }
    return df, normalized, {"ok": True}


def test_run_training_task_registers_candidate_when_gate_passes(monkeypatch):
    monkeypatch.setattr("services.train_service.normalize_training_payload", lambda *args, **kwargs: _normalized_payload())

    class _PipelineModule:
        @staticmethod
        def run_pipeline_and_update_state(**kwargs):
            return {
                "metrics": {"test": {"nrmse": 0.2}},
                "artifacts": {"randomforest_params": {"n_estimators": 10}},
                "data": {"degraded": False},
            }

    monkeypatch.setattr("services.train_service.load_pipeline_module", lambda: _PipelineModule())
    monkeypatch.setattr("services.train_service.list_models", lambda limit=20: [])
    monkeypatch.setattr("services.train_service._write_latest_report", lambda *args, **kwargs: None)
    monkeypatch.setattr("services.train_service._purge_old_runs", lambda *args, **kwargs: None)

    captured = {}

    def _fake_register_model(**kwargs):
        captured.update(kwargs)
        return {"id": "m1", **kwargs}

    monkeypatch.setattr("services.train_service.register_model", _fake_register_model)

    task_id = "gate-pass-case"
    out = run_training_task({"rows": []}, task_id=task_id, emit_metrics=False)
    assert captured["stage"] == "candidate"
    assert out["model_record"]["stage"] == "candidate"
    training_params_path = captured["artifacts"].get("training_params_path")
    assert isinstance(training_params_path, str)
    payload = json.loads(Path(training_params_path).read_text(encoding="utf-8"))
    assert payload.get("n_estimators") == 10


def test_run_training_task_registers_archived_when_gate_fails(monkeypatch):
    monkeypatch.setattr("services.train_service.normalize_training_payload", lambda *args, **kwargs: _normalized_payload("xgboost"))

    class _PipelineModule:
        @staticmethod
        def run_pipeline_and_update_state(**kwargs):
            return {
                "metrics": {"test": {"nrmse": 1.8}},
                "artifacts": {"xgboost_params": {"max_depth": 4}},
                "data": {"degraded": True, "degraded_reason": "fallback"},
            }

    monkeypatch.setattr("services.train_service.load_pipeline_module", lambda: _PipelineModule())
    monkeypatch.setattr("services.train_service.list_models", lambda limit=20: [])
    monkeypatch.setattr("services.train_service._write_latest_report", lambda *args, **kwargs: None)
    monkeypatch.setattr("services.train_service._purge_old_runs", lambda *args, **kwargs: None)

    captured = {}

    def _fake_register_model(**kwargs):
        captured.update(kwargs)
        return {"id": "m2", **kwargs}

    monkeypatch.setattr("services.train_service.register_model", _fake_register_model)

    out = run_training_task({"rows": []}, task_id="gate-fail-case", emit_metrics=False)
    assert captured["stage"] == "archived"
    assert out["model_record"]["stage"] == "archived"
    gate = captured["params"].get("quality_gate")
    assert isinstance(gate, dict)
    assert gate.get("passed") is False


def test_run_training_task_falls_back_to_summary_training_params(monkeypatch):
    monkeypatch.setattr("services.train_service.normalize_training_payload", lambda *args, **kwargs: _normalized_payload("prophet"))

    class _PipelineModule:
        @staticmethod
        def run_pipeline_and_update_state(**kwargs):
            return {
                "metrics": {"test": {"nrmse": 0.2}},
                "artifacts": {},
                "data": {"degraded": False},
            }

    monkeypatch.setattr("services.train_service.load_pipeline_module", lambda: _PipelineModule())
    monkeypatch.setattr("services.train_service.list_models", lambda limit=20: [])
    monkeypatch.setattr("services.train_service._write_latest_report", lambda *args, **kwargs: None)
    monkeypatch.setattr("services.train_service._purge_old_runs", lambda *args, **kwargs: None)

    captured = {}

    def _fake_register_model(**kwargs):
        captured.update(kwargs)
        return {"id": "m3", **kwargs}

    monkeypatch.setattr("services.train_service.register_model", _fake_register_model)

    out = run_training_task({"rows": []}, task_id="gate-summary-case", emit_metrics=False)
    assert out["training_params"]["model_name"] == "prophet"
    training_params_path = captured["artifacts"].get("training_params_path")
    payload = json.loads(Path(training_params_path).read_text(encoding="utf-8"))
    assert payload.get("model_name") == "prophet"
