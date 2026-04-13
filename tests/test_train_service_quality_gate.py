import pandas as pd


from services import train_service  # noqa: E402


def _mock_normalized():
    df = pd.DataFrame(
        [
            {"date": "2024-01-01", "value": 10.0},
            {"date": "2024-01-02", "value": 11.0},
            {"date": "2024-01-03", "value": 12.0},
        ]
    )
    normalized = {
        "model_name": "informer",
        "model_alias": "gate-test-model",
        "time_col": "date",
        "value_col": "value",
        "feature_cols": ["value"],
        "allow_degrade": False,
    }
    return df, normalized, {}


def _gate_cfg():
    return {
        "quality_gate": {
            "enabled": True,
            "profile": "staging",
            "templates": {
                "staging": {
                    "required_metrics": {"MAPE": 0.2, "RMSE": 5.0, "MAE": 5.0},
                    "baseline": {
                        "enabled": True,
                        "max_degradation": {"MAPE": 0.1, "RMSE": 0.1, "MAE": 0.1},
                    },
                    "missing_metric_policy": "fail",
                },
                "dev": {
                    "required_metrics": {"MAPE": 0.5, "RMSE": 10.0, "MAE": 10.0},
                    "baseline": {"enabled": False},
                    "missing_metric_policy": "pass",
                },
            },
            "suggestion": {"enabled": True, "recent_runs": 5, "quantile": 0.8, "metrics": ["MAPE", "RMSE", "MAE"]},
        }
    }


def test_quality_gate_pass_registers_candidate(monkeypatch):
    monkeypatch.setattr(train_service, "normalize_training_payload", lambda payload, auto_select_features=True: _mock_normalized())
    monkeypatch.setattr(train_service, "load_yaml_config", lambda: _gate_cfg())

    class _Pipeline:
        @staticmethod
        def run_pipeline_and_update_state(**kwargs):
            return {
                "metrics": {
                    "test": {"MAPE": 0.10, "RMSE": 1.0, "MAE": 0.8},
                    "baseline": {"MAPE": 0.12, "RMSE": 1.1, "MAE": 0.85},
                },
                "artifacts": {"model_path": "artifacts/runs/good/model.bin"},
                "data": {},
            }

    monkeypatch.setattr(train_service, "load_pipeline_module", lambda: _Pipeline())
    monkeypatch.setattr(train_service, "list_models", lambda limit=5: [])
    captured = {}

    def _mock_register_model(**kwargs):
        captured.update(kwargs)
        return {"id": "rec-1", **kwargs}

    monkeypatch.setattr(train_service, "register_model", _mock_register_model)

    result = train_service.run_training_task(payload={"rows": []}, task_id="task-gate-pass", emit_metrics=False)
    assert result["gate_failed_reason"] is None
    assert isinstance(result["gate_decision_trace"], list)
    assert result["model_record"]["stage"] == "candidate"
    assert captured["stage"] == "candidate"


def test_quality_gate_fail_archives_model_and_returns_reason(monkeypatch):
    monkeypatch.setattr(train_service, "normalize_training_payload", lambda payload, auto_select_features=True: _mock_normalized())
    monkeypatch.setattr(train_service, "load_yaml_config", lambda: _gate_cfg())

    class _Pipeline:
        @staticmethod
        def run_pipeline_and_update_state(**kwargs):
            return {
                "metrics": {
                    "test": {"MAPE": 0.45, "RMSE": 10.0, "MAE": 8.0},
                    "baseline": {"MAPE": 0.20, "RMSE": 5.0, "MAE": 4.0},
                },
                "artifacts": {"model_path": "artifacts/runs/bad/model.bin"},
                "data": {},
            }

    monkeypatch.setattr(train_service, "load_pipeline_module", lambda: _Pipeline())
    monkeypatch.setattr(train_service, "list_models", lambda limit=5: [])
    captured = {}

    def _mock_register_model(**kwargs):
        captured.update(kwargs)
        return {"id": "rec-2", **kwargs}

    monkeypatch.setattr(train_service, "register_model", _mock_register_model)

    result = train_service.run_training_task(payload={"rows": []}, task_id="task-gate-fail", emit_metrics=False)
    assert isinstance(result["gate_failed_reason"], str)
    assert "MAPE" in result["gate_failed_reason"]
    assert result["model_record"]["stage"] == "archived"
    assert captured["stage"] == "archived"


def test_quality_gate_missing_metrics_respects_policy(monkeypatch):
    monkeypatch.setattr(train_service, "normalize_training_payload", lambda payload, auto_select_features=True: _mock_normalized())
    cfg = _gate_cfg()
    cfg["quality_gate"]["profile"] = "dev"
    monkeypatch.setattr(train_service, "load_yaml_config", lambda: cfg)
    monkeypatch.setattr(train_service, "list_models", lambda limit=5: [])

    class _Pipeline:
        @staticmethod
        def run_pipeline_and_update_state(**kwargs):
            return {
                "metrics": {"test": {"MAPE": 0.15}},
                "artifacts": {"model_path": "artifacts/runs/missing/model.bin"},
                "data": {},
            }

    monkeypatch.setattr(train_service, "load_pipeline_module", lambda: _Pipeline())
    monkeypatch.setattr(train_service, "register_model", lambda **kwargs: {"id": "rec-3", **kwargs})
    result = train_service.run_training_task(payload={"rows": []}, task_id="task-gate-missing", emit_metrics=False)
    assert result["gate_failed_reason"] is None
    assert result["model_record"]["stage"] == "candidate"


def test_quality_gate_suggestion_uses_recent_quantile(monkeypatch):
    cfg = _gate_cfg()
    monkeypatch.setattr(train_service, "load_yaml_config", lambda: cfg)
    monkeypatch.setattr(
        train_service,
        "list_models",
        lambda limit=5: [
            {"metrics": {"test": {"MAPE": 0.10, "RMSE": 1.0, "MAE": 0.9}}},
            {"metrics": {"test": {"MAPE": 0.20, "RMSE": 2.0, "MAE": 1.4}}},
            {"metrics": {"test": {"MAPE": 0.40, "RMSE": 4.0, "MAE": 3.0}}},
        ],
    )
    out = train_service._quality_gate_threshold_suggestion({"quality_gate": cfg["quality_gate"]})
    assert isinstance(out, dict)
    assert out["sample_size"]["MAPE"] == 3
    assert 0.20 <= out["suggested_required_metrics"]["MAPE"] <= 0.40
