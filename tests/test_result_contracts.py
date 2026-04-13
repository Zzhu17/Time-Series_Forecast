from services.result_contracts import ensure_run_result, make_prediction_result  # noqa: E402


def test_ensure_run_result_normalizes_shape():
    payload = ensure_run_result({"status": "ok", "data": None, "metrics": None, "artifacts": None})
    assert payload["status"] == "ok"
    assert payload["data"] == {}
    assert payload["metrics"] == {}
    assert payload["artifacts"] == {}


def test_make_prediction_result_builds_standard_payload():
    payload = make_prediction_result(
        predictions=[1.0, 2.0],
        degraded=True,
        degraded_reason="fallback",
        fallback_model="baseline",
        used_model="xgboost->baseline",
        reason="fallback",
        contract_report={"feature_cols": ["value"]},
    )
    assert payload["status"] == "ok"
    assert payload["predictions"] == [1.0, 2.0]
    assert payload["degraded"] is True
    assert payload["fallback_model"] == "baseline"
    assert payload["contract_report"]["feature_cols"] == ["value"]
