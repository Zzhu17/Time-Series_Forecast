# TEST_MATRIX

Date: 2026-04-08

| Area | Command | Result | Notes |
|---|---|---|---|
| Full pytest (initial baseline) | `pytest -q` | Failed | Collection failed before patch due to missing `httpx` in environment. |
| Model and payload/core tests | `pytest -q tests/test_training_payloads.py tests/test_prediction_payloads.py tests/test_model_informer.py tests/test_model_lstm.py tests/test_model_arima.py tests/test_model_randomforest.py tests/test_model_xgboost.py tests/test_model_template.py` | Passed (with expected skips) | Optional heavy deps tests may skip based on installed packages. |
| API tests in missing-httpx env | `pytest -q tests/test_api.py tests/test_api_contracts.py tests/test_api_integration.py` | Skipped (expected) | Guarded by `pytest.importorskip("httpx")` to avoid collection crash. |
