import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT / "Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics import normalize_degrade_reason


def test_normalize_degrade_reason_is_bounded_for_free_form_errors() -> None:
    assert normalize_degrade_reason("xgboost_fallback: ValueError: boom") == "fallback_error"


def test_normalize_degrade_reason_recognizes_known_codes() -> None:
    assert normalize_degrade_reason("model_not_available") == "model_not_available"
    assert normalize_degrade_reason("residual_skipped:missing_features") == "residual_skipped"
    assert normalize_degrade_reason("no feature contract found; using raw columns") == "feature_contract_fallback"
    assert normalize_degrade_reason(None) == "unknown"
