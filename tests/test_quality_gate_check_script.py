import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality_gate_check import build_report


def _rec(mape, rmse, mae, gate_passed=True):
    return {
        "id": f"id-{mape}",
        "name": "model-a",
        "version": "v1",
        "created_at": "2026-01-01T00:00:00",
        "params": {"gate_passed": gate_passed},
        "metrics": {"test": {"MAPE": mape, "RMSE": rmse, "MAE": mae}},
    }


def test_build_report_marks_trend_warning_when_consecutive_degradation_reaches_threshold():
    # load order: newest -> oldest (same as DB query)
    recent = [
        _rec(0.30, 3.0, 2.0),
        _rec(0.20, 2.0, 1.5),
        _rec(0.10, 1.0, 1.0),
        _rec(0.05, 0.8, 0.9),
    ]

    report = build_report(recent, history_size=4, streak_threshold=3, thresholds={"MAPE": None, "RMSE": None, "MAE": None})

    assert report["gate_passed"] is True
    assert report["trend_status"] == "warning_degrading"
    assert report["risk_level"] == "medium"
    assert report["metrics"]["MAPE"]["consecutive_degradation"] == 3


def test_build_report_fails_gate_when_latest_gate_passed_false():
    recent = [
        _rec(0.30, 3.0, 2.0, gate_passed=False),
        _rec(0.20, 2.0, 1.5, gate_passed=True),
    ]

    report = build_report(recent, history_size=2, streak_threshold=3, thresholds={"MAPE": 0.5, "RMSE": 4.0, "MAE": 3.0})

    assert report["gate_passed"] is False
    assert report["risk_level"] == "high"
