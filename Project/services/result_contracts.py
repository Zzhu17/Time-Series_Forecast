from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


RUN_RESULT_DATA_KEYS = (
    "split",
    "val_dense",
    "test_dense",
    "val_long",
    "test_long",
    "baseline_metrics",
    "drift",
    "backtest",
    "backtest_metrics",
    "degraded",
    "degraded_mode",
    "degraded_reason",
    "degraded_error",
    "missing_required_core",
    "dropped_optional_features",
)


def ensure_mapping(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def ensure_run_result(res: Any = None) -> Dict[str, Any]:
    out = dict(res) if isinstance(res, dict) else {}
    out.setdefault("status", "ok")
    out.setdefault("message", None)
    out["metrics"] = ensure_mapping(out.get("metrics"))
    out["data"] = ensure_mapping(out.get("data"))
    out["artifacts"] = ensure_mapping(out.get("artifacts"))
    return out


def backfill_missing(dst: Dict[str, Any], src: Dict[str, Any], keys: Iterable[str]) -> None:
    for key in keys:
        if key not in dst and key in src:
            dst[key] = src.get(key)


def ensure_metric_slot(metrics: Dict[str, Any], name: str) -> Dict[str, Any]:
    slot = metrics.get(name)
    if isinstance(slot, dict):
        return slot
    slot = {}
    metrics[name] = slot
    return slot


def backfill_metric_slot(metrics: Dict[str, Any], name: str, *candidates: Any) -> None:
    if name in metrics:
        return
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            metrics[name] = candidate
            return


def make_prediction_result(
    *,
    predictions,
    used_model: str,
    degraded: bool = False,
    degraded_reason: Optional[str] = None,
    fallback_model: Optional[str] = None,
    contract_report: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "status": "ok",
        "predictions": list(predictions),
        "degraded": bool(degraded),
        "degraded_reason": degraded_reason or reason,
        "fallback_model": fallback_model,
        "used_model": used_model,
        "reason": reason,
        "contract_report": contract_report or {},
    }
