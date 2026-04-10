from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Optional

try:  # pragma: no cover - optional dependency in some environments
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest  # type: ignore
except Exception:  # pragma: no cover
    Counter = None  # type: ignore
    Histogram = None  # type: ignore
    generate_latest = None  # type: ignore
    CONTENT_TYPE_LATEST = "text/plain"

_ENABLED = Counter is not None and Histogram is not None and generate_latest is not None


if _ENABLED:
    HTTP_REQUESTS = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "path", "status"],
    )
    HTTP_LATENCY = Histogram(
        "http_request_duration_seconds",
        "HTTP request duration in seconds",
        ["method", "path"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
    )
    TASK_DURATION = Histogram(
        "task_duration_seconds",
        "Task duration in seconds",
        ["task_type", "model", "status"],
        buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300),
    )
    TASK_FAILURES = Counter(
        "task_failures_total",
        "Task failures total",
        ["task_type", "model"],
    )
    PREDICT_LATENCY = Histogram(
        "predict_latency_seconds",
        "Prediction latency in seconds",
        ["model"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
    )
    DEGRADE_EVENTS = Counter(
        "degrade_events_total",
        "Count of degraded runs/predictions",
        ["stage", "model", "reason"],
    )
    TSF_DEGRADE_TOTAL = Counter(
        "tsf_degrade_total",
        "Total degraded predictions/trainings",
        ["model", "reason"],
    )
else:  # pragma: no cover
    HTTP_REQUESTS = None
    HTTP_LATENCY = None
    TASK_DURATION = None
    TASK_FAILURES = None
    PREDICT_LATENCY = None
    DEGRADE_EVENTS = None
    TSF_DEGRADE_TOTAL = None


_DEGRADE_EVENTS_WINDOW = timedelta(hours=24)
_DEGRADE_EVENTS: deque[tuple[datetime, str, str]] = deque()
_DEGRADE_EVENTS_LOCK = Lock()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _trim_degrade_events(now: datetime) -> None:
    cutoff = now - _DEGRADE_EVENTS_WINDOW
    while _DEGRADE_EVENTS and _DEGRADE_EVENTS[0][0] < cutoff:
        _DEGRADE_EVENTS.popleft()


def metrics_enabled() -> bool:
    return bool(_ENABLED)


def render_metrics() -> tuple[bytes, str]:
    if not _ENABLED:
        return b"metrics disabled", "text/plain"
    return generate_latest(), CONTENT_TYPE_LATEST  # type: ignore[return-value]


def observe_http_request(*, method: str, path: str, status: int, duration: float) -> None:
    if not _ENABLED:
        return
    HTTP_REQUESTS.labels(method=method, path=path, status=str(status)).inc()
    HTTP_LATENCY.labels(method=method, path=path).observe(max(0.0, float(duration)))


def observe_task(
    *,
    task_type: str,
    model: Optional[str],
    duration: float,
    status: str,
) -> None:
    if not _ENABLED:
        return
    model_name = model or "unknown"
    TASK_DURATION.labels(task_type=task_type, model=model_name, status=status).observe(max(0.0, float(duration)))
    if status != "success":
        TASK_FAILURES.labels(task_type=task_type, model=model_name).inc()


def observe_predict(*, model: Optional[str], duration: float) -> None:
    if not _ENABLED:
        return
    model_name = model or "unknown"
    PREDICT_LATENCY.labels(model=model_name).observe(max(0.0, float(duration)))


def normalize_degrade_reason(reason: Optional[str]) -> str:
    raw_reason = str(reason or "").strip().lower()
    if not raw_reason:
        return "unknown"
    if "model_not_supported" in raw_reason:
        return "model_not_supported"
    if "model_not_available" in raw_reason:
        return "model_not_available"
    if "multi_step_not_supported" in raw_reason:
        return "multi_step_not_supported"
    if "required_core_missing" in raw_reason:
        return "required_core_missing"
    if "non_informer_one_step_only" in raw_reason:
        return "non_informer_one_step_only"
    if "inverse_target_failed" in raw_reason:
        return "inverse_target_failed"
    if "residual_skipped" in raw_reason:
        return "residual_skipped"
    if "feature contract" in raw_reason:
        return "feature_contract_fallback"
    if "fallback" in raw_reason:
        return "fallback_error"
    return "other"


def observe_degrade(*, model: Optional[str], reason: Optional[str], stage: Optional[str] = None) -> None:
    model_name = model or "unknown"
    degrade_reason = normalize_degrade_reason(reason)

    with _DEGRADE_EVENTS_LOCK:
        now = _utc_now()
        _DEGRADE_EVENTS.append((now, model_name, degrade_reason))
        _trim_degrade_events(now)

    if not _ENABLED:
        return
    TSF_DEGRADE_TOTAL.labels(model=model_name, reason=degrade_reason).inc()
    if DEGRADE_EVENTS is not None:
        stage_name = str(stage or "unknown").strip() or "unknown"
        DEGRADE_EVENTS.labels(stage=stage_name, model=model_name, reason=degrade_reason).inc()


def get_degrade_summary(*, window_minutes: int = 60, limit: int = 5) -> dict[str, object]:
    safe_window = max(1, int(window_minutes))
    safe_limit = max(1, int(limit))
    now = _utc_now()
    cutoff = now - timedelta(minutes=safe_window)

    with _DEGRADE_EVENTS_LOCK:
        _trim_degrade_events(now)
        window_events = [item for item in _DEGRADE_EVENTS if item[0] >= cutoff]

    by_model: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for _, model_name, reason in window_events:
        by_model[model_name] = by_model.get(model_name, 0) + 1
        by_reason[reason] = by_reason.get(reason, 0) + 1

    top_models = sorted(by_model.items(), key=lambda item: item[1], reverse=True)[:safe_limit]
    top_reasons = sorted(by_reason.items(), key=lambda item: item[1], reverse=True)[:safe_limit]

    return {
        "window_minutes": safe_window,
        "total_degrade": len(window_events),
        "top_models": [{"model": model, "count": count} for model, count in top_models],
        "top_reasons": [{"reason": reason, "count": count} for reason, count in top_reasons],
        "updated_at": now.isoformat(),
    }
