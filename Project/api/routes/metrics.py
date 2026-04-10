from fastapi import APIRouter, Query, Response

from utils.metrics import get_degrade_summary, metrics_enabled, render_metrics

router = APIRouter()


@router.get("/metrics")
def metrics():
    payload, content_type = render_metrics()
    status = 200 if metrics_enabled() else 503
    return Response(content=payload, media_type=content_type, status_code=status)


@router.get("/metrics/degrade_metric")
def degrade_metric_name():
    return {"metric": "tsf_degrade_total{model,reason}"}


@router.get("/metrics/degrade_summary")
def degrade_summary(window_minutes: int = Query(default=60, ge=1, le=1440), limit: int = Query(default=5, ge=1, le=20)):
    return get_degrade_summary(window_minutes=window_minutes, limit=limit)
