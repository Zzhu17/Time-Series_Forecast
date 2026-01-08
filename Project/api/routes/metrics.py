from fastapi import APIRouter, Response

from utils.metrics import metrics_enabled, render_metrics

router = APIRouter()


@router.get("/metrics")
def metrics():
    payload, content_type = render_metrics()
    status = 200 if metrics_enabled() else 503
    return Response(content=payload, media_type=content_type, status_code=status)
