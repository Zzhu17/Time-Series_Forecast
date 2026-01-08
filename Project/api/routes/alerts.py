from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from utils.logging_utils import log_json, setup_json_logger

router = APIRouter()
LOGGER = setup_json_logger()


@router.post("/alerts")
async def receive_alerts(request: Request) -> dict[str, Any]:
    payload = await request.json()
    alerts = payload.get("alerts") if isinstance(payload, dict) else None
    alert_count = len(alerts) if isinstance(alerts, list) else 0
    log_json(
        LOGGER,
        "alertmanager_webhook",
        status=payload.get("status") if isinstance(payload, dict) else None,
        receiver=payload.get("receiver") if isinstance(payload, dict) else None,
        alert_count=alert_count,
    )
    return {"status": "ok", "received": alert_count}
