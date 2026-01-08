from __future__ import annotations

import os
import time
import uuid
from typing import Optional

from fastapi import FastAPI, Request

from utils.logging_utils import log_json, setup_json_logger
from utils.metrics import observe_http_request


def _get_log_level() -> Optional[int]:
    name = os.getenv("TSF_LOG_LEVEL")
    if not name:
        return None
    try:
        import logging

        return int(getattr(logging, name.upper()))
    except Exception:
        return None


def add_observability(app: FastAPI) -> None:
    logger = setup_json_logger()
    level = _get_log_level()
    if level is not None:
        logger.setLevel(level)

    @app.middleware("http")
    async def _log_and_measure(request: Request, call_next):
        trace_id = (
            request.headers.get("x-request-id")
            or request.headers.get("x-trace-id")
            or str(uuid.uuid4())
        )
        request.state.trace_id = trace_id
        start = time.time()
        response = None
        error: Optional[BaseException] = None
        try:
            response = await call_next(request)
            return response
        except BaseException as exc:
            error = exc
            raise
        finally:
            duration = time.time() - start
            route = request.scope.get("route")
            path = getattr(route, "path", request.url.path)
            status = response.status_code if response is not None else 500
            observe_http_request(
                method=request.method,
                path=path,
                status=status,
                duration=duration,
            )
            log_json(
                logger,
                "http_request",
                trace_id=trace_id,
                method=request.method,
                path=path,
                status=status,
                duration_ms=int(duration * 1000),
                client_ip=getattr(request.client, "host", None),
                error=str(error) if error else None,
            )
            if response is not None:
                response.headers.setdefault("X-Trace-Id", trace_id)
