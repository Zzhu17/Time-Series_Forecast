from __future__ import annotations

import os

from celery import Celery


def _env(key: str, default: str = "") -> str:
    return str(os.getenv(key, default)).strip()


def create_celery_app() -> Celery:
    broker_url = _env("CELERY_BROKER_URL", "")
    result_backend = _env("CELERY_RESULT_BACKEND", "")
    if not broker_url:
        broker_url = "redis://localhost:6379/0"
    if not result_backend:
        result_backend = broker_url
    app = Celery("tsf", broker=broker_url, backend=result_backend)
    app.conf.update(
        task_track_started=True,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
    )
    return app


celery_app = create_celery_app()
