from __future__ import annotations

import os
from typing import Any, Callable

try:
    from celery import Celery
    _CELERY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    Celery = None  # type: ignore[assignment]
    _CELERY_AVAILABLE = False


def _env(key: str, default: str = "") -> str:
    return str(os.getenv(key, default)).strip()


def create_celery_app() -> Any:
    broker_url = _env("CELERY_BROKER_URL", "")
    result_backend = _env("CELERY_RESULT_BACKEND", "")
    if not broker_url:
        broker_url = "redis://localhost:6379/0"
    if not result_backend:
        result_backend = broker_url
    if not _CELERY_AVAILABLE:
        return _CeleryStub()
    app = Celery("tsf", broker=broker_url, backend=result_backend)
    app.conf.update(
        task_track_started=True,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
    )
    setattr(app, "available", True)
    return app


class _CeleryStub:
    available = False

    def task(self, *args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            def _delay(*_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError("Celery is not installed. Install celery[redis] to enable async tasks.")

            setattr(fn, "delay", _delay)
            return fn

        return decorator


celery_app = create_celery_app()
