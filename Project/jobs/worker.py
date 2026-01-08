from jobs import tasks  # noqa: F401
from jobs.celery_app import celery_app

__all__ = ("celery_app",)
