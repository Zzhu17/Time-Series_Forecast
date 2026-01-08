from __future__ import annotations

import os
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict

from jobs.celery_app import celery_app
from services.db import SessionLocal, TaskRecord, init_db
from utils.logging_utils import log_json, setup_json_logger
from services.train_service import run_training_task
from utils.metrics import observe_task

try:  # optional in some minimal envs
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # optional in some minimal envs
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

# Initialize DB (sqlite by default)
init_db()

# Fallback executor when Celery is disabled/unavailable.
EXECUTOR = ThreadPoolExecutor(max_workers=2)
LOGGER = setup_json_logger()


def _now():
    return datetime.utcnow()


def _env_flag(key: str) -> bool:
    return str(os.getenv(key, "")).strip().lower() in ("1", "true", "yes", "on")


def _celery_enabled() -> bool:
    if not _env_flag("CELERY_ENABLED"):
        return False
    return bool(str(os.getenv("CELERY_BROKER_URL", "")).strip())


def _sanitize_value(value: Any) -> Any:
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if pd is not None and isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _sanitize_payload(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_payload(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_payload(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_payload(v) for v in obj]
    return _sanitize_value(obj)


@celery_app.task(name="tsf.train_task")
def run_training_task_async(task_id: str, payload: Dict[str, Any]):
    _run_task(task_id, payload)
    return {"task_id": task_id}


def _run_task(task_id: str, payload: Dict[str, Any]):
    session = SessionLocal()
    start_ts = _now()
    start_time = time.time()
    status = "success"
    try:
        rec = session.get(TaskRecord, task_id)
        if not rec:
            return
        rec.status = "running"
        rec.updated_at = _now()
        session.commit()
        log_json(LOGGER, "task_started", task_id=task_id, model=payload.get("model_name"))

        result = run_training_task(payload, task_id=task_id, emit_metrics=False)
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        artifacts = result.get("artifacts", {}) if isinstance(result, dict) else {}
        model_record = result.get("model_record") if isinstance(result, dict) else None
        degraded = bool(result.get("degraded", False)) if isinstance(result, dict) else False
        reason = result.get("degraded_reason") if isinstance(result, dict) else None

        rec.metrics = TaskRecord.dumps(metrics)
        rec.artifacts = TaskRecord.dumps(
            {"artifacts": artifacts, "model_record": model_record, "degraded_reason": reason}
        )
        rec.status = "success"
        rec.degraded = degraded
        rec.updated_at = _now()
        session.commit()
        log_json(
            LOGGER,
            "task_succeeded",
            task_id=task_id,
            model=payload.get("model_name"),
            degraded=degraded,
            reason=reason,
            duration_ms=int((rec.updated_at - start_ts).total_seconds() * 1000),
        )
    except Exception as e:
        status = "failed"
        try:
            rec = session.get(TaskRecord, task_id)
            if rec:
                rec.status = "failed"
                rec.error = f"{e}\n{traceback.format_exc()}"
                rec.updated_at = _now()
                session.commit()
                log_json(
                    LOGGER,
                    "task_failed",
                    task_id=task_id,
                    model=payload.get("model_name"),
                    error=str(e),
                )
        except Exception:
            pass
    finally:
        observe_task(
            task_type="train",
            model=str(payload.get("model_name") or "unknown"),
            duration=time.time() - start_time,
            status=status,
        )
        session.close()


def submit_train_task(payload: Dict[str, Any]) -> str:
    task_id = str(uuid.uuid4())
    safe_payload = _sanitize_payload(payload)
    session = SessionLocal()
    try:
        rec = TaskRecord(
            id=task_id,
            status="pending",
            model_name=str(payload.get("model_name", "baseline")),
            params=TaskRecord.dumps(safe_payload),
            created_at=_now(),
            updated_at=_now(),
        )
        session.add(rec)
        session.commit()
    finally:
        session.close()
    if _celery_enabled():
        try:
            run_training_task_async.delay(task_id, safe_payload)
            log_json(LOGGER, "task_enqueued", task_id=task_id, queue="celery")
            return task_id
        except Exception as exc:
            log_json(LOGGER, "task_enqueue_failed", task_id=task_id, error=str(exc))
    EXECUTOR.submit(_run_task, task_id, safe_payload)
    return task_id


def get_task(task_id: str) -> Dict[str, Any] | None:
    session = SessionLocal()
    try:
        rec = session.get(TaskRecord, task_id)
        return rec.to_dict() if rec else None
    finally:
        session.close()


def list_tasks(limit: int = 20, offset: int = 0):
    session = SessionLocal()
    try:
        q = session.query(TaskRecord).order_by(TaskRecord.created_at.desc())
        if offset:
            q = q.offset(offset)
        if limit:
            q = q.limit(limit)
        return [r.to_dict() for r in q.all()]
    finally:
        session.close()
