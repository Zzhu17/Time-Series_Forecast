from __future__ import annotations

import uuid
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict

from services.db import SessionLocal, TaskRecord, init_db
from utils.logging_utils import setup_json_logger, log_json
from services.train_service import run_training_task

# Initialize DB (sqlite by default)
init_db()

# Lightweight executor; can be replaced by Celery/RQ later.
EXECUTOR = ThreadPoolExecutor(max_workers=2)
LOGGER = setup_json_logger()


def _now():
    return datetime.utcnow()


def _run_task(task_id: str, payload: Dict[str, Any]):
    session = SessionLocal()
    start_ts = _now()
    try:
        rec = session.get(TaskRecord, task_id)
        if not rec:
            return
        rec.status = "running"
        rec.updated_at = _now()
        session.commit()
        log_json(LOGGER, "task_started", task_id=task_id, model=payload.get("model_name"))

        result = run_training_task(payload, task_id=task_id)
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
        session.close()


def submit_train_task(payload: Dict[str, Any]) -> str:
    task_id = str(uuid.uuid4())
    session = SessionLocal()
    try:
        rec = TaskRecord(
            id=task_id,
            status="pending",
            model_name=str(payload.get("model_name", "baseline")),
            params=TaskRecord.dumps(payload),
            created_at=_now(),
            updated_at=_now(),
        )
        session.add(rec)
        session.commit()
    finally:
        session.close()
    EXECUTOR.submit(_run_task, task_id, payload)
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
