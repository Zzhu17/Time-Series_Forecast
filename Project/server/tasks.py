from __future__ import annotations

import uuid
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict

import pandas as pd

from server.db import SessionLocal, TaskRecord, init_db
from server.logging_utils import setup_json_logger, log_json
from server.predict_utils import baseline_predict, predict_with_xgboost

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

        model = str(payload.get("model_name", "baseline")).lower()
        time_col = str(payload.get("time_col", "date"))
        value_col = str(payload.get("value_col", "value"))
        horizon = int(payload.get("horizon", 1))
        rows = payload.get("rows") or []
        df = pd.DataFrame(rows)

        degraded = False
        reason = None
        preds = None

        if model == "xgboost":
            try:
                preds, degraded, used_model, reason = predict_with_xgboost(
                    df,
                    time_col=time_col,
                    value_col=value_col,
                    horizon=horizon,
                    baseline_fallback=True,
                )
                model = used_model
            except Exception as e:
                degraded = True
                reason = f"xgboost failed: {e}"

        if preds is None:
            preds = baseline_predict(df, value_col, horizon)
            if model != "baseline" and not reason:
                reason = f"{model} not available; baseline used"

        metrics = {"model": model, "degraded": degraded, "reason": reason}
        rec.metrics = TaskRecord.dumps(metrics)
        rec.artifacts = TaskRecord.dumps({"predictions": preds.tolist() if preds is not None else []})
        rec.status = "success"
        rec.degraded = bool(degraded)
        rec.updated_at = _now()
        session.commit()
        log_json(
            LOGGER,
            "task_succeeded",
            task_id=task_id,
            model=model,
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
