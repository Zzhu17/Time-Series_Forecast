from __future__ import annotations

import uuid
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from server.db import SessionLocal, TaskRecord, init_db
from server.xgb_loader import XGBPredictor
from server.logging_utils import setup_json_logger, log_json

# Initialize DB (sqlite by default)
init_db()

# Lightweight executor; can be replaced by Celery/RQ later.
EXECUTOR = ThreadPoolExecutor(max_workers=2)
LOGGER = setup_json_logger()


def _now():
    return datetime.utcnow()


def _baseline_predict(df: pd.DataFrame, value_col: str, horizon: int) -> Tuple[np.ndarray, bool, str | None]:
    if value_col not in df.columns:
        raise KeyError(f"Missing target column '{value_col}' in rows.")
    y = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if len(y) == 0:
        raise ValueError("No numeric values found for target column.")
    last = float(y.iloc[-1])
    return np.array([last for _ in range(horizon)], dtype=float), False, None


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
                predictor = XGBPredictor(
                    model_path="Project/artifacts/xgboost_model.json",
                    feature_contract_path="Project/artifacts/feature_cols.json",
                    target_transform=None,
                    time_col=time_col,
                    value_col=value_col,
                )
                preds, meta, degraded, reason = predictor.predict(df, horizon=horizon)
            except Exception as e:
                degraded = True
                reason = f"xgboost failed: {e}"

        if preds is None:
            preds, degraded2, reason2 = _baseline_predict(df, value_col, horizon)
            degraded = degraded or degraded2
            if reason2:
                reason = (reason or "") + f"|{reason2}"
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
