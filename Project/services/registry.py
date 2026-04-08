from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import desc

from services.db import ModelRecord, SessionLocal, init_db

init_db()

ALLOWED_STAGES = {"candidate", "production", "archived"}


def _now():
    return datetime.utcnow()


def register_model(
    name: str,
    *,
    version: Optional[str],
    stage: str = "candidate",
    params: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    stage_value = str(stage or "candidate").strip().lower()
    if stage_value not in ALLOWED_STAGES:
        stage_value = "candidate"
    rec_id = str(uuid.uuid4())
    session = SessionLocal()
    try:
        rec = ModelRecord(
            id=rec_id,
            name=name,
            version=version,
            stage=stage_value,
            params=ModelRecord.dumps(params),
            metrics=ModelRecord.dumps(metrics),
            artifacts=ModelRecord.dumps(artifacts),
            created_at=_now(),
            updated_at=_now(),
            promoted_at=_now() if stage_value == "production" else None,
        )
        session.add(rec)
        session.commit()
        return rec.to_dict()
    finally:
        session.close()


def promote_model(rec_id: str, stage: str = "production") -> Optional[Dict[str, Any]]:
    session = SessionLocal()
    try:
        rec = session.get(ModelRecord, rec_id)
        if not rec:
            return None
        rec.stage = stage
        rec.promoted_at = _now() if stage == "production" else None
        rec.updated_at = _now()
        session.commit()
        return rec.to_dict()
    finally:
        session.close()


def get_model(rec_id: str) -> Optional[Dict[str, Any]]:
    session = SessionLocal()
    try:
        rec = session.get(ModelRecord, rec_id)
        return rec.to_dict() if rec else None
    finally:
        session.close()


def list_models(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    session = SessionLocal()
    try:
        q = session.query(ModelRecord).order_by(desc(ModelRecord.created_at))
        if offset:
            q = q.offset(offset)
        if limit:
            q = q.limit(limit)
        return [r.to_dict() for r in q.all()]
    finally:
        session.close()


def latest_production() -> Optional[Dict[str, Any]]:
    session = SessionLocal()
    try:
        rec = (
            session.query(ModelRecord)
            .filter(ModelRecord.stage == "production")
            .order_by(desc(ModelRecord.promoted_at))
            .first()
        )
        return rec.to_dict() if rec else None
    finally:
        session.close()


def latest_model_for_name(name: str, stage: Optional[str] = None) -> Optional[Dict[str, Any]]:
    session = SessionLocal()
    try:
        q = session.query(ModelRecord).filter(ModelRecord.name == name)
        if stage:
            q = q.filter(ModelRecord.stage == stage)
        rec = q.order_by(desc(ModelRecord.created_at)).first()
        return rec.to_dict() if rec else None
    finally:
        session.close()
