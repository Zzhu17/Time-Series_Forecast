from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Any, Dict
from urllib.parse import urlparse

from sqlalchemy import Boolean, Column, DateTime, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_DB_PATH = os.path.join(PROJECT_DIR, "output", "tasks.db")


def _choose_database_url() -> str:
    """Prefer env DATABASE_URL unless it points at placeholder host; else fall back to local sqlite."""
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        try:
            parsed = urlparse(env_url)
            if parsed.hostname and parsed.hostname.lower() == "host":
                # Common placeholder value; avoid failing tests when a real DB isn't available.
                return f"sqlite:///{DEFAULT_DB_PATH}"
        except Exception:
            pass
        return env_url
    return f"sqlite:///{DEFAULT_DB_PATH}"


DATABASE_URL = _choose_database_url()
_is_sqlite = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)
Base = declarative_base()


class TaskRecord(Base):
    __tablename__ = "tasks"
    id = Column(String(64), primary_key=True)
    status = Column(String(32), nullable=False, default="pending")
    model_name = Column(String(64), nullable=False)
    params = Column(Text, nullable=True)
    metrics = Column(Text, nullable=True)
    artifacts = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    degraded = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        def _sanitize(obj: Any):
            if isinstance(obj, float):
                return obj if math.isfinite(obj) else None
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            return obj

        def _loads(x):
            try:
                return _sanitize(json.loads(x)) if x else None
            except Exception:
                return None

        parsed_params = _loads(self.params)
        parsed_metrics = _loads(self.metrics)
        parsed_artifacts = _loads(self.artifacts)
        return {
            "id": self.id,
            "status": self.status,
            "model_name": self.model_name,
            "params": parsed_params,
            "metrics": parsed_metrics,
            "artifacts": parsed_artifacts,
            "error": self.error,
            "degraded": bool(self.degraded),
            "degraded_reason": parsed_artifacts.get("degraded_reason") if isinstance(parsed_artifacts, dict) else None,
            "fallback_model": parsed_artifacts.get("fallback_model") if isinstance(parsed_artifacts, dict) else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @staticmethod
    def dumps(obj: Any) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return ""


class ModelRecord(Base):
    __tablename__ = "models"
    id = Column(String(64), primary_key=True)
    name = Column(String(128), nullable=False)
    version = Column(String(64), nullable=True)
    stage = Column(String(32), nullable=False, default="candidate")  # candidate/production/archived
    params = Column(Text, nullable=True)
    metrics = Column(Text, nullable=True)
    artifacts = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    promoted_at = Column(DateTime, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        def _sanitize(obj: Any):
            if isinstance(obj, float):
                return obj if math.isfinite(obj) else None
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            return obj

        def _loads(x):
            try:
                return _sanitize(json.loads(x)) if x else None
            except Exception:
                return None

        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "stage": self.stage,
            "params": _loads(self.params),
            "metrics": _loads(self.metrics),
            "artifacts": _loads(self.artifacts),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
        }

    @staticmethod
    def dumps(obj: Any) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return ""


def init_db():
    os.makedirs(os.path.dirname(DEFAULT_DB_PATH), exist_ok=True)
    Base.metadata.create_all(bind=engine)
