from __future__ import annotations

from typing import Any, Dict, List, Optional

from services import registry


ALLOWED_STAGES = {"candidate", "production", "archived"}


def _validate_stage(stage: str) -> str:
    st = str(stage or "").strip()
    if st not in ALLOWED_STAGES:
        raise ValueError("stage must be candidate/production/archived")
    return st


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model name is required")
    cleaned = name.strip()
    if cleaned.lower() in ("none", "null"):
        raise ValueError("model name is required")
    return cleaned


def list_model_catalog() -> List[Dict[str, str]]:
    return [
        {"name": "baseline", "description": "Naive last-value persistence."},
        {"name": "informer", "description": "Heavy model; requires artifacts (not loaded by default)."},
        {"name": "lstm", "description": "Heavy model; requires artifacts (not loaded by default)."},
        {"name": "xgboost", "description": "Requires trained artifacts."},
        {"name": "randomforest", "description": "Requires trained artifacts."},
        {"name": "arima", "description": "Requires trained artifacts."},
        {"name": "prophet", "description": "Requires trained artifacts."},
    ]


def register_model_entry(
    *,
    name: str,
    version: Optional[str],
    stage: str,
    params: Optional[Dict[str, Any]],
    metrics: Optional[Dict[str, Any]],
    artifacts: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    name = _validate_name(name)
    stage = _validate_stage(stage)
    return registry.register_model(
        name=name,
        version=version,
        stage=stage,
        params=params,
        metrics=metrics,
        artifacts=artifacts,
    )


def promote_model_entry(model_id: str, *, stage: str) -> Optional[Dict[str, Any]]:
    stage = _validate_stage(stage)
    return registry.promote_model(model_id, stage=stage)


def list_models_registry(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    try:
        lim = int(limit)
    except Exception:
        lim = 50
    try:
        off = int(offset)
    except Exception:
        off = 0
    lim = max(1, min(lim, 500))
    off = max(0, off)
    return registry.list_models(limit=lim, offset=off)


def latest_production_model() -> Optional[Dict[str, Any]]:
    return registry.latest_production()
