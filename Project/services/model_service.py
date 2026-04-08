from __future__ import annotations

import importlib.util
from typing import Any, Dict, List, Optional

from services import registry
from models.registry import MODEL_CATALOG


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


def _module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _check_deps(deps: List[str]) -> tuple[bool, List[str]]:
    missing = [dep for dep in deps if dep and not _module_available(dep)]
    return (len(missing) == 0), missing


def list_model_catalog() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for name, item in MODEL_CATALOG.items():
        deps = item.get("deps") or []
        available, missing = _check_deps(deps)
        out.append(
            {
                "name": name,
                "description": item["description"],
                "available": available,
                "missing_deps": missing or None,
            }
        )
    return out


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
