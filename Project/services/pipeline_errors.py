from __future__ import annotations

from typing import Any, Dict, Tuple


def infer_error_stage_and_action(exc: Exception, default_stage: str = "train") -> Tuple[str, str]:
    msg = str(exc).lower()
    stage = default_stage
    if any(k in msg for k in ["dataframe", "feature", "missing column", "data prep", "preprocess"]):
        stage = "data_prep"
    elif any(k in msg for k in ["validate", "validation", "val_loss"]):
        stage = "validate"
    elif any(k in msg for k in ["predict", "forecast", "inference"]):
        stage = "predict"
    action = "retry" if any(k in msg for k in ["timeout", "temporar", "resource busy", "connection reset"]) else "fail"
    return stage, action


def build_error_payload(exc: Exception, *, stage: str, action: str, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "error",
        "message": str(exc),
        "error_stage": stage,
        "error_type": type(exc).__name__,
        "action": action,
        "metrics": {},
        "data": {},
        "artifacts": artifacts,
    }
