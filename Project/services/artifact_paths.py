from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def resolve_run_dir_from_artifacts(artifacts: Dict[str, Any], *, default_dir: str = "") -> str:
    run_dir = artifacts.get("run_dir") or artifacts.get("artifact_dir")
    if isinstance(run_dir, str) and run_dir.strip():
        return run_dir

    for key in ("model_path", "scaler_path", "feature_cols_path", "y_scaler_path"):
        p = artifacts.get(key)
        if isinstance(p, str) and p.strip():
            return str(Path(p).expanduser().resolve().parent)

    return default_dir
