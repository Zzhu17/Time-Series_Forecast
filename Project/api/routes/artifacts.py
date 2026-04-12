from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()


def _project_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_latest_meta() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    output_dir = _project_dir() / "output"
    meta_path = output_dir / "latest_report.json"
    run_dir: Optional[str] = None
    leaderboard: Optional[str] = None
    report: Optional[str] = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            run_dir = meta.get("run_dir")
            leaderboard = meta.get("leaderboard")
            report = meta.get("report")
        except Exception:
            pass

    if run_dir:
        return run_dir, leaderboard, report

    # Fallback: newest run dir by mtime
    runs_root = _repo_root() / "artifacts" / "runs"
    if not runs_root.exists():
        return None, None, None
    run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    if not run_dirs:
        return None, None, None
    latest = max(run_dirs, key=lambda p: p.stat().st_mtime)
    run_dir = str(latest)
    lb_path = latest / "leaderboard.csv"
    rpt_path = latest / "report.html"
    leaderboard = str(lb_path) if lb_path.exists() else None
    report = str(rpt_path) if rpt_path.exists() else None
    return run_dir, leaderboard, report


def _load_leaderboard(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    lb_path = Path(path)
    if not lb_path.exists():
        return []
    try:
        df = pd.read_csv(lb_path)
        return df.to_dict(orient="records")
    except Exception:
        return []


def _load_registry_record(run_id: Optional[str]) -> Dict[str, Any]:
    if not run_id:
        return {}
    registry_path = _project_dir() / "output" / "run_registry.json"
    if not registry_path.exists():
        return {}
    try:
        data = json.loads(registry_path.read_text())
    except Exception:
        return {}
    records = data if isinstance(data, list) else [data]
    for record in reversed(records):
        if isinstance(record, dict) and record.get("run_id") == run_id:
            return record
    return {}


def _load_latest_registry_record() -> Dict[str, Any]:
    registry_path = _project_dir() / "output" / "run_registry.json"
    if not registry_path.exists():
        return {}
    try:
        data = json.loads(registry_path.read_text())
    except Exception:
        return {}
    records = data if isinstance(data, list) else [data]
    for record in reversed(records):
        if isinstance(record, dict) and record.get("run_id"):
            return record
    return {}


@router.get("/artifacts/latest")
def latest_artifacts():
    run_dir, leaderboard_path, report_path = _load_latest_meta()
    registry: Dict[str, Any] = {}
    if not run_dir and not leaderboard_path and not report_path:
        registry = _load_latest_registry_record()
        artifacts = registry.get("artifacts") if isinstance(registry, dict) else {}
        if not isinstance(artifacts, dict):
            artifacts = {}
        run_dir = artifacts.get("run_dir")
        if not run_dir:
            model_path = artifacts.get("model_path")
            if isinstance(model_path, str) and model_path:
                run_dir = str(Path(model_path).parent)
        if run_dir:
            leaderboard_path = artifacts.get("leaderboard_path") or str(Path(run_dir) / "leaderboard.csv")
            report_path = artifacts.get("report_path") or str(Path(run_dir) / "report.html")
    if not run_dir and not leaderboard_path and not report_path:
        raise HTTPException(status_code=404, detail="no artifacts found")
    run_id = Path(run_dir).name if run_dir else None
    leaderboard = _load_leaderboard(leaderboard_path)
    if not registry:
        registry = _load_registry_record(run_id)
    payload = {
        "run_id": run_id,
        "model_name": registry.get("model_name") if registry else None,
        "metrics": registry.get("metrics") if registry else None,
        "artifacts": registry.get("artifacts") if registry else None,
        "data": {
            "leaderboard": leaderboard,
            "leaderboard_path": leaderboard_path,
            "report_path": report_path,
        },
    }
    return payload


@router.get("/artifacts/{run_id}/report")
def get_report(run_id: str):
    report_path = _repo_root() / "artifacts" / "runs" / run_id / "report.html"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="report not found")
    return FileResponse(report_path, media_type="text/html")


@router.get("/artifacts/{run_id}/leaderboard")
def get_leaderboard(run_id: str):
    leaderboard_path = _repo_root() / "artifacts" / "runs" / run_id / "leaderboard.csv"
    if not leaderboard_path.exists():
        raise HTTPException(status_code=404, detail="leaderboard not found")
    return FileResponse(leaderboard_path, media_type="text/csv")
