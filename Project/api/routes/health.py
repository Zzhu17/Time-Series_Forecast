import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from configs.config import load_yaml_config
from services.db import engine

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/health/live")
def live():
    return {"status": "ok"}


@router.get("/health/ready")
def ready():
    checks = {}
    ok = True

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["db"] = "ok"
    except Exception as e:
        ok = False
        checks["db"] = f"error: {e}"

    try:
        load_yaml_config()
        checks["config"] = "ok"
    except Exception as e:
        ok = False
        checks["config"] = f"error: {e}"

    try:
        project_dir = Path(__file__).resolve().parents[2]
        for name in ("artifacts", "output"):
            path = project_dir / name
            path.mkdir(parents=True, exist_ok=True)
            if not os.access(path, os.W_OK):
                ok = False
                checks[name] = "error: not writable"
            else:
                checks[name] = "ok"
    except Exception as e:
        ok = False
        checks["storage"] = f"error: {e}"

    status = "ok" if ok else "error"
    payload = {"status": status, **checks}
    return JSONResponse(content=payload, status_code=200 if ok else 503)
