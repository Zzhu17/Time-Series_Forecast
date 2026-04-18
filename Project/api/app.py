from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.observability import add_observability
from api.routes import alerts, artifacts, health, metrics, models, predict, tasks, train
from api.security import get_cors_allow_origins, validate_runtime_security, verify_api_token

app = FastAPI(title="TS Forecast API", version="0.1.0", dependencies=[Depends(verify_api_token)])

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_allow_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-Id", "X-Trace-Id"],
)

# Optional: serve built frontend (if present) from Project/frontend/dist
frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
legacy_dist = Path(__file__).resolve().parents[1] / "Universal Time-Series Forecast" / "dist"
if not frontend_dist.exists() and legacy_dist.exists():
    frontend_dist = legacy_dist
if frontend_dist.exists():
    app.mount("/ui", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

add_observability(app)
app.add_event_handler("startup", validate_runtime_security)

app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(alerts.router)
app.include_router(predict.router)
app.include_router(train.router)
app.include_router(tasks.router)
app.include_router(models.router)
app.include_router(artifacts.router)
