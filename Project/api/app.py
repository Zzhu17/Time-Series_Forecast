from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.observability import add_observability
from api.routes import alerts, health, metrics, models, predict, tasks, train
from api.security import verify_api_token

app = FastAPI(title="TS Forecast API", version="0.1.0", dependencies=[Depends(verify_api_token)])

# CORS for React frontend (adjust origins if you want to lock down)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: serve built frontend (if present) from Universal Time-Series Forecast/dist
frontend_dist = Path(__file__).resolve().parents[1] / "Universal Time-Series Forecast" / "dist"
if frontend_dist.exists():
    app.mount("/ui", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

add_observability(app)

app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(alerts.router)
app.include_router(predict.router)
app.include_router(train.router)
app.include_router(tasks.router)
app.include_router(models.router)
