from fastapi import APIRouter, HTTPException

from services.registry import (
    latest_production,
    list_models as list_models_rec,
    promote_model,
    register_model,
)
from schemas.api import ModelInfo, ModelRegisterRequest, ModelResponse

router = APIRouter()


@router.get("/models", response_model=list[ModelInfo])
def list_models():
    return [
        ModelInfo(name="baseline", description="Naive last-value persistence."),
        ModelInfo(name="informer", description="Heavy model; requires artifacts (not loaded by default)."),
        ModelInfo(name="lstm", description="Heavy model; requires artifacts (not loaded by default)."),
        ModelInfo(name="xgboost", description="Requires trained artifacts."),
        ModelInfo(name="randomforest", description="Requires trained artifacts."),
        ModelInfo(name="arima", description="Requires trained artifacts."),
        ModelInfo(name="prophet", description="Requires trained artifacts."),
    ]


@router.post("/models/register", response_model=ModelResponse)
def register_model_api(req: ModelRegisterRequest):
    rec = register_model(
        name=req.name,
        version=req.version,
        stage=req.stage,
        params=req.params,
        metrics=req.metrics,
        artifacts=req.artifacts,
    )
    return rec


@router.post("/models/{model_id}/promote", response_model=ModelResponse)
def promote_model_api(model_id: str, stage: str = "production"):
    rec = promote_model(model_id, stage=stage)
    if rec is None:
        raise HTTPException(status_code=404, detail="model not found")
    return rec


@router.get("/models/registry", response_model=list[ModelResponse])
def list_models_api(limit: int = 50, offset: int = 0):
    return list_models_rec(limit=limit, offset=offset)


@router.get("/models/production", response_model=ModelResponse)
def get_latest_production_model():
    rec = latest_production()
    if rec is None:
        raise HTTPException(status_code=404, detail="no production model")
    return rec
