from fastapi import APIRouter, HTTPException

from schemas.api import ModelInfo, ModelRegisterRequest, ModelResponse
from services.model_service import (
    latest_production_model,
    list_model_catalog,
    list_models_registry,
    promote_model_entry,
    register_model_entry,
)

router = APIRouter()


@router.get("/models", response_model=list[ModelInfo])
def list_models():
    return [ModelInfo(**item) for item in list_model_catalog()]


@router.post("/models/register", response_model=ModelResponse)
def register_model_api(req: ModelRegisterRequest):
    try:
        return register_model_entry(
            name=req.name,
            version=req.version,
            stage=req.stage,
            params=req.params,
            metrics=req.metrics,
            artifacts=req.artifacts,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/models/{model_id}/promote", response_model=ModelResponse)
def promote_model_api(model_id: str, stage: str = "production"):
    try:
        rec = promote_model_entry(model_id, stage=stage)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if rec is None:
        raise HTTPException(status_code=404, detail="model not found")
    return rec


@router.get("/models/registry", response_model=list[ModelResponse])
def list_models_api(limit: int = 50, offset: int = 0):
    return list_models_registry(limit=limit, offset=offset)


@router.get("/models/production", response_model=ModelResponse)
def get_latest_production_model():
    rec = latest_production_model()
    if rec is None:
        raise HTTPException(status_code=404, detail="no production model")
    return rec
