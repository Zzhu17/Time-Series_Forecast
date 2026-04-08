from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, validator


class Row(BaseModel):
    __root__: dict

    @property
    def data(self) -> dict:
        return self.__root__


class PredictRequest(BaseModel):
    model_name: str = Field(..., description="Model to use (e.g., informer/lstm/xgboost/baseline).")
    model_id: Optional[str] = Field(None, description="Optional model registry id.")
    model_version: Optional[str] = Field(None, description="Optional model registry version.")
    time_col: str = Field(..., description="Timestamp column name in the rows.")
    value_col: str = Field(..., description="Target column name.")
    rows: List[Row] = Field(..., description="List of records (dict-like).")
    horizon: int = Field(1, description="Forecast horizon (points).")
    feature_cols: Optional[List[str]] = Field(None, description="Optional feature column ordering.")
    residual_modeling: Optional[dict] = Field(None, description="Residual options (matches app config).")
    allow_degrade: Optional[bool] = Field(None, description="Allow degraded fallback for missing features.")

    @validator("model_name", "time_col", "value_col")
    def _non_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("must be a non-empty string")
        return v.strip()

    @validator("rows")
    def _non_empty_rows(cls, v: List[Row]):
        if not v:
            raise ValueError("rows must not be empty")
        return v

    @validator("horizon")
    def _horizon_pos(cls, v: int):
        if v <= 0:
            raise ValueError("horizon must be > 0")
        return v

    class Config:
        extra = "ignore"


class PredictResponse(BaseModel):
    status: str
    degraded: bool = False
    degraded_reason: Optional[str] = None
    fallback_model: Optional[str] = None
    reason: Optional[str] = None
    predictions: List[float]
    used_model: str


class TrainRequest(PredictRequest):
    pass


class TaskResponse(BaseModel):
    id: str
    status: str
    model_name: str
    params: Optional[dict] = None
    metrics: Optional[dict] = None
    artifacts: Optional[dict] = None
    error: Optional[str] = None
    degraded: bool = False
    degraded_reason: Optional[str] = None
    fallback_model: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ModelRegisterRequest(BaseModel):
    name: str
    version: Optional[str] = None
    stage: str = Field(default="candidate", description="candidate/production/archived")
    params: Optional[dict] = None
    metrics: Optional[dict] = None
    artifacts: Optional[dict] = None

    @validator("name")
    def _non_empty_name(cls, v: str):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("name must be non-empty")
        return v.strip()

    @validator("stage")
    def _valid_stage(cls, v: str):
        if v not in ("candidate", "production", "archived"):
            raise ValueError("stage must be candidate/production/archived")
        return v


class ModelResponse(BaseModel):
    id: str
    name: str
    version: Optional[str] = None
    stage: str
    params: Optional[dict] = None
    metrics: Optional[dict] = None
    artifacts: Optional[dict] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    promoted_at: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    description: str
    trainer_key: Optional[str] = None
    forecaster_key: Optional[str] = None
    listed: bool = True
    trainable: bool = False
    buildable: bool = False
    forecastable: bool = False
    available: bool = True
    missing_deps: Optional[List[str]] = None
