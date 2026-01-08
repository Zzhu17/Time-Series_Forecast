from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class TrainingPayload(BaseModel):
    model_name: str = Field(..., description="Model to train (e.g., informer/xgboost/arima).")
    time_col: str = Field(..., description="Timestamp column name.")
    value_col: str = Field(..., description="Target column name.")
    horizon: int = Field(1, description="Forecast horizon (points).")
    rows: List[Dict[str, Any]] = Field(..., description="List of records (dict-like).")
    feature_cols: Optional[List[str]] = Field(None, description="Optional feature column ordering.")
    residual_modeling: Optional[dict] = Field(None, description="Residual options (matches app config).")
    allow_degrade: bool = Field(False, description="Allow degraded fallback for missing features.")
    device: str = Field("cpu", description="Preferred device (cpu/cuda).")
    uploaded_name: Optional[str] = Field(None, description="Original upload filename (optional).")

    @validator("model_name", "time_col", "value_col")
    def _non_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("must be a non-empty string")
        return v.strip()

    @validator("rows")
    def _non_empty_rows(cls, v: List[Dict[str, Any]]):
        if not isinstance(v, list) or not v:
            raise ValueError("rows must not be empty")
        for item in v:
            if not isinstance(item, dict):
                raise ValueError("rows must be a list of dict-like objects")
        return v

    @validator("horizon")
    def _horizon_pos(cls, v: int):
        if v <= 0:
            raise ValueError("horizon must be > 0")
        return v

    @validator("feature_cols", pre=True)
    def _clean_features(cls, v):
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("feature_cols must be a list or None")
        cleaned = [str(c).strip() for c in v if isinstance(c, str) and str(c).strip()]
        return cleaned if cleaned else None

    @validator("device")
    def _device_string(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            return "cpu"
        return v.strip()

    class Config:
        extra = "ignore"


 
