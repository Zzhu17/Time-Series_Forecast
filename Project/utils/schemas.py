from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, validator


class ResidualConfigModel(BaseModel):
    enabled: bool = Field(default=False)
    model_type: Optional[str] = Field(default=None)

    class Config:
        extra = "ignore"


class PipelineRunModel(BaseModel):
    time_col: str
    value_col: str
    model_name: str
    feature_cols: Optional[List[str]] = None
    residual_modeling: Optional[ResidualConfigModel] = None

    @validator("time_col", "value_col", "model_name")
    def _non_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("must be a non-empty string")
        return v.strip()

    @validator("feature_cols", pre=True, always=True)
    def _clean_features(cls, v):
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("feature_cols must be a list or None")
        cleaned = [c for c in v if isinstance(c, str) and c.strip()]
        return cleaned if cleaned else None

    class Config:
        extra = "ignore"
