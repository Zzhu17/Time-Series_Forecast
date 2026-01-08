from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class FeatureContractReport(BaseModel):
    feature_cols: List[str] = Field(default_factory=list)
    missing_required_cols: List[str] = Field(default_factory=list)
    missing_feature_cols: List[str] = Field(default_factory=list)
    recomputable_missing_cols: List[str] = Field(default_factory=list)
    duplicate_features: List[str] = Field(default_factory=list)
    invalid_features: List[str] = Field(default_factory=list)
    extra_columns: List[str] = Field(default_factory=list)

    class Config:
        extra = "ignore"
