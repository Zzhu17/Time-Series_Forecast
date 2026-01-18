from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import ValidationError

from schemas.training import TrainingPayload
from services.contract_utils import (
    apply_hybrid_preset,
    build_feature_contract_report,
    coerce_rows,
    normalize_feature_cols,
    normalize_model_name,
    validate_required_columns,
)
from services.request_utils import auto_feature_cols


def normalize_training_payload(
    payload: Dict[str, Any],
    *,
    auto_select_features: bool = True,
    df_override: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    raw = dict(payload)
    raw["model_name"] = normalize_model_name(raw.get("model_name"))
    raw["model_name"], raw["residual_modeling"], model_alias = apply_hybrid_preset(
        raw["model_name"],
        raw.get("residual_modeling") if isinstance(raw.get("residual_modeling"), dict) else None,
    )
    raw["rows"] = coerce_rows(raw.get("rows"))
    try:
        parsed = TrainingPayload(**raw)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    df = df_override if isinstance(df_override, pd.DataFrame) else pd.DataFrame(parsed.rows)
    if df.empty:
        raise ValueError("rows must not be empty")

    validate_required_columns(df, parsed.time_col, parsed.value_col)

    feature_cols = list(parsed.feature_cols or [])
    if not feature_cols and auto_select_features:
        feature_cols = auto_feature_cols(df.copy(), parsed.time_col, parsed.value_col)

    feature_cols, normalize_report = normalize_feature_cols(
        feature_cols,
        time_col=parsed.time_col,
        value_col=parsed.value_col,
    )
    contract_report = build_feature_contract_report(
        df,
        time_col=parsed.time_col,
        value_col=parsed.value_col,
        feature_cols=feature_cols,
        normalize_report=normalize_report,
    )

    normalized = parsed.dict()
    normalized["feature_cols"] = list(feature_cols)
    normalized["contract_report"] = contract_report
    if model_alias:
        normalized["model_alias"] = model_alias
    return df, normalized, contract_report


def prepare_training_payload(
    payload: Dict[str, Any],
    *,
    df_override: Optional[pd.DataFrame] = None,
    auto_select_features: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    _, normalized, contract_report = normalize_training_payload(
        payload,
        auto_select_features=auto_select_features,
        df_override=df_override,
    )
    return normalized, contract_report
