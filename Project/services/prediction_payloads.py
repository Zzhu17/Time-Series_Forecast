from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd
from pydantic import ValidationError

from schemas.api import PredictRequest
from services.contract_utils import (
    apply_hybrid_preset,
    coerce_rows,
    normalize_model_name,
    resolve_feature_contract,
    validate_required_columns,
)


def normalize_prediction_payload(
    payload: Dict[str, Any],
    *,
    df_override: pd.DataFrame | None = None,
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
        parsed = PredictRequest(**raw)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    df = df_override if isinstance(df_override, pd.DataFrame) else pd.DataFrame(raw["rows"])
    if df.empty:
        raise ValueError("rows must not be empty")
    validate_required_columns(df, parsed.time_col, parsed.value_col)

    feature_cols = list(parsed.feature_cols or [])
    contract_report: Dict[str, Any] = {}
    if feature_cols:
        feature_cols, contract_report = resolve_feature_contract(
            df,
            time_col=parsed.time_col,
            value_col=parsed.value_col,
            feature_cols=feature_cols,
        )

    normalized = parsed.dict()
    normalized["feature_cols"] = list(feature_cols) if feature_cols else None
    normalized["rows"] = raw["rows"]
    if contract_report:
        normalized["contract_report"] = contract_report
    if model_alias:
        normalized["model_alias"] = model_alias
    return df, normalized, contract_report
