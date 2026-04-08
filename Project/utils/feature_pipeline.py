from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils.feature_contract import align_df_to_feature_contract
from utils.feature_missing_policy import prepare_df_for_non_informer_models
from utils.feature_selection import FeatureContract, save_feature_contract

PREPROCESS_VERSION = "contract-v1"


def build_train_features(
    df: pd.DataFrame,
    *,
    time_col: str,
    value_col: str,
    candidate_cols: List[str],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    cleaned, feature_cols, report = prepare_df_for_non_informer_models(
        df,
        time_col=time_col,
        value_col=value_col,
        candidate_cols=candidate_cols,
        config=config,
    )
    return cleaned, feature_cols, report


def save_feature_contract_if_any(report: Dict[str, Any], artifacts: Dict[str, Any]) -> None:
    """
    Persist feature contract to artifacts.feature_cols_path if present in report.
    Accepts either a FeatureContract dict (preferred) or falls back to saving the dict as JSON.
    """
    if not isinstance(artifacts, dict):
        return
    contract_dict = (report or {}).get("feature_contract") if isinstance(report, dict) else None
    if not contract_dict:
        return
    path = artifacts.get("feature_cols_path")
    if not isinstance(path, str) or not path:
        return
    try:
        if isinstance(contract_dict, dict):
            feature_order = [str(c) for c in (contract_dict.get("feature_cols") or []) if str(c).strip()]
            required = [str(c) for c in (contract_dict.get("required_core_cols") or contract_dict.get("core_cols") or []) if str(c).strip()]
            repairable = [str(c) for c in (contract_dict.get("repairable_core_cols") or contract_dict.get("recomputable_cols") or []) if str(c).strip()]
            optional = [str(c) for c in (contract_dict.get("optional_cols") or []) if str(c).strip()]
            if not optional and feature_order:
                opt_set = [c for c in feature_order if c not in set(required) | set(repairable)]
                optional = list(opt_set)
            contract_dict["feature_order"] = list(feature_order)
            contract_dict["required_core_cols"] = list(required)
            contract_dict["repairable_core_cols"] = list(repairable)
            contract_dict["optional_cols"] = list(optional)
            contract_dict["preprocess_version"] = str(contract_dict.get("preprocess_version") or PREPROCESS_VERSION)
        # Try to serialize via FeatureContract helper if shape matches
        if isinstance(contract_dict, dict) and "feature_cols" in contract_dict:
            try:
                fc = FeatureContract(**contract_dict)  # type: ignore[arg-type]
                save_feature_contract(path, fc)
                return
            except Exception:
                pass
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(contract_dict, f, ensure_ascii=False, indent=2)
    except Exception:
        return


def align_predict_df(
    df: pd.DataFrame,
    *,
    contract: Dict[str, Any],
    time_col: str,
    value_col: str,
    tail_rows: Optional[int] = None,
    allow_degrade: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """
    Align incoming dataframe to a saved feature contract for predict-time:
      - rebuild calendar features
      - recompute recomputable features (lag/rolling/diff) if needed
      - drop optional columns that are missing/NaN
    Returns aligned_df, report, usable_cols.
    """
    contract = contract or {}
    feature_cols = list(contract.get("feature_cols") or [])
    aligned_df, report, usable_cols = align_df_to_feature_contract(
        df,
        time_col=time_col,
        value_col=value_col,
        feature_cols=feature_cols,
        contract=contract,
        recompute_policy="recompute",
        tail_rows=tail_rows,
        allow_degrade=allow_degrade,
    )
    return aligned_df, report, usable_cols
