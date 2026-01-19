from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils.feature_contract import align_df_to_feature_contract
from utils.feature_selection import FeatureContract, save_feature_contract


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
    )
    return aligned_df, report, usable_cols
