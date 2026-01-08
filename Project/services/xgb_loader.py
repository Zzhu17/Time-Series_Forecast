from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.feature_pipeline import align_predict_df
from utils.target_transform import inverse_transform_array


class XGBPredictor:
    def __init__(
        self,
        *,
        model_path: str,
        feature_contract_path: Optional[str],
        target_transform: Optional[dict],
        time_col: str,
        value_col: str,
    ):
        self.model_path = model_path
        self.feature_contract_path = feature_contract_path
        self.target_transform = target_transform
        self.time_col = time_col
        self.value_col = value_col
        self._model = None
        self._contract: Optional[Dict[str, Any]] = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            import xgboost as xgb  # type: ignore
        except Exception as e:
            raise RuntimeError(f"xgboost not installed: {e}")
        mdl = xgb.XGBRegressor()
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"xgboost model not found at {self.model_path}")
        mdl.load_model(self.model_path)
        self._model = mdl
        return mdl

    def _load_contract(self):
        if self._contract is not None:
            return self._contract
        if self.feature_contract_path and Path(self.feature_contract_path).exists():
            try:
                import json

                with open(self.feature_contract_path, "r", encoding="utf-8") as f:
                    self._contract = json.load(f)
            except Exception:
                self._contract = None
        return self._contract

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> Tuple[np.ndarray, dict, bool, Optional[str]]:
        mdl = self._load_model()

        contract = self._load_contract()
        degraded = False
        degraded_reason = None

        if contract and isinstance(contract.get("feature_cols"), list):
            expected_cols = [str(c) for c in contract.get("feature_cols") if str(c).strip()]
            if not expected_cols:
                contract = None
            else:
                aligned, _report, usable_cols = align_predict_df(
                    df,
                    contract=contract,
                    time_col=self.time_col,
                    value_col=self.value_col,
                    tail_rows=horizon + 5,  # minimal history for lags/rolling
                )
                if list(usable_cols) != list(expected_cols):
                    dropped = sorted(set(expected_cols) - set(usable_cols))
                    raise ValueError(f"optional features dropped: {dropped}")
                df_feat = aligned
                feature_cols = [c for c in expected_cols if c != self.value_col]
        if not contract:
            degraded = True
            degraded_reason = "no feature contract found; using raw columns"
            feature_cols = [c for c in df.columns if c not in (self.time_col, self.value_col)]
            df_feat = df.copy()

        X = df_feat[feature_cols].copy() if feature_cols else pd.DataFrame(index=df_feat.index)
        for c in feature_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X_np = X.to_numpy(dtype=np.float32)

        # Use last row as context; predict horizon steps independently (naive strategy)
        if len(X_np) == 0:
            raise ValueError("No usable features for prediction.")
        last_row = X_np[-1:].repeat(horizon, axis=0)
        preds = mdl.predict(last_row)
        preds = np.asarray(preds, dtype=float).reshape(-1)

        # Inverse target transform if provided
        if self.target_transform:
            try:
                preds = inverse_transform_array(preds, self.target_transform)
            except Exception:
                degraded = True
                degraded_reason = (degraded_reason or "") + "|inverse_target_failed"

        return preds, {"feature_cols": feature_cols}, degraded, degraded_reason
