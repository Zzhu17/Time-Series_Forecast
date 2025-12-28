from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.feature_contract import (
    ensure_calendar_features,
    is_recomputable_name,
    recompute_feature_column,
)
from utils.feature_selection import select_features_train_only


CALENDAR_FEATURES = ["month", "day_of_month", "day_of_week", "hour", "day_of_year"]


@dataclass
class FeatureTiers:
    required_core: List[str]
    repairable_core: List[str]
    optional: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required_core": list(self.required_core),
            "repairable_core": list(self.repairable_core),
            "optional": list(self.optional),
        }


def _cfg_informer(config: Dict[str, Any]) -> Dict[str, Any]:
    return ((config.get("model_config") or {}).get("Informer") or {}) if isinstance(config, dict) else {}


def infer_feature_tiers(
    *,
    candidate_cols: List[str],
    time_col: str,
    value_col: str,
    config: Dict[str, Any],
) -> FeatureTiers:
    """
    Tiering rules:
      - Required Core: always includes value_col; plus explicit required_core_cols/core_cols.
      - Repairable Core: calendar features + explicit repairable_core_cols/safe_default_cols + recomputable name-pattern features.
      - Optional: everything else in candidate_cols (excluding time_col).
    """
    inf = _cfg_informer(config)
    fs = dict(inf.get("feature_selection") or {})
    required_extra = list(fs.get("required_core_cols") or fs.get("core_cols") or [])
    repairable_extra = list(fs.get("repairable_core_cols") or fs.get("safe_default_cols") or CALENDAR_FEATURES)

    cand = [c for c in candidate_cols if c and c != time_col]
    cand = [value_col] + [c for c in cand if c != value_col]

    required = [value_col] + [c for c in required_extra if c and c != value_col and c in cand]
    repairable = []
    for c in repairable_extra:
        if c and c != time_col and c not in required and c in cand:
            repairable.append(c)

    # auto include any recomputable features that appear in candidates (rolling/lag/diff)
    for c in cand:
        if c in required or c in repairable:
            continue
        if is_recomputable_name(c):
            repairable.append(c)

    optional = [c for c in cand if c not in required and c not in repairable]
    return FeatureTiers(required_core=required, repairable_core=repairable, optional=optional)


def prepare_df_train_strict(
    df: pd.DataFrame,
    *,
    time_col: str,
    value_col: str,
    candidate_cols: List[str],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, List[str], FeatureTiers, Dict[str, Any]]:
    """
    Training-stage strict policy:
      - Required core: missing/NaN -> error
      - Repairable core: try recompute; if still missing -> error
      - Optional: if missing/NaN -> drop from feature set and record
      - Never silently fill 0

    Also trims a leading prefix if repairable recomputation creates expected initial NaNs (e.g., rolling windows),
    but requires that after the first valid row, there are no gaps for required+repairable.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("prepare_df_train_strict expects a non-empty DataFrame.")

    tiers = infer_feature_tiers(candidate_cols=candidate_cols, time_col=time_col, value_col=value_col, config=config)
    report: Dict[str, Any] = {
        "dropped_optional": [],
        "recomputed": [],
        "trimmed_prefix_rows": 0,
        "tiers": tiers.to_dict(),
    }

    work = df.copy()

    # Ensure calendar features exist if they are requested as repairable core
    if any(c in CALENDAR_FEATURES for c in tiers.repairable_core):
        work = ensure_calendar_features(work, time_col=time_col)

    # Coerce numeric for candidate feature columns (exclude time_col)
    for c in [c for c in candidate_cols if c != time_col and c in work.columns]:
        s = pd.to_numeric(work[c], errors="coerce")
        try:
            s = s.mask(~np.isfinite(s.to_numpy(dtype=float)), np.nan)
        except Exception:
            pass
        work[c] = s

    # Required core checks
    for c in tiers.required_core:
        if c not in work.columns:
            raise KeyError(f"Required core feature missing: '{c}'")
        if work[c].isna().any():
            raise ValueError(f"Required core feature has NaN: '{c}'")

    # Repairable core recompute if needed
    for c in tiers.repairable_core:
        if c in CALENDAR_FEATURES:
            if c not in work.columns:
                # ensure_calendar_features should have created it; treat as failure
                raise KeyError(f"Repairable core calendar feature missing after rebuild: '{c}'")
        if c not in work.columns or work[c].isna().any():
            # try recompute
            try:
                work[c] = recompute_feature_column(work, c, value_col=value_col, time_col=time_col)
                report["recomputed"].append(c)
            except Exception as e:
                raise ValueError(f"Repairable core feature '{c}' cannot be recomputed: {e}") from e

    # Trim prefix to drop expected initial NaNs (rolling/lag/diff), but do not allow gaps later.
    core_cols = [c for c in (tiers.required_core + tiers.repairable_core) if c in work.columns]
    if core_cols:
        mask = ~work[core_cols].isna().any(axis=1)
        if not bool(mask.any()):
            raise ValueError("All rows invalid after repairable recomputation (no row has full required+repairable core).")
        first_valid = int(np.argmax(mask.to_numpy()))
        if first_valid > 0:
            report["trimmed_prefix_rows"] = int(first_valid)
            work = work.iloc[first_valid:].reset_index(drop=True)
            mask = ~work[core_cols].isna().any(axis=1)
        if not bool(mask.all()):
            # gaps remain -> strict failure
            bad_idx = work.index[~mask].tolist()[:5]
            raise ValueError(f"Repairable core still has missing values after recompute (non-prefix gaps), e.g. rows {bad_idx}")

    # Optional: drop columns that are missing or contain any NaN
    kept_optional: List[str] = []
    for c in tiers.optional:
        if c not in work.columns:
            report["dropped_optional"].append({"col": c, "reason": "missing_column"})
            continue
        if work[c].isna().any():
            report["dropped_optional"].append({"col": c, "reason": "has_nan"})
            continue
        kept_optional.append(c)

    final_feature_cols = tiers.required_core + tiers.repairable_core + kept_optional
    # ensure unique and value_col first
    seen = set()
    ordered: List[str] = []
    for c in [value_col] + [x for x in final_feature_cols if x != value_col]:
        if c in seen:
            continue
        if c in work.columns and c != time_col:
            ordered.append(c)
            seen.add(c)
    final_feature_cols = ordered

    return work, final_feature_cols, tiers, report


def prepare_df_for_non_informer_models(
    df: pd.DataFrame,
    *,
    time_col: str,
    value_col: str,
    candidate_cols: List[str],
    config: Dict[str, Any],
    split_ratio: Tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Unified feature cleaning for non-Informer models (LSTM/RF/ARIMA/Prophet/...),
    matching Informer’s training-time feature policy:
      1) Strict missing-feature policy on the FULL series first:
         - Required core: missing/NaN => error
         - Repairable core: recompute; remaining NaN => error
         - Optional: any NaN/missing => drop
      2) Train-only feature selection on the TRAIN split (same logic as Informer).

    This avoids model-by-model NaN debugging and keeps all models aligned with Informer.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("prepare_df_for_non_informer_models expects a non-empty DataFrame.")

    work = df.copy()
    if time_col in work.columns:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce", utc=True)
        try:
            work[time_col] = work[time_col].dt.tz_localize(None)
        except Exception:
            pass
        work = work.sort_values(time_col)
        # de-dup on time (keep last)
        try:
            work = work.drop_duplicates(subset=[time_col], keep="last")
        except Exception:
            pass
    else:
        # If time_col missing, keep row order stable and let downstream metrics infer timestamps.
        work = work.reset_index(drop=True)

    # Required: target must be numeric and non-missing
    if value_col in work.columns:
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
        work = work.dropna(subset=[value_col]).reset_index(drop=True)
    else:
        raise KeyError(f"Missing target column '{value_col}'.")

    # Ensure calendar features exist (Informer-style repairable core defaults)
    try:
        if time_col in work.columns:
            work = ensure_calendar_features(work, time_col=time_col)
    except Exception:
        pass

    # Normalize candidate list (exclude time_col, ensure value_col first)
    cand_in = [c for c in (candidate_cols or []) if isinstance(c, str) and c and c != time_col]
    if time_col in work.columns:
        for c in CALENDAR_FEATURES:
            if c not in cand_in:
                cand_in.append(c)
    if not cand_in:
        numeric_cols = [c for c in work.select_dtypes(include="number").columns if c != time_col]
        cand_in = [c for c in numeric_cols if c != value_col]
    cand = [value_col] + [c for c in cand_in if c != value_col]
    cand = [c for c in cand if c in work.columns and c != time_col]
    if not cand:
        cand = [value_col]

    # Train split for train-only feature selection
    n = int(len(work))
    tr = float(split_ratio[0]) if split_ratio and len(split_ratio) >= 1 else 0.6
    va = float(split_ratio[1]) if split_ratio and len(split_ratio) >= 2 else 0.2
    t_len = max(1, int(n * tr)) if n > 0 else 0
    v_len = max(0, int(n * va)) if n > 0 else 0
    if t_len + v_len >= n:
        v_len = max(0, n - t_len - 1)
    # 1) Apply strict missing policy first (same as Informer).
    cleaned, base_feat_cols, tiers, strict_report = prepare_df_train_strict(
        work,
        time_col=time_col,
        value_col=value_col,
        candidate_cols=cand,
        config=config,
    )

    # 2) Train-only selection (same as Informer).
    train_df = cleaned.iloc[:t_len].copy() if t_len > 0 else cleaned.copy()
    val_df = cleaned.iloc[t_len : t_len + v_len].copy() if v_len > 0 else cleaned.iloc[0:0].copy()
    test_df = cleaned.iloc[t_len + v_len :].copy()

    final_feature_cols = list(base_feat_cols)
    selection_report: Dict[str, Any] | None = None
    try:
        final_feature_cols, contract = select_features_train_only(
            train_df,
            time_col=time_col,
            value_col=value_col,
            candidate_cols=list(base_feat_cols),
            config=config,
        )
        try:
            selection_report = dict(getattr(contract, "selection_report", None) or {})  # type: ignore[attr-defined]
        except Exception:
            selection_report = None
    except Exception as e:
        selection_report = {"error": str(e), "fallback": True, "candidates": list(base_feat_cols)}
        final_feature_cols = list(base_feat_cols)

    # Guardrail: selected features must exist and contain no NaN after strict policy.
    for c in final_feature_cols:
        if c not in train_df.columns:
            raise KeyError(f"Selected feature missing after strict missing-policy: {c}")
        if train_df[c].isna().any() or val_df[c].isna().any() or test_df[c].isna().any():
            raise ValueError(f"Selected feature contains NaN after strict missing-policy (should not happen): {c}")

    report: Dict[str, Any] = {
        "candidate_cols": list(cand),
        "base_feature_cols": list(base_feat_cols),
        "final_feature_cols": list(final_feature_cols),
        "tiers": tiers.to_dict(),
        "strict_report": strict_report,
        "selection_report": selection_report,
        "split": {"train_len": int(t_len), "val_len": int(v_len), "test_len": int(len(cleaned) - t_len - v_len)},
    }
    return cleaned, final_feature_cols, report
