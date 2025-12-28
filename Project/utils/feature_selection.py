from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SAFE_TIME_FEATURES_DEFAULT = ["month", "day_of_month", "day_of_week", "hour", "day_of_year"]


def _as_float_frame(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _default_cfg_get(config: Dict[str, Any], path: List[str], default: Any) -> Any:
    cur: Any = config
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _leakage_name_hit(col: str, patterns: List[str]) -> bool:
    s = str(col).lower()
    for p in patterns:
        if p and p.lower() in s:
            return True
    return False


def _corr_drop_redundant(
    train_df: pd.DataFrame,
    cols: List[str],
    value_col: str,
    threshold: float,
) -> Tuple[List[str], Dict[str, Any]]:
    report: Dict[str, Any] = {"dropped_pairs": [], "dropped_cols": []}
    if threshold is None:
        return cols, report
    thr = float(threshold)
    if thr <= 0 or thr >= 1 or len(cols) <= 2:
        return cols, report

    df_num = _as_float_frame(train_df, cols).dropna(axis=0, how="any")
    if len(df_num) < 10:
        return cols, report

    # Corr among features only (exclude target itself from redundancy comparisons)
    feats = [c for c in cols if c != value_col]
    if len(feats) <= 1:
        return cols, report

    corr = df_num[feats].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop: set[str] = set()

    # Use correlation with target as tie-breaker (computed on same train-only data)
    try:
        y = pd.to_numeric(df_num[value_col], errors="coerce")
        target_corr = {}
        for c in feats:
            try:
                target_corr[c] = float(df_num[c].corr(y))
            except Exception:
                target_corr[c] = 0.0
    except Exception:
        target_corr = {c: 0.0 for c in feats}

    for c in upper.columns:
        if c in to_drop:
            continue
        high = upper[c][upper[c] > thr]
        for r, v in high.items():
            if r in to_drop:
                continue
            # drop the one with weaker |corr(feature, target)|
            c_sc = abs(float(target_corr.get(c, 0.0)))
            r_sc = abs(float(target_corr.get(r, 0.0)))
            drop = r if r_sc < c_sc else c
            keep = c if drop == r else r
            to_drop.add(drop)
            report["dropped_pairs"].append({"keep": keep, "drop": drop, "corr": float(v)})

    if to_drop:
        report["dropped_cols"] = sorted(to_drop)
    kept = [c for c in cols if c not in to_drop]
    return kept, report


def _compute_mi_scores(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Optional[np.ndarray]:
    try:
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.impute import SimpleImputer
    except Exception:
        return None

    # Downsample for speed on very long series (keeps selection lightweight)
    n = int(X.shape[0])
    if n > 5000:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(n, size=5000, replace=False)
        X = X[idx]
        y = y[idx]

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    try:
        mi = mutual_info_regression(X_imp, y, random_state=int(seed))
    except TypeError:
        mi = mutual_info_regression(X_imp, y)
    return np.asarray(mi, dtype=float)


def _compute_rf_importance(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Optional[np.ndarray]:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
    except Exception:
        return None

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    n = X_imp.shape[0]
    # Keep this lightweight; downsample rows for very long series
    if n > 5000:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(n, size=5000, replace=False)
        X_fit = X_imp[idx]
        y_fit = y[idx]
    else:
        X_fit = X_imp
        y_fit = y

    rf = RandomForestRegressor(
        n_estimators=150,
        random_state=int(seed),
        n_jobs=-1,
        max_depth=8,
        min_samples_leaf=2,
    )
    rf.fit(X_fit, y_fit)
    return np.asarray(getattr(rf, "feature_importances_", None), dtype=float)


@dataclass
class FeatureContract:
    version: int
    time_col: str
    value_col: str
    feature_cols: List[str]
    required_core_cols: List[str]
    repairable_core_cols: List[str]
    optional_cols: List[str]
    safe_default_cols: List[str]
    recomputable_cols: List[str]
    core_cols: List[str]
    train_median: Dict[str, float]
    selection_report: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "time_col": self.time_col,
            "value_col": self.value_col,
            "feature_cols": list(self.feature_cols),
            "required_core_cols": list(self.required_core_cols),
            "repairable_core_cols": list(self.repairable_core_cols),
            "optional_cols": list(self.optional_cols),
            "safe_default_cols": list(self.safe_default_cols),
            "recomputable_cols": list(self.recomputable_cols),
            "core_cols": list(self.core_cols),
            "train_median": dict(self.train_median),
            "selection_report": self.selection_report,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }


def save_feature_contract(path: str, contract: FeatureContract) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(contract.to_dict(), f, ensure_ascii=False, indent=2)


def load_feature_contract(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "feature_cols" in obj:
            return obj
    except Exception:
        return None
    return None


def select_features_train_only(
    train_df: pd.DataFrame,
    *,
    time_col: str,
    value_col: str,
    candidate_cols: List[str],
    config: Dict[str, Any],
) -> Tuple[List[str], FeatureContract]:
    """
    Train-only feature selection:
      - leakage name filter
      - missing-rate filter
      - low-variance filter
      - redundancy drop (corr only for de-dup)
      - MI + RF importance scoring on y(t+1)
      - freeze order and save train medians for imputation at predict-time
    """
    inf_cfg = _default_cfg_get(config, ["model_config", "Informer"], {}) or {}
    fs_cfg = dict(inf_cfg.get("feature_selection") or {})
    seed = int(_default_cfg_get(config, ["default", "seed"], 42) or 42)

    missing_thr = float(fs_cfg.get("missing_rate_threshold", 0.4))
    var_thr = float(fs_cfg.get("low_variance_threshold", 1e-8))
    redundant_thr = float(fs_cfg.get("redundant_corr_threshold", 0.95))
    max_features = fs_cfg.get("max_features", None)
    try:
        max_features_i = int(max_features) if max_features is not None else None
    except Exception:
        max_features_i = None

    leakage_patterns = list(fs_cfg.get("leakage_name_patterns") or ["label", "target", "future", "t+", "lead", "yhat", "predict"])

    # --- 1) Candidate normalization ---
    cand = [c for c in candidate_cols if c != time_col]
    cand = [value_col] + [c for c in cand if c != value_col]
    cand = [c for c in cand if c in train_df.columns]

    # safe defaults are time-derived features (if present) -> Repairable Core by default
    safe_defaults = [c for c in (fs_cfg.get("repairable_core_cols") or fs_cfg.get("safe_default_cols") or SAFE_TIME_FEATURES_DEFAULT) if c in train_df.columns]

    report: Dict[str, Any] = {
        "input_candidates": list(candidate_cols),
        "normalized_candidates": list(cand),
        "removed": [],
        "scores": {},
        "redundancy": {},
        "params": {
            "missing_rate_threshold": missing_thr,
            "low_variance_threshold": var_thr,
            "redundant_corr_threshold": redundant_thr,
            "max_features": max_features_i,
            "selector": "mi+rf",
        },
    }

    # --- 2) Leakage guardrails (name-based, hard drop) ---
    safe_cols = []
    for c in cand:
        if c == value_col:
            safe_cols.append(c)
            continue
        if _leakage_name_hit(c, leakage_patterns):
            report["removed"].append({"col": c, "reason": "leakage_name"})
            continue
        safe_cols.append(c)

    # --- 3) Train-only missing-rate filter ---
    df_num = _as_float_frame(train_df, safe_cols)
    miss_rate = df_num.isna().mean(axis=0)
    keep = []
    for c in safe_cols:
        if c == value_col:
            keep.append(c)
            continue
        if float(miss_rate.get(c, 0.0)) > missing_thr:
            report["removed"].append({"col": c, "reason": "missing_rate", "missing_rate": float(miss_rate.get(c, 0.0))})
            continue
        keep.append(c)

    # --- 4) Low-variance filter (train-only) ---
    df_k = _as_float_frame(train_df, keep)
    var = df_k.var(axis=0, ddof=0, numeric_only=False)
    keep2 = []
    for c in keep:
        if c == value_col:
            keep2.append(c)
            continue
        v = float(var.get(c, np.nan))
        if not np.isfinite(v) or v <= var_thr:
            report["removed"].append({"col": c, "reason": "low_variance", "variance": v})
            continue
        keep2.append(c)

    # --- 5) Redundancy removal (corr only for de-dup, train-only) ---
    keep3, corr_rep = _corr_drop_redundant(train_df, keep2, value_col=value_col, threshold=redundant_thr)
    report["redundancy"] = corr_rep

    # --- 6) MI + RF importance on y(t+1) ---
    feats = [c for c in keep3 if c != value_col]
    if not feats:
        selected = [value_col] + [c for c in safe_defaults if c != value_col and c in keep3]
    else:
        df_sel = _as_float_frame(train_df, [value_col] + feats)
        y_next = pd.to_numeric(df_sel[value_col], errors="coerce").shift(-1)
        X0 = df_sel[feats].copy()
        # align rows (drop last, and any y_next NaN)
        mask = y_next.notna()
        X = X0.loc[mask].to_numpy(dtype=np.float32)
        y = y_next.loc[mask].to_numpy(dtype=np.float32)

        mi = _compute_mi_scores(X, y, seed=seed)
        rf = _compute_rf_importance(X, y, seed=seed)

        # Normalize and combine (robust to None)
        scores = np.zeros(len(feats), dtype=float)
        if mi is not None and len(mi) == len(feats):
            mi_n = mi / (np.nanmax(mi) + 1e-12)
            scores += np.nan_to_num(mi_n, nan=0.0, posinf=0.0, neginf=0.0)
            report["scores"]["mi"] = {feats[i]: float(mi[i]) for i in range(len(feats))}
        else:
            report["scores"]["mi"] = None
        if rf is not None and len(rf) == len(feats):
            rf_n = rf / (np.nanmax(rf) + 1e-12)
            scores += np.nan_to_num(rf_n, nan=0.0, posinf=0.0, neginf=0.0)
            report["scores"]["rf_importance"] = {feats[i]: float(rf[i]) for i in range(len(feats))}
        else:
            report["scores"]["rf_importance"] = None

        order = np.argsort(-scores)
        ranked = [feats[i] for i in order.tolist()]
        report["scores"]["combined_rank"] = ranked

        # Always keep safe default time features if they exist and survived filters
        safe_keep = [c for c in safe_defaults if c in keep3 and c != value_col]
        ranked = [c for c in ranked if c not in safe_keep]

        selected_feats = safe_keep + ranked
        if max_features_i is not None and max_features_i > 0:
            # reserve 1 slot for value_col
            selected_feats = selected_feats[: max(0, max_features_i - 1)]

        selected = [value_col] + [c for c in selected_feats if c != value_col]

    # Keep only columns that actually exist in train_df (selection can include safe defaults that are absent)
    selected = [c for c in selected if c in train_df.columns]
    if selected and selected[0] != value_col and value_col in selected:
        selected = [value_col] + [c for c in selected if c != value_col]
    if not selected:
        selected = [value_col]

    # Identify recomputable cols by name pattern (for predict-time policy)
    recomputable_cols: List[str] = []
    recompute_patterns = list(fs_cfg.get("recomputable_name_regex") or [])
    if not recompute_patterns:
        recompute_patterns = [
            r"^(?:(?P<base>[A-Za-z0-9_]+)_)?rolling_(mean|std|min|max|median)_(?P<win>\d+)$",
            r"^(?:(?P<base>[A-Za-z0-9_]+)_)?lag_(?P<k>\d+)$",
            r"^(?:(?P<base>[A-Za-z0-9_]+)_)?diff_(?P<k>\d+)$",
        ]
    compiled = [re.compile(p) for p in recompute_patterns]
    for c in selected:
        if c == value_col:
            continue
        for rx in compiled:
            if rx.match(str(c)):
                recomputable_cols.append(str(c))
                break

    # Required Core (explicit) + Repairable Core (calendar + recomputable)
    required_extra = list(fs_cfg.get("required_core_cols") or fs_cfg.get("core_cols") or [])
    required_core_cols = [value_col] + [c for c in required_extra if c in selected and c != value_col]
    repairable_core_cols = [c for c in safe_defaults if c in selected and c != value_col] + [c for c in recomputable_cols if c in selected and c != value_col]
    # Backward-compat: core_cols keeps only required core (fail-fast)
    core_cols = list(required_core_cols)
    optional_cols = [c for c in selected if c not in required_core_cols and c not in repairable_core_cols and c != value_col]

    # Train medians for imputation at predict-time (exclude core cols; core missing => error)
    train_median: Dict[str, float] = {}
    try:
        df_med = _as_float_frame(train_df, selected)
        for c in selected:
            if c in core_cols:
                continue
            m = float(np.nanmedian(df_med[c].to_numpy(dtype=float)))
            if np.isfinite(m):
                train_median[c] = m
    except Exception:
        train_median = {}

    contract = FeatureContract(
        version=1,
        time_col=str(time_col),
        value_col=str(value_col),
        feature_cols=list(selected),
        required_core_cols=list(required_core_cols),
        repairable_core_cols=list(repairable_core_cols),
        optional_cols=list(optional_cols),
        safe_default_cols=list(safe_defaults),
        recomputable_cols=list(recomputable_cols),
        core_cols=list(core_cols),
        train_median=train_median,
        selection_report=report,
    )
    return list(selected), contract
