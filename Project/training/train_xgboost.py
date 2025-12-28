from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from models.xgboost import build_xgboost_regressor
from utils.array_utils import clean_and_unify_arrays


def _split_lengths(n: int, cfg: dict) -> Tuple[int, int, int]:
    data_blk = (cfg.get("data") or {}) if isinstance(cfg, dict) else {}
    split = data_blk.get("split") if isinstance(data_blk, dict) else None
    if isinstance(split, dict):
        try:
            t = int(split.get("train_len", 0))
            v = int(split.get("val_len", 0))
            te = int(split.get("test_len", 0))
            if t > 0 and v >= 0 and te >= 0 and (t + v + te) <= n + 1:
                if t + v + te != n:
                    te = max(0, n - t - v)
                return t, v, te
        except Exception:
            pass
    t = int(n * 0.6)
    v = int(n * 0.2)
    te = max(0, n - t - v)
    return t, v, te


def _as_float_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c not in df.columns:
            out[c] = np.nan
        else:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def train_xgboost_model(df: pd.DataFrame, config: dict):
    """
    XGBoost trainer aligned with the unified pipeline/app expectations:
      - Uses 6/2/2 time split (or split injected via config['data']['split'])
      - Multi-feature inputs, single-target output (value_col)
      - Returns unified 7-tuple and writes val/test dense frames into config['data']
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("train_xgboost_model expects a pandas DataFrame as input.")

    dft = config.get("default", {}) or {}
    time_col = config.get("time_col", dft.get("time_col", "date"))
    value_col = config.get("value_col", dft.get("value_col", "value"))
    data_blk = config.setdefault("data", {})
    artifacts = config.setdefault("artifacts", {})

    # Feature list: prefer pipeline-provided final list
    all_feature_cols = (
        data_blk.get("all_feature_cols")
        or (artifacts.get("feature_cols") if isinstance(artifacts.get("feature_cols"), (list, tuple)) else None)
        or []
    )
    if not isinstance(all_feature_cols, (list, tuple)) or not all_feature_cols:
        # Fallback: numeric columns except time_col
        numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != time_col]
        all_feature_cols = [value_col] + [c for c in numeric_cols if c != value_col]
    all_feature_cols = [str(value_col)] + [str(c) for c in all_feature_cols if c and str(c) not in (value_col, time_col)]
    data_blk["all_feature_cols"] = list(all_feature_cols)
    artifacts["feature_cols"] = list(all_feature_cols)

    # Sort by time if possible (pipeline usually already did this)
    work = df.copy()
    if time_col in work.columns:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce", utc=True)
        try:
            work[time_col] = work[time_col].dt.tz_localize(None)
        except Exception:
            pass
        work = work.sort_values(time_col).reset_index(drop=True)

    n = int(len(work))
    t_len, v_len, te_len = _split_lengths(n, config)
    data_blk["split"] = {"train_len": int(t_len), "val_len": int(v_len), "test_len": int(te_len)}

    # Build timestamps for plotting (val/test)
    ts = None
    if time_col in work.columns:
        ts = pd.to_datetime(work[time_col], errors="coerce", utc=True)
        try:
            ts = ts.dt.tz_localize(None)
        except Exception:
            pass
    if ts is not None and len(ts) == n:
        data_blk["val_timestamps"] = ts.iloc[t_len : t_len + v_len].tolist()
        data_blk["test_timestamps"] = ts.iloc[t_len + v_len : t_len + v_len + te_len].tolist()

    # Target
    y_all = pd.to_numeric(work[value_col], errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(y_all).any():
        raise ValueError(f"Target column '{value_col}' has no usable numeric values.")

    # Features exclude target itself
    feature_cols = [c for c in all_feature_cols if c != value_col]
    if not feature_cols:
        # No exogenous features -> baseline persistence (still returns valid dense frames)
        yhat = np.roll(y_all.astype(float), 1)
        yhat[0] = float(np.nan)
        val_true = y_all[t_len : t_len + v_len]
        val_pred = yhat[t_len : t_len + v_len]
        test_true = y_all[t_len + v_len : t_len + v_len + te_len]
        test_pred = yhat[t_len + v_len : t_len + v_len + te_len]
        val_true_u, val_pred_u, _ = clean_and_unify_arrays(val_true, val_pred)
        test_true_u, test_pred_u, _ = clean_and_unify_arrays(test_true, test_pred)
        data_blk["val_dense"] = pd.DataFrame({"y_true": val_true_u, "yhat": val_pred_u})
        data_blk["test_dense"] = pd.DataFrame({"y_true": test_true_u, "yhat": test_pred_u})
        params = {"model": "xgboost", "mode": "baseline_persistence", "features": []}
        return val_true_u, val_pred_u, test_true_u, test_pred_u, None, None, params

    X_all = _as_float_frame(work, feature_cols).to_numpy(dtype=np.float32)

    # Split
    X_train = X_all[:t_len]
    y_train = y_all[:t_len]
    X_val = X_all[t_len : t_len + v_len]
    y_val = y_all[t_len : t_len + v_len]
    X_test = X_all[t_len + v_len : t_len + v_len + te_len]
    y_test = y_all[t_len + v_len : t_len + v_len + te_len]

    # Hyperparameters
    mcfg = (config.get("model_config") or {}).get("XGBoost", {}) or {}
    early_stopping_rounds = int(mcfg.get("early_stopping_rounds", 50))
    model = build_xgboost_regressor(config)

    eval_set = []
    if X_val is not None and len(X_val) > 0 and np.isfinite(y_val).any():
        eval_set = [(X_val, y_val)]

    # Fit with version-compatible early stopping:
    # - Some XGBoost versions accept `early_stopping_rounds` in .fit(...)
    # - Others require callbacks (xgb.callback.EarlyStopping)
    import inspect

    fit_kwargs: Dict[str, Any] = {}
    try:
        sig = inspect.signature(model.fit)
        fit_params = sig.parameters
    except Exception:
        fit_params = {}

    if eval_set and "eval_set" in fit_params:
        fit_kwargs["eval_set"] = eval_set
    if "verbose" in fit_params:
        fit_kwargs["verbose"] = False

    if eval_set and int(early_stopping_rounds) > 0:
        es = max(1, int(early_stopping_rounds))
        if "early_stopping_rounds" in fit_params:
            fit_kwargs["early_stopping_rounds"] = es
        elif "callbacks" in fit_params:
            try:
                import xgboost as xgb  # type: ignore

                fit_kwargs["callbacks"] = [xgb.callback.EarlyStopping(rounds=es, save_best=True)]
            except Exception:
                pass

    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        # Retry with minimal kwargs in case a version rejects some keys.
        minimal: Dict[str, Any] = {}
        if eval_set and "eval_set" in fit_params:
            minimal["eval_set"] = eval_set
        if "verbose" in fit_params:
            minimal["verbose"] = False
        model.fit(X_train, y_train, **minimal)

    # Predict
    val_pred = model.predict(X_val).astype(np.float32) if len(X_val) else np.array([], dtype=np.float32)
    test_pred = model.predict(X_test).astype(np.float32) if len(X_test) else np.array([], dtype=np.float32)

    # Align/clean
    val_true_u, val_pred_u, _ = clean_and_unify_arrays(y_val, val_pred)
    test_true_u, test_pred_u, _ = clean_and_unify_arrays(y_test, test_pred)

    # Dense frames for pipeline/app (timestamps are attached later in pipeline)
    data_blk["val_dense"] = pd.DataFrame({"y_true": val_true_u, "yhat": val_pred_u})
    data_blk["test_dense"] = pd.DataFrame({"y_true": test_true_u, "yhat": test_pred_u})

    # Artifacts: save model if configured
    model_path = (config.get("artifacts") or {}).get("xgboost_model_path") or (config.get("artifacts") or {}).get("model_path")
    if isinstance(model_path, str) and model_path:
        try:
            model.save_model(model_path)
            artifacts["xgboost_model_path"] = model_path
        except Exception:
            pass

    params: Dict[str, Any] = {
        "model": "xgboost",
        "feature_cols": list(feature_cols),
        "early_stopping_rounds": early_stopping_rounds,
    }
    # Echo key hparams for debugging/traceability
    for k in (
        "n_estimators",
        "learning_rate",
        "max_depth",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "min_child_weight",
        "gamma",
        "tree_method",
        "n_jobs",
    ):
        if k in mcfg:
            params[k] = mcfg.get(k)
    try:
        params["best_iteration"] = int(getattr(model, "best_iteration", -1))
    except Exception:
        pass

    return val_true_u, val_pred_u, test_true_u, test_pred_u, model, None, params
