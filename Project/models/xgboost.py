from __future__ import annotations

from typing import Any, Dict


def build_xgboost_regressor(config: Dict[str, Any]):
    """
    Build an XGBoost regressor from config (no training here).

    Reads:
      config['model_config']['XGBoost'] (preferred)
      config['default']['seed'] (for random_state)
    """
    mcfg = (config.get("model_config") or {}).get("XGBoost", {}) or {}
    dft = config.get("default", {}) or {}

    n_estimators = int(mcfg.get("n_estimators", 2000))
    learning_rate = float(mcfg.get("learning_rate", 0.03))
    max_depth = int(mcfg.get("max_depth", 6))
    subsample = float(mcfg.get("subsample", 0.8))
    colsample_bytree = float(mcfg.get("colsample_bytree", 0.8))
    reg_lambda = float(mcfg.get("reg_lambda", 1.0))
    min_child_weight = float(mcfg.get("min_child_weight", 1.0))
    gamma = float(mcfg.get("gamma", 0.0))
    random_state = int(dft.get("seed", 42))
    tree_method = str(mcfg.get("tree_method", "hist"))
    n_jobs = int(mcfg.get("n_jobs", -1))

    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:
        raise RuntimeError("xgboost is not installed. Install with: `pip install xgboost`.") from e

    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=random_state,
        tree_method=tree_method,
        n_jobs=n_jobs,
    )

