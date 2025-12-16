from typing import Dict, Any, Tuple
import pandas as pd
import torch
import numpy as np
import random
import os
from models.registry import TRAINER_REGISTRY


def run_training(model_name: str, config: Dict[str, Any]) -> Tuple[Any, pd.DataFrame]:
    """
    统一训练入口（严格版）：
    - 仅通过 TRAINER_REGISTRY 派发
    - 只接受 (df, config) 形参
    - 只接受 7 元组返回：
      (val_true, val_forecast, test_true, test_forecast, final_model, test_forecast_df, best_params)
    """
    model_name_lower = model_name.lower().replace(" ", "_").replace("-", "_")
    model_key = model_name_lower

    # === Device / Dtype fallback (prefer GPU, then config/default, else cpu) ===
    auto_device = "cuda" if torch.cuda.is_available() else (
        config.get("device") or config.get("default", {}).get("device") or "cpu"
    )
    auto_dtype = config.get("dtype") or config.get("default", {}).get("dtype") or "float32"

    # Write back so downstream (models/informer/train.py, forward.py, etc.) see consistent values
    config["device"] = auto_device
    config["dtype"] = auto_dtype

    # Debug: show final dispatch device/dtype
    print(f"[training.dispatch] device={auto_device}, dtype={auto_dtype}")

    # --- Rolling prediction config (log only; do not modify pred_len here) ---
    roll_cfg = (config.get("prediction", {}) or {}).get("rolling", {}) or {}
    roll_enabled = bool(roll_cfg.get("enabled", True))
    roll_step = roll_cfg.get("step", None)
    roll_mode = str(roll_cfg.get("mode", "overwrite"))
    print("[training.dispatch] rolling config:")
    print(f"  enabled={roll_enabled}, step={roll_step}, mode={roll_mode}")

    # UI horizon (if any) — only display; horizon to be applied by predict.rolling_predict
    ui_cfg = (config.get("prediction", {}) or {}).get("ui", {}) or {}
    horizons_hours = ui_cfg.get("horizons_hours", None)
    if horizons_hours is not None:
        print(f"  ui.horizons_hours={horizons_hours}")

    # Also mirror into model-specific section if present (non-destructive)
    try:
        if "model_config" in config and "Informer" in config["model_config"]:
            config["model_config"]["Informer"].setdefault("device", auto_device)
            config["model_config"]["Informer"].setdefault("dtype", auto_dtype)
    except Exception:
        pass

    # === 严格：从注册表取 trainer ===
    runner = TRAINER_REGISTRY.get(model_key)
    if runner is None:
        raise ValueError(
            f"Unsupported model '{model_name}'. Available: {list(TRAINER_REGISTRY.keys())}"
        )

    # === 严格：取 df，并以 (df, config) 调用 ===
    data_blk = config.get("data", {}) or {}
    df = data_blk.get("df") or data_blk.get("dataframe") or data_blk.get("raw_df")
    if df is None:
        raise ValueError("[training.dispatch] Missing config['data']['df'] for trainer input.")

    # === 严格：必须返回 7 元组 ===
    (val_true, val_pred, test_true, test_pred,
     final_model, test_forecast_df, best_params) = runner(df, config)

    # 组装 result_df（供本入口的下游复用）
    result_df = pd.concat(
        [
            pd.DataFrame({"phase": "val",  "y_true": val_true,  "yhat": val_pred}),
            pd.DataFrame({"phase": "test", "y_true": test_true, "yhat": test_pred}),
        ],
        ignore_index=True,
    )

    # 写回 artifacts（RF 面板固定键）
    arts = config.setdefault("artifacts", {})
    if best_params is not None:
        arts[f"{model_key}_params"] = best_params
        if model_key == "randomforest":
            arts["randomforest_params"] = best_params

    # Debug: show artifact paths if available
    model_path = arts.get("model_path")
    scaler_path = arts.get("scaler_path")
    resid_path  = arts.get("residual_model_path")
    print("[training.dispatch] artifacts:")
    print(f"  model_path={model_path}")
    print(f"  scaler_path={scaler_path}")
    print(f"  residual_model_path={resid_path}")

    # Data/feature write-backs for downstream consistency
    data_blk = config.setdefault("data", {})
    all_cols = data_blk.get("all_feature_cols")
    if all_cols is not None:
        try:
            print(f"[training.dispatch] all_feature_cols (len={len(all_cols)}): {list(all_cols)}")
        except Exception:
            print("[training.dispatch] all_feature_cols present (unprintable type)")
    else:
        print("[training.dispatch] all_feature_cols: <missing>")

    # Rolling evaluation artifacts (optional)
    val_result = data_blk.get("val_result_df")
    test_result = data_blk.get("test_result_df")
    if isinstance(val_result, pd.DataFrame):
        print(f"[training.dispatch] val_result_df: shape={val_result.shape}")
    else:
        print("[training.dispatch] val_result_df: <missing>")
    if isinstance(test_result, pd.DataFrame):
        print(f"[training.dispatch] test_result_df: shape={test_result.shape}")
    else:
        print("[training.dispatch] test_result_df: <missing>")

    # Scaler quick info (if available)
    scaler_obj = arts.get("scaler") or None
    if scaler_obj is None and scaler_path:
        # avoid loading here; just record the expected path
        print("[training.dispatch] scaler object not attached; will rely on scaler_path during predict.")
    else:
        try:
            n_in = getattr(scaler_obj, "n_features_in_", None)
            names = getattr(scaler_obj, "feature_names_in_", None)
            print(f"[training.dispatch] scaler.n_features_in_={n_in}")
            if names is not None:
                print(f"[training.dispatch] scaler.feature_names_in_={list(names)}")
        except Exception:
            pass

    return final_model, result_df