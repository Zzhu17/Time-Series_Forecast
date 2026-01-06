from typing import Optional, List

import pandas as pd
import streamlit as st

import services.snapshot as snapshot_mod
from services.pipeline_loader import load_pipeline_module
from services.snapshot import cacheable_results, reset_snapshot
from ui.model_config import load_xgboost_hparams_from_configs_yaml
from ui.results import render_cached_summary


def render_train_button():
    return st.button("Train & Predict", type="primary")


def run_training_and_prediction(
    *,
    df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
    residual_learner: str,
    device_choice: str,
    time_col: str,
    value_col: str,
    allow_degrade: bool,
    uploaded_name: Optional[str],
    results_container,
):
    """Trigger training/prediction pipeline and render results."""
    try:
        results_container.empty()
    except Exception:
        pass

    st.session_state["is_training"] = True
    _proj_dir = snapshot_mod.PROJECT_DIR
    _art_dir = _proj_dir / "artifacts"
    _out_dir = _proj_dir / "output"
    try:
        _art_dir.mkdir(parents=True, exist_ok=True)
        _out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    config = {
        "model": {"name": model_name},
        "default": {
            "time_col": time_col,
            "value_col": value_col,
            "device": device_choice,
            "dtype": "float32",
        },
        "visualization": {
            "pipeline_plot": False,
            "build_continuous": False,
        },
        "target_transform": {
            "enabled": model_name in ("informer", "lstm"),
            "method": "log1p",
        },
        "post_calibration": {
            "enabled": True,
            "a_clip": [0.8, 1.2],
            "b_clip_ratio": 0.1,
            "ridge": 1e-6,
            "mape_guard_rel": 1.02,
        },
        "model_config": {
            "Informer": {
                "seq_len": 96,
                "label_len": 48,
                "pred_len": 8,
                "auto_feature_cols": True,
                "lock_feature_order": True,
                "feature_cols": feature_cols,
                "feature_selection": {
                    "missing_rate_threshold": 0.4,
                    "low_variance_threshold": 1e-8,
                    "redundant_corr_threshold": 0.95,
                    "max_features": None,
                    "leakage_name_patterns": ["label", "target", "future", "t+", "lead", "yhat", "predict"],
                    "safe_default_cols": ["month", "day_of_month", "day_of_week", "hour", "day_of_year"],
                    "required_core_cols": [],
                    "repairable_core_cols": ["month", "day_of_month", "day_of_week", "hour", "day_of_year"],
                    "core_cols": [],
                },
            }
        },
        "artifacts": {
            "model_path": str(_art_dir / "informer_model.pth"),
            "scaler_path": str(_art_dir / "scaler.pkl"),
            "residual_model_path": str(_art_dir / "residual_model.pkl"),
            "y_scaler_path": str(_art_dir / "value_scaler.pkl"),
            "feature_cols_path": str(_art_dir / "feature_cols.json"),
            "feature_report_path": str(_art_dir / "feature_report.json"),
        }
    }

    _res_choice = str(residual_learner or "none").strip().lower()
    if _res_choice in ("linear", "xgboost"):
        rm_cfg = {
            "enabled": True,
            "model_type": "ridge" if _res_choice == "linear" else "xgboost",
        }
        config["residual_modeling"] = rm_cfg
        if model_name == "informer":
            try:
                config.setdefault("model_config", {}).setdefault("Informer", {})["use_residual"] = False
            except Exception:
                pass
            try:
                config.setdefault("post_calibration", {})["enabled"] = False
            except Exception:
                pass

    if model_name == "xgboost" or _res_choice == "xgboost":
        try:
            xgb_hp = load_xgboost_hparams_from_configs_yaml()
            if isinstance(xgb_hp, dict) and xgb_hp:
                config.setdefault("model_config", {})["XGBoost"] = xgb_hp
        except Exception:
            pass
    try:
        if model_name == "xgboost":
            config.setdefault("artifacts", {})["xgboost_model_path"] = str(_art_dir / "xgboost_model.json")
    except Exception:
        pass
    try:
        if _res_choice == "xgboost":
            config.setdefault("artifacts", {})["xgboost_residual_model_path"] = str(_art_dir / "xgboost_residual_model.json")
    except Exception:
        pass

    config["device"] = device_choice
    config.setdefault("callbacks", {})
    config.setdefault("model", {})["name"] = model_name
    config["model_type"] = model_name
    config.setdefault("data", {})
    config["data"]["dataframe"] = df.copy()

    _status = st.empty()
    _bar = st.progress(0.0)

    def _set_progress(pct: float, msg: str):
        try:
            pct = float(pct)
        except Exception:
            pct = 0.0
        pct = 0.0 if pct < 0 else (1.0 if pct > 1 else pct)
        try:
            _bar.progress(pct)
        except Exception:
            try:
                _bar.progress(int(pct * 100))
            except Exception:
                pass
        try:
            _status.info(msg)
        except Exception:
            pass

    def _progress_cb(stage: str = "pipeline", pct: Optional[float] = None, **info):
        try:
            if stage == "train":
                return
            if pct is None:
                return
            _set_progress(float(pct), f"{stage}: {info.get('msg') or ''}".strip())
        except Exception:
            pass

    config["callbacks"]["progress"] = _progress_cb

    try:
        reset_snapshot()
    except Exception:
        pass
    try:
        for _k in ("last_results", "last_meta", "last_results_source"):
            st.session_state.pop(_k, None)
    except Exception:
        pass

    _set_progress(0.02, "Starting pipeline...")
    try:
        pipeline_mod = load_pipeline_module()
        results = pipeline_mod.run_pipeline_and_update_state(
            df=df.copy(),
            config=config,
            feature_cols=feature_cols,
            uploaded_name=uploaded_name,
            model_name=model_name,
            time_col=time_col,
            value_col=value_col,
            allow_degrade=bool(allow_degrade),
            progress_cb=_progress_cb,
        )
        _set_progress(0.99, "Rendering UI...")
    except Exception as e:
        st.error("Pipeline failed.")
        st.exception(e)
        st.stop()
    finally:
        try:
            st.session_state["is_training"] = False
        except Exception:
            pass

    status = results.get("status", "error")
    if status not in ("ok", "success"):
        st.error(results.get("message", "Training/prediction failed."))
        tb = results.get("traceback")
        if tb:
            st.code(tb)
        st.stop()

    try:
        cached = st.session_state.get("last_results")
        if not isinstance(cached, dict):
            cached = cacheable_results(results)
        meta = st.session_state.get("last_meta") if isinstance(st.session_state.get("last_meta"), dict) else {}
        with results_container.container():
            render_cached_summary(
                cached,
                model_name=str(meta.get("model_name") or model_name),  # type: ignore
                time_col=str(meta.get("time_col") or time_col),  # type: ignore
                value_col=str(meta.get("value_col") or value_col),  # type: ignore
            )
    except Exception as _e:
        st.error(f"Failed to render results: {_e}")
    finally:
        st.session_state["is_training"] = False
    st.stop()
