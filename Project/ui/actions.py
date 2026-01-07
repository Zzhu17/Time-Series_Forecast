from typing import Optional, List

import pandas as pd
import streamlit as st

from services.snapshot import reset_snapshot
from ui.api_client import train_file_streamlit


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
    api_url: str,
    uploaded_bytes: bytes,
):
    """Trigger training/prediction via API and cache results."""
    try:
        results_container.empty()
    except Exception:
        pass

    st.session_state["is_training"] = True

    try:
        reset_snapshot()
    except Exception:
        pass
    try:
        for _k in ("last_results", "last_meta", "last_results_source"):
            st.session_state.pop(_k, None)
    except Exception:
        pass

    residual_cfg = _build_residual_config(residual_learner)

    with st.spinner("Calling API for training..."):
        try:
            results = train_file_streamlit(
                api_url=api_url,
                file_bytes=uploaded_bytes,
                filename=uploaded_name or "data.csv",
                model_name=model_name,
                time_col=time_col,
                value_col=value_col,
                horizon=24,
                feature_cols=feature_cols,
                residual_modeling=residual_cfg,
                allow_degrade=allow_degrade,
                device=device_choice,
            )
        except Exception as e:
            st.session_state["is_training"] = False
            st.error("Training failed.")
            st.exception(e)
            return

    st.session_state["last_results"] = results
    st.session_state["last_meta"] = {
        "uploaded_name": uploaded_name,
        "model_name": model_name,
        "time_col": time_col,
        "value_col": value_col,
    }
    st.session_state["last_results_source"] = "api"
    st.session_state["is_training"] = False
    try:
        if results.get("status") not in ("ok", "success"):
            st.error(results.get("message", "Training failed."))
        else:
            st.success("Training complete.")
    except Exception:
        pass


def _build_residual_config(residual_learner: str) -> Optional[dict]:
    choice = str(residual_learner or "none").strip().lower()
    if choice == "linear":
        return {"enabled": True, "model_type": "ridge"}
    if choice == "xgboost":
        return {"enabled": True, "model_type": "xgboost"}
    return None
