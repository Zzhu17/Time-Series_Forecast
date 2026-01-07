import pandas as pd
import streamlit as st

from ui.api_client import predict_batch


MODEL_OPTIONS = ["Informer", "ARIMA", "Prophet", "RandomForest", "LSTM", "XGBoost"]


def render_controls():
    col_a, col_b, col_c = st.columns([2, 1, 2])
    with col_a:
        model_name = st.selectbox("Predict model", MODEL_OPTIONS, index=0, key="predict_model_name")
    with col_b:
        horizon = st.number_input("Forecast horizon", min_value=1, max_value=365, value=24, step=1)
    with col_c:
        allow_degrade = st.checkbox(
            "Allow degrade (fallback to baseline if Required Core is missing)",
            value=False,
            key="predict_allow_degrade",
        )
    run_click = st.button("Predict only", type="secondary")
    return model_name, int(horizon), bool(allow_degrade), run_click


def run_predict_only(
    *,
    api_url: str,
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    model_name: str,
    horizon: int,
    allow_degrade: bool,
    results_container,
):
    payload = {
        "model_name": model_name,
        "time_col": time_col,
        "value_col": value_col,
        "horizon": int(horizon),
        "rows": df.to_dict(orient="records"),
        "allow_degrade": bool(allow_degrade),
    }
    with st.spinner("Calling API for prediction..."):
        try:
            resp = predict_batch(api_url=api_url, payload=payload)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

    if resp.get("status") not in ("ok", "success"):
        st.error(resp.get("message", "Prediction failed."))
        return

    preds = resp.get("predictions") or []
    degraded = bool(resp.get("degraded", False))
    reason = resp.get("reason")

    with results_container.container():
        st.markdown("<div class='tsf-card'>", unsafe_allow_html=True)
        st.markdown("### 🔮 Predict-only result", unsafe_allow_html=True)
        if degraded:
            st.error("⚠️ Prediction is degraded (fallback baseline used).")
            if reason:
                st.caption(f"reason={reason}")
        st.caption(f"Model: {resp.get('used_model')} | Horizon: {len(preds)}")
        out_df = pd.DataFrame({"step": list(range(1, len(preds) + 1)), "prediction": preds})
        st.dataframe(out_df, width="stretch")
        try:
            st.download_button(
                label="Download predictions CSV",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception:
            pass
        st.markdown("</div>", unsafe_allow_html=True)
