import os
import pandas as pd
import streamlit as st

from ui import actions, data_load, model_config, online, results, state, theme

# Disable pipeline-side Matplotlib plotting (can hang on macOS); the app renders plots itself.
os.environ["TSF_PIPELINE_PLOT"] = "0"
os.environ["TSF_BUILD_CONTINUOUS"] = "0"
os.environ["TSF_DEBUG_CONTINUOUS"] = "0"

MIME_CSV = "text/csv"

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore[assignment]


def main():
    theme.setup_page()
    theme.render_hero()
    state.init_state()

    with st.container():
        st.markdown("<div class='tsf-card'>", unsafe_allow_html=True)
        st.markdown("### Data & Model setup", unsafe_allow_html=True)
        st.markdown("<p class='section-note'>Upload your CSV, choose a preset or customize the model/residual pairing, then run training.</p>", unsafe_allow_html=True)

        uploaded = data_load.upload_csv()
        preset_name, preset_model, preset_residual = model_config.render_preset_selector()
        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            model_name, residual_learner = model_config.render_model_selectors(preset_model, preset_residual)
        with col_cfg2:
            device_choice, _mps_ok = model_config.render_device_selector()
            if model_name == "randomforest":
                st.caption("⚙️ RandomForest runs Optuna tuning on the validation split (n_trials from configs.yaml: optimization.n_trials).")
            if model_name == "lstm":
                st.caption("🧩 LSTM uses configs.yaml: model_config.LSTM (seq_len/hidden_dim/num_layers/n_epochs/learning_rate) and produces dense val/test predictions.")
            if model_name == "xgboost":
                st.caption("🌲 XGBoost uses configs.yaml: model_config.XGBoost and produces dense val/test predictions.")
            if residual_learner != "none":
                st.caption(f"🧪 Residual modeling enabled: {residual_learner}.")

        run_click = actions.render_train_button()
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='tsf-card'>", unsafe_allow_html=True)
        st.markdown("### Online rolling inference", unsafe_allow_html=True)
        st.markdown("<p class='section-note'>Run fast rolling forecasts without retraining; choose horizon and step.</p>", unsafe_allow_html=True)
        horizon_days, step_mode, allow_degrade, online_click = online.render_controls()
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is None:
        st.info("Upload a CSV file to begin.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    time_col, value_col, feature_cols, _profiles = data_load.select_columns(
        df, run_click=run_click, online_click=online_click
    )

    data_load.render_preview(df, time_col, value_col)
    results_container = st.empty()

    if online_click:
        online.run_online_inference(
            df=df.copy(),
            time_col=time_col,
            value_col=value_col,
            feature_cols=feature_cols,
            device_choice=device_choice,
            allow_degrade=allow_degrade,
            horizon_days=int(horizon_days),
            step_mode=step_mode,
            mime_csv=MIME_CSV,
            torch_module=torch,
        )

    if run_click:
        actions.run_training_and_prediction(
            df=df.copy(),
            feature_cols=feature_cols,
            model_name=model_name,
            residual_learner=residual_learner,
            device_choice=device_choice,
            time_col=time_col,
            value_col=value_col,
            allow_degrade=allow_degrade,
            uploaded_name=getattr(uploaded, "name", None),
            results_container=results_container,
        )

    results.render_cached_results(
        results_container,
        model_name=model_name,
        time_col=time_col,
        value_col=value_col,
    )

    if st.button("Clear cached results", type="secondary", key="clear_cached_results"):
        state.clear_cached_results()


if __name__ == "__main__":
    main()
