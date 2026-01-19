import numpy as np
import pandas as pd
import streamlit as st

from ui.api_client import file_hash, list_model_registry, predict_online_file_cached
from visualizations.plot import df_from_long, render_true_pred


def render_controls():
    col_r1, col_r2, col_r3 = st.columns([1, 1, 2])
    with col_r1:
        horizon_days = st.selectbox("Online forecast horizon (days)", [1, 3, 7], index=0)
    with col_r2:
        step_mode = st.selectbox("Rolling step", ["Block step (= horizon)", "Step-by-step (= 1)"], index=0)
    with col_r3:
        st.caption("Block step is faster with no error accumulation; step-by-step is smoother but slower and accumulates errors.")
        allow_degrade = st.checkbox("Allow degrade (fallback to baseline if Required Core is missing)", value=False)
        st.caption("When enabled: online inference falls back to a baseline if required features are missing; training/eval may also return baseline with degraded=True.")
        online_model_name = st.selectbox(
            "Online model",
            ["Informer", "ARIMA", "Prophet", "RandomForest", "LSTM", "XGBoost"],
            index=0,
            key="online_model_name",
        )
    online_click = st.button("Predict only (online rolling)", type="secondary")
    return horizon_days, step_mode, allow_degrade, online_click, online_model_name


def render_model_version_selector(api_url: str, model_name: str) -> tuple[str | None, str | None]:
    if str(model_name).lower() != "informer":
        st.caption("Online inference model selection is available for Informer only.")
        return None, None

    try:
        models = list_model_registry(api_url)
    except Exception as e:
        st.caption(f"Model registry unavailable: {e}")
        return None, None
    if not models:
        st.caption("No registered models found yet.")
        return None, None

    filtered = [m for m in models if str(m.get("name", "")).lower() == str(model_name).lower()]
    if not filtered:
        st.caption(f"No registered models for {model_name}.")
        return None, None

    options = ["latest"]
    lookup = {"latest": (None, None)}
    for rec in filtered:
        mid = str(rec.get("id") or "")
        ver = str(rec.get("version") or "")
        stage = str(rec.get("stage") or "")
        label = f"{ver} | {mid[:8]} | {stage}"
        options.append(label)
        lookup[label] = (mid, ver)

    choice = st.selectbox("Online model version", options, index=0)
    return lookup.get(choice, (None, None))


def run_online_inference(
    *,
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    feature_cols,
    model_name: str,
    device_choice: str,
    allow_degrade: bool,
    horizon_days: int,
    step_mode: str,
    mime_csv: str,
    api_url: str,
    uploaded_bytes: bytes,
    uploaded_name: str,
    model_id: str | None,
    model_version: str | None,
):
    if str(model_name).lower() != "informer":
        st.info("Non-Informer online inference runs one-step rolling. Horizon/step settings may be ignored.")

    st.caption(f"Feature columns (fixed order for train/predict): {feature_cols}")
    horizon_steps = int(24 * horizon_days)
    step_val = None if step_mode.startswith("Block") else 1

    file_bytes = uploaded_bytes or df.to_csv(index=False).encode("utf-8")
    file_name = uploaded_name or "online.csv"
    with st.spinner("Calling API for online rolling inference..."):
        try:
            resp = predict_online_file_cached(
                api_url=api_url,
                file_hash_value=file_hash(file_bytes),
                file_bytes=file_bytes,
                filename=file_name,
                model_name=model_name,
                time_col=time_col,
                value_col=value_col,
                horizon_days=horizon_days,
                step_mode=step_mode,
                allow_degrade=allow_degrade,
                device=device_choice,
                model_id=model_id,
                model_version=model_version,
            )
        except Exception as e:
            st.error("Online rolling inference failed.")
            st.exception(e)
            st.stop()

    merged = np.asarray(resp.get("predictions") or []).reshape(-1)
    if merged.size != len(df):
        st.error("Online prediction length mismatch. Re-train the model and try again.")
        return
    degraded = bool(resp.get("degraded", False))
    reason = resp.get("degraded_reason", "unknown")
    mode = resp.get("degraded_mode", "baseline")
    missing_req = resp.get("missing_required_core")
    dropped_opt = resp.get("dropped_optional_features")

    if degraded:
        st.error(
            "⚠️ Prediction is degraded (degraded=True): features are missing/mismatched; "
            "switched to a baseline predictor. Do not interpret as normal results."
        )
        st.caption(f"degraded_reason={reason} | degraded_mode={mode}")
        if missing_req:
            st.caption(f"Missing Required Core: {missing_req}")
        if dropped_opt:
            st.caption(f"Dropped optional features: {dropped_opt}")
    mask = ~np.isnan(merged)
    if mask.sum() == 0:
        st.warning("No valid prediction region (data too short or parameters mismatch).")
        return

    y_true = pd.to_numeric(df.loc[mask, value_col], errors="coerce").to_numpy()
    y_hat = merged[mask]
    rmse = float(np.sqrt(np.nanmean((y_hat - y_true) ** 2)))
    std = float(np.nanstd(y_true)) + 1e-8
    nrmse = rmse / std if np.isfinite(std) and std > 1e-8 else float("nan")
    denom = np.abs(y_true) + np.abs(y_hat) + 1e-8
    smape = float(np.nanmean(2.0 * np.abs(y_hat - y_true) / denom)) * 100.0
    mean_abs = float(np.nanmean(np.abs(y_true)))
    tau = max(1e-8, 0.01 * mean_abs) if np.isfinite(mean_abs) and mean_abs > 0 else 1e-8
    mape_mask = np.abs(y_true) > tau
    if int(np.sum(mape_mask)) == 0:
        mape = float("nan")
    else:
        mape = float(np.nanmean(np.abs((y_hat[mape_mask] - y_true[mape_mask]) / (np.abs(y_true[mape_mask]) + 1e-8)))) * 100.0

    st.markdown("<div class='tsf-card'>", unsafe_allow_html=True)
    st.markdown("### ⚡ Online rolling inference — Metrics", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Online RMSE", f"{rmse:.4f}")
    with c2:
        st.metric("Online MAPE", f"{mape:.2f}%")
    with c3:
        st.metric("Online nRMSE", f"{nrmse:.4f}")
    with c4:
        st.metric("Online sMAPE", f"{smape:.2f}%")

    online_long = {
        "timestamps": pd.to_datetime(df[time_col]).astype(str).tolist(),
        "y_true": pd.to_numeric(df[value_col], errors="coerce").astype(float).tolist(),
        "yhat": merged.astype(float).tolist(),
        "degraded": degraded,
        "degraded_reason": reason,
        "degraded_mode": mode,
        "missing_required_core": missing_req,
        "dropped_optional_features": dropped_opt,
    }

    st.markdown("### 📈 Online rolling inference — Curve", unsafe_allow_html=True)
    online_plot_n = st.selectbox("Points to plot (last N)", ["1000", "2000", "4000", "ALL"], index=1, key="plot_n_online")
    online_marker_every = st.number_input("Marker every k points", min_value=20, max_value=2000, value=200, step=20, key="plot_marker_online")
    n_points = None if online_plot_n == "ALL" else int(online_plot_n)
    if online_plot_n == "ALL":
        st.warning("Rendering ALL points may be slow for large datasets.")
    df_series = pd.DataFrame(
        {
            "ts": pd.to_datetime(df[time_col], errors="coerce", utc=True),
            "true": pd.to_numeric(df[value_col], errors="coerce"),
            "pred": merged.astype(float),
        }
    )
    render_true_pred(
        df_series,
        title=f"Online Rolling Inference (H={horizon_steps}, step={'H' if step_val is None else step_val})",
        n_points=n_points,
        marker_every=int(online_marker_every),
    )

    with st.expander("🔎 Online details (last 200)", expanded=False):
        view_df = pd.DataFrame({
            time_col: online_long["timestamps"],
            "y_true": online_long["y_true"],
            "yhat": online_long["yhat"],
        }).tail(200)
        st.dataframe(view_df, width="stretch")

    with st.expander("🧾 Online details (full)", expanded=False):
        full_df = df_from_long(online_long, time_col)
        if degraded:
            full_df["degraded"] = True
            full_df["degraded_reason"] = str(reason)
            full_df["degraded_mode"] = str(mode)
            full_df["missing_required_core"] = str(missing_req)
            full_df["dropped_optional_features"] = str(dropped_opt)
        st.dataframe(full_df, width="stretch")
        try:
            st.download_button(
                label="Download online full details CSV",
                data=full_df.to_csv(index=False).encode('utf-8'),
                file_name="online_long.csv",
                mime=mime_csv,
            )
        except Exception:
            pass
    st.markdown("</div>", unsafe_allow_html=True)
