import numpy as np
import pandas as pd
import streamlit as st

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
    online_click = st.button("Predict only (online rolling)", type="secondary")
    return horizon_days, step_mode, allow_degrade, online_click


def run_online_inference(
    *,
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    feature_cols,
    device_choice: str,
    allow_degrade: bool,
    horizon_days: int,
    step_mode: str,
    mime_csv: str,
    torch_module=None,
):
    config_pred = {
        "device": device_choice,
        "default": {
            "time_col": time_col,
            "value_col": value_col,
            "device": device_choice,
            "dtype": "float32",
        },
        "model_config": {
            "Informer": {
                "seq_len": 96,
                "label_len": 48,
                "pred_len": 24,
                "feature_cols": feature_cols,
            }
        },
        "artifacts": {
            "model_path": "artifacts/informer_model.pth",
            "scaler_path": "artifacts/scaler.pkl",
            "residual_model_path": "artifacts/residual_model.pkl",
            "feature_cols_path": "artifacts/feature_cols.json",
        },
        "prediction": {
            "rolling": {
                "enabled": True,
                "step": None,
                "mode": "overwrite",
            },
            "degrade": {"enabled": bool(allow_degrade), "mode": "naive_last"},
        },
    }

    st.caption(f"Feature columns (fixed order for train/predict): {feature_cols}")

    horizon_steps = int(24 * horizon_days)
    step_val = None if step_mode.startswith("Block") else 1

    try:
        _torch = torch_module
        if _torch is None:
            try:
                import torch as _t  # type: ignore
                _torch = _t
            except Exception:
                _torch = None
        if _torch is None:
            raise RuntimeError("Torch is not installed; cannot load InformerPredictor. Install: `pip install torch`")
        from models.informer.predict import InformerPredictor  # lazy import (requires torch)

        predictor = InformerPredictor(config_pred)
    except Exception as e:
        st.error("Failed to load trained model (train once first, or check dependencies/paths).")
        st.exception(e)
        st.stop()

    with st.spinner("Running online rolling inference..."):
        try:
            merged = predictor.rolling_predict(df.copy(), horizon=horizon_steps, step=step_val, mode="overwrite")
        except Exception as e:
            st.error("Online rolling inference failed.")
            st.exception(e)
            st.stop()

    dblk = (config_pred.get("data", {}) or {})
    if bool(dblk.get("degraded", False)):
        missing_req = dblk.get("missing_required_core")
        dropped_opt = dblk.get("dropped_optional_features")
        reason = dblk.get("degraded_reason", "unknown")
        mode = dblk.get("degraded_mode", "baseline")
        st.error(
            "⚠️ Prediction is degraded (degraded=True): features are missing/mismatched; "
            "switched to a baseline predictor. Do not interpret as normal results."
        )
        st.caption(f"degraded_reason={reason} | degraded_mode={mode}")
        if missing_req:
            st.caption(f"Missing Required Core: {missing_req}")
        if dropped_opt:
            st.caption(f"Dropped optional features: {dropped_opt}")
        if dblk.get("degraded_error"):
            st.caption(f"Error: {dblk.get('degraded_error')}")

    merged = np.asarray(merged).reshape(-1)
    mask = ~np.isnan(merged)
    if mask.sum() == 0:
        st.warning("No valid prediction region (data too short or parameters mismatch).")
        return

    y_true = pd.to_numeric(df.loc[mask, value_col], errors="coerce").to_numpy()
    y_hat = merged[mask]
    rmse = float(np.sqrt(np.nanmean((y_hat - y_true) ** 2)))
    denom = np.where(y_true == 0, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs((y_hat - y_true) / denom)) * 100.0)

    st.markdown("<div class='tsf-card'>", unsafe_allow_html=True)
    st.markdown("### ⚡ Online rolling inference — Metrics", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Online RMSE", f"{rmse:.4f}")
    with c2:
        st.metric("Online MAPE", f"{mape:.2f}%")

    online_long = {
        "timestamps": pd.to_datetime(df[time_col]).astype(str).tolist(),
        "y_true": pd.to_numeric(df[value_col], errors="coerce").astype(float).tolist(),
        "yhat": merged.astype(float).tolist(),
        "degraded": bool(dblk.get("degraded", False)),
        "degraded_reason": dblk.get("degraded_reason"),
        "degraded_mode": dblk.get("degraded_mode"),
        "missing_required_core": dblk.get("missing_required_core"),
        "dropped_optional_features": dblk.get("dropped_optional_features"),
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
        st.dataframe(view_df, use_container_width=True)

    with st.expander("🧾 Online details (full)", expanded=False):
        full_df = df_from_long(online_long, time_col)
        if bool(dblk.get("degraded", False)):
            full_df["degraded"] = True
            full_df["degraded_reason"] = str(dblk.get("degraded_reason"))
            full_df["degraded_mode"] = str(dblk.get("degraded_mode"))
            full_df["missing_required_core"] = str(dblk.get("missing_required_core"))
            full_df["dropped_optional_features"] = str(dblk.get("dropped_optional_features"))
        st.dataframe(full_df, use_container_width=True)
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
