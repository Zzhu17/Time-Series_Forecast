import os
import pandas as pd
import numpy as np
from typing import Optional, Dict
import streamlit as st
import services.snapshot as snapshot_mod

from services.pipeline_loader import load_pipeline_module
from services.snapshot import (
    as_int,
    cacheable_results,
    load_last_results_json,
)
from visualizations.plot import df_from_long, render_true_pred, render_val_test

# Disable pipeline-side Matplotlib plotting (can hang on macOS); the app renders plots itself.
os.environ["TSF_PIPELINE_PLOT"] = "0"
os.environ["TSF_BUILD_CONTINUOUS"] = "0"
os.environ["TSF_DEBUG_CONTINUOUS"] = "0"

# ---- Module-level constants ----
MIME_CSV = "text/csv"

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore[assignment]

def _load_xgboost_hparams_from_configs_yaml() -> Optional[dict]:
    """
    Load model_config.XGBoost from `configs/configs.yaml` without requiring PyYAML.
    Only supports the simple scalar key/value block we use for XGBoost.
    """
    try:
        cfg_path = os.path.join(os.path.dirname(__file__), "configs", "configs.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return None

    in_model_config = False
    xgb_indent = None
    out: dict = {}

    def _strip_comment(s: str) -> str:
        # Good-enough: strip trailing comments starting with '#'
        # (configs.yaml doesn't use quoted strings with '#').
        return s.split("#", 1)[0].rstrip("\n")

    def _parse_scalar(v: str):
        v = v.strip()
        if v == "":
            return ""
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            return v[1:-1]
        vl = v.lower()
        if vl in ("true", "false"):
            return vl == "true"
        # number?
        try:
            if any(ch in v for ch in (".", "e", "E")) and v.replace(".", "", 1).replace("-", "", 1).replace("+", "", 1).replace("e", "", 1).replace("E", "", 1).isdigit():
                return float(v)
            if v.lstrip("+-").isdigit():
                return int(v)
        except Exception:
            pass
        return v

    for raw in lines:
        s = _strip_comment(raw)
        if not s.strip():
            continue
        indent = len(s) - len(s.lstrip(" "))
        txt = s.strip()

        if not in_model_config:
            if txt == "model_config:":
                in_model_config = True
            continue

        # leave model_config section
        if indent == 0 and ":" in txt:
            break

        if xgb_indent is None:
            if txt == "XGBoost:":
                xgb_indent = indent
            continue

        # leave XGBoost block
        if indent <= xgb_indent:
            break

        if ":" not in txt:
            continue
        k, v = txt.split(":", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        out[k] = _parse_scalar(v)

    return out or None

def _to_df_plot_blob(blob):
    if not isinstance(blob, dict):
        return None
    try:
        return pd.DataFrame({"ts": blob.get("ts") or [], "true": blob.get("true") or [], "pred": blob.get("pred") or []})
    except Exception:
        return None

def _render_cached_summary(results: dict, *, model_name: str, time_col: str, value_col: str):
    """
    Render a minimal-but-complete UI from cached results, so users still see outputs
    even if Streamlit reruns cancel the original run-click execution.
    """
    dblk = (results.get("data", {}) or {})
    if bool(dblk.get("degraded", False)):
        st.error(
            "⚠️ Results are degraded (degraded=True): Required Core features are missing/invalid; "
            "baseline predictions were returned. Do not interpret them as normal results."
        )
        st.caption(f"degraded_reason={dblk.get('degraded_reason')} | degraded_mode={dblk.get('degraded_mode')}")
        if dblk.get("missing_required_core"):
            st.caption(f"Missing Required Core: {dblk.get('missing_required_core')}")
        if dblk.get("dropped_optional_features"):
            st.caption(f"Dropped optional features: {dblk.get('dropped_optional_features')}")
        if dblk.get("degraded_error"):
            st.caption(f"Error: {dblk.get('degraded_error')}")

    metrics = results.get("metrics", {}) or {}
    val_metrics = metrics.get("validation", {}) or {}
    test_metrics = metrics.get("test", {}) or {}

    data_blob = results.get("data", {}) or {}
    split_info = data_blob.get("split", {}) or {}
    train_len = as_int(split_info.get("train_len"))
    val_len = as_int(split_info.get("val_len"))
    test_len = as_int(split_info.get("test_len"))
    if train_len is not None and val_len is not None and test_len is not None:
        total = int(train_len) + int(val_len) + int(test_len)
        if total > 0:
            st.caption(
                f"Split: train={train_len}, val={val_len}, test={test_len} "
                f"(ratio ~ {train_len/total:.2f}/{val_len/total:.2f}/{test_len/total:.2f})"
            )
        else:
            st.caption(f"Split: train={train_len}, val={val_len}, test={test_len}")

    def _fmt(x, pct=False, safe=False, metrics=None):
        if safe and isinstance(metrics, dict):
            if metrics.get("mape_safe") is not None:
                x = metrics.get("mape_safe")
            elif metrics.get("mape") is not None:
                x = metrics.get("mape")
        if x is None:
            return "—"
        try:
            xv = float(x)
            if pct:
                xv = xv * 100.0
                return f"{xv:.2f}%"
            return f"{xv:.4f}"
        except Exception:
            return str(x)

    import numpy as _np

    # nRMSE: rmse / mean(abs(y_true)); cached results usually only contain plot_data (last N points)
    _plot_blob0 = (data_blob.get("plot_data") or {}) if isinstance(data_blob, dict) else {}
    _val_plot0 = _plot_blob0.get("val") if isinstance(_plot_blob0, dict) else None
    _test_plot0 = _plot_blob0.get("test") if isinstance(_plot_blob0, dict) else None

    def _mean_abs_from_plot_blob(blob: Optional[dict]) -> Optional[float]:
        if not isinstance(blob, dict):
            return None
        y = blob.get("true") or []
        try:
            arr = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
        except Exception:
            try:
                arr = _np.asarray(y, dtype=float)
            except Exception:
                return None
        if arr.size == 0:
            return None
        mu = float(_np.nanmean(_np.abs(arr)))
        return mu if _np.isfinite(mu) and mu != 0.0 else None

    # Prefer persisted scalars (more stable than computing from truncated plot_data)
    try:
        mu_v = float(data_blob.get("mean_abs_true_val")) if isinstance(data_blob, dict) and data_blob.get("mean_abs_true_val") is not None else None # type: ignore
    except Exception:
        mu_v = None
    try:
        mu_t = float(data_blob.get("mean_abs_true_test")) if isinstance(data_blob, dict) and data_blob.get("mean_abs_true_test") is not None else None # type: ignore
    except Exception:
        mu_t = None
    if not (isinstance(mu_v, (int, float)) and _np.isfinite(mu_v) and mu_v > 0):
        mu_v = _mean_abs_from_plot_blob(_val_plot0)
    if not (isinstance(mu_t, (int, float)) and _np.isfinite(mu_t) and mu_t > 0):
        mu_t = _mean_abs_from_plot_blob(_test_plot0)
    rv = val_metrics.get("rmse")
    rt = test_metrics.get("rmse")
    # Prefer direct nrmse if provided by trainer/pipeline; fallback to rmse/mean(|y|).
    rv_nrmse = val_metrics.get("nrmse")
    rt_nrmse = test_metrics.get("nrmse")
    if rv_nrmse is None:
        rv_nrmse = (float(rv) / float(mu_v)) if (rv is not None and mu_v) else None
    if rt_nrmse is None:
        rt_nrmse = (float(rt) / float(mu_t)) if (rt is not None and mu_t) else None
    rv_pct = (rv_nrmse * 100.0) if rv_nrmse is not None else None
    rt_pct = (rt_nrmse * 100.0) if rt_nrmse is not None else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Val nRMSE", _fmt(rv_nrmse, pct=True))
        st.caption(f"RMSE: {_fmt(rv)}")
    with c2:
        st.metric("Val MAPE", _fmt(None, pct=True, safe=True, metrics=val_metrics))
    with c3:
        st.metric("Test nRMSE", _fmt(rt_nrmse, pct=True))
        st.caption(f"RMSE: {_fmt(rt)}")
    with c4:
        st.metric("Test MAPE", _fmt(None, pct=True, safe=True, metrics=test_metrics))

    if rv_pct is not None or rt_pct is not None:
        st.caption(f"Relative RMSE (vs mean |y|): Val {rv_pct:.3f}% | Test {rt_pct:.3f}%")

    st.subheader("📈 Forecast (Val/Test)")
    plot_blob = (data_blob.get("plot_data") or {}) if isinstance(data_blob, dict) else {}
    val_plot = plot_blob.get("val") if isinstance(plot_blob, dict) else None
    test_plot = plot_blob.get("test") if isinstance(plot_blob, dict) else None

    has_plot_data = isinstance(val_plot, dict) or isinstance(test_plot, dict)

    plot_n_opt = st.selectbox("Points to plot (last N)", ["1000", "2000", "4000", "ALL"], index=1, key="plot_n_cached")
    marker_every = st.number_input("Marker every k points", min_value=20, max_value=2000, value=200, step=20, key="plot_marker_cached")
    if has_plot_data:
        n_points = None if plot_n_opt == "ALL" else int(plot_n_opt)
        if plot_n_opt == "ALL":
            st.warning("Rendering ALL points may be slow for large datasets.")
        render_val_test(
            _to_df_plot_blob(val_plot),
            _to_df_plot_blob(test_plot),
            time_col=time_col,
            n_points=n_points,
            marker_every=int(marker_every),
        )
    else:
        st.caption("No curve data available yet. Run training once to generate plots.")

    with st.expander("🧳 Artifacts", expanded=False):
        arts = results.get("artifacts", {}) or {}
        try:
            paths = {}
            for k in ("model_path", "scaler_path", "residual_model_path", "y_scaler_path", "feature_cols_path", "feature_report_path"):
                v = (arts.get(k) if isinstance(arts, dict) else None)
                if v:
                    paths[k] = str(v)
            if paths:
                st.caption("Artifact paths")
                st.json(paths)
        except Exception:
            pass


# ==========================
# ======= Streamlit UI =====
# ==========================

st.set_page_config(page_title="Universal TS Forecast", layout="wide")
st.title("🧠 Universal Time Series Forecast")
st.caption(
    "Note: Streamlit file watcher is disabled (see `.streamlit/config.toml`) "
    "to avoid reruns interrupting training while writing artifacts."
)

# ---- Explicit state management: persist results & user actions across reruns ----
st.session_state.setdefault("last_results", None)          # dict (cacheable)
st.session_state.setdefault("last_meta", None)             # dict
st.session_state.setdefault("last_results_source", None)   # fresh/degraded/snapshot
st.session_state.setdefault("is_training", False)          # avoid stale snapshot rendering mid-run

# Main: upload + model + run
uploaded = st.file_uploader("Upload CSV", type=["csv"])

# Presets are convenience shortcuts; they only configure (base model + residual learner).
preset_name = st.selectbox(
    "Preset",
    ["Default", "Informer + XGBoost (residual)", "LSTM + XGBoost (residual)"],
    index=0,
)
_preset_model = None
_preset_residual = None
if preset_name == "Informer + XGBoost (residual)":
    _preset_model = "Informer"
    _preset_residual = "xgboost"
elif preset_name == "LSTM + XGBoost (residual)":
    _preset_model = "LSTM"
    _preset_residual = "XGBoost"

model_options = ["Informer", "ARIMA", "Prophet", "RandomForest", "LSTM", "XGBoost"]
try:
    if _preset_model:
        st.session_state["model_name"] = _preset_model
    st.session_state.setdefault("model_name", model_options[0])
except Exception:
    pass
model_name = st.selectbox("Model", model_options, index=0, key="model_name", disabled=bool(_preset_model))

residual_options = ["none", "linear", "XGBoost"]
try:
    if _preset_residual:
        st.session_state["residual_learner"] = _preset_residual
    st.session_state.setdefault("residual_learner", "none")
except Exception:
    pass
residual_learner = st.selectbox(
    "Residual learner",
    residual_options,
    index=0,
    key="residual_learner",
    disabled=bool(_preset_residual),
    help="Learns residual = y_true - y_hat_main, then adds it back to the main prediction.",
)

if torch is None:
    st.warning("`torch` is not installed: Informer/LSTM are unavailable. Install with: `pip install torch`.")
    device_choice = st.selectbox("Compute device", ["cpu"], index=0, help="Torch is missing; CPU only.")
    mps_ok = False
else:
    device_choice = st.selectbox(
        "Compute device",
        ["auto", "mps", "cpu"],
        index=0,
        help="auto: prefer MPS, then CPU; mps: force MPS (no fallback)",
    )
    try:
        mps_ok = bool(getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_built() and torch.backends.mps.is_available())
    except Exception:
        mps_ok = False
if device_choice == 'mps' and not mps_ok:
    st.error("You selected MPS (forced), but MPS is not available in this PyTorch build. Upgrade PyTorch or switch to auto/cpu.")
    st.stop()

if model_name == "randomforest":
    st.caption("⚙️ RandomForest runs Optuna tuning on the validation split (n_trials from configs.yaml: optimization.n_trials).")
if model_name == "lstm":
    st.caption("🧩 LSTM uses configs.yaml: model_config.LSTM (seq_len/hidden_dim/num_layers/n_epochs/learning_rate) and produces dense val/test predictions.")
if model_name == "xgboost":
    st.caption("🌲 XGBoost uses configs.yaml: model_config.XGBoost and produces dense val/test predictions.")
if residual_learner != "none":
    st.caption(f"🧪 Residual modeling enabled: {residual_learner}.")
run_click = st.button("Train & Predict", type="primary")

# Online rolling inference (no retraining)
col_r1, col_r2, col_r3 = st.columns([1,1,2])
with col_r1:
    horizon_days = st.selectbox("Online forecast horizon (days)", [1, 3, 7], index=0)
with col_r2:
    step_mode = st.selectbox("Rolling step", ["Block step (= horizon)", "Step-by-step (= 1)"], index=0)
with col_r3:
    st.caption("Block step is faster with no error accumulation; step-by-step is smoother but slower and accumulates errors.")
    allow_degrade = st.checkbox("Allow degrade (fallback to baseline if Required Core is missing)", value=False)
    st.caption("When enabled: online inference falls back to a baseline if required features are missing; training/eval may also return baseline with degraded=True.")

online_click = st.button("Predict only (online rolling)", type="secondary")

if uploaded is None:
    st.info("Upload a CSV file to begin.")
else:
    # Load data
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # Basic column inference
    time_col = 'date' if 'date' in df.columns else df.columns[0]

    # More robust target selection: prefer numeric-like columns
    def _numeric_profile(frame: pd.DataFrame, col: str):
        s = frame[col]
        if pd.api.types.is_numeric_dtype(s):
            num = pd.to_numeric(s, errors="coerce")
        else:
            try:
                ss = s.astype(str).str.replace(",", "", regex=False).str.strip()
            except Exception:
                ss = s
            num = pd.to_numeric(ss, errors="coerce")
        notna = float(num.notna().mean()) if len(num) else 0.0
        miss = float(num.isna().mean()) if len(num) else 1.0
        try:
            var = float(num.var()) # type: ignore
        except Exception:
            var = 0.0
        return num, notna, miss, var

    candidates_all = [c for c in df.columns if c != time_col]
    profiles = {c: _numeric_profile(df, c) for c in candidates_all}

    # numeric-like: some parseable numeric values (can be sparse, but not all-NaN)
    numeric_like = [c for c in candidates_all if profiles[c][1] > 0.01]

    # Target candidates: prefer numeric-like to avoid categorical columns
    value_candidates = numeric_like if numeric_like else candidates_all

    # Default target: prefer 'value', else choose low-missing + higher-variance numeric column
    if "value" in df.columns and "value" in numeric_like:
        default_value_col = "value"
    elif numeric_like:
        default_value_col = sorted(numeric_like, key=lambda c: (profiles[c][2], -profiles[c][3]))[0]
    else:
        default_value_col = candidates_all[0] if candidates_all else df.columns[0]

    def _fmt_col(c: str) -> str:
        _notna = profiles[c][1]
        _miss = profiles[c][2]
        return f"{c}  (numeric={_notna:.0%}, missing={_miss:.0%})"

    value_col = st.selectbox(
        "Target column (value_col)",
        options=value_candidates,
        index=value_candidates.index(default_value_col) if default_value_col in value_candidates else 0,
        format_func=_fmt_col,
    )


    # Guardrail: target must be numeric-like
    try:
        _num_rate = float(profiles[value_col][1])
        if _num_rate < 0.5:
            st.error(
                f"Target '{value_col}' is not reliably numeric (numeric={_num_rate:.0%}). "
                "Choose a numeric column as the prediction target."
            )
            if run_click or online_click:
                st.stop()
    except Exception:
        pass

    # Auto feature_cols (uni/multi-var): numeric-like candidates only
    feature_cols = [value_col] + [c for c in numeric_like if c != value_col]

    # Warning: missing values in target may trigger Required Core fail-fast
    try:
        y_num = profiles[value_col][0]
        n_nan = int(y_num.isna().sum())
        if n_nan > 0:
            st.warning(
                f"Target '{value_col}' has {n_nan} missing/unparseable values; "
                "training may fail-fast under the Required Core policy."
            )
    except Exception:
        pass

    missing_cols = [c for c in (time_col, value_col) if c not in df.columns]
    if missing_cols:
        st.error(f"CSV is missing required columns: {missing_cols}")
        st.stop()

    st.subheader("📄 Data preview")
    st.caption(f"Time column: {time_col} | Target: {value_col}")
    st.dataframe(df.head(10), use_container_width=True)
    # Results area: cleared at training start, filled when training completes (or from cached snapshot).
    results_container = st.empty()

    # ==========================
    # Online rolling inference (no retraining)
    # ==========================
    if online_click:
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
                    "pred_len": 24,  # actual horizon is overridden by UI
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
                }
                ,
                # degrade mode: when Required Core missing, fallback to baseline and mark degraded=True
                "degrade": {"enabled": bool(allow_degrade), "mode": "naive_last"}
            }
        }

        st.caption(f"Feature columns (fixed order for train/predict): {feature_cols}")

        # Horizon & step
        horizon_steps = int(24 * horizon_days)  # adjust multiplier if your data is not hourly
        step_val = None if step_mode.startswith("Block") else 1

        try:
            if torch is None:
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

        # --- Degraded warning (platform-safety) ---
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

        # Metrics on the overlapping (non-NaN) region
        merged = np.asarray(merged).reshape(-1)
        mask = ~np.isnan(merged)
        if mask.sum() == 0:
            st.warning("No valid prediction region (data too short or parameters mismatch).")
        else:
            y_true = pd.to_numeric(df.loc[mask, value_col], errors='coerce').to_numpy()
            y_hat = merged[mask]
            rmse = float(np.sqrt(np.nanmean((y_hat - y_true) ** 2)))
            denom = np.where(y_true == 0, np.nan, np.abs(y_true))
            mape = float(np.nanmean(np.abs((y_hat - y_true) / denom)) * 100.0)

            st.subheader("⚡ Online rolling inference — Metrics")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Online RMSE", f"{rmse:.4f}")
            with c2:
                st.metric("Online MAPE", f"{mape:.2f}%")

            # Long-series payload
            online_long = {
                "timestamps": pd.to_datetime(df[time_col]).astype(str).tolist(),
                "y_true": pd.to_numeric(df[value_col], errors='coerce').astype(float).tolist(),
                "yhat": merged.astype(float).tolist(),
                "degraded": bool(dblk.get("degraded", False)),
                "degraded_reason": dblk.get("degraded_reason"),
                "degraded_mode": dblk.get("degraded_mode"),
                "missing_required_core": dblk.get("missing_required_core"),
                "dropped_optional_features": dblk.get("dropped_optional_features"),
            }

            st.subheader("📈 Online rolling inference — Curve")
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
                # annotate degraded columns for download/traceability
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
                        mime=MIME_CSV,
                    )
                except Exception:
                    pass
    # ==========================
    # Train + predict + unified plotting (6/2/2 + long series)
    # ==========================
    if run_click:
        # Train + predict + unified plotting (6/2/2 + long series)
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
                # App handles plotting; disable pipeline-side Matplotlib helpers by default.
                "pipeline_plot": False,
                "build_continuous": False,
            },
            # Target transform: compress volatility during training (applied for deep models in data prep)
            "target_transform": {
                "enabled": model_name in ("informer", "lstm"),
                "method": "log1p",
            },
            # Post-hoc calibration: fit affine (a,b) on val to slightly reduce RMSE without hurting MAPE.
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
                    # Shorter training horizon reduces over-smoothing and improves generalization;
                    # long-horizon is handled via rolling inference / dense 1-step outputs.
                    "pred_len": 8,
                    "auto_feature_cols": True,
                    "lock_feature_order": True,
                    # Initial candidate set (final set will be train-only selected and frozen).
                    "feature_cols": feature_cols,
                    # Train-only feature selection (MI + RF importance)
                    "feature_selection": {
                        "missing_rate_threshold": 0.4,
                        "low_variance_threshold": 1e-8,
                        "redundant_corr_threshold": 0.95,
                        "max_features": None,
                        "leakage_name_patterns": ["label", "target", "future", "t+", "lead", "yhat", "predict"],
                        "safe_default_cols": ["month", "day_of_month", "day_of_week", "hour", "day_of_year"],
                        # Tiering (generic missing-feature solution)
                        "required_core_cols": [],
                        "repairable_core_cols": ["month", "day_of_month", "day_of_week", "hour", "day_of_year"],
                        # Extra core cols can be configured here if needed
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

        # Residual modeling (combo models): configure via UI.
        _res_choice = str(residual_learner or "none").strip().lower()
        if _res_choice in ("linear", "xgboost"):
            rm_cfg = {
                "enabled": True,
                "model_type": "ridge" if _res_choice == "linear" else "xgboost",
            }
            config["residual_modeling"] = rm_cfg
            # Avoid double-correction for Informer: disable its internal residual + post_calibration
            if model_name == "informer":
                try:
                    config.setdefault("model_config", {}).setdefault("Informer", {})["use_residual"] = False
                except Exception:
                    pass
                try:
                    config.setdefault("post_calibration", {})["enabled"] = False
                except Exception:
                    pass

        # XGBoost: load hyperparameters from configs.yaml (if present).
        # - used by the standalone XGBoost model
        # - also used as residual learner when residual_learner == "xgboost"
        if model_name == "xgboost" or _res_choice == "xgboost":
            try:
                xgb_hp = _load_xgboost_hparams_from_configs_yaml()
                if isinstance(xgb_hp, dict) and xgb_hp:
                    config.setdefault("model_config", {})["XGBoost"] = xgb_hp
            except Exception:
                pass
        # Dedicated artifact paths: avoid overwriting other models.
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

        # Keep back-compat keys for the pipeline
        config.setdefault("model", {})["name"] = model_name
        config["model_type"] = model_name

        # Provide raw DataFrame for registry models (e.g. arima)
        config.setdefault("data", {})
        config["data"]["dataframe"] = df.copy()
        # Pipeline call + normalize + snapshot are handled in services.pipeline (keeps this file small).

        # Progress UI (training/pipeline/app)
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

        # Pipeline/training callback (called from deep modules)
        def _progress_cb(stage: str = "pipeline", pct: Optional[float] = None, **info):
            try:
                # IMPORTANT: avoid calling Streamlit APIs inside the training loop.
                # Streamlit may interrupt long runs at st.* yield points if a rerun is requested,
                # which looks like "training stops at epoch k/10". We only update UI at coarse pipeline stages.
                if stage == "train":
                    return
                if pct is None:
                    return
                _set_progress(float(pct), f"{stage}: {info.get('msg') or ''}".strip())
            except Exception:
                pass

        config["callbacks"]["progress"] = _progress_cb
        _set_progress(0.02, "Starting pipeline...")
        try:
            pipeline_mod = load_pipeline_module()
            results = pipeline_mod.run_pipeline_and_update_state(
                df=df.copy(),
                config=config,
                feature_cols=feature_cols,
                uploaded_name=getattr(uploaded, "name", None),
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
            # Always release training flag even if the pipeline raised.
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

        # Stable UI path: render from minimal cached snapshot so plot controls survive reruns.
        try:
            cached = st.session_state.get("last_results")
            if not isinstance(cached, dict):
                cached = cacheable_results(results)
            meta = st.session_state.get("last_meta") if isinstance(st.session_state.get("last_meta"), dict) else {}
            with results_container.container():
                _render_cached_summary(
                    cached,
                    model_name=str(meta.get("model_name") or model_name), # type: ignore
                    time_col=str(meta.get("time_col") or time_col), # type: ignore
                    value_col=str(meta.get("value_col") or value_col), # type: ignore
                )
        except Exception as _e:
            st.error(f"Failed to render results: {_e}")
        finally:
            st.session_state["is_training"] = False
        st.stop()

    # ==========================
    # Cached results (state-driven; survives reruns)
    # ==========================
    if st.session_state.get("last_results") is None:
        snap = load_last_results_json()
        if isinstance(snap, dict) and isinstance(snap.get("results"), dict):
            st.session_state["last_results"] = snap.get("results")
            st.session_state["last_meta"] = snap.get("meta") if isinstance(snap.get("meta"), dict) else {}
            st.session_state["last_results_source"] = "snapshot"

    cached = st.session_state.get("last_results")
    if isinstance(cached, dict) and not bool(st.session_state.get("is_training", False)):
        meta = st.session_state.get("last_meta") if isinstance(st.session_state.get("last_meta"), dict) else {}
        with results_container.container():
            _render_cached_summary(
                cached,
                model_name=str(meta.get("model_name") or model_name), # type: ignore
                time_col=str(meta.get("time_col") or time_col), # type: ignore
                value_col=str(meta.get("value_col") or value_col), # type: ignore
            )

    if st.button("Clear cached results", type="secondary", key="clear_cached_results"):
        st.session_state["last_results"] = None
        st.session_state["last_meta"] = None
        st.session_state["last_results_source"] = None
