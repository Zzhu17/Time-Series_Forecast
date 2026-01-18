from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from services.snapshot import as_int, load_last_results_json
from visualizations.plot import df_from_long, render_true_pred, render_val_test


def _to_df_plot_blob(blob):
    if not isinstance(blob, dict):
        return None
    try:
        return pd.DataFrame({"ts": blob.get("ts") or [], "true": blob.get("true") or [], "pred": blob.get("pred") or []})
    except Exception:
        return None


def render_cached_summary(results: dict, *, model_name: str, time_col: str, value_col: str):
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
                arr = np.asarray(y, dtype=float)
            except Exception:
                return None
        if arr.size == 0:
            return None
        mu = float(np.nanmean(np.abs(arr)))
        return mu if np.isfinite(mu) and mu != 0.0 else None

    try:
        mu_v = float(data_blob.get("mean_abs_true_val")) if isinstance(data_blob, dict) and data_blob.get("mean_abs_true_val") is not None else None  # type: ignore
    except Exception:
        mu_v = None
    try:
        mu_t = float(data_blob.get("mean_abs_true_test")) if isinstance(data_blob, dict) and data_blob.get("mean_abs_true_test") is not None else None  # type: ignore
    except Exception:
        mu_t = None
    if not (isinstance(mu_v, (int, float)) and np.isfinite(mu_v) and mu_v > 0):
        mu_v = _mean_abs_from_plot_blob(_val_plot0)
    if not (isinstance(mu_t, (int, float)) and np.isfinite(mu_t) and mu_t > 0):
        mu_t = _mean_abs_from_plot_blob(_test_plot0)
    rv = val_metrics.get("rmse")
    rt = test_metrics.get("rmse")
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

    st.markdown("<div class='tsf-card'>", unsafe_allow_html=True)
    st.markdown("### 📈 Forecast (Val/Test)", unsafe_allow_html=True)
    plot_blob = (data_blob.get("plot_data") or {}) if isinstance(data_blob, dict) else {}
    train_plot = plot_blob.get("train") if isinstance(plot_blob, dict) else None
    val_plot = plot_blob.get("val") if isinstance(plot_blob, dict) else None
    test_plot = plot_blob.get("test") if isinstance(plot_blob, dict) else None

    has_plot_data = isinstance(train_plot, dict) or isinstance(val_plot, dict) or isinstance(test_plot, dict)
    plot_n_opt = st.selectbox("Points to plot (last N)", ["1000", "2000", "4000", "ALL"], index=1, key="plot_n_cached")
    marker_every = st.number_input("Marker every k points", min_value=20, max_value=2000, value=200, step=20, key="plot_marker_cached")
    if has_plot_data:
        n_points = None if plot_n_opt == "ALL" else int(plot_n_opt)
        if plot_n_opt == "ALL":
            st.warning("Rendering ALL points may be slow for large datasets.")
        if isinstance(train_plot, dict):
            render_true_pred(
                _to_df_plot_blob(train_plot),
                title="Train: True vs Pred",
                n_points=n_points,
                marker_every=int(marker_every),
            )
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
            for k in (
                "model_path",
                "scaler_path",
                "residual_model_path",
                "y_scaler_path",
                "feature_cols_path",
                "feature_report_path",
                "leaderboard_path",
                "report_path",
                "processed_data_path",
                "data_profile_path",
            ):
                v = (arts.get(k) if isinstance(arts, dict) else None)
                if v:
                    paths[k] = str(v)
            if paths:
                st.caption("Artifact paths")
                st.json(paths)
        except Exception:
            pass
    if isinstance(data_blob, dict) and data_blob.get("leaderboard"):
        with st.expander("🏁 Leaderboard", expanded=False):
            try:
                st.dataframe(pd.DataFrame(data_blob.get("leaderboard")))
            except Exception:
                st.json(data_blob.get("leaderboard"))
    st.markdown("</div>", unsafe_allow_html=True)


def render_cached_results(results_container, *, model_name: str, time_col: str, value_col: str):
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
            render_cached_summary(
                cached,
                model_name=str(meta.get("model_name") or model_name),  # type: ignore
                time_col=str(meta.get("time_col") or time_col),  # type: ignore
                value_col=str(meta.get("value_col") or value_col),  # type: ignore
            )
