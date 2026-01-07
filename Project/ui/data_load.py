from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st


def upload_csv():
    """Render upload widget and return uploaded file."""
    return st.file_uploader("Upload CSV", type=["csv"])


def _numeric_profile(frame: pd.DataFrame, col: str) -> Tuple[pd.Series, float, float, float]:
    """Profile a column for numeric-like heuristics."""
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
        var = float(num.var())  # type: ignore
    except Exception:
        var = 0.0
    return num, notna, miss, var


def _infer_columns(df: pd.DataFrame) -> Tuple[str, List[str], List[str], Dict[str, Tuple[pd.Series, float, float, float]]]:
    time_col = "date" if "date" in df.columns else df.columns[0]
    candidates_all = [c for c in df.columns if c != time_col]
    profiles = {c: _numeric_profile(df, c) for c in candidates_all}
    numeric_like = [c for c in candidates_all if profiles[c][1] > 0.01]
    value_candidates = numeric_like if numeric_like else candidates_all

    if "value" in df.columns and "value" in numeric_like:
        default_value_col = "value"
    elif numeric_like:
        default_value_col = sorted(numeric_like, key=lambda c: (profiles[c][2], -profiles[c][3]))[0]
    else:
        default_value_col = candidates_all[0] if candidates_all else df.columns[0]

    return default_value_col, value_candidates, numeric_like, profiles


def _fmt_col(profiles: Dict[str, Tuple[pd.Series, float, float, float]], col: str) -> str:
    _notna = profiles[col][1]
    _miss = profiles[col][2]
    return f"{col}  (numeric={_notna:.0%}, missing={_miss:.0%})"


def select_columns(df: pd.DataFrame, *, run_click: bool, online_click: bool):
    """Select target/time/feature columns and emit warnings."""
    time_col = "date" if "date" in df.columns else df.columns[0]
    default_value_col, value_candidates, numeric_like, profiles = _infer_columns(df)
    value_col = st.selectbox(
        "Target column (value_col)",
        options=value_candidates,
        index=value_candidates.index(default_value_col) if default_value_col in value_candidates else 0,
        format_func=lambda c: _fmt_col(profiles, c),
    )

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

    feature_cols = [value_col] + [c for c in numeric_like if c != value_col]

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

    return time_col, value_col, feature_cols, profiles


def render_preview(df: pd.DataFrame, time_col: str, value_col: str):
    st.markdown("<div class='tsf-card'>", unsafe_allow_html=True)
    st.markdown("### 📄 Data preview", unsafe_allow_html=True)
    st.caption(f"Time column: {time_col} | Target: {value_col}")
    st.dataframe(df.head(10), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)
