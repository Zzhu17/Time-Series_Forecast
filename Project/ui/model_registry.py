from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.api_client import list_model_registry


def render_model_registry(api_url: str) -> None:
    st.markdown("<div class='tsf-card'>", unsafe_allow_html=True)
    st.markdown("### 🧾 Model Registry", unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 1])
    with col_b:
        if st.button("Refresh registry", type="secondary"):
            try:
                list_model_registry.clear()
            except Exception:
                pass

    try:
        models = list_model_registry(api_url)
    except Exception as e:
        st.error(f"Registry unavailable: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    if not models:
        st.caption("No models registered yet. Train a model to populate the registry.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    df = pd.DataFrame(models)
    if "created_at" in df.columns:
        df = df.sort_values("created_at", ascending=False)

    names = sorted({str(n) for n in df.get("name", []) if isinstance(n, str)})
    sel_name = col_a.selectbox("Filter by model name", ["All"] + names, index=0)
    if sel_name != "All":
        df = df[df["name"] == sel_name]

    st.dataframe(df, width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)
