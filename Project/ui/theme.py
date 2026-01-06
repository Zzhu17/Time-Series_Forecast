import streamlit as st

from services.ui_theme import inject_global_theme


def setup_page():
    """Configure page and inject shared theme."""
    st.set_page_config(page_title="Universal TS Forecast", layout="wide")
    inject_global_theme()


def render_hero():
    """Render hero banner."""
    st.markdown(
        """
        <div class="tsf-hero">
          <div class="tsf-hero-icon">📈</div>
          <div>
            <div class="tsf-pill">Universal Time-Series Forecast</div>
            <h1>Upload, configure, and forecast with one unified pipeline.</h1>
            <p class="section-note">File watcher is disabled (see .streamlit/config.toml) to prevent reruns while artifacts are written.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
