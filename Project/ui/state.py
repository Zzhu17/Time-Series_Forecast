import streamlit as st


STATE_KEYS = ("last_results", "last_meta", "last_results_source")


def init_state():
    """Initialize session state keys used by the app."""
    st.session_state.setdefault("last_results", None)
    st.session_state.setdefault("last_meta", None)
    st.session_state.setdefault("last_results_source", None)
    st.session_state.setdefault("is_training", False)


def clear_cached_results():
    """Clear cached results stored in session state."""
    for key in STATE_KEYS:
        st.session_state[key] = None
