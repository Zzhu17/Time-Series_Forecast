"""
Lightweight theme helpers to mirror the “Universal Time-Series Forecast” Figma styling
inside Streamlit without adding front-end frameworks.
"""

from __future__ import annotations

import textwrap


_CSS = textwrap.dedent(
    """
    <style>
    :root {
        --tsf-indigo-50: #eef2ff;
        --tsf-indigo-100: #e0e7ff;
        --tsf-indigo-600: #4f46e5;
        --tsf-indigo-700: #4338ca;
        --tsf-slate-50: #f8fafc;
        --tsf-slate-100: #eef2f7;
        --tsf-slate-300: #d9e2ec;
        --tsf-slate-700: #1f2937;
        --tsf-card-bg: #ffffff;
        --tsf-card-border: #e6ebf5;
        --tsf-text-primary: #1f2a44;
        --tsf-text-muted: #5b6477;
    }

    .stApp {
        background: radial-gradient(circle at 10% 20%, #f3f5ff 0, #eef4ff 25%, #f8fbff 50%, #eef5ff 75%, #f1f5ff 100%);
    }

    .block-container {
        padding: 1.2rem 1.8rem 2.6rem;
        max-width: 1180px;
    }

    .tsf-hero {
        background: linear-gradient(135deg, #eef2ff 0%, #f7f9ff 40%, #eef7ff 100%);
        border: 1px solid var(--tsf-card-border);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 18px 40px rgba(38, 60, 123, 0.12);
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 1rem;
        align-items: center;
    }

    .tsf-hero-icon {
        width: 56px;
        height: 56px;
        border-radius: 18px;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        display: grid;
        place-items: center;
        font-size: 28px;
        box-shadow: 0 12px 32px rgba(79, 70, 229, 0.35);
    }

    .tsf-hero h1 {
        margin: 0;
        font-size: 2rem;
        color: var(--tsf-text-primary);
        letter-spacing: -0.01em;
    }

    .tsf-pill {
        display: inline-flex;
        padding: 0.25rem 0.9rem;
        border-radius: 999px;
        background: rgba(79, 70, 229, 0.08);
        color: #4338ca;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }

    .tsf-card {
        background: var(--tsf-card-bg);
        border: 1px solid var(--tsf-card-border);
        border-radius: 16px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 12px 30px rgba(31, 45, 91, 0.08);
        margin-top: 1rem;
    }

    .tsf-card h3 {
        margin-top: 0;
        margin-bottom: 0.35rem;
        color: var(--tsf-text-primary);
    }

    .tsf-card .section-note {
        color: var(--tsf-text-muted);
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: #fff;
        border-radius: 12px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 14px 30px rgba(79, 70, 229, 0.25);
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%);
        transform: translateY(-1px);
    }

    .stButton>button:focus {
        outline: 2px solid rgba(79, 70, 229, 0.35);
        outline-offset: 2px;
    }

    /* Inputs */
    .stSelectbox, .stNumberInput, .stTextInput, .stFileUploader {
        border-radius: 12px !important;
    }

    /* Reduce caption spacing */
    .stMarkdown p {
        margin-bottom: 0.3rem;
    }
    </style>
    """
)


def inject_global_theme() -> None:
    """Inject the custom CSS theme into the current Streamlit app."""
    try:
        import streamlit as st

        st.markdown(_CSS, unsafe_allow_html=True)
    except Exception:
        # Keep UI functional even if Streamlit is not available in some contexts.
        pass
