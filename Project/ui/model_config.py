import os
from typing import Optional, Tuple

import streamlit as st

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore[assignment]


def render_preset_selector() -> Tuple[str, Optional[str], Optional[str]]:
    preset_name = st.selectbox(
        "Preset",
        ["Default", "Informer + XGBoost (residual)", "LSTM + XGBoost (residual)"],
        index=0,
    )
    preset_model = None
    preset_residual = None
    if preset_name == "Informer + XGBoost (residual)":
        preset_model = "Informer"
        preset_residual = "xgboost"
    elif preset_name == "LSTM + XGBoost (residual)":
        preset_model = "LSTM"
        preset_residual = "XGBoost"
    return preset_name, preset_model, preset_residual


def render_model_and_device_selectors(preset_model: Optional[str], preset_residual: Optional[str]):
    model_name, residual_learner = render_model_selectors(preset_model, preset_residual)
    device_choice, mps_ok = render_device_selector()

    if model_name == "randomforest":
        st.caption("⚙️ RandomForest runs Optuna tuning on the validation split (n_trials from configs.yaml: optimization.n_trials).")
    if model_name == "lstm":
        st.caption("🧩 LSTM uses configs.yaml: model_config.LSTM (seq_len/hidden_dim/num_layers/n_epochs/learning_rate) and produces dense val/test predictions.")
    if model_name == "xgboost":
        st.caption("🌲 XGBoost uses configs.yaml: model_config.XGBoost and produces dense val/test predictions.")
    if residual_learner != "none":
        st.caption(f"🧪 Residual modeling enabled: {residual_learner}.")

    return model_name, residual_learner, device_choice


def render_model_selectors(preset_model: Optional[str], preset_residual: Optional[str]):
    model_options = ["Informer", "ARIMA", "Prophet", "RandomForest", "LSTM", "XGBoost"]
    try:
        if preset_model:
            st.session_state["model_name"] = preset_model
        st.session_state.setdefault("model_name", model_options[0])
    except Exception:
        pass
    model_name = st.selectbox("Model", model_options, index=0, key="model_name", disabled=bool(preset_model))

    residual_options = ["none", "linear", "XGBoost"]
    try:
        if preset_residual:
            st.session_state["residual_learner"] = preset_residual
        st.session_state.setdefault("residual_learner", "none")
    except Exception:
        pass
    residual_learner = st.selectbox(
        "Residual learner",
        residual_options,
        index=0,
        key="residual_learner",
        disabled=bool(preset_residual),
        help="Learns residual = y_true - y_hat_main, then adds it back to the main prediction.",
    )

    return model_name, residual_learner


def render_device_selector():
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

    return device_choice, mps_ok


def load_xgboost_hparams_from_configs_yaml() -> Optional[dict]:
    """
    Load model_config.XGBoost from `configs/configs.yaml` without requiring PyYAML.
    Only supports the simple scalar key/value block we use for XGBoost.
    """
    try:
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "configs.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return None

    in_model_config = False
    xgb_indent = None
    out: dict = {}

    def _strip_comment(s: str) -> str:
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

        if indent == 0 and ":" in txt:
            break

        if xgb_indent is None:
            if txt == "XGBoost:":
                xgb_indent = indent
            continue

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
