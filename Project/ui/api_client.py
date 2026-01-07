from __future__ import annotations

import json
import os
import hashlib
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_API_URL = "http://localhost:8000"
_SESSION: Optional[requests.Session] = None


def get_api_url() -> str:
    api_url = st.session_state.get("api_url")
    if isinstance(api_url, str) and api_url.strip():
        return api_url.strip().rstrip("/")
    env_url = os.getenv("TSF_API_URL") or os.getenv("API_URL")
    if env_url:
        return env_url.strip().rstrip("/")
    return DEFAULT_API_URL


def get_api_token() -> str:
    token = st.session_state.get("api_token")
    if isinstance(token, str) and token.strip():
        return token.strip()
    env_token = os.getenv("TSF_API_TOKEN") or os.getenv("API_TOKEN")
    if env_token:
        return env_token.strip()
    return ""


def render_api_settings() -> str:
    api_url = get_api_url()
    val = st.sidebar.text_input("API URL", value=api_url)
    st.session_state["api_url"] = val.strip()
    token_val = get_api_token()
    token = st.sidebar.text_input("API Token (optional)", value=token_val, type="password")
    st.session_state["api_token"] = token.strip()
    return get_api_url()


def _raise_for_response(resp: requests.Response) -> None:
    if resp.status_code >= 400:
        try:
            data = resp.json()
            detail = data.get("detail") or data
        except Exception:
            detail = resp.text
        raise RuntimeError(f"API error ({resp.status_code}): {detail}")


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    _SESSION = session
    return session


def _headers() -> Dict[str, str]:
    token = get_api_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _post(api_url: str, path: str, *, files=None, data=None, timeout: int = 300) -> requests.Response:
    session = _get_session()
    return session.post(f"{api_url}{path}", files=files, data=data, headers=_headers(), timeout=timeout)


def _get(api_url: str, path: str, *, params=None, timeout: int = 60) -> requests.Response:
    session = _get_session()
    return session.get(f"{api_url}{path}", params=params, headers=_headers(), timeout=timeout)


def _post_json(api_url: str, path: str, *, payload: Dict[str, Any], timeout: int = 120) -> requests.Response:
    session = _get_session()
    return session.post(f"{api_url}{path}", json=payload, headers=_headers(), timeout=timeout)


def train_file_streamlit(
    *,
    api_url: str,
    file_bytes: bytes,
    filename: str,
    model_name: str,
    time_col: str,
    value_col: str,
    horizon: int,
    feature_cols: Optional[List[str]],
    residual_modeling: Optional[dict],
    allow_degrade: bool,
    device: str,
) -> Dict[str, Any]:
    files = {"file": (filename or "data.csv", file_bytes, "text/csv")}
    data = {
        "model_name": model_name,
        "time_col": time_col,
        "value_col": value_col,
        "horizon": str(int(horizon)),
        "feature_cols": json.dumps(feature_cols) if feature_cols is not None else "",
        "residual_modeling": json.dumps(residual_modeling) if residual_modeling else "",
        "allow_degrade": str(bool(allow_degrade)).lower(),
        "device": device,
    }
    resp = _post(api_url, "/train_file_streamlit", files=files, data=data, timeout=600)
    _raise_for_response(resp)
    return resp.json()


def predict_online_file(
    *,
    api_url: str,
    file_bytes: bytes,
    filename: str,
    model_name: str,
    time_col: str,
    value_col: str,
    horizon_days: int,
    step_mode: str,
    allow_degrade: bool,
    device: str,
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    files = {"file": (filename or "data.csv", file_bytes, "text/csv")}
    data = {
        "model_name": model_name,
        "time_col": time_col,
        "value_col": value_col,
        "horizon_days": str(int(horizon_days)),
        "step_mode": step_mode,
        "allow_degrade": str(bool(allow_degrade)).lower(),
        "device": device,
    }
    if model_id:
        data["model_id"] = model_id
    if model_version:
        data["model_version"] = model_version
    resp = _post(api_url, "/predict_online_file", files=files, data=data, timeout=300)
    _raise_for_response(resp)
    return resp.json()


def predict_batch(
    *,
    api_url: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    resp = _post_json(api_url, "/predict", payload=payload, timeout=120)
    _raise_for_response(resp)
    return resp.json()


@st.cache_data(ttl=300)
def predict_online_file_cached(
    *,
    api_url: str,
    file_hash_value: str,
    file_bytes: bytes,
    filename: str,
    model_name: str,
    time_col: str,
    value_col: str,
    horizon_days: int,
    step_mode: str,
    allow_degrade: bool,
    device: str,
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    return predict_online_file(
        api_url=api_url,
        file_bytes=file_bytes,
        filename=filename,
        model_name=model_name,
        time_col=time_col,
        value_col=value_col,
        horizon_days=horizon_days,
        step_mode=step_mode,
        allow_degrade=allow_degrade,
        device=device,
        model_id=model_id,
        model_version=model_version,
    )


@st.cache_data(ttl=120)
def list_model_registry(api_url: str, limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
    resp = _get(api_url, "/models/registry", params={"limit": limit, "offset": offset})
    _raise_for_response(resp)
    data = resp.json()
    return data if isinstance(data, list) else []


def promote_model(api_url: str, model_id: str, stage: str = "production") -> Dict[str, Any]:
    resp = _post(api_url, f"/models/{model_id}/promote", data={"stage": stage}, timeout=60)
    _raise_for_response(resp)
    return resp.json()


def file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()
