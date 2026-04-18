from __future__ import annotations

import ipaddress
import os
from typing import Optional

from fastapi import Header, HTTPException


DEV_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://localhost:8501",
]


def get_runtime_env() -> str:
    return str(os.getenv("TSF_ENV") or os.getenv("ENV") or "dev").strip().lower()


def is_protected_env(env_name: Optional[str] = None) -> bool:
    return (env_name or get_runtime_env()) in {"prod", "production", "staging"}


def get_api_token() -> str:
    return str(os.getenv("TSF_API_TOKEN") or os.getenv("API_TOKEN") or "").strip()


def parse_cors_allow_origins(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


def get_cors_allow_origins() -> list[str]:
    configured = parse_cors_allow_origins(os.getenv("TSF_CORS_ALLOW_ORIGINS"))
    if configured:
        return configured
    if is_protected_env():
        return []
    return list(DEV_CORS_ORIGINS)


def validate_runtime_security() -> None:
    if is_protected_env() and not get_api_token():
        raise RuntimeError("TSF_API_TOKEN (or API_TOKEN) is required in staging/prod environments.")


def should_log_client_ip() -> bool:
    return str(os.getenv("TSF_LOG_CLIENT_IP") or "").strip().lower() in {"1", "true", "yes", "on"}


def redact_client_ip(client_ip: Optional[str]) -> Optional[str]:
    if not client_ip:
        return None
    try:
        ip_obj = ipaddress.ip_address(client_ip)
    except ValueError:
        return None

    if isinstance(ip_obj, ipaddress.IPv4Address):
        network = ipaddress.ip_network(f"{client_ip}/24", strict=False)
        return str(network.network_address)

    network = ipaddress.ip_network(f"{client_ip}/48", strict=False)
    return str(network.network_address)


def verify_api_token(authorization: Optional[str] = Header(default=None)) -> None:
    token = get_api_token()
    if not token:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="missing API token")
    parts = authorization.split()
    provided = parts[-1] if len(parts) >= 2 else authorization
    if provided != token:
        raise HTTPException(status_code=403, detail="invalid API token")
