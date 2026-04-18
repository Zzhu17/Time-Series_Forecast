from __future__ import annotations

import pytest

from api.security import (
    DEV_CORS_ORIGINS,
    get_cors_allow_origins,
    redact_client_ip,
    validate_runtime_security,
    verify_api_token,
)


def test_validate_runtime_security_allows_dev_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TSF_ENV", "dev")
    monkeypatch.delenv("TSF_API_TOKEN", raising=False)
    monkeypatch.delenv("API_TOKEN", raising=False)

    validate_runtime_security()


def test_validate_runtime_security_requires_token_in_prod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TSF_ENV", "prod")
    monkeypatch.delenv("TSF_API_TOKEN", raising=False)
    monkeypatch.delenv("API_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="TSF_API_TOKEN"):
        validate_runtime_security()


def test_get_cors_allow_origins_defaults_to_local_origins_in_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TSF_ENV", "dev")
    monkeypatch.delenv("TSF_CORS_ALLOW_ORIGINS", raising=False)

    assert get_cors_allow_origins() == DEV_CORS_ORIGINS


def test_get_cors_allow_origins_is_empty_in_prod_without_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TSF_ENV", "prod")
    monkeypatch.delenv("TSF_CORS_ALLOW_ORIGINS", raising=False)

    assert get_cors_allow_origins() == []


def test_verify_api_token_rejects_missing_or_invalid_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TSF_API_TOKEN", "super-secret-token-value")

    with pytest.raises(Exception):
        verify_api_token(None)

    with pytest.raises(Exception):
        verify_api_token("Bearer wrong-token")


def test_verify_api_token_accepts_matching_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TSF_API_TOKEN", "super-secret-token-value")

    verify_api_token("Bearer super-secret-token-value")


def test_redact_client_ip_masks_ipv4_and_ipv6() -> None:
    assert redact_client_ip("192.168.10.24") == "192.168.10.0"
    assert redact_client_ip("2001:db8:abcd:1234:1111:2222:3333:4444") == "2001:db8:abcd::"
