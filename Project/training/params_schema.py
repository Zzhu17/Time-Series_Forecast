from __future__ import annotations

from typing import Any, Dict

SCHEMA_VERSION = "training_params.v1"


def _to_int(v: Any) -> int:
    try:
        out = int(v)
    except Exception:
        return 0
    return max(0, out)


def build_training_params(
    *,
    model: str,
    split: Dict[str, Any],
    core_hparams: Dict[str, Any] | None = None,
    runtime: Dict[str, Any] | None = None,
    data_signature: Dict[str, Any] | None = None,
    trainer_version: str = SCHEMA_VERSION,
    legacy_fields: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": str(model).strip().lower(),
        "split": {
            "train_len": _to_int((split or {}).get("train_len")),
            "val_len": _to_int((split or {}).get("val_len")),
            "test_len": _to_int((split or {}).get("test_len")),
        },
        "core_hparams": dict(core_hparams or {}),
        "runtime": dict(runtime or {}),
        "data_signature": dict(data_signature or {}),
        "trainer_version": str(trainer_version or SCHEMA_VERSION),
    }
    if isinstance(legacy_fields, dict):
        for k, v in legacy_fields.items():
            if k not in payload:
                payload[k] = v
    return payload


def validate_training_params_schema(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("training_params must be a dict")
    for key in ("model", "split", "core_hparams", "runtime", "data_signature", "trainer_version"):
        if key not in payload:
            raise ValueError(f"missing required field: {key}")
    if not isinstance(payload["model"], str) or not payload["model"].strip():
        raise ValueError("model must be a non-empty string")
    split = payload["split"]
    if not isinstance(split, dict):
        raise ValueError("split must be a dict")
    for k in ("train_len", "val_len", "test_len"):
        if k not in split:
            raise ValueError(f"split missing field: {k}")
        try:
            raw = split.get(k)
            if int(raw) < 0:
                raise ValueError(f"split.{k} must be >= 0")
        except Exception as e:
            raise ValueError(f"split.{k} must be int-like") from e
    if not isinstance(payload["core_hparams"], dict):
        raise ValueError("core_hparams must be a dict")
    if not isinstance(payload["runtime"], dict):
        raise ValueError("runtime must be a dict")
    if not isinstance(payload["data_signature"], dict):
        raise ValueError("data_signature must be a dict")
    if not isinstance(payload["trainer_version"], str) or not payload["trainer_version"].strip():
        raise ValueError("trainer_version must be a non-empty string")
