from __future__ import annotations

from typing import Any, Dict


def build_rolling_snapshot(config: Dict[str, Any], informer_cfg: Dict[str, Any]) -> Dict[str, Any]:
    roll_cfg = (config.get("prediction", {}) or {}).get("rolling", {}) or {}
    try:
        pred_len_snap = int(informer_cfg.get("pred_len", 24))
    except Exception:
        pred_len_snap = 24
    step_snap = roll_cfg.get("step", pred_len_snap)
    try:
        step_snap = int(step_snap)
    except Exception:
        step_snap = pred_len_snap
    return {
        "enabled": bool(roll_cfg.get("enabled", True)),
        "mode": str(roll_cfg.get("mode", "overwrite")),
        "step": step_snap,
        "pred_len": pred_len_snap,
        "calibrate": bool(roll_cfg.get("calibrate", True)),
    }


def maybe_auto_adjust_windows(
    informer_cfg: Dict[str, Any],
    train_len: int,
    val_len: int,
) -> str | None:
    seq_len = int(informer_cfg.get("seq_len", 96))
    label_len = int(informer_cfg.get("label_len", 48))
    pred_len = int(informer_cfg.get("pred_len", 24))
    if label_len > seq_len:
        label_len = seq_len
    n_min = int(min(train_len, val_len))
    required = seq_len + pred_len
    if n_min >= required:
        return None

    pred_len_new = max(1, min(pred_len, max(1, int(n_min * 0.2))))
    seq_len_new = max(4, n_min - pred_len_new)
    label_len_new = min(label_len, seq_len_new)
    informer_cfg["seq_len"] = int(seq_len_new)
    informer_cfg["label_len"] = int(label_len_new)
    informer_cfg["pred_len"] = int(pred_len_new)
    return (
        f"[informer] train/val 数据不足自动缩短窗口: seq_len={seq_len_new}, "
        f"label_len={label_len_new}, pred_len={pred_len_new} "
        f"(train={train_len}, val={val_len})"
    )
