import logging
import os
from typing import Optional

import yaml

LOGGER = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs.yaml")


def _env_name() -> str:
    return str(os.getenv("TSF_ENV") or os.getenv("ENV") or "").strip()


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out.get(key, {}), val)
        else:
            out[key] = val
    return out

# Helper function to load a YAML configuration file
def load_yaml_config(path: str = CONFIG_PATH, env: Optional[str] = None) -> dict:
    """
    Load a YAML configuration file and return its contents as a dict.
    Defaults to loading from CONFIG_PATH.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
        env_name = str(env or _env_name())
        if env_name:
            env_path = os.path.join(os.path.dirname(path), f"configs.{env_name}.yaml")
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    env_cfg = yaml.safe_load(f) or {}
                base = _deep_merge(base, env_cfg)
        return base
    except FileNotFoundError:
        LOGGER.warning("Config file not found at %s; using empty config", path)
        return {}
    except Exception as e:
        LOGGER.error("Failed to load config from %s: %s", path, e)
        raise

def get_informer_config(layer: dict) -> dict:
    layer = dict(layer or {})
    d = {
        "seq_len":   int(layer.get("seq_len", 48)),
        "label_len": int(layer.get("label_len", 24)),
        "pred_len":  int(layer.get("pred_len", 24)),
        "batch_size": int(layer.get("batch_size", 32)),
        "n_epochs":  int(layer.get("n_epochs", 10)),
        "d_model":   int(layer.get("d_model", 64)),
        "d_ff":      int(layer.get("d_ff", 128)),
        "n_heads":   int(layer.get("n_heads", 2)),
        "e_layers":  int(layer.get("e_layers", 2)),
        "d_layers":  int(layer.get("d_layers", 1)),
        "dropout":   float(layer.get("dropout", 0.0)),
        "factor":    int(layer.get("factor", 5)),
        "attn":      str(layer.get("attn", "prob")),
        "embed":     str(layer.get("embed", "fixed")),
        "freq":      str(layer.get("freq", "t")),
        "enc_in":    int(layer.get("enc_in", 1)),
        "dec_in":    int(layer.get("dec_in", 1)),
        "c_out":     int(layer.get("c_out", 1)),
        "activation": str(layer.get("activation", "gelu")),
        "device":     str(layer.get("device", "cpu")),
        "feature_cols": list(layer.get("feature_cols", ["value"])),
        # 可选：训练细节
        "stride": int(layer.get("stride", 1)),
        "drop_last": bool(layer.get("drop_last", False)),
    }
    # 简单健壮性（防负数/零）
    for k in ("seq_len","label_len","pred_len","batch_size","n_epochs","d_model","d_ff","n_heads","e_layers","d_layers"):
        if d[k] <= 0:
            d[k] = {"batch_size":1}.get(k, max(1, d[k]))
    return d
