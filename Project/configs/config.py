import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs.yaml')

# 支持的模型结构规则：限制允许传入构造器的字段
MODEL_CONFIG_RULES = {
    "informer": {
        'seq_len', 'label_len', 'pred_len',
        'enc_in', 'dec_in', 'c_out',
        'd_model', 'n_heads', 'e_layers', 'd_layers',
        'd_ff', 'dropout', 'activation', 'device'
    },
    "lstm": {
        'input_size', 'hidden_size', 'num_layers', 'output_size',
        'dropout', 'device'
    },
    "prophet": set(),  # Prophet 无需额外参数
    "random_forest": {
        'n_estimators', 'max_depth', 'random_state'
    }
    # 可以继续添加其他模型的规则
}

def get_default_config():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    config.setdefault("columns", {
        "time_col": "date",
        "value_col": "value"
    })
    return config

# Helper function to load a YAML configuration file
def load_yaml_config(path: str = CONFIG_PATH) -> dict:
    """
    Load a YAML configuration file and return its contents as a dict.
    Defaults to loading from CONFIG_PATH.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        return {}

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

def get_thresholds(config):
    """
    返回通用的 MAPE 和 RMSE 阈值。
    """
    return config.get('thresholds', {"MAPE": 0.1, "RMSE": 10})

def is_residual_enabled(config):
    """
    是否启用残差建模。
    """
    return config.get('residual_modeling', {}).get('enabled', False)

def get_logging_config(config):
    return config.get("logging", {})