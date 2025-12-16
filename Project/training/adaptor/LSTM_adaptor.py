# training/adaptor/lstm_adaptor.py
from __future__ import annotations
import pandas as pd

from training.train_lstm import train_lstm_model
from utils.array_utils import clean_dataframe

def _read_defaults(config: dict) -> tuple[str, str]:
    # 兼容 default 段与根级两种写法
    dft = config.get("default", {})
    time_col = config.get("time_col", dft.get("time_col", "date"))
    value_col = config.get("value_col", dft.get("value_col", "value"))
    return time_col, value_col

def _read_lstm_hparams(config: dict) -> dict:
    """
    统一从 model_config.LSTM 读取；若没有则兜底到旧键。
    不碰训练循环，只整理出 train_lstm_model 期望看到的键。
    """
    mcfg = (config.get("model_config") or {}).get("LSTM", {})
    h = {
        # train_lstm_model 目前读取的键（保持兼容，不改它）
        "hidden_size":   mcfg.get("hidden_dim",  config.get("hidden_size", 50)),
        "num_layers":    mcfg.get("num_layers",  config.get("num_layers", 1)),
        "learning_rate": mcfg.get("learning_rate", config.get("learning_rate", 1e-3)),
        "epochs":        mcfg.get("n_epochs",    config.get("epochs", 10)),
        "weight_decay":  mcfg.get("weight_decay", config.get("weight_decay", 0.0)),
        "dropout":       mcfg.get("dropout",     config.get("dropout", 0.0)),

        # 序列长度等（train_lstm_model 里用到了 seq_len）
        "seq_len":       mcfg.get("seq_len", 60),
        # label_len / pred_len 暂不强耦合 train_lstm_model；保留在 config 便于未来扩展
        "label_len":     mcfg.get("label_len", 30),
        "pred_len":      mcfg.get("pred_len", 30),

        # 其他可选
        "batch_size":    mcfg.get("batch_size", 32),
    }
    return h

def _build_subconfig_for_train(call_base: dict, time_col: str, value_col: str, hparams: dict) -> dict:
    """
    生成传给 train_lstm_model 的最小配置字典（不改变其读取逻辑）。
    """
    cfg = dict(call_base or {})
    cfg.update({
        "time_col":  time_col,
        "value_col": value_col,
        # 兼容 train_lstm_model 的键名
        "hidden_size":   hparams["hidden_size"],
        "num_layers":    hparams["num_layers"],
        "learning_rate": hparams["learning_rate"],
        "epochs":        hparams["epochs"],
        "seq_len":       hparams["seq_len"],
        # 预留但不强依赖
        "batch_size":    hparams["batch_size"],
        "label_len":     hparams["label_len"],
        "pred_len":      hparams["pred_len"],
        "weight_decay":  hparams["weight_decay"],
        "dropout":       hparams["dropout"],
    })
    return cfg

def train_lstm_model_7tuple(df: pd.DataFrame, config: dict):
    """
    统一适配：
    - 读配置（含兜底）
    - 清洗 df（只做安全处理，不改变语义）
    - 调用新的 train_lstm_model（单次即可同时给出 val/test）
    - 返回统一 7 元组
    """
    # 1) 读取字段
    time_col, value_col = _read_defaults(config)
    hparams = _read_lstm_hparams(config)

    # 2) 清洗/类型安全（不改变你的训练逻辑）
    _df_clean = clean_dataframe(df, value_col=value_col, time_col=time_col, feature_cols=None)
    if _df_clean is not None:
        df = _df_clean

    # 3) 将归一化后的超参写回 config，保证 train_lstm_model 看到一致的键
    lstm_cfg = config.setdefault("model_config", {}).setdefault("LSTM", {})
    lstm_cfg.setdefault("hidden_dim", hparams["hidden_size"])
    lstm_cfg.setdefault("num_layers", hparams["num_layers"])
    lstm_cfg.setdefault("learning_rate", hparams["learning_rate"])
    lstm_cfg.setdefault("n_epochs", hparams["epochs"])
    lstm_cfg.setdefault("seq_len", hparams["seq_len"])
    lstm_cfg.setdefault("label_len", hparams["label_len"])
    lstm_cfg.setdefault("pred_len", hparams["pred_len"])
    lstm_cfg.setdefault("batch_size", hparams["batch_size"])
    lstm_cfg.setdefault("weight_decay", hparams["weight_decay"])
    lstm_cfg.setdefault("dropout", hparams["dropout"])

    # 保证顶层/默认 time/value col 可用（训练/绘图一致）
    config.setdefault("time_col", time_col)
    config.setdefault("value_col", value_col)
    config.setdefault("default", {}).setdefault("time_col", time_col)
    config.setdefault("default", {}).setdefault("value_col", value_col)

    # 将原始 df 透传（供训练/绘图/缓存使用）
    config.setdefault("data", {})
    config["data"].setdefault("df", df.copy())
    config["data"].setdefault("dataframe", df.copy())

    # 4) 直接调用新 LSTM 训练（单次即可获得 val/test 结果）
    cfg_for_train = _build_subconfig_for_train(config, time_col, value_col, hparams)
    return train_lstm_model(df, cfg_for_train)
