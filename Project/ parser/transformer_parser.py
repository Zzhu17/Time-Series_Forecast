from typing import Dict, Any

def get_transformer_config() -> Dict[str, Any]:
    """
    返回用于构建 Transformer 或 Informer 模型的默认配置参数。
    后续可接入 argparse、streamlit 或 YAML 配置覆盖。
    """
    return {
        # 模型结构参数
        "seq_len": 96,
        "label_len": 48,
        "pred_len": 48,
        "e_layers": 2,
        "d_layers": 1,
        "d_model": 256,
        "d_ff": 512,
        "n_heads": 8,
        "dropout": 0.1,
        "activation": "gelu",

        # 输入输出通道（适用于多变量）
        "enc_in": 1,
        "dec_in": 1,
        "c_out": 1,

        # 训练参数
        "batch_size": 16,
        "lr": 0.001,
        "n_epochs": 10,

        # 通用列定义
        "time_col": "date",
        "value_col": "value",

        # 控制项
        "device": "cpu",
    }