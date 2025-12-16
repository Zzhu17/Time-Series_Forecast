import torch
from typing import Dict, Optional, Any

# 单一来源的默认配置（便于外部查看与复用）
DEFAULT_INFORMER_CONFIG: Dict[str, Any] = {
    # 核心长度配置（统一命名）
    'seq_len': 192,       # Encoder 输入序列长度
    'label_len': 96,      # Decoder 输入历史长度
    'pred_len': 48,       # 预测长度

    # 模型结构
    'enc_in': 1,
    'dec_in': 1,
    'c_out': 1,
    'd_model': 512,
    'n_heads': 8,
    'e_layers': 2,
    'd_layers': 1,
    'd_ff': 2048,
    'dropout': 0.05,
    'factor': 5,
    'attn': 'prob',
    'embed': 'fixed',
    'freq': 'h',
    'activation': 'gelu',

    # 运行环境与训练相关
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'dtype': torch.float32,
    'batch_size': 32,
    'lr': 1e-3,
    'use_residual': True,
    'num_workers': 0,     # DataLoader 进程数
    'weight_decay': 0.0   # Adam/L2 正则
}


def _normalize_types(cfg: Dict[str, Any]) -> None:
    """就地规范 device / dtype 的类型表示。"""
    if isinstance(cfg.get('device'), str):
        cfg['device'] = torch.device(cfg['device'])
    if isinstance(cfg.get('dtype'), str):
        # 允许传入 'float32' / 'float64' 等字符串
        dtype_name = cfg['dtype']
        if hasattr(torch, dtype_name):
            cfg['dtype'] = getattr(torch, dtype_name)


def get_informer_config(user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    合并用户配置与默认配置，并进行必要的健壮性修复（如 None 值、device/dtype 规范）。

    参数
    ----
    user_config: 可选的用户覆盖字典；若为 None 则使用空字典。

    返回
    ----
    dict: 可直接用于模型/数据加载的完整配置。
    """
    user_cfg = {} if user_config is None else dict(user_config)

    # 先合并：用户优先覆盖
    config: Dict[str, Any] = {**DEFAULT_INFORMER_CONFIG, **user_cfg}

    # 将显式传入的 None 改回默认值（避免下游报错）
    for k, default_v in DEFAULT_INFORMER_CONFIG.items():
        if config.get(k) is None:
            config[k] = default_v

    # 规范类型
    _normalize_types(config)

    return config


# 保留一个兼容别名，避免外部旧代码引用失败（去除重复实现）
complete_informer_config = get_informer_config