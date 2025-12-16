import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Tuple, Optional, Callable
from utils.array_utils import debug_stat
from utils.sliding_windows import create_windows_for_informer
def auto_transformer_shape(x):
    """
    全局 3D reshape 工具。任何 1D/2D/3D 输入自动转为 3D (batch, seq, feature)。
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1, 1)
    elif x.ndim == 2:
        return x.reshape(x.shape[0], x.shape[1], 1)
    elif x.ndim == 3:
        return x
    else:
        raise ValueError(f"Unsupported input ndim for transformer: {x.shape}")

def get_x_feature(x_enc, flatten_to_2d=True):
    """
    获取 residual 所需特征向量。
    - x_enc: 输入编码器的 3D 张量
    - flatten_to_2d: 是否 flatten 成 (batch, -1)
    返回: (batch, n_feature_flattened) or (batch, seq, n_feature)
    """
    x_enc = np.asarray(x_enc)
    if x_enc.ndim == 3:
        if flatten_to_2d:
            return x_enc.reshape(x_enc.shape[0], -1)
        return x_enc
    elif x_enc.ndim == 2:
        return x_enc if not flatten_to_2d else x_enc
    elif x_enc.ndim == 1:
        return x_enc.reshape(-1, 1)
    raise ValueError(f"get_x_feature: only support 1D/2D/3D, got {x_enc.shape}")

def build_main_and_feature_windows(
    values, features=None, seq_len=48, pred_len=24, label_len=24
):
    """
    构建 Transformer 主输入、特征输入滑窗（3D）。
    values: shape [N] 或 [N, n_features]
    features: shape [N, n_extra_features]（可选）
    返回:
        x_enc: (batch, seq_len, n_feature)
        x_dec: (batch, label_len + pred_len, n_feature)
        y_label: (batch, pred_len, n_feature)
        x_feature: (batch, seq_len, n_feature) or (batch, seq_len, n_feature+n_extra) 取决于 features
    """
    values = np.asarray(values)
    features = np.asarray(features) if features is not None else None
    n = len(values)
    window_size = seq_len + label_len + pred_len
    X_enc, X_dec, Y, X_feature = [], [], [], []
    for i in range(n - window_size + 1):
        main_seq = values[i : i + seq_len]
        dec_seq = values[i + seq_len : i + seq_len + label_len + pred_len]
        label_seq = values[i + seq_len + label_len : i + seq_len + label_len + pred_len]
        X_enc.append(main_seq)
        X_dec.append(dec_seq)
        Y.append(label_seq)
        # 特征窗口
        if features is not None:
            feat_window = features[i : i + seq_len]
            X_feature.append(feat_window)
    X_enc = np.array(X_enc)
    X_dec = np.array(X_dec)
    Y = np.array(Y)
    X_feature = np.array(X_feature) if len(X_feature) > 0 else None
    # reshape
    X_enc = auto_transformer_shape(X_enc)
    X_dec = auto_transformer_shape(X_dec)
    Y = auto_transformer_shape(Y)
    if X_feature is not None:
        X_feature = auto_transformer_shape(X_feature)
    # --- 可视化断点检查滑窗采样 ---
    # 打印前3个x/y滑窗实际样本（便于人工检查是否与原始数据对齐）
    for k in range(min(3, len(X_enc))):
        print(f"[DEBUG][滑窗检查] x_enc[{k}]: {X_enc[k].flatten()}")
        print(f"[DEBUG][滑窗检查] Y[{k}]: {Y[k].flatten()}")
    debug_stat('build_main_and_feature_windows/X_enc', X_enc)
    debug_stat('build_main_and_feature_windows/X_dec', X_dec)
    debug_stat('build_main_and_feature_windows/Y', Y)
    if X_feature is not None:
        debug_stat('build_main_and_feature_windows/X_feature', X_feature)
    # --- end ---
    return X_enc, X_dec, Y, X_feature

def prepare_informer_inputs(
    df_scaled: pd.DataFrame, 
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    【桥梁函数】从已经归一化好的 DataFrame 创建 Informer 的输入数组。

    此函数职责单一：
    1. 从 DataFrame 中提取数值。
    2. 调用核心的滑动窗口生成器。
    3. 返回生成好的四个 NumPy 数组。

    Args:
        df_scaled (pd.DataFrame): 已经过归一化处理的数据帧。
        config (dict): 完整的配置字典（可包含 data.all_feature_cols 或 model_config.Informer.feature_cols）。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        返回 x_enc, x_dec, y_label, x_feature 四个数组。
    """
    informer_cfg = config['model_config']['Informer']

    # --- 1) 选择要作为“值通道”的列集合（优先 pipeline 注入的全集） ---
    data_cfg = config.get('data', {})  # 安全读取
    feature_cols = data_cfg.get('all_feature_cols')
    if not feature_cols:
        feature_cols = informer_cfg.get('feature_cols')
    if not feature_cols:
        # 再兜底：从 default.value_col 回退
        default_cfg = config.get('default', {})
        fallback_col = default_cfg.get('value_col', 'value')
        feature_cols = [fallback_col]

    # 校验列名是否存在
    missing = [c for c in feature_cols if c not in df_scaled.columns]
    if missing:
        raise KeyError(f"prepare_informer_inputs: 缺少必要列 {missing}，当前 df_scaled.columns={list(df_scaled.columns)}")

    # --- 2) 取窗口参数并做长度预检（可读错误早抛出） ---
    seq_len = int(informer_cfg.get('seq_len', 96))
    label_len = int(informer_cfg.get('label_len', 48))
    pred_len = int(informer_cfg.get('pred_len', 24))
    ensure_valid_window_length(df_scaled, seq_len, label_len, pred_len)

    # --- 3) 取值矩阵（float32） ---
    data_values = np.asarray(df_scaled[feature_cols].values, dtype=np.float32)

    # --- 4) 真正生成滑窗 ---
    x_enc, x_dec, y_label, x_feature = create_windows_for_informer(
        data_values=data_values, 
        config=informer_cfg  # 传递 Informer 的特定配置（包含 seq_len/label_len/pred_len 等）
    )

    return x_enc, x_dec, y_label, x_feature

def ensure_valid_window_length(arr, seq_len, label_len, pred_len):
    """
    检查窗口长度是否足够，保险处理 DataFrame/ndarray/1D/2D/3D。
    """
    if hasattr(arr, "shape"):
        n = arr.shape[0]
    elif hasattr(arr, "__len__"):
        n = len(arr)
    else:
        raise ValueError("arr must be a DataFrame or array-like object")
    required = seq_len + label_len + pred_len
    if n < required:
        raise ValueError(
            f"❌ 样本数不足，当前{n}，需要至少{required}（窗口参数：seq_len={seq_len}, label_len={label_len}, pred_len={pred_len}）。"
        )


# 新增：适配 config 的 DataLoader 构造器
def make_informer_loader(
    x_enc: np.ndarray,
    x_dec: np.ndarray,
    y_label: np.ndarray,
    config: Dict[str, Any],
    shuffle: bool = True,
    *,
    generator: Optional[torch.Generator] = None,
    worker_init_fn: Optional[Callable[[int], None]] = None,
    **dloader_kwargs,
) -> DataLoader:
    """
    创建 Informer 的 DataLoader。
    """
    informer_cfg = config['model_config']['Informer']
    batch_size = informer_cfg.get('batch_size', 32)
    
    # 将所有数据打包成一个 TensorDataset
    # 注意：y_label 包含了 label_len 和 pred_len 两部分
    dataset = TensorDataset(
        torch.from_numpy(x_enc).float(),
        torch.from_numpy(x_dec).float(),
        torch.from_numpy(y_label).float()
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # 本地调试建议为0；如需改为>0，配合 worker_init_fn 保证可复现
        generator=generator,
        worker_init_fn=worker_init_fn,
        **dloader_kwargs,
    )
    return loader