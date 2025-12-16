from __future__ import annotations
import io
import math
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Dict, Any, Protocol
import pandas as pd
class GetInformerConfig(Protocol):
    """Callable contract for informer config merging."""
    def __call__(self, user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
# === 从你的 config.py 读取默认配置（至少包含 INFORMER_DEFAULT_CONFIG）===
try:
    from models.informer.config import (  # type: ignore
        get_informer_config as _get_informer_config,
        DEFAULT_INFORMER_CONFIG,
    )
    # Ensure the callable matches our accepted signature (allows None).
    get_informer_config: GetInformerConfig = _get_informer_config  # type: ignore[assignment]
except Exception:
    DEFAULT_INFORMER_CONFIG = {
        "seq_len": 96,
        "label_len": 48,
        "pred_len": 24,
        "stride": 1,
        "batch_size": 32,
        "drop_last": False,
    }
    def get_informer_config(config: Dict[str, Any]) -> Dict[str, Any]:
        merged = {**DEFAULT_INFORMER_CONFIG, **config}
        return merged

def _safe_to_int(x: Any, default: int) -> int:
    if x in (None, "", "None"):
        return int(default)
    try:
        return int(x)
    except Exception:
        return int(default)

def _infer_freq_from_index(idx: pd.DatetimeIndex) -> Optional[str]:
    """
    尝试从 DatetimeIndex 推断采样频率 (pandas offset alias)。
    """
    try:
        # pandas 会返回最常见的推断频率；推断失败则为 None
        return pd.infer_freq(idx)
    except Exception:
        return None

def _split_df(df: pd.DataFrame, test_size: float, val_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    顺序切分：先按比例划出 test，再从剩余中划出 val，剩余为 train。
    """
    n = len(df)
    n_test = int(round(n * float(test_size)))
    n_val  = int(round(n * float(val_size)))
    # 顺序不打乱，保证时间序列的因果性
    test_df = df.iloc[-n_test:] if n_test > 0 else df.iloc[0:0]
    remain  = df.iloc[: max(0, n - n_test)]
    val_df  = remain.iloc[-n_val:] if n_val > 0 else remain.iloc[0:0]
    train_df = remain.iloc[: max(0, len(remain) - n_val)]
    return train_df.copy(), val_df.copy(), test_df.copy()

def load_csv(
    uploaded_file,
    time_col: str = "date",
    value_col: str = "value",
    test_size: float = 0.2,
    val_size: float = 0.2,
    model_name: Optional[str] = None,
    extra_config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    读取 CSV -> 基础清洗 -> 切分 -> 返回 (df, train_df, val_df, test_df, config)

    参数：
        uploaded_file: Streamlit 的文件对象或路径字符串
        time_col/value_col: 列名
        test_size/val_size: 比例（0~1）
        model_name: 当前选择的模型名（用于加载该模型默认配置）
        extra_config: 额外要写入 config 的键值（可选）

    返回：
        df, train_df, val_df, test_df, config
    """
    # === 读 CSV ===
    if hasattr(uploaded_file, "read"):
        # Streamlit 上传对象
        content = uploaded_file.read()
        df = pd.read_csv(io.BytesIO(content))
    else:
        # 路径字符串
        df = pd.read_csv(uploaded_file)

    # === 基础清洗 ===
    if time_col not in df.columns:
        # 尝试大小写/常见命名兜底
        candidates = [c for c in df.columns if c.lower() in ("date", "datetime", "time", "timestamp")]
        if candidates:
            time_col = candidates[0]
        else:
            raise ValueError(f"找不到时间列 `{time_col}`，且未发现可替代候选。列清单：{list(df.columns)}")

    if value_col not in df.columns:
        # 尝试自动找一个数值列
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            value_col = num_cols[0]
        else:
            raise ValueError(f"找不到值列 `{value_col}`，且未能自动识别任何数值列。列清单：{list(df.columns)}")

    # 时间解析 & 排序
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.sort_values(by=time_col).reset_index(drop=True)

    # 仅保留必要列（其它特征工程在后续模块中做）
    base_cols = [time_col, value_col]
    keep_cols = [c for c in base_cols if c in df.columns]
    df = df[keep_cols].copy()

    # 缺失值简单处理（值列可前向填充）
    if df[value_col].isna().any():
        df[value_col] = df[value_col].ffill().bfill()

    # === 切分 ===
    train_df, val_df, test_df = _split_df(df, test_size=test_size, val_size=val_size)

    # === 构造 config 初始信息 ===
    config: Dict[str, Any] = {
        "time_col": time_col,
        "value_col": value_col,
        "test_size": float(test_size),
        "val_size": float(val_size),
        "model_name": model_name,
    }

    # 推断频率（有助于下游窗口构造）
    freq = _infer_freq_from_index(
    pd.DatetimeIndex(pd.to_datetime(df[time_col], errors="coerce"))
    )
    if freq:
        config["freq"] = freq

    # 合并外部 extra_config（优先级低于显性赋值，后面再做“模型默认 + 显性覆盖”）
    if isinstance(extra_config, dict):
        for k, v in extra_config.items():
            if k not in config:
                config[k] = v

    # === 合并各模型默认配置（目前针对 informer）===
    if (model_name or "").lower() == "informer":
        # 使用新 config.py 的接口合并默认配置
        config = get_informer_config({**config})

    # 统一：避免字符串的 "None"
    for k, v in list(config.items()):
        if isinstance(v, str) and v.strip().lower() == "none":
            config[k] = None

    return df, train_df, val_df, test_df, config

def split_data(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    【核心新增】根据配置文件中的比例，将 DataFrame 划分为训练集、验证集和测试集。

    Args:
        df (pd.DataFrame): 已经过预处理和特征工程的完整数据集。
        config (Dict[str, Any]): 包含 'split_ratios' 的配置字典。

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
        返回 train_df, val_df, test_df 三个数据帧。
    """
    # 从配置中安全地获取划分比例
    split_ratios = config.get('split_ratios', {'train': 0.6, 'validation': 0.2, 'test': 0.2})
    train_ratio = split_ratios.get('train', 0.6)
    val_ratio = split_ratios.get('validation', 0.2)

    # 确保比例总和不超过1
    if train_ratio + val_ratio > 1.0:
        raise ValueError(f"Train ratio ({train_ratio}) and validation ratio ({val_ratio}) sum to more than 1.")

    n_samples = len(df)
    
    # 计算切分点
    train_end_idx = int(n_samples * train_ratio)
    val_end_idx = train_end_idx + int(n_samples * val_ratio)

    # 执行切分
    train_df = df.iloc[:train_end_idx].copy()
    val_df = df.iloc[train_end_idx:val_end_idx].copy()
    test_df = df.iloc[val_end_idx:].copy()

    print(f"Data split successfully: "
          f"Train set size = {len(train_df)}, "
          f"Validation set size = {len(val_df)}, "
          f"Test set size = {len(test_df)}")

    return train_df, val_df, test_df

# 新增: 通用且更稳健的 DataLoader 构造函数
def build_loader(
    X,
    Y,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    generator=None,
):

    # 基本校验
    n_x = len(X)
    n_y = len(Y)
    if n_x != n_y:
        raise ValueError(f"X/Y 样本数不一致: len(X)={n_x}, len(Y)={n_y}")
    if n_x == 0:
        raise ValueError("空数据：没有样本可用于构造 DataLoader")

    # 张量化 + 类型统一
    x_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(Y, dtype=torch.float32)
    if y_tensor.ndim == 1:
        y_tensor = y_tensor.unsqueeze(-1)  # 保证有通道维

    # 批大小自适应
    bs = max(1, min(int(batch_size), n_x))

    # pin_memory 默认行为：若有 CUDA，则开启
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    ds = TensorDataset(x_tensor, y_tensor)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        generator=generator,
        persistent_workers=bool(num_workers > 0),
    )
