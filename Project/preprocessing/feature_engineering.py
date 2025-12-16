import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import List, Dict, Any, Tuple, cast

def generate_features(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """
    根据配置生成时间序列特征。

    Args:
        df (pd.DataFrame): 输入的数据帧。
        config (Dict[str, Any]): 配置字典，可以是完整 config（含 'default' 键）或直接是包含 'time_col'/'value_col' 的子字典。

    Returns:
        Tuple[pd.DataFrame, List[str]]: 增加了特征的数据帧，以及新增的时间特征列名列表。
    """
    df = df.copy()

    time_col: str
    value_col: str

    # ---- 1) 解析 time_col / value_col（兼容完整 config 或 default 子字典） ----
    if isinstance(config, dict):
        if 'time_col' in config and 'value_col' in config:
            time_col = config['time_col']
            value_col = config['value_col']
        elif 'default' in config and isinstance(config['default'], dict):
            _t = config['default'].get('time_col')
            _v = config['default'].get('value_col')
            if _t is None or _v is None:
                raise KeyError("'default' must contain non-null 'time_col' and 'value_col'.")
            time_col = str(_t)
            value_col = str(_v)
        else:
            raise KeyError("Config must contain 'time_col' and 'value_col' (directly or inside 'default').")
    else:
        raise TypeError("config must be a dict.")

    if not isinstance(time_col, str) or not isinstance(value_col, str):
        raise TypeError("'time_col' and 'value_col' must be strings.")

    if time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' not in DataFrame columns: {list(df.columns)}")
    if value_col not in df.columns:
        # 容错：若 value_col 是可转数字的字符串列，后续仍可使用；否则抛错
        try:
            pd.to_numeric(df[value_col])
        except Exception as e:
            raise KeyError(f"value_col '{value_col}' not in DataFrame columns and cannot be interpreted as numeric.") from e

    # ---- 2) 基本预处理与排序 ----
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=cast(str, time_col))

    # ---- 3) 生成时间派生特征（全部为数值列，便于被自动纳入特征） ----
    df['month'] = df[time_col].dt.month
    df['day_of_month'] = df[time_col].dt.day
    df['day_of_week'] = df[time_col].dt.dayofweek
    df['hour'] = df[time_col].dt.hour
    df['day_of_year'] = df[time_col].dt.dayofyear
    time_feature_cols = ['month', 'day_of_month', 'day_of_week', 'hour', 'day_of_year']

    # ---- 4) 丢弃缺失行并重建索引 ----
    df = df.dropna().reset_index(drop=True)

    # ---- 5) 解析 / 推断 feature_cols（单/多变量自适配） ----
    # 优先读取配置；当 auto_feature_cols=True 或未显式指定 feature_cols 时走自动推断
    inf_cfg = config.setdefault('model_config', {}).setdefault('Informer', {})
    cfg_cols = inf_cfg.get('feature_cols')
    auto_on = bool(inf_cfg.get('auto_feature_cols', True))

    feature_cols: List[str]
    if auto_on and (not cfg_cols or len(cfg_cols) == 0):
        # 自动：取所有数值列，排除时间列；确保 value_col 在首位
        numeric_cols: List[str] = list(df.select_dtypes(include=[np.number]).columns)
        # 移除 time_col（通常是 datetime，不会出现在 numeric_cols，但稳妥起见）
        if time_col in numeric_cols:
            numeric_cols.remove(time_col)
        # 确保 value_col 在首位
        others = [c for c in numeric_cols if c != value_col]
        feature_cols = [value_col] + others
    else:
        # 使用配置给定列，但要与 df 取交集，并保证 value_col 在首位
        cfg_list: List[str] = list(cfg_cols) if cfg_cols else [value_col]
        present: List[str] = [str(c) for c in cfg_list if c in df.columns]
        missing: List[str] = [str(c) for c in cfg_list if c not in df.columns]
        if missing:
            pass
        others: List[str] = [c for c in present if c != value_col]
        head: List[str] = [value_col] if value_col in df.columns else []
        feature_cols = head + others
        if not feature_cols:
            feature_cols = [value_col]

    # ---- 6) 回写已决策的特征列，供下游（scaler/切窗/模型）统一使用 ----
    inf_cfg['feature_cols'] = feature_cols

    # 保证输出 DataFrame 至少包含 time_col 与 feature_cols（按既定顺序）
    ordered_cols: List[str] = [time_col] + [c for c in feature_cols if c in df.columns]
    # 将其他非特征列原样保留在 df 中（不改变已有下游依赖），但特征列顺序按 ordered_cols 保证
    df = df[[c for c in ordered_cols if c in df.columns] + [c for c in df.columns if c not in ordered_cols]]

    return df, time_feature_cols

def fit_and_transform_scaler(
    train_df: pd.DataFrame, 
    feature_cols: List[str], 
    scaler_path: str
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    在训练数据上拟合 StandardScaler，转换数据，并保存 scaler 对象。

    Args:
        train_df (pd.DataFrame): 训练数据集。
        feature_cols (List[str]): 需要进行归一化的特征列名列表。
        scaler_path (str): 保存 scaler 对象的文件路径。

    Returns:
        Tuple[pd.DataFrame, StandardScaler]: 归一化后的数据帧和拟合好的 scaler 对象。
    """
    scaler = StandardScaler()
    
    # 复制 DataFrame 以避免 SettingWithCopyWarning
    df_scaled = train_df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    
    # 确保保存路径的目录存在
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    # 保存 scaler 对象
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    return df_scaled, scaler

def transform_with_scaler(
    df: pd.DataFrame, 
    scaler: StandardScaler, 
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    使用一个已经存在的 scaler 来转换数据。

    Args:
        df (pd.DataFrame): 需要转换的数据帧 (例如验证集或测试集)。
        scaler (StandardScaler): 已经拟合好的 scaler 对象。
        feature_cols (List[str]): 需要进行归一化的特征列名列表。

    Returns:
        pd.DataFrame: 归一化后的数据帧。
    """
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])
    return df_scaled
