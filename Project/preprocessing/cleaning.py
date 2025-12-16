import pandas as pd

def clean_data(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df = df.copy()

    # 删除 value_col 为 NaN 的行
    df = df.dropna(subset=[value_col])

    # 转换为数值型，非法字符变为 NaN
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

    # 替换 inf/-inf 并删除
    df = df.replace([float('inf'), float('-inf')], pd.NA).dropna(subset=[value_col])

    # 提示数据量不足
    if df.shape[0] < 10:
        print("⚠️ 数据过少，模型可能无法训练")

    return df.reset_index(drop=True)

def check_columns(df: pd.DataFrame, time_col: str, value_col: str):
    if time_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"缺少列: {time_col} 或 {value_col}")
    df[time_col] = pd.to_datetime(df[time_col])
    return df


# 智能自动检测特征列和目标列
def auto_detect_columns(df, target_col=None, time_keywords=None):
    """
    智能自动检测 feature_cols（输入特征）和 target_col（预测目标）
    适配任何行业/表头，默认排除时间戳列和目标列。
    """
    if time_keywords is None:
        time_keywords = ["date", "time", "datetime", "timestamp"]

    # 自动猜测目标列
    if target_col is None:
        # 按常用目标列名优先级猜测
        for col in df.columns:
            if col.lower() in ["value", "target", "close", "temperature", "label"]:
                target_col = col
                break
        if target_col is None:
            target_col = df.columns[-1]  # 默认最后一列

    # 自动检测时间戳列
    ts_cols = [col for col in df.columns if any(kw in col.lower() for kw in time_keywords)]
    # 自动识别特征列（排除时间戳和目标）
    feature_cols = [col for col in df.columns if col not in ts_cols and col != target_col]
    return feature_cols, target_col