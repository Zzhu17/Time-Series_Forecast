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
