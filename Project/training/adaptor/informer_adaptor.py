import pandas as pd

def train_informer_model_7tuple(df, config):
    """
    适配器：不改 informer/train.py；
    调用原 train_informer_model(config) -> (model, result_df)
    再转换为统一 7 元组。
    """
    # 延迟导入，避免循环依赖
    from models.informer.train import train_informer_model

    # 原 informer 训练入口通常只吃 config，这里按你现状调用
    final_model, result_df = train_informer_model(config)

    # 尝试从 result_df 中拆出 val/test 的 y_true/yhat
    val_true = val_forecast = test_true = test_forecast = None

    if isinstance(result_df, pd.DataFrame):
        df_ = result_df.copy()
        # 常见两种形态：1) 有 phase 列 2) 只有 y_true / yhat 的分段
        if "phase" in df_.columns:
            is_val = df_["phase"].astype(str).str.lower().eq("val")
            is_tst = df_["phase"].astype(str).str.lower().eq("test")
            if {"y_true", "yhat"} <= set(df_.columns):
                val_true      = df_.loc[is_val, "y_true"].to_numpy()
                val_forecast  = df_.loc[is_val, "yhat"].to_numpy()
                test_true     = df_.loc[is_tst, "y_true"].to_numpy()
                test_forecast = df_.loc[is_tst, "yhat"].to_numpy()
        else:
            # 兜底：若没有 phase，但有 y_true/yhat，则按 80/20 时间顺序切分
            if {"y_true", "yhat"} <= set(df_.columns):
                n = len(df_)
                cut = int(n * 0.8)
                val_true      = df_.iloc[:cut]["y_true"].to_numpy()
                val_forecast  = df_.iloc[:cut]["yhat"].to_numpy()
                test_true     = df_.iloc[cut:]["y_true"].to_numpy()
                test_forecast = df_.iloc[cut:]["yhat"].to_numpy()

    # 最佳超参：Informer 若无调参可设 None（保持统一位置）
    best_params = None
    test_forecast_df = None  # Informer 若有单独 test df，可在此返回；否则 None

    return (
        val_true, val_forecast,
        test_true, test_forecast,
        final_model, test_forecast_df,
        best_params
    )