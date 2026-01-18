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

    # 优先使用训练过程写回的 dense 输出（严格对齐 6:2:2 的 val/test 段长度）
    import numpy as np
    data_blk = config.get("data", {}) or {}

    def _extract_from_df(df_like):
        if not isinstance(df_like, pd.DataFrame):
            return None, None
        if not {"y_true", "yhat"} <= set(df_like.columns):
            return None, None
        y_t = np.asarray(df_like["y_true"].to_numpy(), dtype=float).reshape(-1)
        y_h = np.asarray(df_like["yhat"].to_numpy(), dtype=float).reshape(-1)
        L = min(len(y_t), len(y_h))
        return y_t[:L], y_h[:L]

    def _pick_df(*candidates):
        for c in candidates:
            if isinstance(c, pd.DataFrame) and not c.empty:
                return c
        return None

    val_df = _pick_df(data_blk.get("val_dense"), data_blk.get("val_result_df"))
    test_df = _pick_df(data_blk.get("test_dense"), data_blk.get("test_result_df"))

    val_true, val_forecast = _extract_from_df(val_df)
    test_true, test_forecast = _extract_from_df(test_df)

    # 兼容：若 dense 未生成，则回退解析 result_df（但不要再做 80/20，优先用 split）
    if val_true is None or test_true is None:
        val_true = val_forecast = test_true = test_forecast = None

        if isinstance(result_df, pd.DataFrame):
            df_ = result_df.copy()
            # 常见两种形态：1) 有 phase 列 2) 只有 y_true / yhat 的分段
            if "phase" in df_.columns:
                is_val = df_["phase"].astype(str).str.lower().eq("val")
                is_tst = df_["phase"].astype(str).str.lower().eq("test")
                if {"y_true", "yhat"} <= set(df_.columns):
                    val_true      = np.asarray(df_.loc[is_val, "y_true"].to_numpy(), dtype=float).reshape(-1)
                    val_forecast  = np.asarray(df_.loc[is_val, "yhat"].to_numpy(), dtype=float).reshape(-1)
                    test_true     = np.asarray(df_.loc[is_tst, "y_true"].to_numpy(), dtype=float).reshape(-1)
                    test_forecast = np.asarray(df_.loc[is_tst, "yhat"].to_numpy(), dtype=float).reshape(-1)
            else:
                # 兜底：若没有 phase，但有 y_true/yhat，则按 config['data']['split'] 切分
                if {"y_true", "yhat"} <= set(df_.columns):
                    split = data_blk.get("split") or {}
                    v = int(split.get("val_len") or 0)
                    te = int(split.get("test_len") or 0)
                    if v > 0 and te > 0 and len(df_) >= v + te:
                        val_part = df_.iloc[-(v + te): -te]
                        test_part = df_.iloc[-te:]
                        val_true, val_forecast = _extract_from_df(val_part)
                        test_true, test_forecast = _extract_from_df(test_part)
                    else:
                        # 最后兜底：全当验证（测试为空）
                        val_true, val_forecast = _extract_from_df(df_)
                        test_true, test_forecast = np.array([], dtype=float), np.array([], dtype=float)

    # 最佳超参：Informer 若无调参可设 None（保持统一位置）
    best_params = None
    test_forecast_df = data_blk.get("test_dense") if isinstance(data_blk.get("test_dense"), pd.DataFrame) else data_blk.get("test_result_df")

    return (
        val_true, val_forecast,
        test_true, test_forecast,
        final_model, test_forecast_df,
        best_params
    )


def get_forecaster():
    from models.base import TrainFunctionForecaster

    return TrainFunctionForecaster("informer", train_informer_model_7tuple)
