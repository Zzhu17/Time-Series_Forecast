import pandas as pd
import random
import numpy as np
from training.adaptor.common import (
    build_adapter_training_params,
    extract_dense_predictions,
    extract_split_predictions,
    infer_split_lengths,
)


def _apply_informer_smoke_config(config: dict) -> None:
    training_cfg = (config.get("training") or {})
    smoke_cfg = (training_cfg.get("smoke") or {})
    if not bool(smoke_cfg.get("enabled", False)):
        return
    inf_cfg = config.setdefault("model_config", {}).setdefault("Informer", {})
    inf_cfg["batch_size"] = min(int(inf_cfg.get("batch_size", 32)), int(smoke_cfg.get("batch_size", 8)))
    inf_cfg["n_epochs"] = min(int(inf_cfg.get("n_epochs", 10)), int(smoke_cfg.get("epochs", 2)))


def _smoke_mode_enabled(config: dict) -> bool:
    if bool(config.get("smoke_mode", False)):
        return True
    smoke_cfg = ((config.get("training") or {}).get("smoke") or {})
    return bool(smoke_cfg.get("enabled", False))


def train_informer_model_7tuple(df, config):
    """
    适配器：不改 informer/train.py；
    调用原 train_informer_model(config) -> (model, result_df)
    再转换为统一 7 元组。
    """
    # 延迟导入，避免循环依赖
    from models.informer.train import train_informer_model
    _apply_informer_smoke_config(config)

    artifacts = config.setdefault("artifacts", {})

    # 原 informer 训练入口通常只吃 config，这里按你现状调用
    seed = int(config.get("seed", config.get("default", {}).get("seed", 42)) or 42)
    random.seed(seed)
    np.random.seed(seed)
    arts = config.setdefault("artifacts", {})
    smoke_mode = _smoke_mode_enabled(config)
    arts["training_meta"] = {"model": "informer", "seed": seed, "smoke_mode": smoke_mode}
    if smoke_mode:
        config.setdefault("model_config", {}).setdefault("Informer", {})["n_epochs"] = 1
    try:
        final_model, result_df = train_informer_model(config)
    except Exception as exc:
        raise RuntimeError(f"informer_train_failed: {type(exc).__name__}: {exc}") from exc

    # 优先使用训练过程写回的 dense 输出（严格对齐 6:2:2 的 val/test 段长度）
    data_blk = config.get("data", {}) or {}

    def _pick_df(*candidates):
        for c in candidates:
            if isinstance(c, pd.DataFrame) and not c.empty:
                return c
        return None

    val_df = _pick_df(data_blk.get("val_dense"), data_blk.get("val_result_df"))
    test_df = _pick_df(data_blk.get("test_dense"), data_blk.get("test_result_df"))

    val_true, val_forecast = extract_dense_predictions(val_df)
    test_true, test_forecast = extract_dense_predictions(test_df)

    # 兼容：若 dense 未生成，则回退解析 result_df（但不要再做 80/20，优先用 split）
    if val_true is None or test_true is None:
        val_true = val_forecast = test_true = test_forecast = None

        if isinstance(result_df, pd.DataFrame):
            val_true, val_forecast, test_true, test_forecast = extract_split_predictions(
                result_df.copy(),
                split=data_blk.get("split") if isinstance(data_blk.get("split"), dict) else None,
            )

    # 统一返回 training_params，便于注册/落盘追踪
    split_info = data_blk.get("split") if isinstance(data_blk.get("split"), dict) else {}
    split = infer_split_lengths(df, val_true, test_true)
    split["train_len"] = int(split_info.get("train_len") or split["train_len"])
    split["val_len"] = int(split_info.get("val_len") or split["val_len"])
    split["test_len"] = int(split_info.get("test_len") or split["test_len"])
    epochs = int(((config.get("model_config") or {}).get("Informer") or {}).get("n_epochs", 0))
    training_params = build_adapter_training_params(
        model="informer",
        df=df,
        config=config,
        split=split,
        core_hparams={"epochs": epochs},
        runtime={"fit_status": "trained", "seed": seed},
        legacy_fields={"model_name": "informer", "fit_status": "trained", "epochs": epochs},
        artifacts=artifacts,
    )
    test_forecast_df = data_blk.get("test_dense") if isinstance(data_blk.get("test_dense"), pd.DataFrame) else data_blk.get("test_result_df")

    return (
        val_true, val_forecast,
        test_true, test_forecast,
        final_model, test_forecast_df,
        training_params
    )


def get_forecaster():
    from models.base import TrainFunctionForecaster

    return TrainFunctionForecaster("informer", train_informer_model_7tuple)
