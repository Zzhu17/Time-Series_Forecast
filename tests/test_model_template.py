"""
模型专项测试模板（复制后替换 `MODEL_NAME` / 训练函数 / 最小配置）.
用于统一 7-tuple 契约校验：
  (val_true, val_pred, test_true, test_pred, model, test_df, best_params)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


MODEL_NAME = "replace_me"


def _minimal_df(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "value": np.linspace(1.0, 5.0, n),
        }
    )


def _assert_7tuple_contract(out) -> None:
    assert isinstance(out, tuple), f"{MODEL_NAME}: trainer output must be tuple"
    assert len(out) == 7, f"{MODEL_NAME}: trainer output must be 7-tuple"

    val_true, val_pred, test_true, test_pred, model, _test_df, best_params = out
    assert model is not None, f"{MODEL_NAME}: model should not be None"
    assert isinstance(best_params, (dict, list, tuple, type(None)))

    if val_true is not None and val_pred is not None:
        assert len(val_true) == len(val_pred), f"{MODEL_NAME}: val y_true/yhat mismatch"
    if test_true is not None and test_pred is not None:
        assert len(test_true) == len(test_pred), f"{MODEL_NAME}: test y_true/yhat mismatch"

