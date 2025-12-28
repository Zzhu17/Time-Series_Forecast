def to_numpy_safe_force_float(x):
    import numpy as np
    arr = np.asarray(x)
    if arr.dtype.kind in {'U', 'S', 'O'}:
        arr = arr.astype(float)
    return arr

def to_numpy_safe(x):
    import numpy as np
    if x is None:
        return np.array([])
    if isinstance(x, str):  # PATCH: 防止字符串传入
        return np.array([])
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

import numpy as np

# sklearn is optional; keep metrics working without it.
try:
    from sklearn.metrics import mean_squared_error as _sk_mse  # type: ignore
except Exception:
    _sk_mse = None

def unify_length_and_flatten(y_true, y_pred):
    """
    自动适配 y_true 与 y_pred 的维度与长度。
    - 转为 numpy 数组
    - 扁平化（reshape(-1)）
    - 统一最短长度
    """
    y_true = to_numpy_safe_force_float(y_true).reshape(-1)
    y_pred = to_numpy_safe_force_float(y_pred).reshape(-1)
    min_len = min(len(y_true), len(y_pred))
    return y_true[-min_len:], y_pred[-min_len:]

def get_metrics(y_true, y_pred):
    y_true = to_numpy_safe_force_float(y_true)
    y_pred = to_numpy_safe_force_float(y_pred)
    y_true, y_pred = unify_length_and_flatten(y_true, y_pred)
    assert len(y_true) == len(y_pred), f"❌ y_true ({len(y_true)}) ≠ y_pred ({len(y_pred)}). 检查模型输出与目标维度是否一致"
    eps = 1e-8
    denom = np.maximum(np.abs(y_true), eps)
    mape = float(np.mean(np.abs(y_true - y_pred) / denom))
    if _sk_mse is not None:
        rmse = float(np.sqrt(_sk_mse(y_true, y_pred)))
    else:
        diff = (y_true - y_pred).astype(float)
        rmse = float(np.sqrt(np.mean(diff * diff)))
    return mape, rmse


# Additional metric functions
def compute_rmse(y_true, y_pred):
    """
    Compute RMSE between y_true and y_pred.
    Both inputs can be array-like, torch tensors, or other convertible types.
    """
    y_true = to_numpy_safe_force_float(y_true)
    y_pred = to_numpy_safe_force_float(y_pred)
    y_true, y_pred = unify_length_and_flatten(y_true, y_pred)
    if _sk_mse is not None:
        return float(np.sqrt(_sk_mse(y_true, y_pred)))
    diff = (y_true - y_pred).astype(float)
    return float(np.sqrt(np.mean(diff * diff)))


def compute_mape(y_true, y_pred, eps: float = 1e-8, masked: bool = True) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = min(len(yt), len(yp))
    yt, yp = yt[:n], yp[:n]
    if masked:
        m = np.abs(yt) > eps
        yt, yp = yt[m], yp[m]
        if yt.size == 0:
            return float("nan")
    return float(np.mean(np.abs((yt - yp) / (np.abs(yt) + eps))))


# 安全版本的MAPE，防止分母为零或极小导致的NaN/inf
def mape_safe(y_true, y_pred, eps: float = 1e-8, masked: bool = True) -> float:
    """
    计算安全的MAPE，避免分母为零或极小导致的NaN或inf。
    参数与 compute_mape 一致。
    """
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = min(len(yt), len(yp))
    yt, yp = yt[:n], yp[:n]
    if masked:
        m = np.abs(yt) > eps
        yt, yp = yt[m], yp[m]
        if yt.size == 0:
            return float("nan")
    denom = np.abs(yt) + eps
    # 防止分母为零或极小
    denom = np.where(denom < eps, eps, denom)
    mape_arr = np.abs((yt - yp) / denom)
    # 去除无穷大和NaN
    mape_arr = mape_arr[np.isfinite(mape_arr)]
    if mape_arr.size == 0:
        return float("nan")
    return float(np.mean(mape_arr))

def compare_metrics(y_true, y_pred_raw, y_pred_corrected):
    y_true = to_numpy_safe_force_float(y_true)
    y_pred_raw = to_numpy_safe_force_float(y_pred_raw)
    y_pred_corrected = to_numpy_safe_force_float(y_pred_corrected)
    raw_mape, raw_rmse = get_metrics(y_true, y_pred_raw)
    corrected_mape, corrected_rmse = get_metrics(y_true, y_pred_corrected)
    return {
        "raw": {"MAPE": raw_mape, "RMSE": raw_rmse},
        "corrected": {"MAPE": corrected_mape, "RMSE": corrected_rmse},
        "improvement": {
            "MAPE": raw_mape - corrected_mape,
            "RMSE": raw_rmse - corrected_rmse
        }
    }
