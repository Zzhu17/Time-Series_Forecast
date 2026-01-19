def to_numpy_safe_force_float(x):
    import numpy as np
    arr = np.asarray(x)
    if arr.dtype.kind in {'U', 'S', 'O'}:
        arr = arr.astype(float)
    return arr

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

def _align_and_mask(y_true, y_pred):
    yt = to_numpy_safe_force_float(y_true).reshape(-1)
    yp = to_numpy_safe_force_float(y_pred).reshape(-1)
    n = min(len(yt), len(yp))
    yt = yt[:n]
    yp = yp[:n]
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def get_metrics(y_true, y_pred):
    y_true, y_pred = _align_and_mask(y_true, y_pred)
    assert len(y_true) == len(y_pred), f"❌ y_true ({len(y_true)}) ≠ y_pred ({len(y_pred)}). 检查模型输出与目标维度是否一致"
    rmse = compute_rmse(y_true, y_pred)
    mape = compute_mape(y_true, y_pred)
    return mape, rmse


# Additional metric functions
def compute_rmse(y_true, y_pred):
    """
    Compute RMSE between y_true and y_pred.
    Both inputs can be array-like, torch tensors, or other convertible types.
    """
    y_true, y_pred = _align_and_mask(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    if _sk_mse is not None:
        return float(np.sqrt(_sk_mse(y_true, y_pred)))
    diff = (y_true - y_pred).astype(float)
    return float(np.sqrt(np.mean(diff * diff)))


def compute_mape(
    y_true,
    y_pred,
    *,
    eps: float = 1e-8,
    tau: float | None = None,
) -> float:
    yt, yp = _align_and_mask(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    mean_abs = float(np.mean(np.abs(yt)))
    if tau is None:
        tau = max(eps, 0.01 * mean_abs) if np.isfinite(mean_abs) and mean_abs > 0 else eps
    mask = np.abs(yt) > tau
    if int(mask.sum()) == 0:
        return float("nan")
    denom = np.abs(yt[mask]) + eps
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / denom)))


def compute_smape(y_true, y_pred, eps: float = 1e-8) -> float:
    yt, yp = _align_and_mask(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    denom = np.abs(yt) + np.abs(yp) + eps
    return float(np.mean(2.0 * np.abs(yt - yp) / denom))


def compute_nrmse(y_true, y_pred, eps: float = 1e-8) -> float:
    yt, yp = _align_and_mask(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    rmse = compute_rmse(yt, yp)
    denom = float(np.std(yt)) + eps
    if not np.isfinite(denom) or denom <= eps:
        return float("nan")
    return float(rmse / denom)


# 安全版本的MAPE，防止分母为零或极小导致的NaN/inf
def mape_safe(y_true, y_pred, eps: float = 1e-8, tau: float | None = None) -> float:
    """
    Safe MAPE wrapper using a threshold mask to avoid near-zero targets.
    """
    return compute_mape(y_true, y_pred, eps=eps, tau=tau)
