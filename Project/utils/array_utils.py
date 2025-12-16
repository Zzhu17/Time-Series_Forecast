import numpy as np
from typing import Optional, Any, Tuple, Union, List
import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def ensure_array_safe(arr):
    if arr is None:
        return np.array([])
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr

def safe_to_numpy(arr: Optional[Any]) -> Optional[np.ndarray]:
    if arr is None:
        return None

    # 尝试处理 torch.Tensor
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
    except ImportError:
        pass  # 如果没有安装 torch，则跳过这一处理

    try:
        arr_np = np.asarray(arr)
    except Exception:
        return None

    # 调用 safe_to_float_array，将字符串/object/unicode 类型转换为浮点数
    from .array_utils import safe_to_float_array  # 或者根据您的文件结构调整 import
    arr_np = safe_to_float_array(arr_np)
    return arr_np

def safe_to_float_array(arr):
    arr = np.asarray(arr)
    # 如果是字符串类型、object、unicode等，全部转 float
    if arr.dtype.kind in {'U', 'S', 'O'}:
        try:
            arr = arr.astype(float)
        except Exception:
            arr = np.full(arr.shape, np.nan, dtype=float)
    return arr

# === 高级一键预处理 ===

def advanced_preprocess(
    df: Optional[pd.DataFrame],
    value_col: str,
    time_col: str,
    feature_cols: Optional[List[str]] = None,
    fillna_method: Optional[str] = 'ffill',
    outlier_method: Optional[str] = 'clip3sigma',
    time_to_datetime: bool = True,
    drop_duplicates: bool = True,
    clip_range: Optional[Tuple[float, float]] = None,
    add_time_feat: bool = True
) -> Optional[pd.DataFrame]:
    """
    一步到位的复杂预处理
    """
    df = clean_dataframe(
        df, value_col, time_col, feature_cols,
        fillna_method, outlier_method, time_to_datetime, drop_duplicates, clip_range
    )
    if df is not None and add_time_feat:
        df = add_time_features(df, time_col)
    if df is not None:
        df = auto_convert_types(df, exclude_cols=[time_col])
    return df

def clean_dataframe(
    df: Optional[pd.DataFrame],
    value_col: Optional[str] = None,
    time_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    fillna: Optional[str] = "ffill",
    outlier_method: Optional[str] = None,
    time_to_datetime: bool = True,
    drop_duplicates: bool = True,
    clip_range: Optional[Tuple[float, float]] = None
) -> Optional[pd.DataFrame]:
    """
    更加明确和安全的DataFrame预处理，专门处理缺失值，不允许乱传参数
    """
    if df is None:
        return None
    df = df.copy()
    if time_col and time_to_datetime and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    if feature_cols:
        cols = [c for c in [time_col, value_col] if c] + list(feature_cols)
        cols = list(dict.fromkeys(cols))
        df = df[[c for c in cols if c in df.columns]]
    if drop_duplicates:
        df = df.drop_duplicates()
        
    if fillna is not None:
        if fillna == "ffill":
            df = df.fillna(method="ffill")  # pyright: ignore[reportCallIssue] # OK
        elif fillna == "bfill":
            df = df.fillna(method="bfill")  # pyright: ignore[reportCallIssue] # OK
        elif fillna == "zero":
            df = df.fillna(0)
        elif fillna == "mean" and value_col and value_col in df.columns:
            df[value_col] = df[value_col].fillna(df[value_col].mean())
        elif isinstance(fillna, (int, float)):
            df = df.fillna(fillna)
        elif fillna is None:
            pass
        else:
            # 强制忽略 VSCode 类型检查
            df = df.fillna(fillna)  # type: ignore
    if df is not None and clip_range and value_col and value_col in df.columns:
        df[value_col] = df[value_col].clip(lower=clip_range[0], upper=clip_range[1])
    return df

def add_time_features(df: Optional[pd.DataFrame], time_col: str) -> Optional[pd.DataFrame]:
    if df is None or time_col not in df.columns:
        return df
    df = df.copy()
    dt = pd.to_datetime(df[time_col], errors='coerce')
    df['year'] = dt.dt.year
    df['month'] = dt.dt.month
    df['day'] = dt.dt.day
    df['weekday'] = dt.dt.weekday
    return df

def auto_convert_types(df: Optional[pd.DataFrame], exclude_cols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    if df is None:
        return df
    df = df.copy()
    for col in df.columns:
        if exclude_cols and col in exclude_cols:
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception:
            pass
    return df

# === scaler相关 ===

def apply_scaler_if_needed(arr: Optional[Any], scaler: Optional[Any] = None, inverse: bool = False) -> Optional[Any]:
    """
    根据需要应用归一化或反归一化处理
    - arr: 输入数组（支持 1D, 2D, 3D）
    - scaler: sklearn 的 scaler 对象（如 MinMaxScaler）
    - inverse: 是否反归一化
    """
    if scaler is None or arr is None:
        return arr

    arr = ensure_numpy(arr)
    if not hasattr(arr, "ndim"):
        return arr
    original_shape = arr.shape if hasattr(arr, "shape") else None

    if arr.ndim == 1:
        arr_reshaped = arr.reshape(-1, 1)
    elif arr.ndim == 2:
        arr_reshaped = arr
    elif arr.ndim == 3:
        arr_reshaped = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    else:
        raise ValueError("❌ 输入维度过高，仅支持最多3维数组。")

    if inverse:
        arr_scaled = scaler.inverse_transform(arr_reshaped)
    else:
        arr_scaled = scaler.transform(arr_reshaped)

    if arr.ndim == 3 and original_shape is not None:
        arr_scaled = arr_scaled.reshape(original_shape)
    elif arr.ndim == 1:
        arr_scaled = arr_scaled.ravel()

    return arr_scaled

# === array/滑窗统一 ===


def clean_and_unify_arrays(
    y_true: Any,
    y_pred: Any,
    *,
    align: str = "min",           # 目前仅支持对齐到最短长度
    drop_nonfinite: bool = True,  # 是否丢弃 NaN/inf
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    将 y_true / y_pred 规整为 1D float ndarray，按策略对齐长度，并保证**总是**返回三元组：
        (true_aligned, pred_aligned, meta)
    其中 meta 为对齐信息或其它附加内容（类型为 Any，避免 Optional 导致的静态检查告警）。

    - 支持输入为 list / tuple / numpy 数组 / pandas Series / (N,1) 等形状
    - 可选丢弃非有限值（NaN/inf），各自独立清洗
    - 当前对齐策略：对齐到最短长度（align='min'）
    """
    def _to_1d_float(x) -> np.ndarray:
        # 兼容 pandas Series / DataFrame 单列 / 列向量等
        try:
            arr = np.asarray(x, dtype=float).ravel()
        except Exception:
            # 无法转为 float 时返回空
            return np.array([], dtype=float)
        if drop_nonfinite and arr.size > 0:
            mask = np.isfinite(arr)
            if not mask.all():
                arr = arr[mask]
        return arr

    a = _to_1d_float(y_true)
    b = _to_1d_float(y_pred)

    if a.size == 0 or b.size == 0:
        # 任何一侧为空时，统一返回空阵列三元组，避免上层解包报错
        empty = np.array([], dtype=float)
        return empty, empty, {"L": 0, "reason": "empty_input"}

    if align == "min":
        L = min(len(a), len(b))
    else:
        # 预留未来策略；目前等价于 'min'
        L = min(len(a), len(b))

    # 对齐并统一为 1D float
    a_aligned = np.asarray(a[:L], dtype=float).ravel()
    b_aligned = np.asarray(b[:L], dtype=float).ravel()

    return a_aligned, b_aligned, {"L": int(L)}

def ensure_3d_or_2d(arr: Any) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1, 1)
    elif arr.ndim == 2:
        return arr.reshape(arr.shape[0], arr.shape[1], 1)
    elif arr.ndim == 3:
        return arr
    else:
        raise ValueError("ensure_3d_or_2d: input arr shape not 1/2/3D")

# === transformer结构array shape ===

def safe_transformer_shape(x: Any, flatten_3d: bool = False) -> Optional[np.ndarray]:
    x = ensure_numpy(x)
    x = safe_to_float_array(x)
    if hasattr(x, "ndim"):
        if x.ndim == 1:
            return x.reshape(-1, 1, 1)
        elif x.ndim == 2:
            return x.reshape(x.shape[0], x.shape[1], 1)
        elif x.ndim == 3:
            if flatten_3d:
                return x.reshape(x.shape[0], -1)
            return x
        else:
            raise ValueError(f"Unsupported ndim: {x.ndim}")
    return x

def ensure_numpy(arr: Any) -> np.ndarray:
    if arr is None:
        return np.array([])
    if hasattr(arr, "detach") and hasattr(arr, "cpu"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    arr = safe_to_float_array(arr)
    return arr

# === 基础类型/NaN/Inf防御 ===

def tensor_to_numpy(x: Optional[Any]) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, str):
        raise ValueError("tensor_to_numpy: input is string, not array")
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass
    try:
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        x = safe_to_float_array(x)
        return x
    except Exception as e:
        raise ValueError(f"tensor_to_numpy: cannot convert input to np.ndarray: {e}")

def ensure_float(x: Optional[Any]) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = tensor_to_numpy(x)
    if arr is not None and hasattr(arr, "dtype") and arr.dtype.kind in "iu":
        if hasattr(arr, "astype"):
            arr = arr.astype(np.float32)  # type: ignore
    return arr

def safe_no_nan(arr: Optional[Any]) -> Optional[np.ndarray]:
    arr = tensor_to_numpy(arr)
    if arr is None:
        return None
    arr = safe_to_float_array(arr)
    if hasattr(arr, "astype"):
        arr = arr.astype(np.float32)  # type: ignore
    # 只在 arr 非 None 时调用 nan_to_num
    if arr is not None and hasattr(np, "nan_to_num"):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

def assert_no_nan(arr: Optional[Any], name: str = "") -> None:
    arr = tensor_to_numpy(arr)
    arr = safe_to_float_array(arr)
    if arr is None:
        print(f"[DEBUG][assert_no_nan]{name}: arr is None")
        return
    n_nan = np.isnan(arr).sum() if hasattr(arr, "size") and arr.size > 0 else 0
    n_inf = np.isinf(arr).sum() if hasattr(arr, "size") and arr.size > 0 else 0
    if n_nan or n_inf:
        raise ValueError(f"[assert_no_nan]{name}: nan={n_nan}, inf={n_inf}, shape={arr.shape if hasattr(arr, 'shape') else 'unknown'}")

def is_valid_array(arr: Optional[Any]) -> bool:
    arr = tensor_to_numpy(arr)
    arr = safe_to_float_array(arr)
    return arr is not None and isinstance(arr, np.ndarray) and hasattr(arr, "size") and arr.size > 0

def safe_shape(arr: Optional[Any]) -> Optional[Tuple[int, ...]]:
    arr = tensor_to_numpy(arr)
    arr = safe_to_float_array(arr)
    if arr is not None and hasattr(arr, "shape"):
        return arr.shape
    return None

# === flatten / reshape / concat ===

def flatten_to_2d(arr: Optional[Any]) -> Optional[np.ndarray]:
    arr = tensor_to_numpy(arr)
    arr = safe_to_float_array(arr)
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    elif arr.ndim == 2:
        return arr
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        return arr.reshape(-1, 1)
    elif arr.ndim == 3:
        return arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    else:
        return arr.reshape(-1, 1)

def safe_concat(arrs: Any, axis: int = 1) -> Optional[np.ndarray]:
    # 过滤 None
    valid = [tensor_to_numpy(x) for x in arrs if x is not None and is_valid_array(x)]
    valid = [safe_to_float_array(v) for v in valid if v is not None]
    if not valid:
        return None
    try:
        valid = [v for v in valid if v is not None]
        if not valid:
            return None
        return np.concatenate(valid, axis=axis)  # type: ignore
    except Exception as e:
        raise ValueError(f"safe_concat: cannot concatenate: {e}")

def inverse_transform(arr: Optional[Any], scaler: Optional[Any]) -> Optional[np.ndarray]:
    arr = tensor_to_numpy(arr)
    arr = safe_to_float_array(arr)
    if scaler is None or arr is None:
        return arr
    try:
        shape = arr.shape if hasattr(arr, "shape") else None
        if shape is None:
            return arr
        if arr.ndim > 1:
            arr2d = arr.reshape(-1, shape[-1])
        else:
            arr2d = arr.reshape(-1, 1)
        inv = scaler.inverse_transform(arr2d)
        inv = safe_to_float_array(inv)
        if shape is not None and hasattr(inv, "reshape"):
            return inv.reshape(shape)
        return inv
    except Exception:
        return arr

# === 调试辅助 ===

def debug_stat(name: str, arr: Optional[Any], max_print: int = 3) -> None:
    arr_np = tensor_to_numpy(arr)
    arr_np = safe_to_float_array(arr_np)
    print(f"[DEBUG][{name}] type={type(arr)}, dtype={getattr(arr_np, 'dtype', None)}, shape={getattr(arr_np, 'shape', None)}")
    if arr_np is not None and hasattr(arr_np, 'min') and hasattr(arr_np, 'max') and hasattr(arr_np, 'mean'):
        print(f"  min={arr_np.min() if arr_np.size>0 else 'nan'}, max={arr_np.max() if arr_np.size>0 else 'nan'}, mean={arr_np.mean() if arr_np.size>0 else 'nan'}")
        flat = arr_np.flatten()
        print(f"  head={flat[:max_print]}, tail={flat[-max_print:]}")

# === 其它实用函数 ===
def safe_reshape_or_asarray(arr: Optional[Any], shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    arr = tensor_to_numpy(arr)
    arr = safe_to_float_array(arr)
    if arr is not None and hasattr(arr, "reshape") and isinstance(shape, tuple) and len(shape) > 0:
        try:
            return arr.reshape(shape)
        except Exception:
            try:
                return np.asarray(arr)
            except Exception:
                return None
    elif arr is not None:
        try:
            return np.asarray(arr)
        except Exception:
            return None
    return None

def match_tensor_shapes(a: Any, b: Any, feature_dim: int = 1, raise_err: bool = True) -> Tuple[Any, Any]:
    """
    自动对齐两个张量/数组的shape到 (batch, seq_len, feature_dim)
    - a, b: 支持 numpy.ndarray / torch.Tensor
    - feature_dim: 最后一个维度默认1（单变量），多变量可以自定
    """
    import numpy as np
    try:
        import torch
    except ImportError:
        torch = None

    def to_np(x):
        if torch and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _ensure_shape(x, ref_shape):
        x = to_np(x)
        if x.ndim == 1:
            # (seq_len,) -> (1, seq_len, feature_dim)
            return x.reshape(1, -1, feature_dim)
        elif x.ndim == 2:
            # (batch, seq_len) or (seq_len, feature) -> (batch, seq_len, feature_dim)
            if x.shape[-1] != feature_dim:
                return x.reshape(x.shape[0], x.shape[1], feature_dim)
            return x
        elif x.ndim == 3:
            return x
        else:
            if raise_err:
                raise ValueError(f"match_tensor_shapes: 不支持的输入 shape={x.shape}")
            return x

    a_ = _ensure_shape(a, None)
    b_ = _ensure_shape(b, None)

    # 如果是 torch，转回 torch，dtype和device跟原来保持一致
    if torch and (isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor)):
        device = a.device if isinstance(a, torch.Tensor) else (b.device if isinstance(b, torch.Tensor) else 'cpu')
        dtype = a.dtype if isinstance(a, torch.Tensor) else (b.dtype if isinstance(b, torch.Tensor) else torch.float32)
        a_ = torch.tensor(a_, device=device, dtype=dtype)
        b_ = torch.tensor(b_, device=device, dtype=dtype)
    return a_, b_