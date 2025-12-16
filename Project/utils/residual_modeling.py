import numpy as np
from typing import Optional, Tuple, Any, Protocol, runtime_checkable, cast
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

@runtime_checkable
class SklearnLike(Protocol):
    def predict(self, X: np.ndarray) -> Any: ...


# =====================================================
# Helpers: robust shape/dtype handling for residuals
# =====================================================

def _to_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is (N, C):
    - (N,)   -> (N, 1)
    - (N,C)  -> (N, C)
    - (W,L,C)-> (W*L, C)
    """
    if arr is None:
        raise ValueError("_to_2d: input is None")
    a = np.asarray(arr)
    if a.ndim == 1:
        return a.reshape(-1, 1).astype(np.float32, copy=False)
    if a.ndim == 2:
        return a.astype(np.float32, copy=False)
    if a.ndim == 3:
        w, l, c = a.shape
        return a.reshape(w * l, c).astype(np.float32, copy=False)
    raise ValueError(f"_to_2d: unsupported ndim={a.ndim} shape={a.shape}")


def _features_to_2d(x: Optional[np.ndarray], n_rows: int) -> Optional[np.ndarray]:
    """Normalize features to (N, F) and align N with target length.
    Accepts None, (N,), (N,F), or (W,L,F). Truncates/exactly aligns rows.
    """
    if x is None:
        return None
    f = np.asarray(x)
    if f.ndim == 1:
        f = f.reshape(-1, 1)
    elif f.ndim == 3:
        w, l, k = f.shape
        f = f.reshape(w * l, k)
    # Align to n_rows (truncate or keep)
    if f.shape[0] != n_rows:
        f = f[:n_rows]
    return f.astype(np.float32, copy=False)


def _align_2d(arr: np.ndarray, N: int, C: int) -> np.ndarray:
    """
    Align a 2D array to shape (N, C) by truncating or padding with zeros.
    If arr has one column and C>1, repeat that column to C.
    """
    a = _to_2d(arr)
    n, c = a.shape
    # Row align
    if n < N:
        pad = np.zeros((N - n, c), dtype=a.dtype)
        a = np.vstack([a, pad])
    elif n > N:
        a = a[:N, :]
    # Col align
    if c == C:
        return a
    if c == 1 and C > 1:
        a = np.repeat(a, C, axis=1)
        return a
    if c > C:
        return a[:, :C]
    # c < C and c != 1
    padc = np.zeros((a.shape[0], C - c), dtype=a.dtype)
    return np.hstack([a, padc])


def _build_residual_estimator(n_outputs: int):
    base = LinearRegression()
    return MultiOutputRegressor(base) if n_outputs > 1 else base


def _shape_info(arr: np.ndarray) -> dict:
    """Return a dict with shape diagnostics.
    Keys: ndim, shape, W, L, C, WL (W*L or N for 1D/2D).
    """
    a = np.asarray(arr)
    info = {"ndim": a.ndim, "shape": a.shape, "W": None, "L": None, "C": None, "WL": None}
    if a.ndim == 3:
        W, L, C = a.shape
        info.update({"W": W, "L": L, "C": C, "WL": W * L})
    elif a.ndim == 2:
        N, C = a.shape
        info.update({"C": C, "WL": N})
    elif a.ndim == 1:
        N = a.shape[0]
        info.update({"C": 1, "WL": N})
    return info


def _format_shape_mismatch(ctx: str, y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Compose an informative mismatch message with WL diagnostics."""
    ti = _shape_info(y_true)
    pi = _shape_info(y_pred)
    return (
        f"{ctx}: shape mismatch.\n"
        f"  y_true: shape={ti['shape']}, ndim={ti['ndim']}, W={ti['W']}, L={ti['L']}, C={ti['C']}, WL={ti['WL']}\n"
        f"  y_pred: shape={pi['shape']}, ndim={pi['ndim']}, W={pi['W']}, L={pi['L']}, C={pi['C']}, WL={pi['WL']}\n"
        f"  Hint: WL should match (W*L or N). Ensure you only flatten the time dimension and keep C intact."
    )


# =====================================================
# Residual modeling (class kept for compatibility)
# =====================================================
class ResidualModeling:
    """Fit residual = y_true - y_pred. Supports optional exogenous x_features.
    Public API mirrors previous version: fit() and predict().
    """
    def __init__(self):
        self.model = None
        self._n_outputs = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray, x_features: Optional[np.ndarray] = None, scaler: Any = None):  # scaler kept for signature compatibility
        y_true_2d = _to_2d(y_true)
        y_pred_2d = _to_2d(y_pred)
        if y_true_2d.shape != y_pred_2d.shape:
            raise ValueError(_format_shape_mismatch("ResidualModeling.fit", y_true, y_pred))
        N, C = y_true_2d.shape
        self._n_outputs = C
        X = _features_to_2d(x_features, N)
        residual = (y_true_2d - y_pred_2d).astype(np.float32, copy=False)
        if X is None:
            # Bias-only model: store mean residual; use dict to keep simple
            self.model = {"type": "bias", "bias": residual.mean(axis=0, keepdims=True)}  # (1, C)
        else:
            est = _build_residual_estimator(C)
            est.fit(X, residual)  # X: (N, F), residual: (N, C)
            self.model = est
        return self

    def predict(self, y_pred: np.ndarray, x_features: Optional[np.ndarray] = None, restore_shape: Optional[tuple] = None, scaler: Any = None) -> np.ndarray:  # scaler kept
        if self.model is None:
            # no model fitted -> return input prediction
            yp = np.asarray(y_pred)
            return yp if restore_shape is None else yp.reshape(restore_shape)
        y_pred_2d = _to_2d(y_pred)
        N, C = y_pred_2d.shape
        # Bias-only
        if isinstance(self.model, dict) and self.model.get("type") == "bias":
            bias = self.model["bias"]  # (1, C)
            corr = np.repeat(bias, N, axis=0)
            out = y_pred_2d + corr
            return out if restore_shape is None else out.reshape(restore_shape)
        # Sklearn-style estimator
        X = _features_to_2d(x_features, N)
        if X is None:
            # If features missing but estimator needs them, fall back
            out = y_pred_2d
        elif isinstance(self.model, SklearnLike):
            residual_hat = self.model.predict(X)
            residual_hat = np.asarray(residual_hat)
            if residual_hat.ndim == 1:
                residual_hat = residual_hat.reshape(-1, 1)
            out = y_pred_2d + residual_hat.astype(np.float32, copy=False)
        else:
            # Unknown model type, fallback
            out = y_pred_2d
        return out if restore_shape is None else out.reshape(restore_shape)

    # Optional helper kept for backward compatibility
    def fit_and_correct(self, y_true: np.ndarray, y_pred: np.ndarray, x_features: Optional[np.ndarray] = None, restore_shape: Optional[tuple] = None, scaler: Any = None) -> np.ndarray:
        self.fit(y_true, y_pred, x_features=x_features, scaler=scaler)
        return self.predict(y_pred, x_features=x_features, restore_shape=restore_shape, scaler=scaler)


# =====================================================
# Module-level functions used by train.py / predict.py
# =====================================================

def mape(a: np.ndarray, b: np.ndarray) -> float:
    a2 = _to_2d(a)
    b2 = _to_2d(b)
    if a2.shape != b2.shape:
        return float("nan")
    denom = np.abs(a2) + 1e-8
    return float(np.mean(np.abs((a2 - b2) / denom)))


def train_and_predict_residual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x_features: Optional[np.ndarray] = None,
    use_if_better: bool = True,
    flatten_out: bool = False,
    scaler: Any = None,
):
    """Fit residuals on (y_true - y_pred) and return corrected predictions.
    Returns: final_pred, residual_model, metrics_dict, used_flag
    Shapes are handled robustly; inputs may be 1D/2D/3D.
    """
    # Keep ref shape for restore
    ref_shape = np.asarray(y_pred).shape

    y_true_2d = _to_2d(y_true)
    y_pred_2d = _to_2d(y_pred)
    if y_true_2d.shape != y_pred_2d.shape:
        raise ValueError(_format_shape_mismatch("train_and_predict_residual", y_true, y_pred))

    N, C = y_true_2d.shape
    X = _features_to_2d(x_features, N)

    model = ResidualModeling().fit(y_true_2d, y_pred_2d, x_features=X, scaler=scaler)
    corrected = model.predict(y_pred_2d, x_features=X, restore_shape=None, scaler=scaler)

    # Metrics
    raw_m = mape(y_true_2d, y_pred_2d)
    cor_m = mape(y_true_2d, _to_2d(corrected))

    used = True
    final = corrected
    if use_if_better and not (np.isnan(raw_m) or np.isnan(cor_m)):
        if cor_m >= raw_m:
            used = False
            final = y_pred_2d

    # Restore shape / flatten as requested
    if flatten_out:
        final_out = _to_2d(final)
    else:
        final_out = np.asarray(final)
        if final_out.shape != ref_shape:
            try:
                final_out = final_out.reshape(ref_shape)
            except Exception:
                # as a last resort, keep 2D
                final_out = _to_2d(final_out)

    metrics = {"raw_mape": raw_m, "corrected_mape": cor_m}
    return final_out.astype(np.float32, copy=False), model, metrics, used


def apply_residual(
    y_pred: np.ndarray,
    x_features: Optional[np.ndarray],
    residual_model: Any,
) -> np.ndarray:
    """Apply a fitted residual model to correct predictions at inference time.
    y_pred may be 1D/2D/3D; x_features may be None/(N,F)/(W,L,F).
    """
    if residual_model is None:
        return np.asarray(y_pred)

    # Support bias-only model dict
    if isinstance(residual_model, dict) and residual_model.get("type") == "bias":
        yp2 = _to_2d(y_pred)
        N, C = yp2.shape
        bias = np.asarray(residual_model["bias"], dtype=np.float32)
        corr = np.repeat(bias, N, axis=0)
        out2d = yp2 + corr
        out = out2d
    elif isinstance(residual_model, SklearnLike):
        yp2 = _to_2d(y_pred)
        N, C = yp2.shape
        # Features to (N, F) with safe truncation/padding
        X = _features_to_2d(x_features, N)
        if X is None:
            return np.asarray(y_pred)  # cannot apply model without features
        residual_hat = residual_model.predict(X)
        residual_hat = _align_2d(np.asarray(residual_hat), N, C).astype(np.float32, copy=False)
        out2d = yp2 + residual_hat
        out = out2d
    else:
        # Unknown model type; fallback
        return np.asarray(y_pred)

    # Try to restore original shape of y_pred
    ref_shape = np.asarray(y_pred).shape
    if out.shape == ref_shape:
        return out.astype(np.float32, copy=False)
    try:
        return out.reshape(ref_shape).astype(np.float32, copy=False)
    except Exception:
        return out.astype(np.float32, copy=False)