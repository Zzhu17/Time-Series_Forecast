import numpy as np
import torch


def debug_stat(name: str, arr, max_print: int = 3) -> None:
    """Lightweight debug printer (only used when debug=True)."""
    try:
        if isinstance(arr, torch.Tensor):
            arr_np = arr.detach().cpu().numpy()
        else:
            arr_np = np.asarray(arr)
        flat = arr_np.reshape(-1) if arr_np.size else arr_np
        print(
            f"[DEBUG][{name}] shape={arr_np.shape}, dtype={arr_np.dtype}, "
            f"min={flat.min() if flat.size else 'nan'}, max={flat.max() if flat.size else 'nan'}, "
            f"mean={flat.mean() if flat.size else 'nan'}"
        )
        if flat.size:
            print(f"  head={flat[:max_print]}")
            print(f"  tail={flat[-max_print:]}")
    except Exception as e:
        print(f"[DEBUG][{name}] <unable to summarize: {e}>")


# --- helpers ---

def _infer_device_dtype(model, device=None, dtype=None):
    """Infer device/dtype from model when not provided."""
    if device is None or dtype is None:
        try:
            p = next(model.parameters())
            if device is None:
                device = p.device
            if dtype is None:
                dtype = p.dtype
        except StopIteration:
            device = device or torch.device("cpu")
            dtype = dtype or torch.float32
    return device, dtype


def _np_dtype_from_torch(dtype: torch.dtype):
    if dtype == torch.float16:
        return np.float16
    if dtype == torch.float64:
        return np.float64
    # default stable dtype
    return np.float32


def _ensure_numpy_3d(x, np_dtype):
    """Return numpy array with shape (N, seq, feat)."""
    a = np.asarray(x)
    if a.dtype != np_dtype:
        a = a.astype(np_dtype, copy=False)
    if a.ndim == 1:
        a = a.reshape(-1, 1, 1)
    elif a.ndim == 2:
        a = a.reshape(a.shape[0], a.shape[1], 1)
    elif a.ndim != 3:
        raise ValueError(f"Unsupported ndim for numpy input: {a.ndim}")
    return np.ascontiguousarray(a)


def _ensure_tensor_3d(x, device, dtype):
    """Return torch tensor with shape (N, seq, feat)."""
    if isinstance(x, torch.Tensor):
        t = x.to(device=device, dtype=dtype, non_blocking=True)
    else:
        t = torch.as_tensor(x, device=device, dtype=dtype)
    if t.dim() == 1:
        t = t.view(-1, 1, 1)
    elif t.dim() == 2:
        t = t.view(t.shape[0], t.shape[1], 1)
    elif t.dim() != 3:
        raise ValueError(f"Unsupported dim for tensor input: {t.dim()}")
    return t.contiguous()


def informer_forward(
    model,
    x_enc,
    x_dec,
    *,
    pred_len: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    return_numpy: bool = True,
    debug: bool = False,
):
    """
    A tiny, predictable forward wrapper for Informer.

    • Accepts 1D/2D/3D numpy or tensor, normalizes to (B, seq, F).
    • In inference (return_numpy=True): returns numpy (B, L, 1). If pred_len is given, returns last pred_len steps.
    • In training (return_numpy=False): returns raw tensor from model (do not .eval() / .no_grad()).
    """

    # 1) Resolve device/dtype from model when missing
    device, dtype = _infer_device_dtype(model, device, dtype)

    if return_numpy:
        # 2) Normalize inputs as numpy, then convert to tensors
        np_dtype = _np_dtype_from_torch(dtype)
        x_enc_np = _ensure_numpy_3d(x_enc, np_dtype)
        x_dec_np = _ensure_numpy_3d(x_dec, np_dtype)
        if debug:
            debug_stat("x_enc", x_enc_np)
            debug_stat("x_dec", x_dec_np)

        x_enc_t = _ensure_tensor_3d(x_enc_np, device, dtype)
        x_dec_t = _ensure_tensor_3d(x_dec_np, device, dtype)
        if debug:
            debug_stat("x_enc_tensor", x_enc_t)
            debug_stat("x_dec_tensor", x_dec_t)

        model.eval()
        with torch.no_grad():
            out = model(x_enc_t, x_dec_t)  # expected (B, label_len+pred_len, 1)
            if isinstance(out, torch.Tensor) and pred_len is not None and pred_len > 0:
                out = out[:, -pred_len:, :]
            # to numpy 3D
            if isinstance(out, torch.Tensor):
                out = out.detach().to(dtype).cpu().numpy()
            if out.ndim == 1:
                out = out.reshape(-1, 1, 1)
            elif out.ndim == 2:
                out = out.reshape(out.shape[0], out.shape[1], 1)
            if debug:
                debug_stat("output", out)
            return np.ascontiguousarray(out)
    else:
        # Training path: keep graph & caller controls slicing of pred_len
        x_enc_t = _ensure_tensor_3d(x_enc, device, dtype)
        x_dec_t = _ensure_tensor_3d(x_dec, device, dtype)
        return model(x_enc_t, x_dec_t)