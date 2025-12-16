import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Tuple

from models.lstm import lstm_model
from utils.array_utils import clean_and_unify_arrays
from utils.sliding_windows import create_windows_for_ml

# A lightweight fallback scaler in case sklearn is unavailable
try:  # pragma: no cover - runtime environment guard
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - minimal fallback for environments without sklearn
    class StandardScaler:  # type: ignore
        def fit(self, X):
            X = np.asarray(X, dtype=np.float32)
            self.mean_ = X.mean(axis=0)
            self.scale_ = X.std(axis=0)
            self.scale_[self.scale_ == 0] = 1.0
            self.n_features_in_ = X.shape[1]
            return self

        def transform(self, X):
            X = np.asarray(X, dtype=np.float32)
            return (X - self.mean_) / self.scale_

        def inverse_transform(self, X):
            X = np.asarray(X, dtype=np.float32)
            return X * self.scale_ + self.mean_


def _dtype_from_config(config: dict):
    """Map string dtype in config to torch dtype (default float32)."""
    dtype_str = str(
        config.get("dtype")
        or config.get("default", {}).get("dtype")
        or "float32"
    ).lower()
    if "16" in dtype_str:
        return torch.float16
    if "64" in dtype_str or "double" in dtype_str:
        return torch.float64
    return torch.float32


def _inverse_target(
    arr: np.ndarray,
    scaler: Optional[StandardScaler],
    feature_cols: List[str],
    value_col: str,
) -> np.ndarray:
    """Inverse-transform 1D target array back to original scale using feature scaler."""
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if scaler is None or not hasattr(scaler, "inverse_transform"):
        return arr
    try:
        idx = feature_cols.index(value_col)
    except Exception:
        idx = 0
    wide = np.zeros((len(arr), len(feature_cols)), dtype=np.float32)
    wide[:, idx] = arr
    try:
        inv = scaler.inverse_transform(wide)
        return inv[:, idx].reshape(-1)
    except Exception:
        return arr


def train_lstm_model(df: pd.DataFrame, config: dict):
    """
    LSTM trainer aligned with the unified pipeline/app expectations:
    - Uses 6:2:2 time split (or pre-split scaled data injected via config['data'])
    - Supports multi-feature inputs with StandardScaler
    - Returns 7-tuple and writes val/test dense frames into config['data'] for the app
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("train_lstm_model expects a pandas DataFrame as input.")

    # ---- config + hparams ----
    dft = config.get("default", {}) or {}
    time_col = config.get("time_col", dft.get("time_col", "date"))
    value_col = config.get("value_col", dft.get("value_col", "value"))
    mcfg = (config.get("model_config") or {}).get("LSTM", {}) or {}

    hidden_size = mcfg.get("hidden_dim", config.get("hidden_size", 50))
    num_layers = mcfg.get("num_layers", config.get("num_layers", 1))
    lr = mcfg.get("learning_rate", config.get("learning_rate", 1e-3))
    weight_decay = mcfg.get("weight_decay", config.get("weight_decay", 0.0))
    seq_len_cfg = mcfg.get("seq_len", config.get("seq_len", 10))
    batch_size = max(1, int(mcfg.get("batch_size", config.get("batch_size", 32))))
    epochs = mcfg.get("n_epochs", config.get("epochs", 10))
    dropout = mcfg.get("dropout", config.get("dropout", 0.0))
    patience = int(mcfg.get("patience", config.get("patience", 5)))
    grad_clip = mcfg.get("grad_clip", config.get("grad_clip", None))

    data_blk = config.setdefault("data", {})
    artifacts = config.setdefault("artifacts", {})

    # ---- feature columns ----
    feature_cols: List[str] = (
        data_blk.get("all_feature_cols")
        or mcfg.get("feature_cols")
        or []
    )
    if not feature_cols:
        numeric_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c != time_col
        ]
        feature_cols = [value_col] + [c for c in numeric_cols if c != value_col]
    else:
        # ensure target comes first
        feature_cols = [value_col] + [c for c in feature_cols if c != value_col]

    if value_col not in df.columns:
        raise ValueError(f"Missing target column '{value_col}' in input DataFrame.")

    # ---- prepare splits + scaler (reuse if pipeline already prepared) ----
    scaler: Optional[StandardScaler] = artifacts.get("scaler")
    train_df_sc = data_blk.get("train_df_sc")
    val_df_sc = data_blk.get("val_df_sc")
    test_df_sc = data_blk.get("test_df_sc")

    if train_df_sc is None or val_df_sc is None or test_df_sc is None:
        work = df.copy()
        if time_col in work.columns:
            work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
            work = work.sort_values(time_col)
        n = len(work)
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)
        train_df = work.iloc[:n_train]
        val_df = work.iloc[n_train:n_train + n_val]
        test_df = work.iloc[n_train + n_val:]

        scaler = scaler or StandardScaler()
        scaler.fit(train_df[feature_cols].astype(np.float32))

        def _tf(part: pd.DataFrame) -> pd.DataFrame:
            out = part.copy()
            out[feature_cols] = scaler.transform(part[feature_cols].astype(np.float32))
            return out

        train_df_sc = _tf(train_df)
        val_df_sc = _tf(val_df)
        test_df_sc = _tf(test_df)

        data_blk["train_df_sc"] = train_df_sc
        data_blk["val_df_sc"] = val_df_sc
        data_blk["test_df_sc"] = test_df_sc
        data_blk["split"] = {
            "train_len": len(train_df),
            "val_len": len(val_df),
            "test_len": len(test_df),
        }
        data_blk["all_feature_cols"] = feature_cols
        artifacts["scaler"] = scaler

    # ---- build sliding windows (global history -> split by target position) ----
    work_sc = pd.concat([train_df_sc, val_df_sc, test_df_sc], axis=0, ignore_index=True)
    total_len = len(work_sc)
    effective_seq_len = seq_len_cfg
    if total_len < effective_seq_len + 1:
        effective_seq_len = max(1, total_len - 1)

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list, ts_val = [], [], []
    X_test_list, y_test_list, ts_test = [], [], []

    values_all = work_sc[feature_cols].to_numpy(dtype=np.float32)
    ts_all = pd.to_datetime(work_sc[time_col], errors="coerce").tolist() if time_col in work_sc.columns else [None] * total_len
    target_idx = feature_cols.index(value_col) if value_col in feature_cols else 0

    for start in range(0, total_len - effective_seq_len):
        end = start + effective_seq_len
        target_pos = end  # single-step forecast uses next point
        if target_pos >= total_len:
            break
        window = values_all[start:end]
        # 使用差分作为预测目标：Δy(t) = y(t) - y(t-1)
        target_val = values_all[target_pos, target_idx] - values_all[target_pos - 1, target_idx]
        target_ts = ts_all[target_pos]

        if target_pos < len(train_df_sc):
            X_train_list.append(window)
            y_train_list.append(target_val)
        elif target_pos < len(train_df_sc) + len(val_df_sc):
            X_val_list.append(window)
            y_val_list.append(target_val)
            ts_val.append(target_ts)
        else:
            X_test_list.append(window)
            y_test_list.append(target_val)
            ts_test.append(target_ts)

    def _to_arr(lst, shape_dim):
        if len(lst) == 0:
            return np.zeros((0, effective_seq_len, shape_dim), dtype=np.float32)
        return np.asarray(lst, dtype=np.float32)

    train_X = _to_arr(X_train_list, len(feature_cols))
    val_X = _to_arr(X_val_list, len(feature_cols))
    test_X = _to_arr(X_test_list, len(feature_cols))
    train_y = np.asarray(y_train_list, dtype=np.float32)
    val_y = np.asarray(y_val_list, dtype=np.float32)
    test_y = np.asarray(y_test_list, dtype=np.float32)
    val_ts = ts_val
    test_ts = ts_test

    # ---- training loop ----
    if train_X.shape[0] == 0:
        # 数据太短时直接跳过训练，返回空结果，避免异常中断 app
        val_true_u = np.asarray(val_y, dtype=float).reshape(-1)
        val_pred_u = np.asarray([], dtype=float)
        test_true_u = np.asarray(test_y, dtype=float).reshape(-1)
        test_pred_u = np.asarray([], dtype=float)
        val_dense = None
        test_dense = None
        best_params = {
            "seq_len": effective_seq_len,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": epochs,
            "dropout": dropout,
        }
        data_blk["val_dense"] = val_dense
        data_blk["test_dense"] = test_dense
        return (
            val_true_u,
            val_pred_u,
            test_true_u,
            test_pred_u,
            None,
            None,
            best_params,
        )
    device = torch.device(config.get("device") or dft.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = _dtype_from_config(config)

    model = lstm_model(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        dropout=dropout,
        num_layers=num_layers,
    ).to(device=device, dtype=dtype)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(
        torch.tensor(train_X, dtype=dtype),
        torch.tensor(train_y, dtype=dtype),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_val_loss = float("inf")
    patience_ctr = 0

    def _compute_loss(X_np: np.ndarray, y_np: np.ndarray) -> float:
        if X_np is None or len(X_np) == 0:
            return float("inf")
        with torch.no_grad():
            xb = torch.tensor(X_np, device=device, dtype=dtype)
            yb = torch.tensor(y_np, device=device, dtype=dtype)
            pred_b = model(xb).squeeze(-1)
            return float(criterion(pred_b, yb).detach().cpu().item())

    for _ in range(int(epochs)):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device=device, dtype=dtype)
            batch_y = batch_y.to(device=device, dtype=dtype)
            pred = model(batch_x).squeeze(-1)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                try:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                except Exception:
                    pass
            optimizer.step()

        # --- val monitoring / early stop ---
        train_loss_epoch = _compute_loss(train_X, train_y)
        val_loss_epoch = _compute_loss(val_X, val_y)
        monitor_loss = val_loss_epoch if np.isfinite(val_loss_epoch) else train_loss_epoch
        if monitor_loss < best_val_loss:
            best_val_loss = monitor_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= max(1, patience):
                break

    if best_state is not None:
        try:
            model.load_state_dict(best_state)
        except Exception:
            pass

    model.eval()

    def _predict(X_np: np.ndarray) -> np.ndarray:
        if X_np is None or len(X_np) == 0:
            return np.array([], dtype=np.float32)
        with torch.no_grad():
            t = torch.tensor(X_np, device=device, dtype=dtype)
            out = model(t).squeeze(-1).detach().cpu().numpy()
            return out.reshape(-1)

    val_pred_sc = _predict(val_X)
    test_pred_sc = _predict(test_X)

    # inverse to original scale
    # 先反归一化差分，再还原为绝对值：y_hat(t) = y_{t-1} + Δy_hat
    val_true_delta = _inverse_target(val_y, scaler, feature_cols, value_col)
    test_true_delta = _inverse_target(test_y, scaler, feature_cols, value_col)
    val_pred_delta = _inverse_target(val_pred_sc, scaler, feature_cols, value_col)
    test_pred_delta = _inverse_target(test_pred_sc, scaler, feature_cols, value_col)

    # 还原为绝对值（使用对应段历史的末值作为起点）
    def _restore_absolute(delta_arr: np.ndarray, history_last: float) -> np.ndarray:
        if delta_arr is None or len(delta_arr) == 0:
            return np.array([], dtype=float)
        base = float(history_last)
        out = []
        cur = base
        for d in delta_arr:
            cur = cur + float(d)
            out.append(cur)
        return np.asarray(out, dtype=float)

    # 找到各段起点的“上一时刻”真实值
    last_train_val = _inverse_target(np.asarray([train_df_sc[value_col].iloc[-1]]) if value_col in train_df_sc.columns else np.asarray([train_df_sc[feature_cols[0]].iloc[-1]]), scaler, feature_cols, value_col)
    last_val_test = _inverse_target(np.asarray([val_df_sc[value_col].iloc[-1]]) if value_col in val_df_sc.columns else np.asarray([val_df_sc[feature_cols[0]].iloc[-1]]), scaler, feature_cols, value_col)
    last_train_val_val = float(last_train_val.reshape(-1)[-1]) if last_train_val is not None else 0.0
    last_val_test_val = float(last_val_test.reshape(-1)[-1]) if last_val_test is not None else 0.0

    val_true = _restore_absolute(val_true_delta, last_train_val_val)
    val_pred = _restore_absolute(val_pred_delta, last_train_val_val)
    test_true = _restore_absolute(test_true_delta, last_val_test_val)
    test_pred = _restore_absolute(test_pred_delta, last_val_test_val)

    # clean + align
    val_true_u, val_pred_u, _ = clean_and_unify_arrays(val_true, val_pred)
    test_true_u, test_pred_u, _ = clean_and_unify_arrays(test_true, test_pred)

    # ---- lightweight residual modeling on flattened windows（提取趋势残差学习波动）----
    rm_cfg = (config.get("residual_modeling") or {})
    if rm_cfg.get("enabled", False):
        try:
            L_val = min(len(val_true_u), len(val_pred_u), val_X.shape[0])
            if L_val > 0:
                Xv_base = val_X[:L_val].reshape(L_val, -1)
                # 差分特征：末步输入 - 前一步输入（近似 Δwindow）
                Xv_diff = Xv_base[:, -len(feature_cols):] - Xv_base[:, -2*len(feature_cols):-len(feature_cols)] if Xv_base.shape[1] >= 2*len(feature_cols) else np.zeros_like(Xv_base)
                Xv = np.concatenate([Xv_base, Xv_diff], axis=1)
                res = val_true_u[:L_val] - val_pred_u[:L_val]
                alpha = float(rm_cfg.get("ridge_alpha", 0.0))
                if alpha > 0:
                    XtX = Xv.T @ Xv + alpha * np.eye(Xv.shape[1])
                    w = np.linalg.solve(XtX, Xv.T @ res)
                else:
                    w, *_ = np.linalg.lstsq(Xv, res, rcond=None)

                val_res_pred = Xv @ w
                val_pred_u = val_pred_u.copy()
                val_pred_u[:L_val] = val_pred_u[:L_val] + val_res_pred

                if test_X.shape[0] > 0 and len(test_pred_u) > 0:
                    L_t = min(len(test_pred_u), test_X.shape[0])
                    Xt_base = test_X[:L_t].reshape(L_t, -1)
                    Xt_diff = Xt_base[:, -len(feature_cols):] - Xt_base[:, -2*len(feature_cols):-len(feature_cols)] if Xt_base.shape[1] >= 2*len(feature_cols) else np.zeros_like(Xt_base)
                    Xt = np.concatenate([Xt_base, Xt_diff], axis=1)
                    test_pred_u = test_pred_u.copy()
                    test_pred_u[:L_t] = test_pred_u[:L_t] + Xt @ w

                artifacts["lstm_residual_coef"] = w.tolist()
                artifacts["lstm_residual_alpha"] = alpha
                data_blk["residual_applied"] = True
        except Exception as _e:
            print(f"[lstm] residual modeling skipped: {_e}")

    # dense frames for pipeline/app
    def _mk_df(y_t: np.ndarray, y_p: np.ndarray, ts_list: List) -> Optional[pd.DataFrame]:
        L = min(len(y_t), len(y_p), len(ts_list) if ts_list else len(y_t))
        if L == 0:
            return None
        idx = None
        if ts_list:
            idx = pd.DatetimeIndex(ts_list[:L], name=time_col)
        df_out = pd.DataFrame({
            "y_true": np.asarray(y_t, dtype=float)[:L],
            "yhat": np.asarray(y_p, dtype=float)[:L],
        })
        if idx is not None:
            df_out.index = idx
        return df_out

    val_dense = _mk_df(val_true_u, val_pred_u, val_ts)
    test_dense = _mk_df(test_true_u, test_pred_u, test_ts)

    data_blk["val_dense"] = val_dense
    data_blk["test_dense"] = test_dense
    data_blk["val_timestamps"] = val_ts
    data_blk["test_timestamps"] = test_ts
    if val_dense is not None:
        data_blk["val_result_df"] = val_dense
    if test_dense is not None:
        data_blk["test_result_df"] = test_dense

    # optional best_params exposure (for transparency)
    best_params = {
        "seq_len": effective_seq_len,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "dropout": dropout,
        "patience": patience,
        "grad_clip": grad_clip,
    }

    # For interface compatibility: test_forecast_df mirrors test_dense
    test_forecast_df = test_dense.copy() if isinstance(test_dense, pd.DataFrame) else None

    return (
        val_true_u,
        val_pred_u,
        test_true_u,
        test_pred_u,
        model.cpu(),
        test_forecast_df,
        best_params,
    )
