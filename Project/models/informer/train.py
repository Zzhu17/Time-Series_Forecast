import os
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import joblib
import random
import math
# 1. 导入所有经过我们重构和验证的模块
from models.informer.informer import build_informer_model
from models.informer.forward import informer_forward
from models.informer.input_utils import prepare_informer_inputs, make_informer_loader
from utils.array_utils import assert_no_nan, safe_to_numpy
from utils.residual_modeling import train_and_predict_residual, apply_residual

log = logging.getLogger('test')

class EarlyStopping:
    """早停机制，防止过拟合"""
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

# === Helper: inverse only target/features using a scaler fitted on all_feature_cols ===
def _inverse_transform_targets(arr2d: np.ndarray, scaler, config: Dict[str, Any]) -> np.ndarray:
    """Safely inverse-transform predictions/labels that contain only target feature columns
    when the scaler was fit on a wider set of columns (e.g., feature_cols + time_cols).
    - arr2d: shape (N, C_target)
    - scaler: sklearn-like scaler with n_features_in_
    - config: to locate feature_cols/all_feature_cols/value_col for correct column mapping
    Returns array of the same shape as arr2d in original scale.
    """
    arr2d = np.asarray(arr2d, dtype=np.float32)
    n_in = getattr(scaler, 'n_features_in_', None)
    if n_in is None:
        try:
            return scaler.inverse_transform(arr2d)
        except Exception:
            return arr2d

    if arr2d.shape[1] == n_in:
        return scaler.inverse_transform(arr2d)

    data_cfg = config.get('data', {})
    all_cols = list(data_cfg.get('all_feature_cols') or [])
    informer_cfg = config.get('model_config', {}).get('Informer', {})
    feature_cols = list(informer_cfg.get('feature_cols') or [config.get('default', {}).get('value_col', 'value')])
    target_name = config.get('default', {}).get('value_col', 'value')

    tmp = np.zeros((arr2d.shape[0], n_in), dtype=np.float32)
    used_indices = []

    if len(feature_cols) == arr2d.shape[1] and len(feature_cols) > 0:
        for j, name in enumerate(feature_cols):
            try:
                idx = all_cols.index(name)
            except ValueError:
                idx = min(j, n_in - 1)
            tmp[:, idx] = arr2d[:, j]
            used_indices.append(idx)
    else:
        try:
            idx0 = all_cols.index(target_name)
        except ValueError:
            idx0 = 0
        if arr2d.shape[1] == 1:
            tmp[:, idx0] = arr2d[:, 0]
            used_indices = [idx0]
        else:
            k = min(arr2d.shape[1], n_in)
            tmp[:, :k] = arr2d[:, :k]
            used_indices = list(range(k))

    inv = scaler.inverse_transform(tmp)
    out = np.zeros_like(arr2d)
    for j, idx in enumerate(used_indices):
        out[:, j] = inv[:, idx]
    return out

# === Helper: ensure x_feature is time-step level (W, L, F) ===
def _ensure_timestep_features(x_feature: Any, pred_len: int):
    if x_feature is None:
        return None
    x = np.asarray(x_feature)
    if x.ndim == 3:
        return x
    if x.ndim == 2:
        return np.repeat(x[:, None, :], repeats=pred_len, axis=1)
    if x.ndim == 1:
        x2 = x.reshape(-1, 1)
        return np.repeat(x2[:, None, :], repeats=pred_len, axis=1)
    return x

# === Helper: generate rolling window start indices ===
def _gen_rolling_starts(values_len: int, seq_len: int, label_len: int, horizon: int, step: int) -> list:
    starts = []
    start = values_len - (seq_len + label_len + horizon)
    if start < 0:
        start = 0
    while start + seq_len + label_len + horizon <= values_len:
        starts.append(start)
        start += max(1, int(step))
    if not starts:
        starts = [max(0, values_len - (seq_len + label_len + horizon))]
    return starts

# === Helper: strict-mean merge buffers (accumulate then finalize) ===
def _strict_mean_finalize(accum: np.ndarray, count: np.ndarray) -> np.ndarray:
    merged = np.full_like(accum, np.nan, dtype=float)
    mask = count > 0
    merged[mask] = accum[mask] / count[mask]
    return merged

# === Helper: clone config and override horizon for rolling forward ===
def _clone_cfg_with_horizon(config: Dict[str, Any], horizon: int, feature_cols: list) -> Dict[str, Any]:
    tmp_cfg = dict(config)
    tmp_cfg.setdefault('model_config', {})
    tmp_cfg['model_config'] = dict(config['model_config'])
    tmp_cfg['model_config']['Informer'] = dict(config['model_config']['Informer'])
    tmp_cfg['model_config']['Informer']['pred_len'] = int(horizon)
    tmp_cfg.setdefault('data', {})
    tmp_cfg['data'] = dict(config.get('data', {}))
    tmp_cfg['data']['all_feature_cols'] = list(feature_cols)
    return tmp_cfg

# === Helper: Dense rolling prediction for last k points ===
def _dense_predict_last_k(
    model,
    df_all_sc: pd.DataFrame,
    k_last: int,
    config: Dict[str, Any],
    feature_cols: list,
    scaler,
    residual_model=None,
) -> pd.DataFrame:
    """
    以 horizon=1, step=1 做整段“密集滚动”预测，并取序列最后 k_last 个点作为输出。
    - df_all_sc: 训练+验证(+测试)拼接后的【标准化】DataFrame（包含 time_col 和 feature_cols）
    - k_last: 需要返回的末端点数（用于验证=val_len，测试=test_len）
    返回：DataFrame([time_col, 'y_true','yhat'])，索引为时间。
    """
    import pandas as _pd
    from models.informer.predict import rolling_predict_segment

    default_cfg = config.get('default', {})
    time_col  = default_cfg.get('time_col', 'date')

    # 基于当前 cfg 的 seq/label 配置做 horizon=1 的整段滚动
    inf_cfg = (config.get('model_config', {}) or {}).get('Informer', {}) or {}
    seq_len   = int(inf_cfg.get('seq_len', 96))
    label_len = int(inf_cfg.get('label_len', 48))

    full_df, _ = rolling_predict_segment(
        model=model,
        df_sc=df_all_sc,
        scaler=scaler,
        feature_cols=feature_cols,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=1,
        step=1,
        mode="overwrite",
        calib=None,
    )
    if not isinstance(full_df, _pd.DataFrame) or full_df.empty:
        return _pd.DataFrame(columns=[time_col, 'y_true', 'yhat']).set_index(time_col)

    # 仅取末端 k_last 个点
    if k_last and k_last > 0:
        full_df = full_df.tail(int(k_last))

    df_out = full_df.copy()
    if time_col not in df_out.index.names and time_col in df_out.columns:
        df_out = df_out.set_index(_pd.to_datetime(df_out[time_col]))
    df_out.index.name = time_col

    # --- Optional: apply residual correction on dense outputs ---
    if residual_model is not None and 'yhat' in df_out.columns:
        try:
            # reshape to (N, 1, 1) to be compatible with residual apply API
            yhat_3d = df_out['yhat'].astype(float).to_numpy().reshape(-1, 1, 1)
            # we don't have timestep features here; pass None (model should handle it)
            yhat_corr_3d = apply_residual(yhat_3d, None, residual_model)
            if yhat_corr_3d is not None:
                df_out['yhat'] = np.asarray(yhat_corr_3d).reshape(-1)
        except Exception as e:
            print(f"[Residual] apply_residual failed in dense mode: {e}; using base predictions.")

    return df_out

# === Helper: Finalize long payloads after loading best model ===
def _finalize_long_payloads_after_training(model, config):
    """
    用最终权重对 val/test 做一次“整段滚动”，把短+长两套载荷都写回 config['data']。
    不改变原有返回接口（model, result_df），仅补充数据。
    """
    import pandas as pd
    from models.informer.predict import rolling_predict_segment

    data_dict   = config.setdefault('data', {})
    art         = config.setdefault('artifacts', {})
    inf_cfg     = (config.get('model_config', {}) or {}).get('Informer', {}) or {}
    seq_len     = int(inf_cfg.get('seq_len', 96))
    label_len   = int(inf_cfg.get('label_len', 48))
    pred_len    = int(inf_cfg.get('pred_len', 24))
    scaler      = art.get('scaler')
    feature_cols = (
        (art.get('feature_cols') if isinstance(art.get('feature_cols'), (list, tuple)) else None)
        or (data_dict.get('all_feature_cols') if isinstance(data_dict.get('all_feature_cols'), (list, tuple)) else None)
        or list((config.get('model_config', {}) or {}).get('Informer', {}).get('feature_cols') or [])
    )
    calib_ab    = (data_dict or {}).get('val_calib')  # {'a':..., 'b':...} 或 None

    # ---- VAL 整段（严格均值滚动；已 inverse + 校准）----
    try:
        val_df_sc = data_dict.get('val_df_sc')
        if isinstance(val_df_sc, pd.DataFrame) and len(val_df_sc) > 0:
            val_full_df, val_long = rolling_predict_segment(
                model=model,
                df_sc=val_df_sc,
                scaler=scaler,
                feature_cols=feature_cols,
                seq_len=seq_len, label_len=label_len, pred_len=pred_len,
                step=1, mode="mean",
                calib=calib_ab,
            )
            data_dict['val_result_df'] = val_full_df  # index 为该段时间索引，列 ['y_true','yhat']
            data_dict['val_long']      = val_long

            _tail = val_full_df.tail(min(pred_len, len(val_full_df)))
            data_dict['val_tail'] = {
                "timestamps": _tail.index.astype(str).tolist(),
                "y_true": _tail["y_true"].astype(float).tolist(),
                "yhat":  _tail["yhat"].astype(float).tolist(),
            }
    except Exception as e:
        import logging; logging.warning("[train] finalize VAL long failed: %s", e)

    # ---- TEST 整段（严格均值滚动；已 inverse + 校准）----
    try:
        test_df_sc = data_dict.get('test_df_sc')
        if isinstance(test_df_sc, pd.DataFrame) and len(test_df_sc) > 0:
            test_full_df, test_long = rolling_predict_segment(
                model=model,
                df_sc=test_df_sc,
                scaler=scaler,
                feature_cols=feature_cols,
                seq_len=seq_len, label_len=label_len, pred_len=pred_len,
                step=1, mode="mean",
                calib=calib_ab,
            )
            data_dict['test_result_df'] = test_full_df
            data_dict['test_long']      = test_long

            _tail = test_full_df.tail(min(pred_len, len(test_full_df)))
            data_dict['test_tail'] = {
                "timestamps": _tail.index.astype(str).tolist(),
                "y_true": _tail["y_true"].astype(float).tolist(),
                "yhat":  _tail["yhat"].astype(float).tolist(),
            }
    except Exception as e:
        import logging; logging.warning("[train] finalize TEST long failed: %s", e)

def train_informer_model(config: Dict[str, Any], seed: Optional[int] = None) -> Tuple[Any, pd.DataFrame]:
    """
    【最终集成版】Informer 模型的完整训练、预测与残差修正流程。
    """
    # === Seed handling (local, in addition to global set_seed) ===
    if seed is None:
        seed = (
            config.get('seed')
            or config.get('training', {}).get('seed')
            or config.get('default', {}).get('seed')
        )
    try:
        seed = int(seed) if seed is not None else None
    except Exception:
        seed = None

    dl_generator = None
    worker_init_fn = None
    if seed is not None:
        dl_generator = torch.Generator()
        dl_generator.manual_seed(seed)

        def _worker_init_fn(worker_id: int):
            s = seed + worker_id
            np.random.seed(s)
            random.seed(s)
            torch.manual_seed(s)

        worker_init_fn = _worker_init_fn

    informer_cfg = config['model_config']['Informer']
    artifacts_cfg = config['artifacts']
    device = torch.device(config.get('device', 'cpu'))
    
    train_df_sc = config['data']['train_df_sc']
    val_df_sc = config['data']['val_df_sc']
    scaler = config['artifacts']['scaler'] # scaler 从 pipeline 传入

    # === Resolve feature columns (auto single/multi-var) and fix target index ===
    default_cfg = config.get('default', {})
    time_col  = default_cfg.get('time_col', 'date')
    value_col = default_cfg.get('value_col', 'value')

    feature_cols = list(informer_cfg.get('feature_cols') or [])
    if not feature_cols:
        numeric_cols = val_df_sc.select_dtypes(include=[np.number]).columns.tolist()
        if time_col in numeric_cols:
            numeric_cols.remove(time_col)
        feature_cols = [value_col] + [c for c in numeric_cols if c != value_col]
        informer_cfg['feature_cols'] = feature_cols

    lock_order = bool(informer_cfg.get('lock_feature_order', True))
    missing = [c for c in feature_cols if c not in val_df_sc.columns]
    if missing and lock_order:
        raise KeyError(f"Informer.feature_cols missing in validation DataFrame: {missing}")
    if missing and not lock_order:
        for m in missing:
            val_df_sc[m] = 0.0
            train_df_sc[m] = 0.0
            if config.get('data', {}).get('test_df_sc') is not None:
                config['data']['test_df_sc'][m] = 0.0

    config.setdefault('data', {})['all_feature_cols'] = feature_cols
    config.setdefault('artifacts', {})['feature_cols'] = feature_cols
    config.setdefault('artifacts', {})['target_idx'] = 0  # value_col fixed at index 0

    _roll_cfg0 = (config.get('prediction', {}) or {}).get('rolling', {}) or {}
    try:
        _pred_len_snap = int(informer_cfg.get('pred_len', 24))
    except Exception:
        _pred_len_snap = 24
    _step_snap = _roll_cfg0.get('step', _pred_len_snap)
    try:
        _step_snap = int(_step_snap)
    except Exception:
        _step_snap = _pred_len_snap
    rolling_snapshot = {
        "enabled": bool(_roll_cfg0.get('enabled', True)),
        "mode": str(_roll_cfg0.get('mode', 'overwrite')),
        "step": _step_snap,
        "pred_len": _pred_len_snap,
    }
    calibrate_enabled = bool(_roll_cfg0.get('calibrate', True))
    rolling_snapshot["calibrate"] = calibrate_enabled
    config.setdefault('data', {})['rolling_snapshot'] = rolling_snapshot

    # --- 2. 准备 Informer 的输入数据 ---
    x_enc_train, x_dec_train, y_train, _ = prepare_informer_inputs(train_df_sc, config)
    x_enc_val, x_dec_val, y_val, x_feature_val = prepare_informer_inputs(val_df_sc, config)

    # --- 3. 创建 DataLoader ---
    train_loader = make_informer_loader(
        x_enc_train, x_dec_train, y_train, config,
        shuffle=True,
        generator=dl_generator,
        worker_init_fn=worker_init_fn,
    )
    val_loader = make_informer_loader(
        x_enc_val, x_dec_val, y_val, config,
        shuffle=False,
        generator=dl_generator,
        worker_init_fn=worker_init_fn,
    )

    # --- 4. 构建模型、优化器、损失函数 ---
    value_col = config.get('default', {}).get('value_col', 'value')
    resolved_feature_cols = config.get('data', {}).get('all_feature_cols') or informer_cfg.get('feature_cols') or [value_col]
    c_in = len(resolved_feature_cols)
    informer_cfg['enc_in'] = c_in
    informer_cfg['dec_in'] = c_in
    informer_cfg['c_out'] = 1  # always predict target only
    model = build_informer_model(informer_cfg).to(device)
    lr = float(informer_cfg.get('learning_rate', 0.0001))
    wd = float(informer_cfg.get('weight_decay', 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=informer_cfg.get('patience', 5), verbose=True)
    residual_model = None
    
    # --- 5. 训练循环 ---
    n_epochs = int(informer_cfg.get('n_epochs', config.get('training', {}).get('n_epochs', 10)))
    pred_len = informer_cfg['pred_len']

    # === Threshold & ES config ===
    thr_cfg = (config.get('thresholds') or {})
    # 明确默认值；若配置缺失/为 None，则回退到默认
    _rmse_cfg = thr_cfg.get('RMSE')
    _mape_cfg = thr_cfg.get('MAPE')

    rmse_thr = float(_rmse_cfg if _rmse_cfg is not None else 10.0)
    mape_thr = float(_mape_cfg if _mape_cfg is not None else 0.05)

    # 兜底，保证为有限数，避免出现 <= inf
    if not math.isfinite(rmse_thr):
        rmse_thr = 10.0
    if not math.isfinite(mape_thr):
        mape_thr = 0.05

    es_metric_name = str(thr_cfg.get('early_stop_metric', 'MAPE')).upper()  # {'MAPE','RMSE','VAL_LOSS'}
    es_logic = str(thr_cfg.get('logic', 'and')).lower()                     # 'and' or 'or'
    es_reset = bool(thr_cfg.get('patience_reset_if_worse', True))

    print("--- Starting Informer Training ---")
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = []
        for i, (batch_x_enc, batch_x_dec, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_x_enc = torch.as_tensor(batch_x_enc, dtype=torch.float32, device=device).contiguous()
            batch_x_dec = torch.as_tensor(batch_x_dec, dtype=torch.float32, device=device).contiguous()
            batch_y     = torch.as_tensor(batch_y,     dtype=torch.float32, device=device).contiguous()
            outputs = informer_forward(model, batch_x_enc, batch_x_dec, device=device, return_numpy=False)
            loss = criterion(outputs[:, -pred_len:, :], batch_y[:, -pred_len:, :].to(device))
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        # --- 验证（同时计算原尺度 RMSE / MAPE）---
                # --- 验证 ---
        model.eval()
        val_loss = []
        # 为了后面算 RMSE/MAPE，把当轮的预测与真值也收集下来
        val_preds_scaled_epoch = []
        val_true_scaled_epoch  = []
        with torch.no_grad():
            for i, (batch_x_enc, batch_x_dec, batch_y) in enumerate(val_loader):
                batch_x_enc = torch.as_tensor(batch_x_enc, dtype=torch.float32, device=device).contiguous()
                batch_x_dec = torch.as_tensor(batch_x_dec, dtype=torch.float32, device=device).contiguous()
                batch_y     = torch.as_tensor(batch_y,     dtype=torch.float32, device=device).contiguous()
                outputs = informer_forward(model, batch_x_enc, batch_x_dec, device=device, return_numpy=False)
                loss = criterion(outputs[:, -pred_len:, :], batch_y[:, -pred_len:, :].to(device))
                val_loss.append(loss.item())

                # 收集用于阈值评估的 scaled 输出
                c_pred = outputs.shape[-1]
                out_np = safe_to_numpy(outputs[:, -pred_len:, :c_pred])
                y_np   = safe_to_numpy(batch_y[:, -pred_len:, :c_pred])
                val_preds_scaled_epoch.append(out_np)
                val_true_scaled_epoch.append(y_np)

        avg_train_loss = np.average(epoch_loss)
        avg_val_loss   = np.average(val_loss)

        # === 计算当轮的 RMSE / MAPE（原始量纲） ===
        #   注意：阈值是原始量纲（例如 RMSE=10, MAPE=0.05）
        try:
            if len(val_preds_scaled_epoch) > 0:
                _pred_sc = np.concatenate(val_preds_scaled_epoch, axis=0)
                _true_sc = np.concatenate(val_true_scaled_epoch,  axis=0)
                c_pred   = int(_pred_sc.shape[-1])
                _true_sc = _true_sc[:, :, :c_pred]
                _pred_f  = _pred_sc.reshape(-1, c_pred)
                _true_f  = _true_sc.reshape(-1, c_pred)

                _pred_inv = _inverse_transform_targets(_pred_f, scaler, config)
                _true_inv = _inverse_transform_targets(_true_f, scaler, config)

                diff  = (_pred_inv - _true_inv).reshape(-1)
                true1 = _true_inv.reshape(-1)
                rmse  = float(np.sqrt(np.mean(diff ** 2)))
                eps   = 1e-8
                mape  = float(np.mean(np.abs(diff) / (np.abs(true1) + eps)))
            else:
                rmse, mape = float('inf'), float('inf')
        except Exception as _e:
            print(f"[ES] warning: compute val RMSE/MAPE failed: {_e}")
            rmse, mape = float('inf'), float('inf')

        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train_loss:.7f} | Val Loss: {avg_val_loss:.7f} | "
              f"Val RMSE: {rmse:.6f} | Val MAPE: {mape:.6f}")

        # === 以配置的 early_stop_metric（RMSE/MAPE/VAL_LOSS）驱动 patience，并记录指标 ===
        # --- Select early-stop driving metric (lower is better) ---
        es_name = es_metric_name  # already resolved above
        if es_name == 'RMSE':
            es_value = rmse
        elif es_name == 'MAPE':
            es_value = mape
        else:
            es_name = 'VAL_LOSS'
            es_value = avg_val_loss

        # Drive patience using the selected metric
        early_stopping(es_value, model, artifacts_cfg['model_path'])

        # Persist latest validation metrics for app/pipeline use
        try:
            cfg_metrics = config.setdefault('metrics', {})
            cfg_metrics['val'] = {'rmse': float(rmse), 'mape': float(mape)}
            # also keep the current early-stop metric value for transparency
            cfg_metrics.setdefault('val_internal', {})['early_stop_metric'] = {'name': es_name, 'value': float(es_value)}
        except Exception:
            pass

        # === 阈值与 patience 联动早停 ===
        if es_logic == 'or':
            thresholds_met = (rmse <= rmse_thr) or (mape <= mape_thr)
        else:
            thresholds_met = (rmse <= rmse_thr) and (mape <= mape_thr)

        print(f"[ES] thresholds met: {thresholds_met} "
              f"(rmse={rmse:.6f}<= {rmse_thr}, mape={mape:.6f}<= {mape_thr}, logic={es_logic}) "
              f"patience={early_stopping.counter}/{early_stopping.patience}")
        print(f"[ES] driver metric: {es_metric_name}={es_value:.6f} | patience={early_stopping.counter}/{early_stopping.patience}")

        # 只有“patience 用尽 且 阈值达标”才真正 early stop
        if early_stopping.counter >= early_stopping.patience:
            if thresholds_met:
                early_stopping.early_stop = True
                print("Early stopping triggered (patience exhausted & thresholds met).")
            else:
                # 阈值未达标 → 允许继续训练；若开启了 reset，就清零 patience 计数
                if es_reset:
                    early_stopping.counter = 0
                    print("[ES] patience reset because thresholds NOT met; continue training.")

        if early_stopping.early_stop:
            break
    print("--- Finished Informer Training ---")

    # --- 6. 加载最佳模型并在验证集上进行最终预测（收集残差训练所需数据） ---
    model.load_state_dict(torch.load(artifacts_cfg['model_path']))
    model.eval()
    
    val_preds_scaled = []
    with torch.no_grad():
        for i, (batch_x_enc, batch_x_dec, _) in enumerate(val_loader):
            outputs = informer_forward(model, batch_x_enc, batch_x_dec, device=device, return_numpy=True)
            val_preds_scaled.append(safe_to_numpy(outputs[:, -pred_len:, :]))
    
    val_preds_scaled = np.concatenate(val_preds_scaled, axis=0)
    assert_no_nan(val_preds_scaled, "Validation predictions (scaled)")
    
    c_pred = int(val_preds_scaled.shape[-1])
    y_val_true_scaled = y_val[:, -pred_len:, :c_pred]

    # --- 7. 反归一化 ---
    val_preds_flat = val_preds_scaled.reshape(-1, val_preds_scaled.shape[-1])
    y_val_true_flat = y_val_true_scaled.reshape(-1, y_val_true_scaled.shape[-1])
    
    val_preds_inversed = _inverse_transform_targets(val_preds_flat, scaler, config)
    y_val_true_inversed = _inverse_transform_targets(y_val_true_flat, scaler, config)

    x_feature_val = _ensure_timestep_features(x_feature_val, pred_len)

    # --- 8. 残差建模与修正（在 val 上拟合，test 可选应用） ---
    final_preds = val_preds_inversed
    use_residual = informer_cfg.get('use_residual', True)
    if use_residual:
        print("--- Applying Residual Modeling ---")
        final_preds, residual_model, _, _ = train_and_predict_residual(
            y_true=y_val_true_inversed,
            y_pred=val_preds_inversed,
            x_features=x_feature_val
        )
        
        residual_model_path = config.get('artifacts', {}).get('residual_model_path')
        if residual_model_path:
            try:
                os.makedirs(os.path.dirname(residual_model_path), exist_ok=True)
                joblib.dump(residual_model, residual_model_path)
                print(f"Residual model saved to {residual_model_path}")
            except Exception as e:
                print(f"Warning: failed to save residual model to {residual_model_path}: {e}")
        else:
            print("Warning: artifacts.residual_model_path not configured; residual model not saved.")

        # （可选）测试集 quick-pass，维持向后兼容；整段滚动由 finalize 统一生成
        test_df_sc = config.get('data', {}).get('test_df_sc')
        if test_df_sc is not None:
            try:
                x_enc_test, x_dec_test, y_test, x_feature_test = prepare_informer_inputs(test_df_sc, config)
                x_feature_test = _ensure_timestep_features(x_feature_test, pred_len)
                test_loader = make_informer_loader(
                    x_enc_test, x_dec_test, y_test, config,
                    shuffle=False,
                    generator=dl_generator,
                    worker_init_fn=worker_init_fn,
                )
                test_preds_scaled = []
                y_test_true_scaled_batches = []
                with torch.no_grad():
                    for i, (batch_x_enc, batch_x_dec, batch_y) in enumerate(test_loader):
                        batch_x_enc = torch.as_tensor(batch_x_enc, dtype=torch.float32, device=device).contiguous()
                        batch_x_dec = torch.as_tensor(batch_x_dec, dtype=torch.float32, device=device).contiguous()
                        batch_y     = torch.as_tensor(batch_y,     dtype=torch.float32, device=device).contiguous()
                        outputs = informer_forward(model, batch_x_enc, batch_x_dec, device=device, return_numpy=True)
                        test_preds_scaled.append(safe_to_numpy(outputs[:, -pred_len:, :]))
                        y_test_true_scaled_batches.append(safe_to_numpy(batch_y[:, -pred_len:, :]))
                if len(test_preds_scaled) > 0:
                    test_preds_scaled = np.concatenate(test_preds_scaled, axis=0)
                    y_test_true_scaled = np.concatenate(y_test_true_scaled_batches, axis=0)
                    c_pred_t = int(test_preds_scaled.shape[-1])
                    y_test_true_scaled = y_test_true_scaled[:, :, :c_pred_t]
                    test_preds_flat = test_preds_scaled.reshape(-1, test_preds_scaled.shape[-1])
                    y_test_true_flat = y_test_true_scaled.reshape(-1, y_test_true_scaled.shape[-1])
                    test_preds_inversed = _inverse_transform_targets(test_preds_flat, scaler, config)
                    y_test_true_inversed = _inverse_transform_targets(y_test_true_flat, scaler, config)
                    test_final_preds = test_preds_inversed
                    if use_residual and residual_model is not None:
                        try:
                            test_preds_3d = test_preds_inversed.reshape(test_preds_scaled.shape)
                            yhat_corr_3d = apply_residual(test_preds_3d, x_feature_test, residual_model)
                            test_final_preds = yhat_corr_3d.reshape(test_preds_inversed.shape)
                        except Exception as e:
                            print(f"Warning: apply_residual on test failed: {e}; using base predictions.")
                    test_result_df = pd.DataFrame({
                        'y_true': y_test_true_inversed.flatten(),
                        'yhat':  test_final_preds.flatten()
                    })
                    data_blk = config.setdefault('data', {})
                    data_blk['test_result_df'] = test_result_df
            except Exception as e:
                print(f"Warning: generating test predictions failed: {e}")

    # --- 7.5 将验证集的“窗口级”结果写入 config['data'] 作为兜底 ---
    try:
        val_fallback_df = pd.DataFrame({
            'y_true': y_val_true_inversed.flatten(),
            'yhat':  final_preds.flatten()
        })
        config.setdefault('data', {})['val_result_df'] = val_fallback_df
    except Exception:
        pass

    # === 8. 改为“整段密集预测”（h=1, step=1），不再写入 val_long/test_long ===
    try:
        data_blk = config.setdefault('data', {})
        artifacts_blk = config.setdefault('artifacts', {})
        scaler = artifacts_blk.get('scaler')
        inf_cfg = (config.get('model_config', {}) or {}).get('Informer', {}) or {}
        feature_cols = list(data_blk.get('all_feature_cols') or inf_cfg.get('feature_cols') or [value_col])

        # 计算各段长度
        train_df_sc = data_blk.get('train_df_sc')
        val_df_sc   = data_blk.get('val_df_sc')
        test_df_sc  = data_blk.get('test_df_sc')
        len(train_df_sc) if isinstance(train_df_sc, pd.DataFrame) else 0
        n_val   = len(val_df_sc)   if isinstance(val_df_sc,   pd.DataFrame) else 0
        n_test  = len(test_df_sc)  if isinstance(test_df_sc,  pd.DataFrame) else 0

        # —— 验证段：用 train+val 的上下文做密集预测，取最后 n_val 个点
        if n_val > 0:
            df_all_val = pd.concat([train_df_sc, val_df_sc], axis=0, ignore_index=True)
            val_dense = _dense_predict_last_k(
                model, df_all_val, n_val, config, feature_cols, scaler, residual_model=residual_model
            )
            data_blk['val_dense'] = val_dense

        # —— 测试段：用 train+val+test 的上下文做密集预测，取最后 n_test 个点
        if n_test > 0:
            df_all_test = pd.concat([train_df_sc, val_df_sc, test_df_sc], axis=0, ignore_index=True)
            test_dense = _dense_predict_last_k(
                model, df_all_test, n_test, config, feature_cols, scaler, residual_model=residual_model
            )
            data_blk['test_dense'] = test_dense
    except Exception as e:
        print(f"Warning: dense prediction failed: {e}")

    # --- Compute final RMSE/MAPE from dense outputs (if available) ---
    def _compute_dense_metrics(df):
        try:
            if isinstance(df, pd.DataFrame) and {'y_true','yhat'}.issubset(df.columns) and len(df) > 0:
                diff = (df['yhat'].astype(float) - df['y_true'].astype(float)).to_numpy()
                true = df['y_true'].astype(float).to_numpy()
                rmse_f = float(np.sqrt(np.mean(diff ** 2)))
                mape_f = float(np.mean(np.abs(diff) / (np.abs(true) + 1e-8)))
                return {'rmse': rmse_f, 'mape': mape_f}
        except Exception:
            return None
        return None

    try:
        data_blk = config.get('data', {}) or {}
        metrics_blk = config.setdefault('metrics', {})
        if isinstance(data_blk.get('val_dense'), pd.DataFrame):
            m_val = _compute_dense_metrics(data_blk['val_dense'])
            if m_val:
                metrics_blk['val'] = m_val
        if isinstance(data_blk.get('test_dense'), pd.DataFrame):
            m_test = _compute_dense_metrics(data_blk['test_dense'])
            if m_test:
                metrics_blk['test'] = m_test
        # Optional echo to console
        if metrics_blk.get('val') or metrics_blk.get('test'):
            print(f"[pipeline] metrics -> val: {metrics_blk.get('val')} | test: {metrics_blk.get('test')}")
    except Exception:
        pass

    # Echo artifacts back (feature cols / target index) for pipeline & app
    config.setdefault('artifacts', {})['feature_cols'] = list(config.get('data', {}).get('all_feature_cols') or informer_cfg.get('feature_cols') or [value_col])
    config['artifacts']['target_idx'] = 0

    # 优先返回密集预测（连续 1-step），否则退回窗口级结果
    data_blk = config.get('data', {})
    # 优先返回密集预测（连续 1-step），否则退回窗口级结果
    if isinstance(data_blk.get('val_dense'), pd.DataFrame) and not data_blk['val_dense'].empty:
        result_df = data_blk['val_dense'].reset_index().rename(columns={config.get('default', {}).get('time_col', 'date'): 'timestamp'})
    elif isinstance(data_blk.get('val_result_df'), pd.DataFrame) and not data_blk['val_result_df'].empty:
        result_df = data_blk['val_result_df']
    else:
        result_df = pd.DataFrame({
            'y_true': y_val_true_inversed.flatten(),
            'yhat': final_preds.flatten()
        })

    return model, result_df