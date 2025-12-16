import torch
import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Any, Tuple, Optional, Callable, cast
from models.informer.forward import informer_forward
from models.informer.input_utils import prepare_informer_inputs, make_informer_loader

# === Helpers: ensure timestep features (W, L, F) and robust inverse of target columns ===

def _ensure_timestep_features(x_feature, pred_len: int):
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
    return None

def _gen_rolling_starts(values_len: int, seq_len: int, label_len: int, horizon: int, step: int) -> list:
    starts = []
    start = max(0, values_len - (seq_len + label_len + horizon))
    while start + seq_len + label_len + horizon <= values_len:
        starts.append(start)
        start += max(1, int(step))
    if not starts:
        starts = [max(0, values_len - (seq_len + label_len + horizon))]
    return starts

def _inverse_transform_targets(arr2d: np.ndarray, scaler, config) -> np.ndarray:
    """
    只对目标通道做 inverse。scaler 可能是按多列 fit 的，这里用“拼宽再 inverse 再取回”的策略。
    """
    arr2d = np.asarray(arr2d, dtype=np.float32)
    n_in = getattr(scaler, 'n_features_in_', None)
    if n_in is None:
        try:
            return scaler.inverse_transform(arr2d)
        except Exception:
            return arr2d

    default_cfg = (config.get('default') or {})
    value_col = default_cfg.get('value_col', 'value')

    # 尝试从 config.artifacts / config.data 拿全量列顺序
    all_cols = (
        (config.get('artifacts', {}) or {}).get('feature_cols') or
        (config.get('data', {}) or {}).get('all_feature_cols') or
        [value_col]
    )

    tmp = np.zeros((arr2d.shape[0], n_in), dtype=np.float32)
    # 只考虑单目标（C=1）或多目标时“按 feature_cols 顺序放回去”的常见场景
    if arr2d.shape[1] == 1:
        try:
            idx = all_cols.index(value_col)
        except ValueError:
            idx = 0
        tmp[:, idx] = arr2d[:, 0]
        inv = scaler.inverse_transform(tmp)
        out = inv[:, [idx]]
    else:
        # 多通道：尽量把每个通道映射到 all_cols 前面的列
        k = min(arr2d.shape[1], len(all_cols), n_in)
        for j in range(k):
            tmp[:, j] = arr2d[:, j]
        inv = scaler.inverse_transform(tmp)
        out = inv[:, :k]
    return out

# 1. 导入所有必需的模块
from models.informer.informer import build_informer_model
from models.informer.forward import informer_forward
from models.informer.input_utils import prepare_informer_inputs
from utils.array_utils import assert_no_nan, tensor_to_numpy
from preprocessing.feature_engineering import generate_features
from utils.residual_modeling import train_and_predict_residual, apply_residual

class InformerPredictor:
    """
    一个封装了完整预测流程的类。
    它负责加载所有必需的工件（主模型、scaler、残差模型），
    并对新数据进行与训练流程完全一致的端到端预测。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化预测器。

        Args:
            config (Dict[str, Any]): 完整的项目配置字典。
        """
        print("--- Initializing InformerPredictor ---")
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))

        # === Sync feature_cols/target_idx with training artifacts ===
        artifacts_cfg = config.get('artifacts', {})
        informer_cfg = config['model_config']['Informer']
        default_cfg = config.get('default', {})
        time_col  = default_cfg.get('time_col', 'date')
        value_col = default_cfg.get('value_col', 'value')

        feature_cols = artifacts_cfg.get('feature_cols') or informer_cfg.get('feature_cols') or [value_col]
        # keep value_col at index 0 and drop time_col if present
        feature_cols = [value_col] + [c for c in feature_cols if c != value_col and c != time_col]
        self.feature_cols = feature_cols
        self.target_idx = int(artifacts_cfg.get('target_idx', 0))

        # align channels with feature_cols length and 1-output head
        F = len(self.feature_cols)
        informer_cfg['enc_in'] = F
        informer_cfg['dec_in'] = F
        informer_cfg['c_out'] = 1
        informer_cfg['feature_cols'] = self.feature_cols  # lock order for prediction

        # 1. 构建并加载主模型（根据训练脚本：使用 artifacts.model_path 作为权重文件）
        self.model = build_informer_model(informer_cfg).to(self.device)
        model_path = artifacts_cfg.get('model_path')
        if not model_path:
            raise RuntimeError("`artifacts.model_path` missing in config. 请在 configs.yaml 的 artifacts 下配置 model_path。")
        if not os.path.isfile(model_path):
            raise RuntimeError(f"Model file not found at {model_path}. Please train the model first.")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Main model loaded successfully from {model_path}")

        # 2. 加载数据缩放器（scaler）
        # 训练脚本优先使用内存中的 scaler: config['artifacts']['scaler']
        self.scaler = artifacts_cfg.get('scaler', None)
        if self.scaler is None:
            # 兼容：如果提供了 scaler_path 则尝试从磁盘加载
            scaler_path = artifacts_cfg.get('scaler_path')
            if not scaler_path:
                raise RuntimeError(
                    "No scaler provided. Expect config['artifacts']['scaler'] (in-memory) or `artifacts.scaler_path` (on-disk)."
                )
            if not os.path.isfile(scaler_path):
                raise RuntimeError(f"Scaler file not found at {scaler_path}. Please train the model first.")
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
        else:
            print("Scaler loaded from in-memory config['artifacts']['scaler']")

        # 4) 加载训练好的残差模型（可选）
        self.residual_model = None
        residual_model_path = artifacts_cfg.get('residual_model_path')
        if residual_model_path and os.path.isfile(residual_model_path):
            try:
                self.residual_model = joblib.load(residual_model_path)
                print(f"Residual model loaded from {residual_model_path}")
            except Exception as e:
                print(f"Warning: Failed to load residual model from {residual_model_path}: {e}")
        else:
            print("Info: Residual model not configured or file not found. Proceeding without residual correction.")

        # 5) Rolling snapshot for pipeline/UI (so pipeline/app can always read it)
        try:
            roll_cfg = (config.get('prediction', {}) or {}).get('rolling', {}) or {}
            inf_cfg  = self.config.get('model_config', {}).get('Informer', {})
            _pred_len_snap = int(inf_cfg.get('pred_len', 24))
            _step_snap = roll_cfg.get('step', _pred_len_snap)
            try:
                _step_snap = int(_step_snap)
            except Exception:
                _step_snap = _pred_len_snap
            snapshot = {
                'enabled': bool(roll_cfg.get('enabled', True)),
                'mode':    str(roll_cfg.get('mode', 'overwrite')),
                'step':    _step_snap,
                'pred_len': _pred_len_snap,
            }
            self.config.setdefault('data', {})['rolling_snapshot'] = snapshot
        except Exception:
            # do not block prediction if snapshot generation fails
            pass

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        对新的 DataFrame 进行预测。

        Args:
            df (pd.DataFrame): 包含新数据的 DataFrame，其长度应至少为 `seq_len`。

        Returns:
            np.ndarray: 经过反归一化和残差修正后的最终预测结果。
        """
        print("--- Starting prediction process ---")
        informer_cfg = self.config['model_config']['Informer']
        residual_cfg = self.config.get('residual_modeling', {})
        residual_enabled = bool(residual_cfg.get('enabled', True))
        use_x_feature = bool(residual_cfg.get('use_x_feature', True))

        seq_len = informer_cfg['seq_len']

        if len(df) < seq_len:
            raise ValueError(f"Input data length ({len(df)}) is less than the required sequence length ({seq_len}).")

        # --- 1. 特征工程 ---
        df_featured, _time_feats = generate_features(df, self.config)

        # --- 2. 数据归一化 (与训练一致，优先使用传入的 scaler 对特征列做变换) ---
        informer_cfg = self.config['model_config']['Informer']
        feature_cols = self.feature_cols

        # 轻量一致性提示：scaler 的特征数 vs feature_cols
        n_in = getattr(self.scaler, 'n_features_in_', None)
        if n_in is not None and n_in != len(feature_cols):
            print(f"[predict] Warning: scaler expects {n_in} features, but feature_cols has {len(feature_cols)}: {feature_cols}")

        df_scaled = df_featured.copy()
        # 如果 scaler 是 sklearn 风格对象（有 transform），按列变换；否则认为输入已被外部流程缩放
        if hasattr(self.scaler, "transform"):
            df_scaled[feature_cols] = self.scaler.transform(df_featured[feature_cols])
        else:
            print("[predict] scaler has no transform; assume input already scaled.")
        
        # --- 3. 创建滑动窗口 ---
        input_data = df_scaled.tail(informer_cfg['seq_len'] + informer_cfg['pred_len'])
        # 确保在预测阶段也提供与训练阶段一致的特征列集合，供 prepare_informer_inputs 使用
        self.config.setdefault('data', {})
        self.config['data']['all_feature_cols'] = feature_cols
        x_enc, x_dec, _, x_feature = prepare_informer_inputs(input_data, self.config)

        # 统一 x_feature 为时间步级 (W, L, F)，残差工具会自动展平为 (W*L, F)
        x_feature = _ensure_timestep_features(x_feature, informer_cfg['pred_len'])

        # --- 4. 模型推理 ---
        with torch.no_grad():
            outputs_scaled = informer_forward(self.model, x_enc, x_dec, device=self.device, return_numpy=False)
        
        _preds_scaled = tensor_to_numpy(outputs_scaled[:, -informer_cfg['pred_len']:, :])
        if _preds_scaled is None:
            raise RuntimeError("tensor_to_numpy returned None. 请检查 informer_forward 的输出是否为有效张量。")
        preds_scaled = cast(np.ndarray, _preds_scaled)
        assert_no_nan(preds_scaled, "Raw model predictions (scaled)")

        # --- 5. 【核心】反归一化（与训练一致，使用 scaler.inverse_transform） ---
        if hasattr(self.scaler, "inverse_transform"):
            C = preds_scaled.shape[-1]
            flat = preds_scaled.reshape(-1, C)
            inv_flat = _inverse_transform_targets(flat, self.scaler, self.config)
            preds_inversed = inv_flat.reshape(preds_scaled.shape)
        else:
            print("Warning: scaler has no `inverse_transform`; predictions will remain in scaled space.")
            preds_inversed = preds_scaled
        
        # --- 6. 【核心】残差修正 ---
        if residual_enabled and self.residual_model is not None:
            print("--- Applying Residual Modeling for prediction ---")
            try:
                # 当 use_x_feature=False 时，不传特征给残差模型
                x_feat_for_res = x_feature if use_x_feature else None
                final_preds = apply_residual(preds_inversed, x_feat_for_res, self.residual_model)
            except Exception as e:
                print(f"Warning: residual correction failed: {e}; falling back to base predictions.")
                final_preds = preds_inversed
        else:
            final_preds = preds_inversed
            
        print("--- Prediction process finished ---")
        return final_preds.flatten()
    
    def rolling_predict(
    self,
    df: pd.DataFrame,
    horizon: int,
    step: Optional[int] = None,
    mode: str = "overwrite",
) -> np.ndarray:
        """
        对整段数据进行滚动预测，返回长度为 len(df) 的一维数组：
        - 未被覆盖的段为 NaN；
        - 覆盖的段为已反归一化、并经过（可选）残差修正的预测值。
        """
        cfg = self.config
        informer_base = cfg["model_config"]["Informer"]

        seq_len   = int(informer_base["seq_len"])
        label_len = int(informer_base.get("label_len", 0))
        pred_len  = int(horizon)                  # 本次滚动的地平线
        step      = pred_len if step is None else int(step)

        if len(df) < seq_len:
            raise ValueError(f"Input length {len(df)} < seq_len {seq_len}")

        # --- 1) 特征工程（与训练一致） ---
        df_featured, _ = generate_features(df, cfg)

        # --- 2) 缩放（按训练特征列 & scaler） ---
        feature_cols = self.feature_cols

        n_in = getattr(self.scaler, 'n_features_in_', None)
        if n_in is not None and n_in != len(feature_cols):
            print(f"[rolling_predict] Warning: scaler expects {n_in} features, but feature_cols has {len(feature_cols)}: {feature_cols}")

        df_scaled = df_featured.copy()
        if hasattr(self.scaler, "transform"):
            df_scaled[feature_cols] = self.scaler.transform(df_featured[feature_cols])
        else:
            print("[rolling_predict] scaler has no transform; assume input already scaled.")

        values_len = len(df_scaled)

        # --- 3) 计算滚动窗口起点（每个起点恰好覆盖一个窗口） ---
        starts: list[int] = []
        start = 0
        while start + seq_len + label_len + pred_len <= values_len:
            starts.append(start)
            start += step
        if not starts:
            # 兜底：至少返回最后一个完整窗口
            last_start = max(0, values_len - (seq_len + label_len + pred_len))
            starts = [last_start]

        # Update snapshot with online rolling stats
        try:
            snap = self.config.setdefault('data', {}).setdefault('rolling_snapshot', {})
            snap.update({
                'online_windows': int(len(starts)),
                'online_points':  int(len(starts) * pred_len),
            })
        except Exception:
            pass

        # --- 4) 合并容器 ---
        merged = np.full(values_len, np.nan, dtype=float)
        used   = np.zeros(values_len, dtype=bool)

        # 为 prepare_informer_inputs 构造一个“覆盖了 pred_len 的临时配置”
        tmp_cfg = dict(cfg)
        tmp_cfg.setdefault('model_config', {})
        tmp_cfg['model_config'] = dict(cfg['model_config'])
        tmp_cfg['model_config']['Informer'] = dict(informer_base)
        tmp_cfg['model_config']['Informer']['pred_len'] = pred_len
        tmp_cfg.setdefault('data', {})
        tmp_cfg['data'] = dict(cfg.get('data', {}))
        tmp_cfg['data']['all_feature_cols'] = feature_cols

        residual_cfg   = cfg.get('residual_modeling', {})
        residual_on    = bool(residual_cfg.get('enabled', True))
        use_x_feature  = bool(residual_cfg.get('use_x_feature', True))

        # --- 5) 逐窗口推理并合并（关键：强制单窗口 + 只写入 pred_len 段） ---
        for s in starts:
            sl = df_scaled.iloc[s : s + seq_len + label_len + pred_len]
            x_enc, x_dec, _, x_feature = prepare_informer_inputs(sl, tmp_cfg)

            # a) 如果 prepare_informer_inputs 返回了多个窗口（B>1），仅取“最后一个窗口”
            if isinstance(x_enc, np.ndarray) and x_enc.ndim == 3 and x_enc.shape[0] > 1:
                x_enc = x_enc[-1:, ...]
                x_dec = x_dec[-1:, ...]
                if x_feature is not None and getattr(x_feature, "ndim", 0) == 3 and x_feature.shape[0] > 1:
                    x_feature = x_feature[-1:, ...]

            # b) 规范 x_feature 为 (1, L, F)（残差阶段会自动展平时间维）
            x_feature = _ensure_timestep_features(x_feature, pred_len)

            # c) 模型推理：输出形状应为 (1, label_len+pred_len, C)，只取最后 pred_len
            with torch.no_grad():
                outputs_scaled = informer_forward(self.model, x_enc, x_dec, device=self.device, return_numpy=False)

            y_scaled_tail = tensor_to_numpy(outputs_scaled[:, -pred_len:, :])  # (1, pred_len, C)
            if y_scaled_tail is None:
                raise RuntimeError("tensor_to_numpy returned None for rolling outputs")

            # d) 只展平“时间维”（保持 C 这一维），得到 (pred_len, C)
            y_flat = y_scaled_tail.reshape(-1, y_scaled_tail.shape[-1])

            # e) 仅对目标列做反归一化，得到 (pred_len, C_target) -> 再扁平成 (pred_len,)
            y_inv_flat = _inverse_transform_targets(y_flat, self.scaler, cfg)  # (pred_len, C_target)
            y_inv = y_inv_flat.reshape(-1)                                     # (pred_len,)

            # f) 残差修正：仅对“尾部 pred_len”进行，并只展平时间维
            if residual_on and self.residual_model is not None:
                try:
                    x_feat_tail = x_feature[:, -pred_len:, :] if (x_feature is not None and x_feature.ndim == 3) else None
                    x_feat_tail = x_feat_tail if use_x_feature else None
                    # y_inv:(pred_len,) -> (pred_len,1)；x_feat_tail:(1,pred_len,F) -> residual里会展平为 (pred_len,F)
                    y_corr = apply_residual(y_inv.reshape(-1, 1), x_feat_tail, self.residual_model).reshape(-1)
                    y_inv  = y_corr
                except Exception as e:
                    print(f"[rolling_predict] residual failed: {e}; fallback to base predictions.")

            # g) 合并回整体序列 —— 注意：一次只写入 pred_len 个点
            st_idx = s + seq_len + label_len
            ed_idx = st_idx + pred_len
            if mode == "overwrite":
                merged[st_idx:ed_idx] = y_inv
                used[st_idx:ed_idx]   = True
            elif mode == "mean":
                prev = merged[st_idx:ed_idx]
                ok   = ~np.isnan(prev)
                prev[ok]  = (prev[ok] + y_inv[ok]) / 2.0
                prev[~ok] = y_inv[~ok]
                merged[st_idx:ed_idx] = prev
                used[st_idx:ed_idx]   = True
            else:
                raise ValueError(f"Unknown merge mode: {mode}")


        return merged

# === Utility: rolling prediction on a pre-scaled segment (strict-mean merge, optional calibration) ===
# 1) 修改签名：默认值改为 {}
from typing import Dict, Any, Optional

def _inverse_targets(arr2d: np.ndarray, scaler, config: Dict[str, Any]) -> np.ndarray:
    arr2d = np.asarray(arr2d, dtype=np.float32)
    n_in = getattr(scaler, 'n_features_in_', None)
    if n_in is None or not hasattr(scaler, "inverse_transform"):
        return arr2d
    data_cfg  = config.get('data', {})
    all_cols  = list(data_cfg.get('all_feature_cols') or [])
    inf_cfg   = (config.get('model_config', {}) or {}).get('Informer', {}) or {}
    feat_cols = list(inf_cfg.get('feature_cols') or [config.get('default', {}).get('value_col', 'value')])
    target    = config.get('default', {}).get('value_col', 'value')

    if arr2d.shape[1] == n_in:
        return scaler.inverse_transform(arr2d)

    tmp = np.zeros((arr2d.shape[0], n_in), dtype=np.float32)
    used = []
    if len(feat_cols) == arr2d.shape[1] and len(feat_cols) > 0:
        for j, name in enumerate(feat_cols):
            try:
                idx = all_cols.index(name)
            except ValueError:
                idx = min(j, n_in - 1)
            tmp[:, idx] = arr2d[:, j]
            used.append(idx)
    else:
        try:
            idx0 = all_cols.index(target)
        except ValueError:
            idx0 = 0
        tmp[:, idx0] = arr2d[:, 0]
        used = [idx0]

    inv = scaler.inverse_transform(tmp)
    out = np.zeros_like(arr2d)
    for j, k in enumerate(used):
        out[:, j] = inv[:, k]
    return out


def rolling_predict_segment(
    model,
    df_sc: pd.DataFrame,
    scaler,
    feature_cols,
    *,
    seq_len: int,
    label_len: int,
    pred_len: int,
    step: int = 1,
    mode: str = "mean",          # "mean" | "overwrite" 等
    calib: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    用固定窗口在整段 df_sc 上滚动预测，并把重叠部分按严格均值合并。
    返回：
      - full_df: index 对齐的 DataFrame(['y_true','yhat'])
      - long_payload: {'timestamps': [...], 'y_true': [...], 'yhat': [...]}
    说明：
      * 只使用传入的 seq_len/label_len/pred_len，不再从 config 顶层读取，避免 'seq_len' KeyError。
      * config 仅用于 inverse 时的列对齐信息（feature_cols/all_feature_cols/value_col）。
    """
    assert isinstance(df_sc, pd.DataFrame) and len(df_sc) > 0
    if config is None:
        config = {}

    # 构造临时 cfg 以复用现有的 prepare_informer_inputs / DataLoader 逻辑
    tmp_cfg = dict(config)
    mc = dict(((config.get('model_config') or {}).get('Informer') or {}))
    mc.update({'seq_len': int(seq_len), 'label_len': int(label_len), 'pred_len': int(pred_len)})
    tmp_cfg['model_config'] = {'Informer': mc}
    tmp_cfg.setdefault('data', {})
    tmp_cfg['data'] = dict(config.get('data', {}))
    tmp_cfg['data']['all_feature_cols'] = list(feature_cols)

    # 准备输入（整个 df_sc 一次性切所有窗口）
    x_enc, x_dec, y_true_scaled, x_feature = prepare_informer_inputs(df_sc, tmp_cfg)
    loader = make_informer_loader(x_enc, x_dec, y_true_scaled, tmp_cfg, shuffle=False)

    device = next(model.parameters()).device
    preds_scaled = []
    y_true_scaled_batches = []
    with torch.no_grad():
        for batch_x_enc, batch_x_dec, batch_y in loader:
            bxe = torch.as_tensor(batch_x_enc, dtype=torch.float32, device=device).contiguous()
            bxd = torch.as_tensor(batch_x_dec, dtype=torch.float32, device=device).contiguous()
            out = informer_forward(model, bxe, bxd, device=device, return_numpy=True)
            preds_scaled.append(out[:, -pred_len:, :])           # 只取最后 pred_len
            y_true_scaled_batches.append(batch_y[:, -pred_len:, :])

    if len(preds_scaled) == 0:
        # 空 df 兜底
        return pd.DataFrame(columns=['y_true','yhat']), {'timestamps': [], 'y_true': [], 'yhat': []}

    preds_scaled = np.concatenate(preds_scaled, axis=0)
    y_true_scaled = np.concatenate(y_true_scaled_batches, axis=0)
    # 对齐预测的输出通道数
    c_pred = int(preds_scaled.shape[-1])
    y_true_scaled = y_true_scaled[:, :, :c_pred]

    # 展平 → 反归一化
    preds_flat = preds_scaled.reshape(-1, c_pred)
    y_true_flat = y_true_scaled.reshape(-1, c_pred)
    preds_inv = _inverse_targets(preds_flat, scaler, tmp_cfg)
    y_true_inv = _inverse_targets(y_true_flat, scaler, tmp_cfg)

    # 校准（可选，线性 y = a*y + b）
    if calib and all(k in calib for k in ('a', 'b')):
        a, b = float(calib['a']), float(calib['b'])
        preds_inv = preds_inv * a + b

    # --- Optional residual correction on inverse predictions (window x pred_len x C) ---
    try:
        res_cfg = (config.get('residual_modeling') or {})
        res_on  = bool(res_cfg.get('enabled', True))
        residual_model = None
        residual_model_path = (config.get('artifacts') or {}).get('residual_model_path')
        if res_on and residual_model_path and os.path.exists(residual_model_path):
            try:
                residual_model = joblib.load(residual_model_path)
            except Exception as e:
                print(f"[rolling_predict_segment] warn: load residual model failed: {e}; skip residual.")
        if residual_model is not None:
            # reshape to (W, pred_len, C)
            n_window = int(preds_scaled.shape[0])
            C = int(preds_inv.shape[-1])
            try:
                yhat_inv_3d = preds_inv.reshape(n_window, int(pred_len), C)
                # align x_feature to tail pred_len if available
                if x_feature is not None and getattr(x_feature, 'ndim', 0) == 3:
                    x_feature_3d = x_feature[:, -int(pred_len):, :]
                else:
                    x_feature_3d = None
                yhat_corr_3d = apply_residual(yhat_inv_3d, x_feature_3d, residual_model)
                # flatten back for downstream merging
                preds_inv = np.asarray(yhat_corr_3d, dtype=np.float32).reshape(-1, C)
            except Exception as e:
                print(f"[rolling_predict_segment] warn: apply_residual failed: {e}; keep base predictions.")
    except Exception as e:
        print(f"[rolling_predict_segment] residual block error: {e}")

    # Silence unused-parameter warnings for step and mode
    _ = (step, mode)  # keep parameters for compatibility

    # 先拿原始时间索引
    try:
        time_col = (config.get('default') or {}).get('time_col', 'date')
        ts = pd.to_datetime(df_sc[time_col])
    except Exception:
        # Fallback to a simple integer Index; avoid RangeIndex to prevent .iloc attribute errors later.
        ts = pd.Index(np.arange(len(df_sc)))

    # 末端每个窗口对应的 pred_len 个时间点索引
    # prepare_informer_inputs 的切窗是顺序滑动的，因此可以构造对齐索引：
    n_window = preds_scaled.shape[0]
    total_len = len(ts)
    # 窗口结束位置从 (seq_len+label_len) 到 total_len，每步 1
    # 与 make_informer_loader 的切窗对齐
    end_positions = list(range(seq_len + label_len, total_len + 1))
    end_positions = end_positions[:n_window]  # 对齐可能的边界
    # ==== PATCH: guard for empty indices ====
    if len(end_positions) == 0:
        return pd.DataFrame(columns=['y_true','yhat']), {'timestamps': [], 'y_true': [], 'yhat': []}
    # 为每个窗口分配 pred_len 个时间戳
    idx_list = []
    for end in end_positions:
        # 预测区间 [end - pred_len, end)
        st = max(0, end - pred_len)
        ed = end
        # slice time index safely; length may be < pred_len at edges
        seg = ts[st:ed]
        try:
            idx_list.append(seg.to_numpy())
        except Exception:
            idx_list.append(np.asarray(seg))

    # ==== PATCH: robust all_idx construction ====
    if len(idx_list) == 0:
        return pd.DataFrame(columns=['y_true','yhat']), {'timestamps': [], 'y_true': [], 'yhat': []}
    try:
        all_idx = pd.Index(np.unique(np.concatenate(idx_list)))
    except Exception:
        # fallback if concat fails due to mixed types
        flat = []
        for arr in idx_list:
            try:
                flat.extend(list(arr))
            except Exception:
                pass
        if len(flat) == 0:
            return pd.DataFrame(columns=['y_true','yhat']), {'timestamps': [], 'y_true': [], 'yhat': []}
        all_idx = pd.Index(flat).unique()
    accum = np.zeros((len(all_idx),), dtype=float)
    count = np.zeros((len(all_idx),), dtype=float)
    truth_accum = np.zeros((len(all_idx),), dtype=float)
    truth_count = np.zeros((len(all_idx),), dtype=float)

    flat_pred = preds_inv.reshape(-1)
    flat_true = y_true_inv.reshape(-1)
    # ==== PATCH: merge loop with bounds checks ====
    k = 0
    W = min(n_window, len(idx_list))
    for w in range(W):
        w_idx = pd.Index(idx_list[w])
        pos = all_idx.get_indexer(w_idx)
        m = len(pos)
        if m <= 0:
            continue
        # clip to available length in flat arrays
        end_k = min(k + m, flat_pred.size, flat_true.size)
        m2 = max(0, end_k - k)
        if m2 <= 0:
            break
        pos2 = pos[:m2]
        accum[pos2] += flat_pred[k:end_k]
        count[pos2] += 1.0
        truth_accum[pos2] += flat_true[k:end_k]
        truth_count[pos2] += 1.0
        k = end_k

    yhat = np.divide(accum, count, out=np.full_like(accum, np.nan), where=count > 0)
    ytru = np.divide(truth_accum, truth_count, out=np.full_like(truth_accum, np.nan), where=truth_count > 0)

    # ==== PATCH: avoid forced datetime conversion ====
    try:
        _idx = pd.to_datetime(all_idx)
    except Exception:
        _idx = all_idx
    full_df = pd.DataFrame({'y_true': ytru, 'yhat': yhat}, index=_idx).sort_index()

    long_payload = {
        'timestamps': full_df.index.astype(str).tolist(),
        'y_true': full_df['y_true'].astype(float).tolist(),
        'yhat':  full_df['yhat'].astype(float).tolist(),
    }
    return full_df, long_payload