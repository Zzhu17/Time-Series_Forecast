import pandas as pd
from typing import Dict, Any, Tuple, List
import os
import numpy as np
import torch
import random
from typing import Optional

# ---------- 连续序列拼接：供 plot.py 连续分支直接使用 ----------
def build_continuous_series(train_df_plot, val_dense, test_dense, time_col=None):
    """
    将 train/val/test 的真值与预测拼接为“连续绘图序列”。

    返回:
      - full_truth (pd.Series): train_true -> val_true -> test_true （一条连续线，DatetimeIndex）
      - full_pred_cont (pd.Series): (训练末端“衔接点”) -> val_pred -> test_pred （一条连续线）
      - phase_mask (pd.DataFrame): 索引为统一时间轴，标记 is_train/is_val/is_test
    """
    # 训练真值时间索引
    if time_col and hasattr(train_df_plot, "columns") and time_col in train_df_plot.columns:
        train_time = pd.to_datetime(train_df_plot[time_col], errors="coerce")
    else:
        idx_src = getattr(train_df_plot, "index", None)
        if idx_src is None or (hasattr(idx_src, "__len__") and len(idx_src) == 0):
            train_time = pd.date_range(
                start=pd.Timestamp.today().normalize(),
                periods=len(train_df_plot),
                freq="D"
            )
        else:
            train_time = pd.to_datetime(idx_src, errors="coerce")

    train_true = pd.Series(
        pd.to_numeric(train_df_plot.get("training_true", pd.Series([], dtype=float)), errors="coerce").to_numpy(),
        index=train_time, name="y_true"
    ).dropna()

    # 小工具：从 df 取列 -> Series（保证索引为时间）
    def _series(df, col):
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.Series(dtype=float)
        idx = pd.to_datetime(df.index, errors="coerce")
        if col not in df.columns:
            return pd.Series(dtype=float)
        val = pd.to_numeric(df[col], errors="coerce")
        return pd.Series(val.to_numpy(), index=idx, name=col).dropna()

    # 关键：这里要取统一列名 y_true/yhat（我们在 pipeline 里已强制保留）
    val_true  = _series(val_dense,  "y_true")
    val_pred  = _series(val_dense,  "yhat")
    test_true = _series(test_dense, "y_true")
    test_pred = _series(test_dense, "yhat")

    # 1) 连续真值
    full_truth = pd.concat([train_true, val_true, test_true]).sort_index()
    full_truth = full_truth[~full_truth.index.duplicated(keep="last")]

    # 2) 连续预测（在训练末尾放一个“衔接点”把预测线接上）
    if len(train_true):
        t_last = train_true.index.max()
        v_last = float(train_true.iloc[-1])
        splice = pd.Series([v_last], index=[t_last], name="yhat")
        full_pred_cont = pd.concat([splice, val_pred, test_pred]).sort_index()
    else:
        full_pred_cont = pd.concat([val_pred, test_pred]).sort_index()
    full_pred_cont = full_pred_cont[~full_pred_cont.index.duplicated(keep="last")]

    # 3) 阶段掩码
    timeline   = full_truth.index.union(full_pred_cont.index).unique().sort_values()
    phase_mask = pd.DataFrame(index=timeline, data={
        "is_train": False, "is_val": False, "is_test": False
    })
    t_train_end = train_true.index.max() if len(train_true) else None
    t_val_end   = val_true.index.max()   if len(val_true)   else t_train_end

    if t_train_end is not None:
        phase_mask.loc[phase_mask.index <= t_train_end, "is_train"] = True
    if t_train_end is not None and t_val_end is not None:
        phase_mask.loc[(phase_mask.index > t_train_end) & (phase_mask.index <= t_val_end), "is_val"] = True
    if t_val_end is not None:
        phase_mask.loc[phase_mask.index > t_val_end, "is_test"] = True

    try:
        print(f"[pipeline] full_truth len={len(full_truth)}, full_pred_cont len={len(full_pred_cont)}")
    except Exception:
        pass

    return full_truth, full_pred_cont, phase_mask


def set_seed(seed: int | None):
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_logging(cfg):
    import logging, os
    lvl = getattr(logging, str(cfg.get('logging',{}).get('level','DEBUG')).upper(), logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if lvl == logging.DEBUG else lvl)

    class _TestOnly(logging.Filter):
        def filter(self, record):
            return record.levelno >= logging.INFO or '[DBG|TEST]' in record.getMessage()
    if lvl == logging.DEBUG:
        console.addFilter(_TestOnly())

    logging.basicConfig(level=lvl, handlers=[console], force=True)
    logging.getLogger().setLevel(lvl)


from models.registry import TRAINER_REGISTRY

# 原有依赖（保留作回退）
from utils.loader import split_data
from preprocessing.cleaning import clean_data
from preprocessing.feature_engineering import generate_features, fit_and_transform_scaler, transform_with_scaler
from evaluation.metrics import compute_rmse, compute_mape
from training.train import run_training
from models.informer.predict import rolling_predict_segment


def run_train_predict_pipeline(config):
    """
    训练并产出 val/test 整段预测，同时把“连续绘图序列”塞进 payload 供 plot.py 连续分支直接使用。
    """
    import numpy as np
    import pandas as pd

    # ---------- 标准化/规范化工具 ----------
    def _normalize_dense(df_like, time_col: str) -> Optional[pd.DataFrame]:
        if df_like is None:
            return None
        try:
            if isinstance(df_like, pd.DataFrame):
                base = df_like.copy()
                cols_keep = [c for c in ["y_true", "yhat"] if c in base.columns]
                df = base[cols_keep].copy() if cols_keep else pd.DataFrame(index=base.index)
                if not isinstance(df.index, pd.DatetimeIndex):
                    ts_col = None
                    for cand in ["timestamp", time_col, "date", "time", "ds"]:
                        if cand in base.columns:
                            ts_col = cand; break
                    if ts_col is not None:
                        idx = pd.to_datetime(base[ts_col], errors="coerce")
                        df = df.set_index(idx)
                    else:
                        df.index = pd.to_datetime(base.index, errors="coerce")
                return df.sort_index()

            if isinstance(df_like, dict):
                ts = df_like.get("timestamps")
                if ts is None:
                    return None
                idx = pd.to_datetime(ts, errors="coerce")
                cols = {}
                if "y_true" in df_like: cols["y_true"] = df_like["y_true"]
                if "yhat"  in df_like: cols["yhat"]  = df_like["yhat"]
                return pd.DataFrame(cols, index=idx).sort_index()

            if isinstance(df_like, (list, tuple)) and len(df_like) > 0 and isinstance(df_like[0], dict):
                df = pd.DataFrame(df_like)
                return _normalize_dense(df, time_col)
        except Exception as e:
            print(f"[pipeline] _normalize_dense failed: {e}")
        return None

    def _standardize_dense_df(df: Optional[pd.DataFrame], time_col: str) -> Optional[pd.DataFrame]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None
        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            if time_col in out.columns:
                out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
                out = out.set_index(time_col)
            else:
                try:
                    out.index = pd.to_datetime(out.index, errors="coerce")
                except Exception:
                    pass
        out.index.name = time_col
        if "y_true" not in out.columns: out["y_true"] = np.nan
        if "yhat"  not in out.columns: out["yhat"]  = np.nan
        out = out[["y_true", "yhat"]].copy()
        out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce").astype("float32")
        out["yhat"]  = pd.to_numeric(out["yhat"],  errors="coerce").astype("float32")
        return out.sort_index()

    def _attach_ts_and_rename(df_dense: Optional[pd.DataFrame], ts_list, which: str, time_col: str) -> Optional[pd.DataFrame]:
        """
        关键修正：改名为 validation_* / test_* 后，**仍然保留** y_true/yhat 两列，连续拼接就不会丢列。
        """
        if not isinstance(df_dense, pd.DataFrame) or df_dense.empty:
            return None
        out = df_dense.copy()

        # 设置索引
        if ts_list is not None:
            idx = pd.to_datetime(ts_list, errors="coerce")
            if isinstance(idx, pd.Series): idx = idx.values
            try:
                out.index = pd.DatetimeIndex(idx, name=time_col)
            except Exception:
                pass
        if not isinstance(out.index, pd.DatetimeIndex) or out.index.isna().all():
            out.index = pd.date_range(start=pd.Timestamp.today().normalize(), periods=len(out), freq="D", name=time_col)

        # 改名（同时保留 y_true/yhat）
        if which == "val":
            out["validation_true"]    = out["y_true"]
            out["validation_predict"] = out["yhat"]
        else:
            out["test_true"]    = out["y_true"]
            out["test_predict"] = out["yhat"]

        # 列顺序：特定列 + 兼容列
        prefer = (["validation_true","validation_predict"] if which=="val" else ["test_true","test_predict"])
        cols = [c for c in prefer + ["y_true","yhat"] if c in out.columns]
        return out[cols]

    def _inverse_series_1d_from_df_scaled(df_sc: pd.DataFrame, scaler, cfg: dict, value_col: str) -> pd.Series:
        arr2d = df_sc[[value_col]].to_numpy().astype(np.float32)
        try:
            artifacts = (cfg.get('artifacts') or {})
            y_scaler_path = artifacts.get('y_scaler_path')
            if y_scaler_path and os.path.exists(y_scaler_path):
                import joblib
                y_scaler = joblib.load(y_scaler_path)
                inv = y_scaler.inverse_transform(arr2d)
                return pd.Series(inv.reshape(-1), index=df_sc.index)
        except Exception as e:
            print(f"[pipeline] y_scaler inverse failed: {e}")
        n_in = getattr(scaler, 'n_features_in_', None)
        if n_in is None or not hasattr(scaler, "inverse_transform"):
            return pd.Series(arr2d.reshape(-1), index=df_sc.index)
        if arr2d.shape[1] == n_in:
            inv = scaler.inverse_transform(arr2d)
            return pd.Series(inv.reshape(-1), index=df_sc.index)
        all_cols = (
            (cfg.get('artifacts', {}) or {}).get('feature_cols') or
            (cfg.get('data', {}) or {}).get('all_feature_cols') or
            [value_col]
        )
        tmp = np.zeros((arr2d.shape[0], n_in), dtype=np.float32)
        try:
            idx = all_cols.index(value_col)
        except ValueError:
            idx = 0
        tmp[:, idx] = arr2d[:, 0]
        try:
            inv_wide = scaler.inverse_transform(tmp)
            out = inv_wide[:, idx]
        except Exception:
            out = arr2d[:, 0]
        return pd.Series(out, index=df_sc.index)

    # ---------- 配置取值 ----------
    model_key = str((config.get("model") or {}).get("name", "") or config.get("model_type", "")).strip().lower()
    data_blk   = config.setdefault('data', {})
    artifacts  = config.setdefault('artifacts', {})
    default_cf = config.get('default', {}) or {}

    time_col   = default_cf.get('time_col', 'date')
    value_col  = default_cf.get('value_col', 'value')
    scaler     = artifacts.get('scaler')

    # ===========================================================
    # 0) 优先通过 TRAINER_REGISTRY（例如 arima）
    # ===========================================================
    if model_key in TRAINER_REGISTRY:
        runner = TRAINER_REGISTRY[model_key]
        _df_candidates = [
            config.get('dataframe'),
            data_blk.get('dataframe'),
            data_blk.get('df'),
            data_blk.get('data'),
        ]
        _df_input = next((x for x in _df_candidates if isinstance(x, pd.DataFrame)), pd.DataFrame())

        val_true, val_pred, test_true, test_pred, final_model, test_df, params = runner(_df_input, config)
        artifacts[f"{model_key}_params"] = params
        # Ensure RF best params are exposed under a stable key for the app panel
        if model_key == "randomforest":
            artifacts["randomforest_params"] = params
        if isinstance(test_df, pd.DataFrame) and not test_df.empty:
            data_blk["test_forecast_df"] = test_df

        # 反推时间戳（若上游没给）
        try:
            _val_len = int(len(np.asarray(val_true).ravel()))
            _test_len = int(len(np.asarray(test_true).ravel()))
            _ts_series = None
            if isinstance(_df_input, pd.DataFrame) and time_col in _df_input.columns:
                _ts_series = pd.to_datetime(_df_input[time_col], errors="coerce")
            if _ts_series is not None and (_val_len + _test_len) > 0:
                _n_total = int(len(_ts_series))
                _n_train = max(0, _n_total - _val_len - _test_len)
                if data_blk.get("val_timestamps") is None and _val_len > 0:
                    data_blk["val_timestamps"] = _ts_series.iloc[_n_train : _n_train + _val_len].tolist()
                if data_blk.get("test_timestamps") is None and _test_len > 0:
                    data_blk["test_timestamps"] = _ts_series.iloc[_n_train + _val_len : _n_train + _val_len + _test_len].tolist()
        except Exception as _e:
            print(f"[pipeline] warn: failed to infer timestamps: {_e}")

        # 组装 dense DataFrame
        def _mk_dense(true_arr, pred_arr):
            true_arr = np.asarray(true_arr, dtype=float).ravel()
            pred_arr = np.asarray(pred_arr, dtype=float).ravel()
            L = min(len(true_arr), len(pred_arr))
            if L <= 0: return None
            return pd.DataFrame({"y_true": true_arr[:L], "yhat": pred_arr[:L]})

        val_dense = _mk_dense(val_true, val_pred)
        test_dense = _mk_dense(test_true, test_pred)
        data_blk["val_dense"] = val_dense
        data_blk["test_dense"] = test_dense

        # --- Optional: residual modeling hook (registry route) ---
        try:
            rm_cfg = (config.get("residual_modeling") or {})
            already_applied = bool(data_blk.get("residual_applied"))
            rm_enabled = bool(rm_cfg.get("enabled", False)) and not already_applied
            if rm_enabled and isinstance(val_dense, pd.DataFrame) and not val_dense.empty:
                # Choose model class
                model_type = str(rm_cfg.get("model_type", "LinearRegression")).strip().lower()
                try:
                    from sklearn.linear_model import LinearRegression, Ridge, Lasso
                    Model = LinearRegression
                    if model_type == "ridge":
                        Model = Ridge
                    elif model_type == "lasso":
                        Model = Lasso
                except Exception as _imp_e:
                    print(f"[pipeline][registry-route] residual model import failed: {_imp_e}")
                    Model = None
                if Model is not None:
                    # simple residual learn: residual = y_true - yhat, features = yhat
                    # —— 关键修复：用纯 NumPy，显式 dtype/shape，避免 Pandas ExtensionArray 参与 fit/predict ——
                    X_val_res = np.asarray(val_dense[["yhat"]].to_numpy(dtype=np.float64), dtype=np.float64)
                    y_val_res = (
                        np.asarray(val_dense["y_true"].to_numpy(dtype=np.float64), dtype=np.float64)
                        - np.asarray(val_dense["yhat"].to_numpy(dtype=np.float64), dtype=np.float64)
                    ).reshape(-1)

                    try:
                        res_mdl = Model()
                        res_mdl.fit(X_val_res, y_val_res)

                        # adjust validation predictions（复制再写，避免链式赋值/扩展数组问题）
                        val_dense = val_dense.copy()
                        _val_yhat = np.asarray(val_dense["yhat"].to_numpy(dtype=np.float64), dtype=np.float64)
                        val_dense["yhat"] = _val_yhat + res_mdl.predict(X_val_res)

                        # adjust test predictions if available
                        if isinstance(test_dense, pd.DataFrame) and not test_dense.empty:
                            X_test_res = np.asarray(test_dense[["yhat"]].to_numpy(dtype=np.float64), dtype=np.float64)
                            test_dense = test_dense.copy()
                            _test_yhat = np.asarray(test_dense["yhat"].to_numpy(dtype=np.float64), dtype=np.float64)
                            test_dense["yhat"] = _test_yhat + res_mdl.predict(X_test_res)

                        # store to artifacts for external reuse
                        artifacts["residual_model"] = res_mdl
                        artifacts["residual_model_type"] = model_type
                        print("[pipeline][registry-route] residual modeling applied.")
                    except Exception as _fit_e:
                        print(f"[pipeline][registry-route] residual modeling skipped (fit failed): {_fit_e}")
        except Exception as _e:
            print(f"[pipeline][registry-route] residual modeling skipped: {_e}")

        # 标准化 + 改名（保留 y_true/yhat）
        val_ts  = data_blk.get("val_timestamps")
        test_ts = data_blk.get("test_timestamps")
        val_dense_std  = _standardize_dense_df(_normalize_dense(val_dense,  time_col), time_col)
        test_dense_std = _standardize_dense_df(_normalize_dense(test_dense, time_col), time_col)
        val_dense_std  = _attach_ts_and_rename(val_dense_std,  val_ts,  "val",  time_col)
        test_dense_std = _attach_ts_and_rename(test_dense_std, test_ts, "test", time_col)

        data_blk["val_dense"]  = val_dense_std
        data_blk["test_dense"] = test_dense_std

        # 计算指标
        def _compute_metrics_from_dense(df_dense: Optional[pd.DataFrame]) -> Optional[dict]:
            if not isinstance(df_dense, pd.DataFrame) or df_dense.empty: return None
            if not all(c in df_dense.columns for c in ["y_true", "yhat"]): return None
            dfm = df_dense[["y_true", "yhat"]].dropna()
            if dfm.empty: return None
            try:
                y_arr    = np.asarray(dfm["y_true"].values, dtype="float64")
                yhat_arr = np.asarray(dfm["yhat"].values, dtype="float64")
                diff     = np.subtract(yhat_arr, y_arr, dtype=float)
                rmse_val = float(np.sqrt(np.mean(diff * diff)))
            except Exception:
                diff = np.subtract(np.asarray(dfm["yhat"].values, dtype="float64"), np.asarray(dfm["y_true"].values, dtype="float64"))
                rmse_val = float(np.sqrt(np.mean(diff * diff)))
            try:
                mape_val = float(compute_mape(dfm["y_true"].values, dfm["yhat"].values))
            except Exception:
                y    = np.asarray(dfm["y_true"].values, dtype="float64")
                yhat = np.asarray(dfm["yhat"].values, dtype="float64")
                denom = np.where(np.abs(y) > 1e-12, np.abs(y), np.nan)
                mape_val = float(np.nanmean(np.abs(yhat - y) / denom))
            return {"rmse": rmse_val, "mape": mape_val}

        val_metrics  = _compute_metrics_from_dense(val_dense_std)
        test_metrics = _compute_metrics_from_dense(test_dense_std)
        metrics_blk = config.setdefault("metrics", {})
        if isinstance(val_metrics, dict):
            metrics_blk["val_rmse"] = val_metrics.get("rmse"); metrics_blk["val_mape"] = val_metrics.get("mape")
        if isinstance(test_metrics, dict):
            metrics_blk["test_rmse"] = test_metrics.get("rmse"); metrics_blk["test_mape"] = test_metrics.get("mape")
        data_blk["val_metrics"]  = val_metrics
        data_blk["test_metrics"] = test_metrics

        # 反归一化训练真值
        train_true = None
        try:
            train_df_sc = data_blk.get('train_df_sc')
            if isinstance(train_df_sc, pd.DataFrame) and len(train_df_sc) > 0 and scaler is not None:
                train_true = _inverse_series_1d_from_df_scaled(train_df_sc, scaler, config, value_col)
        except Exception:
            pass
        train_df_plot = train_true.to_frame("training_true") if isinstance(train_true, pd.Series) else pd.DataFrame(columns=["training_true"])

        # --- 构建连续序列 + 调试打印 ---
        try:
            full_truth, full_pred_cont, phase_mask = build_continuous_series(
                train_df_plot, val_dense_std, test_dense_std, time_col=time_col
            )
            data_blk["full_truth"]     = full_truth
            data_blk["full_pred_cont"] = full_pred_cont
            data_blk["phase_mask"]     = phase_mask
            print(f"[pipeline] continuous ready: truth={isinstance(full_truth, pd.Series)}, "
                  f"pred={isinstance(full_pred_cont, pd.Series)}")
        except Exception as _e:
            print(f"[pipeline][continuous-series] skipped (registry route): {_e}")
            full_truth = None; full_pred_cont = None; phase_mask = None

        # --- 调 plot ---
        try:
            from visualizations.plot import plot_results
            split_info = (data_blk.get('split') or {})
            train_len = split_info.get('train_len'); val_len = split_info.get('val_len'); test_len = split_info.get('test_len')

            payload = {
                "val_dense": val_dense_std,
                "test_dense": test_dense_std,
                "val_long": None, "test_long": None,
                "split": {"train_len": train_len, "val_len": val_len, "test_len": test_len},
                # 关键：传给连续分支
                "full_truth": full_truth,
                "full_pred_cont": full_pred_cont,
                "phase_mask": phase_mask,
            }

            # 调试：连续分支入参检查
            try:
                print(f"[pipeline] payload check -> truth:{type(payload['full_truth'])}, "
                      f"pred:{type(payload['full_pred_cont'])}, "
                      f"lens: {len(payload['full_truth']) if isinstance(payload['full_truth'], pd.Series) else 'NA'} / "
                      f"{len(payload['full_pred_cont']) if isinstance(payload['full_pred_cont'], pd.Series) else 'NA'}")
            except Exception:
                pass

            plot_results(
                train_df=train_df_plot,
                val_df_aligned=val_dense_std if isinstance(val_dense_std, pd.DataFrame) else None,
                test_df_aligned=test_dense_std if isinstance(test_dense_std, pd.DataFrame) else None,
                time_col=time_col,
                value_col=value_col,
                title=f"Training / Validation / Test - Full Span (Dense 1-step) [{model_key}]",
                payload=payload,
                val_long=None, test_long=None,
                train_len=int(train_len) if train_len is not None else (len(train_true) if isinstance(train_true, pd.Series) else None),
                val_len=int(val_len) if val_len is not None else None,
                test_len=int(test_len) if test_len is not None else None,
            )
        except Exception as e:
            print(f"[pipeline] Info: plot skipped or failed: {e}")

        # 返回首选 result_df（优先 val）
        result_df = _standardize_dense_df(_normalize_dense(val_dense, time_col), time_col)
        if result_df is None:
            result_df = _standardize_dense_df(_normalize_dense(test_dense, time_col), time_col)
        return final_model, (result_df if isinstance(result_df, pd.DataFrame) else pd.DataFrame())

    from models.informer.train import train_informer_model
    model, result_df = train_informer_model(config)

    data_blk   = config.setdefault('data', {})
    artifacts  = config.setdefault('artifacts', {})
    default_cf = config.get('default', {}) or {}

    time_col   = default_cf.get('time_col', 'date')
    value_col  = default_cf.get('value_col', 'value')
    scaler     = artifacts.get('scaler')

    val_dense = data_blk.get('val_dense')
    test_dense = data_blk.get('test_dense')

    val_dense  = _standardize_dense_df(_normalize_dense(val_dense, time_col), time_col)
    test_dense = _standardize_dense_df(_normalize_dense(test_dense, time_col), time_col)

    split_info = (data_blk.get('split') or {})
    train_len = split_info.get('train_len'); val_len = split_info.get('val_len'); test_len = split_info.get('test_len')

    # 维持原指标计算
    def _compute_metrics_from_dense(df_dense: Optional[pd.DataFrame]) -> Optional[dict]:
        if not isinstance(df_dense, pd.DataFrame) or df_dense.empty: return None
        if not all(c in df_dense.columns for c in ["y_true", "yhat"]): return None
        dfm = df_dense[["y_true", "yhat"]].dropna()
        if dfm.empty: return None
        try:
            y_arr    = np.asarray(dfm["y_true"].values, dtype="float64")
            yhat_arr = np.asarray(dfm["yhat"].values, dtype="float64")
            diff     = np.subtract(yhat_arr, y_arr, dtype=float)
            rmse_val = float(np.sqrt(np.mean(diff * diff)))
        except Exception:
            diff = np.subtract(np.asarray(dfm["yhat"].values, dtype="float64"), np.asarray(dfm["y_true"].values, dtype="float64"))
            rmse_val = float(np.sqrt(np.mean(diff * diff)))
        try:
            mape_val = float(compute_mape(dfm["y_true"].values, dfm["yhat"].values))
        except Exception:
            y    = np.asarray(dfm["y_true"].values, dtype="float64")
            yhat = np.asarray(dfm["yhat"].values, dtype="float64")
            denom = np.where(np.abs(y) > 1e-12, np.abs(y), np.nan)
            mape_val = float(np.nanmean(np.abs(yhat - y) / denom))
        return {"rmse": rmse_val, "mape": mape_val}

    val_metrics  = _compute_metrics_from_dense(val_dense)
    test_metrics = _compute_metrics_from_dense(test_dense)
    metrics_blk = config.setdefault("metrics", {})
    if isinstance(val_metrics, dict):
        metrics_blk["val_rmse"] = val_metrics.get("rmse"); metrics_blk["val_mape"] = val_metrics.get("mape")
    if isinstance(test_metrics, dict):
        metrics_blk["test_rmse"] = test_metrics.get("rmse"); metrics_blk["test_mape"] = test_metrics.get("mape")
    data_blk["val_metrics"]  = val_metrics
    data_blk["test_metrics"] = test_metrics

    try:
        print(f"[pipeline] metrics -> val: {val_metrics} | test: {test_metrics}")
    except Exception:
        pass

    # 训练真值
    train_true = None
    try:
        train_df_sc = data_blk.get('train_df_sc')
        if isinstance(train_df_sc, pd.DataFrame) and len(train_df_sc) > 0 and scaler is not None:
            train_true = _inverse_series_1d_from_df_scaled(train_df_sc, scaler, config, value_col)
    except Exception as e:
        print(f"[pipeline] Warning: failed to build training_true series: {e}")

    # 构造连续序列（回退分支）
    try:
        from visualizations.plot import plot_results
        train_df_plot = train_true.to_frame("training_true") if isinstance(train_true, pd.Series) else pd.DataFrame(columns=["training_true"])
        val_dense2  = None if (isinstance(val_dense, pd.DataFrame) and val_dense.empty) else val_dense
        test_dense2 = None if (isinstance(test_dense, pd.DataFrame) and test_dense.empty) else test_dense

        try:
            full_truth, full_pred_cont, phase_mask = build_continuous_series(
                train_df_plot, val_dense2, test_dense2, time_col=time_col
            )
            data_blk["full_truth"]     = full_truth
            data_blk["full_pred_cont"] = full_pred_cont
            data_blk["phase_mask"]     = phase_mask
            print(f"[pipeline] continuous ready (fallback): truth={isinstance(full_truth, pd.Series)}, "
                  f"pred={isinstance(full_pred_cont, pd.Series)}")
        except Exception as _e:
            print(f"[pipeline][continuous-series] skipped (fallback route): {_e}")
            full_truth = None; full_pred_cont = None; phase_mask = None

        payload = {
            "val_dense": val_dense2, "test_dense": test_dense2,
            "val_long": None, "test_long": None,
            "split": {"train_len": train_len, "val_len": val_len, "test_len": test_len},
            "full_truth": full_truth, "full_pred_cont": full_pred_cont, "phase_mask": phase_mask,
        }

        try:
            print(f"[pipeline] payload check (fallback) -> truth:{type(payload['full_truth'])}, "
                  f"pred:{type(payload['full_pred_cont'])}, "
                  f"lens: {len(payload['full_truth']) if isinstance(payload['full_truth'], pd.Series) else 'NA'} / "
                  f"{len(payload['full_pred_cont']) if isinstance(payload['full_pred_cont'], pd.Series) else 'NA'}")
        except Exception:
            pass

        plot_results(
            train_df=train_df_plot,
            val_df_aligned=val_dense2 if isinstance(val_dense2, pd.DataFrame) else None,
            test_df_aligned=test_dense2 if isinstance(test_dense2, pd.DataFrame) else None,
            time_col=time_col, value_col=value_col,
            title="Training / Validation / Test - Full Span (Dense 1-step)",
            payload=payload,
            val_long=None, test_long=None,
            train_len=int(train_len) if train_len is not None else (len(train_true) if isinstance(train_true, pd.Series) else None),
            val_len=int(val_len) if val_len is not None else None,
            test_len=int(test_len) if test_len is not None else None,
        )
    except Exception as e:
        print(f"[pipeline] Info: plot skipped or failed: {e}")

    return model, result_df
