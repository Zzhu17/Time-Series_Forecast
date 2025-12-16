import sys
import importlib
import pandas as pd
import numpy as np
from typing import Optional, Dict
import streamlit as st

# ---- Module-level constants ----
PLOT_MOD = "visualizations.plot"
MIME_CSV = "text/csv"

from services.pipeline import run_train_predict_pipeline
from types import SimpleNamespace
from models.informer.predict import InformerPredictor

# --------- always load latest plot module (Patch C) ---------
def load_plot_module():
    """
    Always return a fresh visualizations.plot module.
    Streamlit reruns can keep module cache; we reload explicitly.
    """
    if PLOT_MOD in sys.modules:
        importlib.reload(sys.modules[PLOT_MOD])
        return sys.modules[PLOT_MOD]
    else:
        mod = importlib.import_module(PLOT_MOD)
        return mod

# ---- safe helpers ----
def _as_int(x, default: Optional[int] = None) -> Optional[int]:
    """Best-effort convert to int; return default on failure."""
    try:
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, str):
            xs = x.strip()
            if xs.isdigit() or (xs.startswith('-') and xs[1:].isdigit()):
                return int(xs)
            return int(float(xs))
        if x is not None:
            return int(x)
    except Exception:
        pass
    return default

# ---- build DataFrame from long JSON safely ----
def _df_from_long(long_obj, time_col: str, value_name_true: str = 'y_true', value_name_pred: str = 'yhat') -> pd.DataFrame:
    try:
        if not isinstance(long_obj, dict):
            return pd.DataFrame(columns=[time_col, value_name_true, value_name_pred])
        ts  = list(long_obj.get('timestamps') or [])
        y_t = list(long_obj.get('y_true') or [])
        y_h = list(long_obj.get('yhat') or [])
        # align lengths
        n = min(len(ts), len(y_t), len(y_h))
        if n == 0:
            return pd.DataFrame(columns=[time_col, value_name_true, value_name_pred])
        df = pd.DataFrame({
            time_col: ts[:n],
            value_name_true: pd.to_numeric(y_t[:n], errors='coerce'),
            value_name_pred: pd.to_numeric(y_h[:n], errors='coerce'),
        })
        return df
    except Exception:
        return pd.DataFrame(columns=[time_col, value_name_true, value_name_pred])

# 新增：将 dense DataFrame 转为标准明细 DataFrame 用于展示/导出
def _df_from_dense_for_display(df_dense: Optional[pd.DataFrame], time_col: str) -> pd.DataFrame:
    if not isinstance(df_dense, pd.DataFrame) or df_dense.empty:
        return pd.DataFrame(columns=[time_col, "y_true", "yhat"])
    out = df_dense.copy()
    # 如果是 DatetimeIndex，则 reset_index 成时间列
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={out.index.name or "index": time_col})
    # 只保留标准列
    keep = [c for c in [time_col, "y_true", "yhat"] if c in out.columns]
    return out[keep]


# ==========================
# ======= Streamlit UI =====
# ==========================

st.set_page_config(page_title="Universal TS Forecast", layout="wide")
st.title("🧠 通用时间序列预测平台（中控台简约版）")

# 主区域：上传文件 + 选择模型 + 运行
uploaded = st.file_uploader("上传 CSV 文件", type=["csv"])
model_name = st.selectbox("选择模型", ["informer", "arima", "prophet", "randomforest", "lstm"], index=0)
if model_name == "randomforest":
    st.caption("⚙️ RandomForest 将强制进行 Optuna 调参（使用验证集，n_trials 由 configs.yaml 的 optimization.n_trials 控制，默认 50）。")
if model_name == "lstm":
    st.caption("🧩 LSTM 将使用 configs.yaml 的 model_config.LSTM 超参（seq_len/hidden_dim/num_layers/n_epochs/learning_rate），并通过适配层生成整段 val/test 预测。")
run_click = st.button("开始训练并预测", type="primary")

# 在线滚动推理参数（不重新训练）
col_r1, col_r2, col_r3 = st.columns([1,1,2])
with col_r1:
    horizon_days = st.selectbox("在线预测地平线（天）", [1, 3, 7], index=0)
with col_r2:
    step_mode = st.selectbox("滚动步幅", ["块推进(=地平线)", "逐步推进(=1)"], index=0)
with col_r3:
    st.caption("块推进速度快、误差不累积；逐步更平滑但更慢且误差递推。")

online_click = st.button("仅预测（在线滚动推理）", type="secondary")

if uploaded is None:
    st.info("请先上传 CSV 文件。")
else:
    # 读取数据并做基本校验
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"读取 CSV 失败：{e}")
        st.stop()

    # 简单的列名推断（若不存在则给出提示）
    time_col = 'date' if 'date' in df.columns else df.columns[0]
    if 'value' in df.columns:
        value_col = 'value'
    elif len(df.columns) > 1:
        value_col = df.columns[1]
    else:
        value_col = df.columns[0]

    # === 自动推断 feature_cols（单/多变量自适配）===
    numeric_cols = [c for c in df.select_dtypes(include='number').columns if c != time_col]
    feature_cols = [value_col] + [c for c in numeric_cols if c != value_col]

    missing_cols = [c for c in (time_col, value_col) if c not in df.columns]
    if missing_cols:
        st.error(f"CSV 中缺少必要列：{missing_cols}")
        st.stop()

    st.subheader("📄 数据概览")
    st.caption(f"时间列: {time_col} | 目标列: {value_col}")
    st.dataframe(df.head(10), use_container_width=True)

    # ==========================
    # 在线滚动推理（不训练，直接加载权重并按整段滚动预测）
    # ==========================
    if online_click:
        config_pred = {
            "default": {
                "time_col": time_col,
                "value_col": value_col,
                "device": "cpu",
                "dtype": "float32",
            },
            "model_config": {
                "Informer": {
                    "seq_len": 96,
                    "label_len": 48,
                    "pred_len": 24,  # 实际滚动地平线由 horizon 覆盖
                    "feature_cols": feature_cols,
                }
            },
            "artifacts": {
                "model_path": "artifacts/informer_model.pth",
                "scaler_path": "artifacts/scaler.pkl",
                "residual_model_path": "artifacts/residual_model.pkl",
            },
            "prediction": {
                "rolling": {
                    "enabled": True,
                    "step": None,
                    "mode": "overwrite",
                }
            }
        }

        st.caption(f"使用特征列（按训练/预测固定顺序）：{feature_cols}")

        # 地平线与步幅
        horizon_steps = int(24 * horizon_days)  # 若是日频自行调整倍数
        step_val = None if step_mode.startswith("块") else 1

        try:
            predictor = InformerPredictor(config_pred)
        except Exception as e:
            st.error("加载已训练模型失败（请先完成一次训练或检查 artifacts 路径）")
            st.exception(e)
            st.stop()

        with st.spinner("在线滚动推理中..."):
            try:
                merged = predictor.rolling_predict(df.copy(), horizon=horizon_steps, step=step_val, mode="overwrite")
            except Exception as e:
                st.error("在线滚动推理失败")
                st.exception(e)
                st.stop()

        # 计算与真值重叠区间的指标
        merged = np.asarray(merged).reshape(-1)
        mask = ~np.isnan(merged)
        if mask.sum() == 0:
            st.warning("没有得到有效的预测区间（数据太短或参数不匹配）")
        else:
            y_true = pd.to_numeric(df.loc[mask, value_col], errors='coerce').to_numpy()
            y_hat = merged[mask]
            rmse = float(np.sqrt(np.nanmean((y_hat - y_true) ** 2)))
            denom = np.where(y_true == 0, np.nan, np.abs(y_true))
            mape = float(np.nanmean(np.abs((y_hat - y_true) / denom)) * 100.0)

            st.subheader("⚡ 在线滚动推理 — 指标")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Online RMSE", f"{rmse:.4f}")
            with c2:
                st.metric("Online MAPE", f"{mape:.2f}%")

            # 在线长序列
            online_long = {
                "timestamps": pd.to_datetime(df[time_col]).astype(str).tolist(),
                "y_true": pd.to_numeric(df[value_col], errors='coerce').astype(float).tolist(),
                "yhat": merged.astype(float).tolist(),
            }

            st.subheader("📈 在线滚动推理 — 长序列曲线")
            vplot = load_plot_module()
            fig_online = vplot.plot_results(
                train_df=df[[time_col, value_col]] if time_col in df.columns and value_col in df.columns else pd.DataFrame(columns=[value_col]),
                val_df_aligned=None,
                test_df_aligned=None,
                time_col=time_col,
                value_col=value_col,
                title=f"Online Rolling Inference (H={horizon_steps}, step={'H' if step_val is None else step_val})",
                payload=None,
                val_long=online_long,
                test_long=None,
                train_len=None,
                val_len=None,
                test_len=None,
            )
            st.pyplot(fig_online)

            with st.expander("🔎 在线滚动推理明细（最近 200 条）", expanded=False):
                view_df = pd.DataFrame({
                    time_col: online_long["timestamps"],
                    "y_true": online_long["y_true"],
                    "yhat": online_long["yhat"],
                }).tail(200)
                st.dataframe(view_df, use_container_width=True)

            with st.expander("🧾 在线滚动推理明细（整段）", expanded=False):
                full_df = _df_from_long(online_long, time_col)
                st.dataframe(full_df, use_container_width=True)
                try:
                    st.download_button(
                        label="下载在线整段明细 CSV",
                        data=full_df.to_csv(index=False).encode('utf-8'),
                        file_name="online_long.csv",
                        mime=MIME_CSV,
                    )
                except Exception:
                    pass
    # ==========================
    # 训练 + 预测 + 统一绘图（6/2/2 + 长序列）
    # ==========================
    if run_click:
        # 训练 + 预测 + 统一绘图（6/2/2 + 长序列）
        config = {
            "model": {"name": model_name},
            "default": {
                "time_col": time_col,
                "value_col": value_col,
            },
            "model_config": {
                "Informer": {
                    "seq_len": 96,
                    "label_len": 48,
                    "pred_len": 24,
                    "feature_cols": feature_cols,
                }
            },
            "artifacts": {
                "model_path": "artifacts/informer_model.pth",
                "scaler_path": "artifacts/scaler.pkl",
                "residual_model_path": "artifacts/residual_model.pkl",
                "y_scaler_path": "artifacts/value_scaler.pkl",
            }
        }
        # 保证新旧配置键都能被 pipeline 识别（放在定义 config 之后）
        config.setdefault("model", {})["name"] = model_name
        config["model_type"] = model_name

        # 为注册表模型（如 arima）提供原始 DataFrame（pipeline 会优先读取这里）
        config.setdefault("data", {})
        config["data"]["dataframe"] = df.copy()

        def _prepare_data_into_config(src_df, cfg, feature_cols):
            """Prepare cfg['data'] for single-arg pipeline: 6:2:2 split + scaler fit/transform."""
            import numpy as _np
            import pandas as _pd

            # Robust StandardScaler import with a minimal fallback implementation
            try:
                from sklearn.preprocessing import StandardScaler as _RealStandardScaler
                SSCls = _RealStandardScaler
            except Exception:
                class _MiniStandardScaler:
                    def fit(self, X):
                        X = _np.asarray(X, dtype=_np.float32)
                        self.mean_ = X.mean(axis=0)
                        self.scale_ = X.std(axis=0)
                        self.scale_[self.scale_ == 0] = 1.0
                        self.n_features_in_ = X.shape[1]
                        return self
                    def transform(self, X):
                        X = _np.asarray(X, dtype=_np.float32)
                        return (X - self.mean_) / self.scale_
                    def inverse_transform(self, X):
                        X = _np.asarray(X, dtype=_np.float32)
                        return X * self.scale_ + self.mean_
                SSCls = _MiniStandardScaler

            time_col = cfg.get('default', {}).get('time_col', 'date')
            value_col = cfg.get('default', {}).get('value_col', 'value')

            df2 = src_df.copy()
            # ensure sorting by time if time_col exists
            if time_col in df2.columns:
                try:
                    df2[time_col] = _pd.to_datetime(df2[time_col])
                    df2 = df2.sort_values(time_col)
                except Exception:
                    pass

            n = len(df2)
            t = int(n * 0.6); v = int(n * 0.2); te = n - t - v
            train_df = df2.iloc[:t].copy()
            val_df   = df2.iloc[t:t+v].copy()
            test_df  = df2.iloc[t+v:].copy()

            # guard: keep only existing feature columns
            feat_cols = [c for c in list(feature_cols) if c in df2.columns]

            # fit scaler on train
            scaler = SSCls()
            if len(feat_cols) == 0:
                raise ValueError("No valid feature columns found for scaling.")
            scaler.fit(train_df[feat_cols].astype('float32'))

            # transform helper
            def _tf(d):
                out = d.copy()
                out[feat_cols] = scaler.transform(d[feat_cols].astype('float32'))
                return out

            cfg.setdefault('data', {})
            cfg['data']['train_df_sc'] = _tf(train_df)
            cfg['data']['val_df_sc']   = _tf(val_df)
            cfg['data']['test_df_sc']  = _tf(test_df)
            cfg['data']['split'] = {"train_len": t, "val_len": v, "test_len": te}
            cfg['data']['all_feature_cols'] = list(feat_cols)
            cfg.setdefault('artifacts', {})['scaler'] = scaler

        def _normalize_results(res, cfg, src_df):
            """统一把 pipeline 返回值规整为 {'status','metrics','data','artifacts'} 结构。支持:
            - (model, result_df)
            - {'status': 'ok', 'metrics': ..., 'data': {...}}
            并从 cfg['data'] 中提取 val_long/test_long/split 信息。
            """
            out = {"status": "ok", "metrics": {}, "data": {}, "artifacts": cfg.get("artifacts", {})}
            data_blk = cfg.get("data", {}) or {}

            def _extract_metrics_from_cfg(_d: dict) -> dict:
                if not isinstance(_d, dict):
                    return {}
                out_m = {}
                # validation candidates
                for k in ("metrics_val", "val_metrics", "validation_metrics", "metrics_validation"):
                    vm = _d.get(k)
                    if isinstance(vm, dict) and vm:
                        out_m["validation"] = vm
                        break
                # test candidates
                for k in ("metrics_test", "test_metrics", "testing_metrics", "metrics_testing"):
                    tm = _d.get(k)
                    if isinstance(tm, dict) and tm:
                        out_m["test"] = tm
                        break
                return out_m

            def _extract_metrics_from_root(_cfg: dict) -> dict:
                """
                支持从 cfg['metrics'] 读取扁平键：val_rmse/val_mape/test_rmse/test_mape
                并映射为 {'validation': {'rmse','mape'}, 'test': {'rmse','mape'}}
                """
                if not isinstance(_cfg, dict):
                    return {}
                root_m = _cfg.get("metrics") or {}
                if not isinstance(root_m, dict):
                    return {}
                out_m = {}
                val_m = {}
                test_m = {}
                # validation
                if "val_rmse" in root_m: val_m["rmse"] = root_m.get("val_rmse")
                if "val_mape" in root_m: val_m["mape"] = root_m.get("val_mape")
                if "val_mape_safe" in root_m: val_m["mape_safe"] = root_m.get("val_mape_safe")
                if val_m:
                    out_m["validation"] = val_m
                # test
                if "test_rmse" in root_m: test_m["rmse"] = root_m.get("test_rmse")
                if "test_mape" in root_m: test_m["mape"] = root_m.get("test_mape")
                if "test_mape_safe" in root_m: test_m["mape_safe"] = root_m.get("test_mape_safe")
                if test_m:
                    out_m["test"] = test_m
                return out_m

            # 1) 标准 dict 返回
            if isinstance(res, dict):
                out.update(res)
                out.setdefault("data", {})

                # 透传长载荷（兼容旧逻辑）
                if "val_long" not in out["data"] and "val_long" in data_blk:
                    out["data"]["val_long"] = data_blk.get("val_long")
                if "test_long" not in out["data"] and "test_long" in data_blk:
                    out["data"]["test_long"] = data_blk.get("test_long")

                # ✅ 新增：透传 dense（对齐好的整段 DataFrame）
                if "val_dense" not in out["data"] and "val_dense" in data_blk:
                    out["data"]["val_dense"] = data_blk.get("val_dense")
                if "test_dense" not in out["data"] and "test_dense" in data_blk:
                    out["data"]["test_dense"] = data_blk.get("test_dense")

                # split 信息兜底
                if "split" not in out["data"]:
                    n = len(src_df)
                    t = int(n * 0.6)
                    v = int(n * 0.2)
                    out["data"]["split"] = {"train_len": t, "val_len": v, "test_len": n - t - v}

                # backfill metrics from cfg['data'] if missing/partial
                cfg_metrics = _extract_metrics_from_cfg(data_blk)
                if cfg_metrics:
                    out.setdefault("metrics", {})
                    # don't overwrite existing sections
                    for sect, md in cfg_metrics.items():
                        if sect not in out["metrics"] or not out["metrics"][sect]:
                            out["metrics"][sect] = md
                # 新增：从 cfg['metrics'] 扁平键提取
                root_metrics = _extract_metrics_from_root(cfg)
                if root_metrics:
                    out.setdefault("metrics", {})
                    for sect, md in root_metrics.items():
                        if sect not in out["metrics"] or not out["metrics"][sect]:
                            out["metrics"][sect] = md
                return out

            # 2) 二元组返回： (model, result_df)
            if isinstance(res, (tuple, list)) and len(res) >= 1:
                # 从 cfg['data'] 里拿载荷
                out["data"]["val_long"]  = data_blk.get("val_long")
                out["data"]["test_long"] = data_blk.get("test_long")
                # ✅ 新增：dense
                out["data"]["val_dense"] = data_blk.get("val_dense")
                out["data"]["test_dense"] = data_blk.get("test_dense")

                # split 兜底
                n = len(src_df)
                t = int(n * 0.6)
                v = int(n * 0.2)
                out["data"]["split"] = {"train_len": t, "val_len": v, "test_len": n - t - v}

                # also try to attach metrics from cfg['data']
                cfg_metrics = _extract_metrics_from_cfg(data_blk)
                if cfg_metrics:
                    out["metrics"] = cfg_metrics
                # 新增：从 cfg['metrics'] 扁平键提取
                root_metrics = _extract_metrics_from_root(cfg)
                if root_metrics:
                    out.setdefault("metrics", {})
                    for sect, md in root_metrics.items():
                        if sect not in out["metrics"] or not out["metrics"][sect]:
                            out["metrics"][sect] = md
                return out

            # 3) 兜底
            cfg_metrics = _extract_metrics_from_cfg(data_blk)
            if cfg_metrics:
                out["metrics"] = cfg_metrics
            # 新增：从 cfg['metrics'] 扁平键提取
            root_metrics = _extract_metrics_from_root(cfg)
            if root_metrics:
                out.setdefault("metrics", {})
                for sect, md in root_metrics.items():
                    if sect not in out["metrics"] or not out["metrics"][sect]:
                        out["metrics"][sect] = md
            out["status"] = "error"
            out["message"] = "Unknown pipeline return type"
            return out

        with st.spinner("训练与预测中，请稍候..."):
            try:
                # 兼容两种 pipeline 签名： (df, config) 或 (config)
                import inspect  # local import to avoid top-level dependency
                sig = inspect.signature(run_train_predict_pipeline)
                params = list(sig.parameters.values())
                # 如果 pipeline 只有一个参数（config），先把数据写入 cfg['data']
                if len(params) < 2:
                    _prepare_data_into_config(df.copy(), config, feature_cols)
                # 根据 pipeline 的签名动态组装参数
                call_args = (df.copy(), config) if len(params) >= 2 else (config,)
                raw_results = run_train_predict_pipeline(*call_args)  # type: ignore[call-arg]
                results = _normalize_results(raw_results, config, df)
            except Exception as e:
                st.error("pipeline 运行失败")
                st.exception(e)
                st.stop()

        status = results.get("status", "error")
        if status not in ("ok", "success"):
            st.error(results.get("message", "训练/预测失败"))
            tb = results.get("traceback")
            if tb:
                st.code(tb)
            st.stop()

        # 指标展示（若没有则显示占位）
        metrics = results.get("metrics", {}) or {}
        val_metrics = metrics.get("validation", {}) or {}
        test_metrics = metrics.get("test", {}) or {}

        # 兼容：如果是扁平键（val_rmse/val_mape/test_rmse/test_mape），转成分组结构
        if (not val_metrics) and any(k in metrics for k in ("val_rmse", "val_mape", "val_mape_safe")):
            val_metrics = {
                "rmse": metrics.get("val_rmse"),
                "mape": metrics.get("val_mape"),
                "mape_safe": metrics.get("val_mape_safe"),
            }
        if (not test_metrics) and any(k in metrics for k in ("test_rmse", "test_mape", "test_mape_safe")):
            test_metrics = {
                "rmse": metrics.get("test_rmse"),
                "mape": metrics.get("test_mape"),
                "mape_safe": metrics.get("test_mape_safe"),
            }

        def _fmt(x, pct=False, safe=False, metrics=None):
            # 如果需要从 metrics 里取（比如 mape_safe 优先），在这里替换 x
            if safe and isinstance(metrics, dict):
                if metrics.get("mape_safe") is not None:
                    x = metrics.get("mape_safe")
                elif metrics.get("mape") is not None:
                    x = metrics.get("mape")
            if x is None:
                return "—"
            try:
                xv = float(x)
                if pct:
                    xv = xv * 100.0  # 将比例转为百分数
                    return f"{xv:.2f}%"
                return f"{xv:.4f}"
            except Exception:
                return str(x)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Val RMSE", _fmt(val_metrics.get("rmse")))
        with c2:
            st.metric("Val MAPE", _fmt(None, pct=True, safe=True, metrics=val_metrics))
        with c3:
            st.metric("Test RMSE", _fmt(test_metrics.get("rmse")))
        with c4:
            st.metric("Test MAPE", _fmt(None, pct=True, safe=True, metrics=test_metrics))

        # 若 artifacts 中包含 RF 最佳超参，则单独展示
        _arts = results.get("artifacts", {}) or {}
        _rf_params = _arts.get("randomforest_params") or _arts.get("best_params") or _arts.get("rf_best_params")
        if _rf_params and model_name == "randomforest":
            with st.expander("🧠 RandomForest 最佳超参（Optuna）", expanded=False):
                st.json(_rf_params)

        # 切分信息
        data_blob = results.get("data", {}) or {}
        split_info = data_blob.get("split", {}) or {}
        train_len = _as_int(split_info.get("train_len"))
        val_len = _as_int(split_info.get("val_len"))
        test_len = _as_int(split_info.get("test_len"))

        if train_len is not None and val_len is not None and test_len is not None:
            t, v, te = int(train_len), int(val_len), int(test_len)
            total = t + v + te
            if total > 0:
                st.caption(f"数据切分：train={t}, val={v}, test={te}（比例约为 {t/total:.2f}/{v/total:.2f}/{te/total:.2f} ）")
            else:
                st.caption(f"数据切分：train={t}, val={v}, test={te}")

        # 长序列载荷和 dense 载荷（如有）
        val_long = data_blob.get("val_long")
        test_long = data_blob.get("test_long")
        val_dense = data_blob.get("val_dense") if "val_dense" in data_blob else None
        test_dense = data_blob.get("test_dense") if "test_dense" in data_blob else None

        # 明细表（整段）：优先 dense DataFrame，如果有
        def _coerce_dense(d):
            if d is None:
                return None
            try:
                if isinstance(d, pd.DataFrame):
                    return d
                return pd.DataFrame(d)
            except Exception:
                return None

        val_df_aligned = _coerce_dense(val_dense)
        test_df_aligned = _coerce_dense(test_dense)

        # 兼容：若 dense 不可用，则回退到 long（字典）形式
        if val_df_aligned is None:
            val_long_df = _df_from_long(val_long, time_col)
        else:
            val_long_df = val_df_aligned.copy()
        if test_df_aligned is None:
            test_long_df = _df_from_long(test_long, time_col)
        else:
            test_long_df = test_df_aligned.copy()

        # 训练段 DataFrame（用于四线图背景）
        if train_len is not None and train_len > 0:
            _train_df_for_plot = df.iloc[:train_len][[time_col, value_col]]
        else:
            _train_df_for_plot = df[[time_col, value_col]]

        # 绘图（整段）。注意：不再传递不存在的形参 val_dense/test_dense；
        # 将 dense 作为对齐好的 DataFrame 直接通过 val_df_aligned/test_df_aligned 传入。
        st.subheader("📈 预测结果（整段）")
        vplot = load_plot_module()
        try:
            # 将 dense 直接作为对齐好的 DataFrame 传入
            fig = vplot.plot_results(
                train_df=_train_df_for_plot,
                val_df_aligned=val_dense if isinstance(val_dense, pd.DataFrame) and not val_dense.empty else None,
                test_df_aligned=test_dense if isinstance(test_dense, pd.DataFrame) and not test_dense.empty else None,
                time_col=time_col,
                value_col=value_col,
                title="Forecast Results（整段）",
                payload=None,           # 如需额外信息（split等）也可以放到 payload
                val_long=val_long,      # 作为后备兜底（当前我们已走 dense，不会用到）
                test_long=test_long,
                train_len=train_len,
                val_len=val_len,
                test_len=test_len,
            )
            st.pyplot(fig)
        except Exception as e:
            st.error("绘图失败")
            st.exception(e)

        # 明细导出（优先 dense，其次 long）
        with st.expander("🧾 验证集明细（整段）", expanded=False):
            val_long_df = _df_from_dense_for_display(val_dense, time_col) \
                          if isinstance(val_dense, pd.DataFrame) else _df_from_long(val_long, time_col)
            if not val_long_df.empty:
                st.dataframe(val_long_df, use_container_width=True)
                try:
                    st.download_button(
                        label="下载验证整段明细 CSV",
                        data=val_long_df.to_csv(index=False).encode('utf-8'),
                        file_name="val_dense.csv" if isinstance(val_dense, pd.DataFrame) else "val_long.csv",
                        mime=MIME_CSV,
                    )
                except Exception:
                    pass
            else:
                st.info("暂无验证整段明细。")

        with st.expander("🧾 测试集明细（整段）", expanded=False):
            test_long_df = _df_from_dense_for_display(test_dense, time_col) \
                           if isinstance(test_dense, pd.DataFrame) else _df_from_long(test_long, time_col)
            if not test_long_df.empty:
                st.dataframe(test_long_df, use_container_width=True)
                try:
                    st.download_button(
                        label="下载测试整段明细 CSV",
                        data=test_long_df.to_csv(index=False).encode('utf-8'),
                        file_name="test_dense.csv" if isinstance(test_dense, pd.DataFrame) else "test_long.csv",
                        mime=MIME_CSV,
                    )
                except Exception:
                    pass
            else:
                st.info("暂无测试整段明细。")

        # 工件路径展示（有就展示）
        with st.expander("🧳 Artifacts", expanded=False):
            st.json(results.get("artifacts", {}))