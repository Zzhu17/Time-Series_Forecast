from __future__ import annotations

import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import os
from pathlib import Path
import numpy as np
import random

from utils.schemas import PipelineRunModel
from utils.feature_pipeline import save_feature_contract_if_any
from services.result_contracts import (
    RUN_RESULT_DATA_KEYS,
    backfill_metric_slot,
    backfill_missing,
    ensure_metric_slot,
    ensure_run_result,
)


def set_seed(seed: int | None):
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # torch is optional: allow ARIMA/Prophet/etc to run without it
        pass


def configure_logging(cfg):
    import logging
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


from models.registry import TRAINER_REGISTRY, FORECASTER_REGISTRY

# NOTE: keep imports minimal at module import time.
# Heavy/optional deps (sklearn/torch/pmdarima/prophet/...) are loaded lazily in trainers.


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
                        idx = pd.to_datetime(base[ts_col], errors="coerce", utc=True)
                        try:
                            idx = idx.dt.tz_localize(None)
                        except Exception:
                            pass
                        df = df.set_index(idx)
                    else:
                        df.index = pd.to_datetime(base.index, errors="coerce", utc=True)
                        try:
                            df.index = df.index.tz_localize(None)
                        except Exception:
                            pass
                return df.sort_index()

            if isinstance(df_like, dict):
                ts = df_like.get("timestamps")
                if ts is None:
                    return None
                idx = pd.to_datetime(ts, errors="coerce", utc=True)
                try:
                    idx = idx.tz_localize(None)
                except Exception:
                    pass
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
                out[time_col] = pd.to_datetime(out[time_col], errors="coerce", utc=True)
                try:
                    out[time_col] = out[time_col].dt.tz_localize(None)
                except Exception:
                    pass
                out = out.set_index(time_col)
            else:
                try:
                    out.index = pd.to_datetime(out.index, errors="coerce", utc=True)
                    try:
                        out.index = out.index.tz_localize(None)
                    except Exception:
                        pass
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
            idx = pd.to_datetime(ts_list, errors="coerce", utc=True)
            try:
                if hasattr(idx, "tz_localize"):
                    idx = idx.tz_localize(None)
            except Exception:
                pass
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
        from utils.target_transform import inverse_transform_array as _inv_tt
        arr2d = df_sc[[value_col]].to_numpy().astype(np.float32)
        tt_params = (cfg.get('artifacts') or {}).get('target_transform')
        try:
            artifacts = (cfg.get('artifacts') or {})
            y_scaler_path = artifacts.get('y_scaler_path')
            if y_scaler_path and os.path.exists(y_scaler_path):
                import joblib
                y_scaler = joblib.load(y_scaler_path)
                inv = y_scaler.inverse_transform(arr2d)
                out = inv.reshape(-1)
                if tt_params:
                    out = _inv_tt(out, tt_params)
                return pd.Series(out, index=df_sc.index)
        except Exception as e:
            print(f"[pipeline] y_scaler inverse failed: {e}")
        n_in = getattr(scaler, 'n_features_in_', None)
        if n_in is None or not hasattr(scaler, "inverse_transform"):
            out = arr2d.reshape(-1)
            if tt_params:
                out = _inv_tt(out, tt_params)
            return pd.Series(out, index=df_sc.index)
        if arr2d.shape[1] == n_in:
            inv = scaler.inverse_transform(arr2d)
            out = inv.reshape(-1)
            if tt_params:
                out = _inv_tt(out, tt_params)
            return pd.Series(out, index=df_sc.index)
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
        if tt_params:
            out = _inv_tt(out, tt_params)
        return pd.Series(out, index=df_sc.index)

    def _resolve_run_dir(arts: dict) -> str:
        run_dir = str((arts or {}).get("run_dir") or (arts or {}).get("artifact_dir") or "")
        if not run_dir:
            model_path = (arts or {}).get("model_path")
            if isinstance(model_path, str) and model_path:
                run_dir = os.path.dirname(model_path)
        if not run_dir:
            run_dir = str(Path(__file__).resolve().parents[1] / "artifacts")
        return run_dir

    def _assign_missing_split_timestamps(data_blk: dict, df_input: pd.DataFrame, *, time_col: str, val_true, test_true) -> None:
        val_len = int(len(np.asarray(val_true).ravel()))
        test_len = int(len(np.asarray(test_true).ravel()))
        ts_series = None
        if isinstance(df_input, pd.DataFrame) and time_col in df_input.columns:
            ts_series = pd.to_datetime(df_input[time_col], errors="coerce", utc=True)
            try:
                ts_series = ts_series.dt.tz_localize(None)
            except Exception:
                pass
        if ts_series is None or (val_len + test_len) <= 0:
            return
        total = int(len(ts_series))
        train_len = max(0, total - val_len - test_len)
        if data_blk.get("val_timestamps") is None and val_len > 0:
            data_blk["val_timestamps"] = ts_series.iloc[train_len: train_len + val_len].tolist()
        if data_blk.get("test_timestamps") is None and test_len > 0:
            data_blk["test_timestamps"] = ts_series.iloc[train_len + val_len: train_len + val_len + test_len].tolist()

    def _to_dense_df(true_arr, pred_arr) -> Optional[pd.DataFrame]:
        true_arr = np.asarray(true_arr, dtype=float).ravel()
        pred_arr = np.asarray(pred_arr, dtype=float).ravel()
        length = min(len(true_arr), len(pred_arr))
        if length <= 0:
            return None
        return pd.DataFrame({"y_true": true_arr[:length], "yhat": pred_arr[:length]})

    # ---------- 配置取值 ----------
    model_key = str((config.get("model") or {}).get("name", "") or config.get("model_type", "")).strip().lower()
    data_blk   = config.setdefault('data', {})
    artifacts  = config.setdefault('artifacts', {})
    default_cf = config.get('default', {}) or {}

    time_col   = default_cf.get('time_col', 'date')
    value_col  = default_cf.get('value_col', 'value')
    scaler     = artifacts.get('scaler')

    # ---------- progress callback ----------
    progress_cb = None
    try:
        progress_cb = (config.get("callbacks") or {}).get("progress")
    except Exception:
        progress_cb = None

    def _progress(pct: float, msg: str):
        if callable(progress_cb):
            try:
                progress_cb(stage="pipeline", pct=float(pct), msg=msg)
            except Exception:
                pass

    def _safe_mape(yt: np.ndarray, yp: np.ndarray, eps: float = 1e-8) -> float:
        mean_abs = float(np.mean(np.abs(yt))) if yt.size else 0.0
        tau = max(eps, 0.01 * mean_abs) if np.isfinite(mean_abs) and mean_abs > 0 else eps
        mask = np.abs(yt) > tau
        if int(mask.sum()) == 0:
            return float("nan")
        denom = np.abs(yt[mask]) + eps
        return float(np.mean(np.abs((yp[mask] - yt[mask]) / denom)))

    def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        yt = np.asarray(y_true, dtype=float).reshape(-1)
        yp = np.asarray(y_pred, dtype=float).reshape(-1)
        n = min(len(yt), len(yp))
        yt = yt[:n]
        yp = yp[:n]
        if n == 0:
            return {"rmse": np.nan, "mape": np.nan, "nrmse": np.nan, "smape": np.nan}
        mask = np.isfinite(yt) & np.isfinite(yp)
        if int(mask.sum()) == 0:
            return {"rmse": np.nan, "mape": np.nan, "nrmse": np.nan, "smape": np.nan}
        yt = yt[mask]
        yp = yp[mask]
        diff = yp - yt
        rmse = float(np.sqrt(np.mean(diff * diff)))
        denom2 = np.abs(yt) + np.abs(yp) + 1e-8
        smape = float(np.mean(2.0 * np.abs(diff) / denom2))
        std = float(np.std(yt)) + 1e-8
        nrmse = float(rmse / std) if np.isfinite(std) and std > 1e-8 else np.nan
        mape = float(_safe_mape(yt, yp))
        return {"rmse": rmse, "mape": mape, "nrmse": nrmse, "smape": smape}

    def _basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return _calc_metrics(y_true, y_pred)

    def _baseline_metrics(y_all: np.ndarray, train_len: int, val_len: int, test_len: int) -> dict:
        y_all = np.asarray(y_all, dtype=float).reshape(-1)
        out = {"naive": {}, "seasonal": {}, "season_len": None}
        if train_len <= 0 or len(y_all) < train_len:
            return out
        last_train = y_all[train_len - 1]
        if val_len > 0:
            y_val = y_all[train_len : train_len + val_len]
            out["naive"]["val"] = _basic_metrics(y_val, np.full(val_len, last_train, dtype=float))
        if test_len > 0:
            y_test = y_all[train_len + val_len : train_len + val_len + test_len]
            last_tv = y_all[train_len + val_len - 1] if train_len + val_len > 0 else last_train
            out["naive"]["test"] = _basic_metrics(y_test, np.full(test_len, last_tv, dtype=float))

        season_len = int((config.get("baseline") or {}).get("season_len", 0) or 0)
        if season_len <= 0:
            # Try infer from data frequency (hourly/daily)
            season_len = int((config.get("prediction", {}) or {}).get("season_len", 0) or 0)
        if season_len > 0 and train_len >= season_len:
            out["season_len"] = season_len
            if val_len > 0:
                y_val = y_all[train_len : train_len + val_len]
                seasonal_val = y_all[train_len - season_len : train_len - season_len + val_len]
                out["seasonal"]["val"] = _basic_metrics(y_val, seasonal_val)
            if test_len > 0:
                y_test = y_all[train_len + val_len : train_len + val_len + test_len]
                test_start = train_len + val_len - season_len
                seasonal_test = y_all[test_start : test_start + test_len] if test_start >= 0 else None
                if seasonal_test is not None and len(seasonal_test) == len(y_test):
                    out["seasonal"]["test"] = _basic_metrics(y_test, seasonal_test)
        return out

    def _compute_metrics_from_dense(df_dense: Optional[pd.DataFrame]) -> Optional[dict]:
        if not isinstance(df_dense, pd.DataFrame) or df_dense.empty:
            return None
        if not all(c in df_dense.columns for c in ["y_true", "yhat"]):
            return None
        dfm = df_dense[["y_true", "yhat"]].dropna()
        if dfm.empty:
            return None
        return _calc_metrics(dfm["y_true"].values, dfm["yhat"].values)

    def _update_dense_metrics(data_block: dict, metrics_block: dict, val_dense_df, test_dense_df) -> tuple[Optional[dict], Optional[dict]]:
        val_metrics_local = _compute_metrics_from_dense(val_dense_df)
        test_metrics_local = _compute_metrics_from_dense(test_dense_df)
        if isinstance(val_metrics_local, dict):
            metrics_block["val_rmse"] = val_metrics_local.get("rmse")
            metrics_block["val_mape"] = val_metrics_local.get("mape")
            metrics_block["val_nrmse"] = val_metrics_local.get("nrmse")
            metrics_block["val_smape"] = val_metrics_local.get("smape")
        if isinstance(test_metrics_local, dict):
            metrics_block["test_rmse"] = test_metrics_local.get("rmse")
            metrics_block["test_mape"] = test_metrics_local.get("mape")
            metrics_block["test_nrmse"] = test_metrics_local.get("nrmse")
            metrics_block["test_smape"] = test_metrics_local.get("smape")
        data_block["val_metrics"] = val_metrics_local
        data_block["test_metrics"] = test_metrics_local
        return val_metrics_local, test_metrics_local

    def _update_baseline_metrics(data_block: dict, metrics_block: dict, y_all_source, val_len: int, test_len: int):
        y_all = pd.to_numeric(y_all_source, errors="coerce").to_numpy(dtype=float)
        n_total = int(len(y_all))
        train_len_local = max(0, n_total - int(val_len) - int(test_len))
        base_metrics_local = _baseline_metrics(y_all, train_len_local, int(val_len), int(test_len))
        data_block["baseline_metrics"] = base_metrics_local
        metrics_block["baseline"] = base_metrics_local

    def _update_drift_metrics(
        data_block: dict,
        metrics_block: dict,
        *,
        val_true_arr,
        val_pred_arr,
        test_true_arr,
        test_pred_arr,
    ):
        from evaluation.drift import compute_residual_drift

        drift = compute_residual_drift(
            val_true=np.asarray(val_true_arr),
            val_pred=np.asarray(val_pred_arr),
            test_true=np.asarray(test_true_arr),
            test_pred=np.asarray(test_pred_arr),
        )
        data_block["drift"] = drift
        metrics_block["drift"] = drift

    def _maybe_run_backtest(data_block: dict, *, series_source, value_col_name: str):
        bt_cfg = (config.get("evaluation") or {}).get("backtest") or {}
        if not bool(bt_cfg.get("enabled", False)):
            return

        from evaluation.backtest import rolling_backtest_naive

        if isinstance(series_source, pd.DataFrame):
            series = pd.to_numeric(series_source[value_col_name], errors="coerce")
        else:
            series = pd.Series(dtype=float)
        bt = rolling_backtest_naive(
            series,
            horizon=int(bt_cfg.get("horizon", 1)),
            step=int(bt_cfg.get("step", 1)),
            window=int(bt_cfg.get("window", 24)),
            seasonal_period=int(bt_cfg.get("seasonal_period", 0)) or None,
        )
        data_block["backtest"] = bt
        if bt.get("y_true") and bt.get("y_pred"):
            data_block["backtest_metrics"] = _basic_metrics(
                np.asarray(bt.get("y_true")),
                np.asarray(bt.get("y_pred")),
            )

    def _build_training_true_series(data_block: dict, *, emit_warning: bool = False) -> Optional[pd.Series]:
        try:
            train_df_sc = data_block.get("train_df_sc")
            if isinstance(train_df_sc, pd.DataFrame) and len(train_df_sc) > 0 and scaler is not None:
                return _inverse_series_1d_from_df_scaled(train_df_sc, scaler, config, value_col)
        except Exception as exc:
            if emit_warning:
                print(f"[pipeline] Warning: failed to build training_true series: {exc}")
        return None

    def _maybe_pipeline_plot(
        *,
        train_true_series: Optional[pd.Series],
        val_dense_df,
        test_dense_df,
        split_info: dict,
        title: str,
    ):
        viz_cfg = (config.get("visualization") or {})
        do_plot = bool(viz_cfg.get("pipeline_plot", False)) or (os.environ.get("TSF_PIPELINE_PLOT", "0") == "1")
        if not do_plot:
            return

        from visualizations.plot import plot_results

        train_len_local = split_info.get("train_len")
        val_len_local = split_info.get("val_len")
        test_len_local = split_info.get("test_len")
        train_df_plot_local = (
            train_true_series.to_frame("training_true")
            if isinstance(train_true_series, pd.Series)
            else pd.DataFrame(columns=["training_true"])
        )
        val_plot_df = None if (isinstance(val_dense_df, pd.DataFrame) and val_dense_df.empty) else val_dense_df
        test_plot_df = None if (isinstance(test_dense_df, pd.DataFrame) and test_dense_df.empty) else test_dense_df
        payload = {
            "val_dense": val_plot_df,
            "test_dense": test_plot_df,
            "val_long": None,
            "test_long": None,
            "split": {
                "train_len": train_len_local,
                "val_len": val_len_local,
                "test_len": test_len_local,
            },
            "full_truth": None,
            "full_pred_cont": None,
            "phase_mask": None,
        }

        plot_results(
            train_df=train_df_plot_local,
            val_df_aligned=val_plot_df if isinstance(val_plot_df, pd.DataFrame) else None,
            test_df_aligned=test_plot_df if isinstance(test_plot_df, pd.DataFrame) else None,
            time_col=time_col,
            value_col=value_col,
            title=title,
            payload=payload,
            val_long=None,
            test_long=None,
            train_len=int(train_len_local) if train_len_local is not None else (len(train_true_series) if isinstance(train_true_series, pd.Series) else None),
            val_len=int(val_len_local) if val_len_local is not None else None,
            test_len=int(test_len_local) if test_len_local is not None else None,
        )

    _progress(0.03, f"pipeline start (model={model_key})")

    # ===========================================================
    # 0) 优先通过 TRAINER_REGISTRY（例如 arima）
    # ===========================================================
    if model_key in TRAINER_REGISTRY or model_key in FORECASTER_REGISTRY:
        _progress(0.08, "trainer dispatch")
        runner = TRAINER_REGISTRY.get(model_key)
        forecaster_factory = FORECASTER_REGISTRY.get(model_key)
        _df_candidates = [
            config.get('dataframe'),
            data_blk.get('dataframe'),
            data_blk.get('df'),
            data_blk.get('data'),
        ]
        _df_input = next((x for x in _df_candidates if isinstance(x, pd.DataFrame)), pd.DataFrame())
        # Ensure raw df is available for trainers that can self-prepare (e.g., Informer)
        try:
            if isinstance(_df_input, pd.DataFrame) and not _df_input.empty:
                data_blk.setdefault("dataframe", _df_input)
        except Exception:
            pass

        # Data preprocessing + versioning (cleaning, feature engineering, profiling).
        try:
            from services.data_versioning import preprocess_dataframe, save_processed_assets

            _df_input, profile = preprocess_dataframe(
                _df_input,
                config=config,
                time_col=time_col,
                value_col=value_col,
            )
            data_blk["dataframe"] = _df_input
            data_blk["df"] = _df_input
            run_dir = _resolve_run_dir(artifacts)
            assets = save_processed_assets(
                _df_input,
                profile=profile,
                artifacts_dir=run_dir or "artifacts",
            )
            artifacts.update(assets)
            data_blk["data_profile"] = profile
        except Exception as _e:
            data_blk["data_profile_error"] = str(_e)

        # Unified feature cleaning for non-Informer models to avoid NaN-heavy feature issues.
        if model_key != "informer":
            try:
                from utils.feature_missing_policy import prepare_df_for_non_informer_models

                candidate_cols = (
                    list((config.get("data", {}) or {}).get("all_feature_cols") or [])
                    or list((config.get("artifacts", {}) or {}).get("feature_cols") or [])
                )
                if not candidate_cols:
                    # Fallback: use all columns except time_col (target will be forced first)
                    candidate_cols = [c for c in _df_input.columns if c != time_col]

                _df_prep, _feat_cols, _prep_report = prepare_df_for_non_informer_models(
                    _df_input,
                    time_col=time_col,
                    value_col=value_col,
                    candidate_cols=candidate_cols,
                    config=config,
                )
                _df_input = _df_prep
                data_blk["dataframe"] = _df_prep
                data_blk["df"] = _df_prep
                data_blk["all_feature_cols"] = list(_feat_cols)
                data_blk["feature_prep_report"] = _prep_report
                try:
                    dropped = []
                    strict_rep = (_prep_report or {}).get("strict_report") if isinstance(_prep_report, dict) else None
                    if isinstance(strict_rep, dict):
                        for item in (strict_rep.get("dropped_optional") or []):
                            if isinstance(item, dict) and isinstance(item.get("col"), str):
                                dropped.append(item["col"])
                    if dropped:
                        data_blk["dropped_optional_features"] = sorted(set(dropped))
                except Exception:
                    pass
                artifacts["feature_cols"] = list(_feat_cols)
            except Exception as _e:
                # Fail-fast: downstream trainers (especially scalers) cannot handle NaNs.
                data_blk["feature_prep_error"] = str(_e)
                raise

        _progress(0.12, "training + predict")
        if forecaster_factory is not None:
            forecaster = forecaster_factory()
            fit = forecaster.fit(_df_input, config)
            val_true = fit.val_true
            val_pred = fit.val_pred
            test_true = fit.test_true
            test_pred = fit.test_pred
            final_model = fit.model
            test_df = fit.test_forecast_df
            params = fit.params
        elif runner is not None:
            val_true, val_pred, test_true, test_pred, final_model, test_df, params = runner(_df_input, config)
        else:
            raise ValueError(f"Unsupported model '{model_key}'")
        _progress(0.80, "postprocess predictions")
        if not isinstance(params, dict):
            params = {"raw_params": params}
        params.setdefault("model_name", model_key)
        artifacts[f"{model_key}_params"] = params
        if isinstance(params, dict):
            artifacts["training_params"] = dict(params)
        # Ensure RF best params are exposed under a stable key for the app panel
        if model_key == "randomforest":
            artifacts["randomforest_params"] = params
        if isinstance(test_df, pd.DataFrame) and not test_df.empty:
            data_blk["test_forecast_df"] = test_df

        # 反推时间戳（若上游没给）
        try:
            _assign_missing_split_timestamps(
                data_blk,
                _df_input,
                time_col=time_col,
                val_true=val_true,
                test_true=test_true,
            )
        except Exception as _e:
            print(f"[pipeline] warn: failed to infer timestamps: {_e}")

        val_dense = _to_dense_df(val_true, val_pred)
        test_dense = _to_dense_df(test_true, test_pred)
        data_blk["val_dense"] = val_dense
        data_blk["test_dense"] = test_dense
        _progress(0.88, "metrics + residual modeling")

        # --- Optional: residual modeling hook (registry route) ---
        try:
            rm_cfg = (config.get("residual_modeling") or {})
            already_applied = bool(data_blk.get("residual_applied"))
            rm_enabled = bool(rm_cfg.get("enabled", False)) and not already_applied
            if rm_enabled and isinstance(val_dense, pd.DataFrame) and not val_dense.empty:
                model_type = str(rm_cfg.get("model_type", "LinearRegression")).strip().lower()

                if model_type in ("xgboost", "xgb"):
                    try:
                        from models.xgboost import build_xgboost_regressor
                    except Exception as _imp_e:
                        print(f"[pipeline][registry-route] residual modeling skipped (xgboost missing): {_imp_e}")
                        build_xgboost_regressor = None  # type: ignore[assignment]

                    if build_xgboost_regressor is not None:
                        try:
                            def _as_np1(s: pd.Series) -> np.ndarray:
                                return np.asarray(pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64), dtype=np.float64).reshape(-1)

                            yhat_val = _as_np1(val_dense["yhat"])
                            ytrue_val = _as_np1(val_dense["y_true"])
                            y_res_val = (ytrue_val - yhat_val).reshape(-1)

                            df_src = _df_input if isinstance(_df_input, pd.DataFrame) else data_blk.get("dataframe")
                            if not isinstance(df_src, pd.DataFrame) or df_src.empty:
                                raise ValueError("missing dataframe")

                            v_len = int(len(val_dense))
                            te_len = int(len(test_dense)) if isinstance(test_dense, pd.DataFrame) else 0
                            t_len = max(0, int(len(df_src)) - v_len - te_len)

                            lags = rm_cfg.get("lags") or [1, 2, 3, 6, 12, 24]
                            rolls = rm_cfg.get("rolling_windows") or [6, 12, 24, 48]
                            diffs = rm_cfg.get("diffs") or [1, 24]
                            base_features = rm_cfg.get("feature_cols")
                            if not isinstance(base_features, list) or not any(isinstance(x, str) and x.strip() for x in base_features):
                                base_features = ["month", "day_of_month", "day_of_week", "hour", "day_of_year"]
                                base_features += [f"lag_{int(k)}" for k in lags if int(k) > 0]
                                for w in rolls:
                                    wi = int(w)
                                    if wi > 0:
                                        base_features += [f"rolling_mean_{wi}", f"rolling_std_{wi}"]
                                base_features += [f"diff_{int(k)}" for k in diffs if int(k) > 0]

                            from utils.feature_contract import ensure_calendar_features, is_recomputable_name, recompute_feature_column

                            feat_df = df_src.copy()
                            try:
                                if time_col in feat_df.columns:
                                    feat_df = ensure_calendar_features(feat_df, time_col=time_col)
                            except Exception:
                                pass

                            computed_cols: List[str] = []
                            for c in base_features:
                                if not isinstance(c, str) or not c.strip() or c == time_col:
                                    continue
                                if c in feat_df.columns:
                                    try:
                                        feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce")
                                        computed_cols.append(c)
                                    except Exception:
                                        continue
                                elif is_recomputable_name(c):
                                    try:
                                        feat_df[c] = recompute_feature_column(feat_df, c, value_col=value_col, time_col=time_col)
                                        computed_cols.append(c)
                                    except Exception:
                                        continue

                            feat_val = feat_df.iloc[t_len : t_len + v_len].reset_index(drop=True)
                            feat_test = feat_df.iloc[t_len + v_len : t_len + v_len + te_len].reset_index(drop=True) if te_len > 0 else None

                            Xv = pd.DataFrame({"yhat": yhat_val[: len(feat_val)]})
                            for c in computed_cols:
                                if c in feat_val.columns:
                                    Xv[c] = pd.to_numeric(feat_val[c], errors="coerce")

                            Xt = None
                            if isinstance(test_dense, pd.DataFrame) and not test_dense.empty and feat_test is not None:
                                yhat_test = _as_np1(test_dense["yhat"])
                                Xt = pd.DataFrame({"yhat": yhat_test[: len(feat_test)]})
                                for c in computed_cols:
                                    if c in feat_test.columns:
                                        Xt[c] = pd.to_numeric(feat_test[c], errors="coerce")

                            y_res_val = y_res_val[: len(Xv)]
                            train_mask = np.isfinite(y_res_val) & np.isfinite(Xv["yhat"].to_numpy(dtype=np.float64))
                            n_all = int(np.sum(train_mask))
                            if n_all < 20:
                                raise ValueError("too few valid rows")

                            Xv_fit = Xv.to_numpy(dtype=np.float32)[train_mask]
                            yv_fit = y_res_val.astype(np.float32, copy=False)[train_mask]

                            try:
                                es_rounds = int(
                                    (rm_cfg.get("early_stopping_rounds") if isinstance(rm_cfg, dict) else None)
                                    or ((config.get("model_config") or {}).get("XGBoost", {}) or {}).get("early_stopping_rounds", 0)
                                    or 0
                                )
                            except Exception:
                                es_rounds = 0
                            split = int(max(10, min(n_all - 10, int(n_all * 0.8))))
                            Xtr, ytr = Xv_fit[:split], yv_fit[:split]
                            Xev, yev = Xv_fit[split:], yv_fit[split:]

                            mdl = build_xgboost_regressor(config)
                            eval_set = [(Xev, yev)] if (int(es_rounds) > 0 and Xev.size and np.isfinite(yev).any()) else []

                            import inspect

                            fit_kwargs: Dict[str, Any] = {}
                            try:
                                sig = inspect.signature(mdl.fit)
                                fit_params = sig.parameters
                            except Exception:
                                fit_params = {}
                            if eval_set and "eval_set" in fit_params:
                                fit_kwargs["eval_set"] = eval_set
                            if "verbose" in fit_params:
                                fit_kwargs["verbose"] = False
                            if eval_set and int(es_rounds) > 0:
                                es = max(1, int(es_rounds))
                                if "early_stopping_rounds" in fit_params:
                                    fit_kwargs["early_stopping_rounds"] = es
                                elif "callbacks" in fit_params:
                                    try:
                                        import xgboost as xgb  # type: ignore

                                        fit_kwargs["callbacks"] = [xgb.callback.EarlyStopping(rounds=es, save_best=True)]
                                    except Exception:
                                        pass
                            try:
                                mdl.fit(Xtr, ytr, **fit_kwargs)
                            except TypeError:
                                minimal: Dict[str, Any] = {}
                                if eval_set and "eval_set" in fit_params:
                                    minimal["eval_set"] = eval_set
                                if "verbose" in fit_params:
                                    minimal["verbose"] = False
                                mdl.fit(Xtr, ytr, **minimal)

                            res_hat_val = mdl.predict(Xv.to_numpy(dtype=np.float32)).astype(np.float64, copy=False).reshape(-1)
                            Lv = int(min(len(val_dense), len(res_hat_val)))
                            if Lv > 0:
                                val_dense = val_dense.copy()
                                y0 = _as_np1(val_dense["yhat"])
                                col_i = int(list(val_dense.columns).index("yhat"))
                                val_dense.iloc[:Lv, col_i] = (y0[:Lv] + res_hat_val[:Lv])

                            if isinstance(test_dense, pd.DataFrame) and not test_dense.empty and Xt is not None:
                                res_hat_test = mdl.predict(Xt.to_numpy(dtype=np.float32)).astype(np.float64, copy=False).reshape(-1)
                                Lt = int(min(len(test_dense), len(res_hat_test)))
                                if Lt > 0:
                                    test_dense = test_dense.copy()
                                    y0t = _as_np1(test_dense["yhat"])
                                    col_it = int(list(test_dense.columns).index("yhat"))
                                    test_dense.iloc[:Lt, col_it] = (y0t[:Lt] + res_hat_test[:Lt])

                            try:
                                path = (config.get("artifacts") or {}).get("xgboost_residual_model_path")
                                if isinstance(path, str) and path:
                                    mdl.save_model(path)
                                    artifacts["xgboost_residual_model_path"] = path
                            except Exception:
                                pass

                            data_blk["residual_applied"] = True
                            artifacts["residual_model_type"] = "xgboost"
                            residual_features = ["yhat"] + list(computed_cols)
                            data_blk["residual_report"] = {
                                "model_type": "xgboost",
                                "features": residual_features,
                                "early_stopping_rounds": int(es_rounds or 0),
                                "n_train_rows": int(n_all),
                            }
                            artifacts["residual_feature_cols"] = residual_features
                            print("[pipeline][registry-route] residual modeling applied (xgboost).")
                        except Exception as _fit_e:
                            print(f"[pipeline][registry-route] residual modeling skipped (xgboost failed): {_fit_e}")

                else:
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
                        X_val_res = np.asarray(val_dense[["yhat"]].to_numpy(dtype=np.float64), dtype=np.float64)
                        y_val_res = (
                            np.asarray(val_dense["y_true"].to_numpy(dtype=np.float64), dtype=np.float64)
                            - np.asarray(val_dense["yhat"].to_numpy(dtype=np.float64), dtype=np.float64)
                        ).reshape(-1)

                        try:
                            res_mdl = Model()
                            res_mdl.fit(X_val_res, y_val_res)

                            val_dense = val_dense.copy()
                            _val_yhat = np.asarray(val_dense["yhat"].to_numpy(dtype=np.float64), dtype=np.float64)
                            val_dense["yhat"] = _val_yhat + res_mdl.predict(X_val_res)

                            if isinstance(test_dense, pd.DataFrame) and not test_dense.empty:
                                X_test_res = np.asarray(test_dense[["yhat"]].to_numpy(dtype=np.float64), dtype=np.float64)
                                test_dense = test_dense.copy()
                                _test_yhat = np.asarray(test_dense["yhat"].to_numpy(dtype=np.float64), dtype=np.float64)
                                test_dense["yhat"] = _test_yhat + res_mdl.predict(X_test_res)

                            artifacts["residual_model"] = res_mdl
                            artifacts["residual_model_type"] = model_type
                            data_blk["residual_applied"] = True
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

        metrics_blk = config.setdefault("metrics", {})
        val_metrics, test_metrics = _update_dense_metrics(data_blk, metrics_blk, val_dense_std, test_dense_std)

        # Baseline metrics (naive / seasonal)
        try:
            v_len = int(len(np.asarray(val_true).ravel())) if val_true is not None else 0
            te_len = int(len(np.asarray(test_true).ravel())) if test_true is not None else 0
            _update_baseline_metrics(data_blk, metrics_blk, _df_input[value_col], v_len, te_len)
        except Exception:
            pass

        try:
            _update_drift_metrics(
                data_blk,
                metrics_blk,
                val_true=np.asarray(val_true),
                val_pred=np.asarray(val_pred),
                test_true=np.asarray(test_true),
                test_pred=np.asarray(test_pred),
            )
        except Exception:
            pass

        # Optional rolling backtest (naive/seasonal naive)
        try:
            _maybe_run_backtest(data_blk, series_source=_df_input, value_col_name=value_col)
        except Exception:
            pass

        # 反归一化训练真值
        train_true = _build_training_true_series(data_blk)

        _progress(0.98, "pipeline done")

        # --- Optional: pipeline-side plot generation (disabled by default; app handles plotting) ---
        try:
            _maybe_pipeline_plot(
                train_true_series=train_true,
                val_dense_df=val_dense_std,
                test_dense_df=test_dense_std,
                split_info=(data_blk.get("split") or {}),
                title=f"Training / Validation / Test - Full Span (Dense 1-step) [{model_key}]",
            )
        except Exception as e:
            print(f"[pipeline] Info: pipeline_plot skipped or failed: {e}")

        # 返回首选 result_df（优先 val）。避免再次做时间索引归一化（可能在某些环境中很慢），
        # 直接复用上面已经标准化过的 val_dense_std/test_dense_std。
        result_df = val_dense_std if isinstance(val_dense_std, pd.DataFrame) else None
        if result_df is None:
            result_df = test_dense_std if isinstance(test_dense_std, pd.DataFrame) else None
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

    metrics_blk = config.setdefault("metrics", {})
    val_metrics, test_metrics = _update_dense_metrics(data_blk, metrics_blk, val_dense, test_dense)

    try:
        if isinstance(data_blk.get("dataframe"), pd.DataFrame):
            y_all_source = data_blk["dataframe"][value_col]
        else:
            y_all_source = config.get("dataframe")[value_col]  # type: ignore[index]
        v_len = int(len(val_dense)) if isinstance(val_dense, pd.DataFrame) else 0
        te_len = int(len(test_dense)) if isinstance(test_dense, pd.DataFrame) else 0
        _update_baseline_metrics(data_blk, metrics_blk, y_all_source, v_len, te_len)
    except Exception:
        pass

    try:
        if isinstance(val_dense, pd.DataFrame) and isinstance(test_dense, pd.DataFrame):
            _update_drift_metrics(
                data_blk,
                metrics_blk,
                val_true=val_dense["y_true"].values,
                val_pred=val_dense["yhat"].values,
                test_true=test_dense["y_true"].values,
                test_pred=test_dense["yhat"].values,
            )
    except Exception:
        pass

    try:
        _maybe_run_backtest(data_blk, series_source=data_blk.get("dataframe"), value_col_name=value_col)
    except Exception:
        pass

    try:
        print(f"[pipeline] metrics -> val: {val_metrics} | test: {test_metrics}")
    except Exception:
        pass

    # 训练真值
    train_true = _build_training_true_series(data_blk, emit_warning=True)

    # 构造连续序列（回退分支） + (可选) pipeline-side plot
    try:
        _maybe_pipeline_plot(
            train_true_series=train_true,
            val_dense_df=val_dense,
            test_dense_df=test_dense,
            split_info={"train_len": train_len, "val_len": val_len, "test_len": test_len},
            title="Training / Validation / Test - Full Span (Dense 1-step)",
        )
    except Exception as e:
        print(f"[pipeline] Info: pipeline_plot skipped or failed: {e}")

    return model, result_df


# ======================================================================================
# Streamlit app helper (keeps Project/app.py small)
# ======================================================================================

def _pick_first_df(*candidates):
    for x in candidates:
        if isinstance(x, pd.DataFrame):
            return x
    return None


def _pick_first_dict(*candidates):
    for x in candidates:
        if isinstance(x, dict):
            return x
    return None


def _normalize_dense_for_plot(df_like: Optional[pd.DataFrame], which: str) -> Optional[pd.DataFrame]:
    if not isinstance(df_like, pd.DataFrame) or df_like.empty:
        return None
    df = df_like.copy()
    if {"y_true", "yhat"} <= set(df.columns):
        return df
    if which == "val":
        if {"validation_true", "validation_predict"} <= set(df.columns):
            df["y_true"] = df["validation_true"]
            df["yhat"] = df["validation_predict"]
            return df
    if which == "test":
        if {"test_true", "test_predict"} <= set(df.columns):
            df["y_true"] = df["test_true"]
            df["yhat"] = df["test_predict"]
            return df
    return None


def normalize_results_for_app(res, cfg: dict, src_df: pd.DataFrame) -> dict:
    """
    Normalize arbitrary pipeline return types into:
      {'status','message','metrics':{'validation','test'}, 'data':{...}, 'artifacts':{...}}
    This is a UI-facing normalization layer; it does not mutate training artifacts.
    """
    out: dict = ensure_run_result(res if isinstance(res, dict) else {})
    out["artifacts"] = (cfg.get("artifacts") or {})
    data_blk = (cfg.get("data") or {}) if isinstance(cfg, dict) else {}
    if not isinstance(data_blk, dict):
        data_blk = {}

    if isinstance(res, dict):
        out.update(ensure_run_result(res))
        out["artifacts"] = out.get("artifacts") or (cfg.get("artifacts") or {})
    elif isinstance(res, (tuple, list)):
        out = ensure_run_result(out)
        out["artifacts"] = (cfg.get("artifacts") or {})
    else:
        out["status"] = "error"
        out["message"] = "Unknown pipeline return type"
        out = ensure_run_result(out)

    # ---- Backfill data from cfg['data'] ----
    out_data = out["data"]
    out_metrics = out["metrics"]
    backfill_missing(out_data, data_blk, RUN_RESULT_DATA_KEYS)

    # Ensure split always exists for UI
    if "split" not in out_data or not isinstance(out_data.get("split"), dict):
        n = int(len(src_df)) if isinstance(src_df, pd.DataFrame) else 0
        t = int(n * 0.6)
        v = int(n * 0.2)
        out_data["split"] = {"train_len": t, "val_len": v, "test_len": n - t - v}

    # data_blk may hold val_metrics/test_metrics
    backfill_metric_slot(out_metrics, "validation", _pick_first_dict(data_blk.get("val_metrics"), data_blk.get("metrics_val"), data_blk.get("validation_metrics")))
    backfill_metric_slot(out_metrics, "test", _pick_first_dict(data_blk.get("test_metrics"), data_blk.get("metrics_test"), data_blk.get("testing_metrics")))
    backfill_metric_slot(out_metrics, "baseline", data_blk.get("baseline_metrics"))
    backfill_metric_slot(out_metrics, "drift", data_blk.get("drift"))

    # root cfg metrics may store flat values
    root_m = cfg.get("metrics") if isinstance(cfg.get("metrics"), dict) else {}
    if isinstance(root_m, dict):
        vm = ensure_metric_slot(out_metrics, "validation")
        tm = ensure_metric_slot(out_metrics, "test")
        if vm.get("rmse") is None and "val_rmse" in root_m:
            vm["rmse"] = root_m.get("val_rmse")
        if vm.get("mape") is None and "val_mape" in root_m:
            vm["mape"] = root_m.get("val_mape")
        if vm.get("nrmse") is None and "val_nrmse" in root_m:
            vm["nrmse"] = root_m.get("val_nrmse")
        if vm.get("smape") is None and "val_smape" in root_m:
            vm["smape"] = root_m.get("val_smape")
        if tm.get("rmse") is None and "test_rmse" in root_m:
            tm["rmse"] = root_m.get("test_rmse")
        if tm.get("mape") is None and "test_mape" in root_m:
            tm["mape"] = root_m.get("test_mape")
        if tm.get("nrmse") is None and "test_nrmse" in root_m:
            tm["nrmse"] = root_m.get("test_nrmse")
        if tm.get("smape") is None and "test_smape" in root_m:
            tm["smape"] = root_m.get("test_smape")

    return ensure_run_result(out)


def looks_like_required_core_error(err: Exception) -> bool:
    msg = str(err)
    keys = [
        "Required core feature",
        "Missing required core",
        "核心特征存在缺失值",
        "缺少必要列",
    ]
    return any(k in msg for k in keys)


def baseline_degraded_results(src_df: pd.DataFrame, cfg: dict, *, error: Exception) -> dict:
    import numpy as _np
    import pandas as _pd

    tcol = (cfg.get("default", {}) or {}).get("time_col", "date")
    vcol = (cfg.get("default", {}) or {}).get("value_col", "value")

    df2 = src_df.copy()
    if tcol in df2.columns:
        ts = _pd.to_datetime(df2[tcol], errors="coerce")
        if ts.isna().all():
            ts = _pd.date_range(start=_pd.Timestamp.today().normalize(), periods=len(df2), freq="D")
        df2["_ts_"] = ts
        df2 = df2.sort_values("_ts_")
        df2 = df2.set_index(_pd.DatetimeIndex(df2["_ts_"], name=tcol))
        df2 = df2.drop(columns=["_ts_"], errors="ignore")
    else:
        df2.index = _pd.date_range(start=_pd.Timestamp.today().normalize(), periods=len(df2), freq="D", name=tcol)

    y = _pd.to_numeric(df2.get(vcol), errors="coerce")
    if int(y.notna().sum()) == 0:
        raise ValueError(f"目标列 '{vcol}' 无可用数值（无法降级预测）。原始错误：{error}")

    y_ffill = y.ffill()
    yhat = y_ffill.shift(1)

    n = len(df2)
    t = int(n * 0.6)
    v = int(n * 0.2)
    te = n - t - v

    val_idx = slice(t, t + v)
    test_idx = slice(t + v, n)
    val_dense = _pd.DataFrame({"y_true": y.iloc[val_idx].to_numpy(), "yhat": yhat.iloc[val_idx].to_numpy()}, index=df2.index[val_idx])
    test_dense = _pd.DataFrame({"y_true": y.iloc[test_idx].to_numpy(), "yhat": yhat.iloc[test_idx].to_numpy()}, index=df2.index[test_idx])

    def _metrics(d: _pd.DataFrame) -> dict:
        yt = d["y_true"].to_numpy(dtype=float)
        yp = d["yhat"].to_numpy(dtype=float)
        mask = _np.isfinite(yt) & _np.isfinite(yp)
        if int(mask.sum()) == 0:
            return {"rmse": _np.nan, "mape": _np.nan, "nrmse": _np.nan, "smape": _np.nan}
        rmse = float(_np.sqrt(_np.mean((yp[mask] - yt[mask]) ** 2)))
        diff = yp[mask] - yt[mask]
        denom = _np.abs(yt[mask]) + _np.abs(yp[mask]) + 1e-8
        smape = float(_np.mean(2.0 * _np.abs(diff) / denom))
        std = float(_np.std(yt[mask])) + 1e-8
        nrmse = float(rmse / std) if _np.isfinite(std) and std > 1e-8 else _np.nan
        mean_abs = float(_np.mean(_np.abs(yt[mask]))) if int(mask.sum()) else 0.0
        tau = max(1e-8, 0.01 * mean_abs) if _np.isfinite(mean_abs) and mean_abs > 0 else 1e-8
        mape_mask = _np.abs(yt[mask]) > tau
        if int(mape_mask.sum()) == 0:
            mape = _np.nan
        else:
            denom_m = _np.abs(yt[mask][mape_mask]) + 1e-8
            mape = float(_np.mean(_np.abs(diff[mape_mask] / denom_m)))
        return {"rmse": rmse, "mape": mape, "nrmse": nrmse, "smape": smape}

    val_m = _metrics(val_dense)
    test_m = _metrics(test_dense)

    data_blk = cfg.setdefault("data", {})
    data_blk["degraded"] = True
    data_blk["degraded_mode"] = "naive_persistence"
    data_blk["degraded_reason"] = "required_core_missing"
    data_blk["degraded_error"] = str(error)
    data_blk["split"] = {"train_len": t, "val_len": v, "test_len": te}
    data_blk["val_dense"] = val_dense
    data_blk["test_dense"] = test_dense
    data_blk["val_metrics"] = val_m
    data_blk["test_metrics"] = test_m

    return {
        "status": "ok",
        "metrics": {"validation": val_m, "test": test_m},
        "data": {
            "split": data_blk["split"],
            "val_dense": val_dense,
            "test_dense": test_dense,
            "degraded": True,
            "degraded_mode": data_blk.get("degraded_mode"),
            "degraded_reason": data_blk.get("degraded_reason"),
            "degraded_error": data_blk.get("degraded_error"),
        },
        "artifacts": cfg.get("artifacts", {}),
    }


def run_pipeline_and_update_state(
    df: pd.DataFrame,
    config: dict,
    feature_cols: list,
    *,
    uploaded_name: str | None,
    model_name: str,
    time_col: str,
    value_col: str,
    allow_degrade: bool = False,
    progress_cb=None,
) -> dict:
    """
    Streamlit-oriented runner:
    - calls run_train_predict_pipeline
    - normalizes results
    - creates minimal snapshot (plot_data + metrics) and updates st.session_state
    """
    from services.snapshot import (
        cacheable_results,
        pack_plot_series,
        safe_artifacts_from_config,
        save_last_results_json,
        strip_heavy_inplace,
        as_int,
    )

    config = config if isinstance(config, dict) else {}
    config.setdefault("callbacks", {})
    if callable(progress_cb):
        config["callbacks"]["progress"] = progress_cb

    def _ensure_result_data_block(obj) -> dict:
        data = obj.get("data")
        if not isinstance(data, dict):
            data = {}
            obj["data"] = data
        return data

    def _sync_result_data_field(key: str, value) -> None:
        for target in (snap_results, results):
            data = _ensure_result_data_block(target)
            data[key] = value

    # Make raw df discoverable by pipeline (both old/new keys)
    config["dataframe"] = df.copy()
    config.setdefault("data", {})
    config["data"]["dataframe"] = df.copy()
    # Persist UI-selected feature candidates for downstream trainers (non-Informer models rely on this).
    def _infer_error_stage_and_action(exc: Exception, default_stage: str = "train") -> tuple[str, str]:
        msg = str(exc).lower()
        stage = default_stage
        if any(k in msg for k in ["dataframe", "feature", "missing column", "data prep", "preprocess"]):
            stage = "data_prep"
        elif any(k in msg for k in ["validate", "validation", "val_loss"]):
            stage = "validate"
        elif any(k in msg for k in ["predict", "forecast", "inference"]):
            stage = "predict"
        action = "retry" if any(k in msg for k in ["timeout", "temporar", "resource busy", "connection reset"]) else "fail"
        return stage, action

    def _build_error_payload(exc: Exception, *, stage: str, action: str) -> dict:
        payload = {
            "status": "error",
            "message": str(exc),
            "error_stage": stage,
            "error_type": type(exc).__name__,
            "action": action,
            "metrics": {},
            "data": {},
            "artifacts": config.get("artifacts", {}),
        }
        config.setdefault("data", {})["last_error"] = {
            "error_stage": stage,
            "error_type": type(exc).__name__,
            "action": action,
            "message": str(exc),
        }
        return payload

    config["data"]["all_feature_cols"] = list(feature_cols or [])

    try:
        # Validate minimal run schema (time_col/value_col/model_name/features).
        PipelineRunModel(
            time_col=time_col,
            value_col=value_col,
            model_name=model_name,
            feature_cols=list(feature_cols or []),
            residual_modeling=config.get("residual_modeling"),
        )
        import inspect

        sig = inspect.signature(run_train_predict_pipeline)
        call_args = (df.copy(), config) if len(sig.parameters) >= 2 else (config,)
        raw_results = run_train_predict_pipeline(*call_args)  # type: ignore[call-arg]
        results = normalize_results_for_app(raw_results, config, df)
    except Exception as e:
        if bool(allow_degrade) and looks_like_required_core_error(e):
            results = baseline_degraded_results(df.copy(), config, error=e)
            results = normalize_results_for_app(results, config, df)
            results["error_stage"] = "data_prep"
            results["error_type"] = type(e).__name__
            results["action"] = "degrade"
        else:
            err_stage, err_action = _infer_error_stage_and_action(e, default_stage="train")
            results = _build_error_payload(e, stage=err_stage, action=err_action)

    # Strip heavy objects and keep artifacts safe
    strip_heavy_inplace(config)
    results["artifacts"] = safe_artifacts_from_config(config)
    train_meta = ((config.get("data") or {}).get("train_run_metadata") or (config.get("artifacts") or {}).get("train_run_metadata"))
    if isinstance(train_meta, dict):
        _ensure_result_data_block(results)["train_run_metadata"] = dict(train_meta)
    try:
        rep = (config.get("data") or {}).get("feature_prep_report")
        save_feature_contract_if_any(rep if isinstance(rep, dict) else {}, config.get("artifacts") or {})
    except Exception:
        pass

    try:
        from evaluation.report import build_leaderboard, write_leaderboard_csv, write_report_html

        arts = config.get("artifacts") or {}
        run_dir = _resolve_run_dir(arts)
        if run_dir:
            leaderboard_path = Path(run_dir) / "leaderboard.csv"
            report_path = Path(run_dir) / "report.html"
            metrics = results.get("metrics", {})
            base_metrics = (config.get("data") or {}).get("baseline_metrics")
            drift = (config.get("data") or {}).get("drift")
            display_name = str(config.get("model_alias") or model_name)
            df_lb = build_leaderboard(
                model_name=display_name,
                metrics=metrics,
                baseline_metrics=base_metrics if isinstance(base_metrics, dict) else {},
            )
            write_leaderboard_csv(df_lb, leaderboard_path)
            write_report_html(
                path=report_path,
                model_name=display_name,
                dataset_id=str(arts.get("dataset_id") or ""),
                metrics=metrics,
                baseline_metrics=base_metrics if isinstance(base_metrics, dict) else {},
                drift=drift if isinstance(drift, dict) else None,
                leaderboard_path=str(leaderboard_path),
                artifacts=arts if isinstance(arts, dict) else {},
            )
            results_data = _ensure_result_data_block(results)
            results_data["leaderboard"] = df_lb.to_dict(orient="records")
            results_data["leaderboard_path"] = str(leaderboard_path)
            results_data["report_path"] = str(report_path)
            if isinstance(arts, dict):
                arts["leaderboard_path"] = str(leaderboard_path)
                arts["report_path"] = str(report_path)
                config["artifacts"] = arts
    except Exception:
        pass

    snap_meta = {
        "uploaded_name": uploaded_name,
        "model_name": model_name,
        "time_col": time_col,
        "value_col": value_col,
        "run_id": (config.get("artifacts") or {}).get("run_id") or config.get("run_id"),
    }
    snap_results = cacheable_results(results)

    def _choose_plot_ts(dfr: pd.DataFrame, fallback_len: int):
        if isinstance(dfr.index, pd.DatetimeIndex):
            return dfr.index
        if time_col in dfr.columns:
            return dfr[time_col]
        if "timestamp" in dfr.columns:
            return dfr["timestamp"]
        return pd.date_range(start=pd.Timestamp.today().normalize(), periods=max(1, int(fallback_len)), freq="D")

    def _build_dense_plot(dense_df: Optional[pd.DataFrame], fallback_len: int):
        if not isinstance(dense_df, pd.DataFrame) or not {"y_true", "yhat"} <= set(dense_df.columns):
            return None
        return pack_plot_series(_choose_plot_ts(dense_df, fallback_len or len(dense_df)), dense_df["y_true"], dense_df["yhat"], max_n=4000)

    def _build_long_plot(long_payload: Optional[dict]):
        if not isinstance(long_payload, dict):
            return None
        return pack_plot_series(long_payload.get("timestamps"), long_payload.get("y_true"), long_payload.get("yhat"), max_n=4000)

    # ---- Build plot_data + mean_abs_true_* (FIX: avoid DataFrame truthiness) ----
    split = (results.get("data", {}) or {}).get("split") or (config.get("data", {}) or {}).get("split") or {}
    t_len = as_int(split.get("train_len"), 0) or 0
    v_len = as_int(split.get("val_len"), 0) or 0
    te_len = as_int(split.get("test_len"), 0) or 0

    mean_abs_true_val = None
    mean_abs_true_test = None
    try:
        if v_len > 0 and value_col in df.columns:
            yv0 = pd.to_numeric(df.iloc[t_len : t_len + v_len][value_col], errors="coerce").to_numpy(dtype=float)
            mean_abs_true_val = float(np.nanmean(np.abs(yv0))) if yv0.size else None
        if te_len > 0 and value_col in df.columns:
            yt0 = pd.to_numeric(df.iloc[t_len + v_len : t_len + v_len + te_len][value_col], errors="coerce").to_numpy(dtype=float)
            mean_abs_true_test = float(np.nanmean(np.abs(yt0))) if yt0.size else None
    except Exception:
        mean_abs_true_val = None
        mean_abs_true_test = None

    train_plot = None
    val_plot = None
    test_plot = None
    try:
        dblk = (config.get("data", {}) or {})
        rdata = (results.get("data", {}) or {})
        vd0 = _pick_first_df(dblk.get("val_dense"), rdata.get("val_dense"), dblk.get("val_result_df"), rdata.get("val_result_df"))
        td0 = _pick_first_df(dblk.get("test_dense"), rdata.get("test_dense"), dblk.get("test_result_df"), rdata.get("test_result_df"))
        vd = _normalize_dense_for_plot(vd0, "val")
        td = _normalize_dense_for_plot(td0, "test")
        vlong = _pick_first_dict(rdata.get("val_long"), dblk.get("val_long"), rdata.get("val_tail"), dblk.get("val_tail"))
        tlong = _pick_first_dict(rdata.get("test_long"), dblk.get("test_long"), rdata.get("test_tail"), dblk.get("test_tail"))

        if t_len > 0 and value_col in df.columns:
            try:
                train_slice = df.iloc[:t_len]
                train_ts = train_slice[time_col] if time_col in train_slice.columns else None
                train_true = train_slice[value_col]
                train_plot = pack_plot_series(train_ts, train_true, train_true, max_n=4000)
            except Exception:
                train_plot = None

        val_plot = _build_dense_plot(vd, v_len)
        if val_plot is None:
            val_plot = _build_long_plot(vlong)

        test_plot = _build_dense_plot(td, te_len)
        if test_plot is None:
            test_plot = _build_long_plot(tlong)
    except Exception as e:
        print(f"[services.pipeline] plot_data build failed: {e}", flush=True)
        train_plot = None
        val_plot = None
        test_plot = None

    if train_plot is None and val_plot is None and test_plot is None:
        try:
            dblk = (config.get("data", {}) or {})
            rdata = (results.get("data", {}) or {})
            vd_dbg = dblk.get("val_dense")
            td_dbg = dblk.get("test_dense")
            print(
                "[services.pipeline] plot_data missing | "
                f"val_dense={type(vd_dbg).__name__} cols={getattr(vd_dbg,'columns',None)} | "
                f"test_dense={type(td_dbg).__name__} cols={getattr(td_dbg,'columns',None)} | "
                f"rdata_keys={list(rdata.keys()) if isinstance(rdata,dict) else None}",
                flush=True,
            )
        except Exception:
            pass

    if train_plot or val_plot or test_plot:
        # Coerce any stringified plot blobs back to dict for safety (avoids cached snapshots with stringified dicts).
        def _coerce_plot(p):
            if isinstance(p, str):
                try:
                    import json, ast
                    try:
                        return json.loads(p)
                    except Exception:
                        return ast.literal_eval(p)
                except Exception:
                    return None
            return p

        plot_blob = {"train": _coerce_plot(train_plot), "val": _coerce_plot(val_plot), "test": _coerce_plot(test_plot)}
        _sync_result_data_field("plot_data", plot_blob)
    if isinstance(mean_abs_true_val, (int, float)) and np.isfinite(float(mean_abs_true_val)) and float(mean_abs_true_val) > 0:
        _sync_result_data_field("mean_abs_true_val", float(mean_abs_true_val))
    if isinstance(mean_abs_true_test, (int, float)) and np.isfinite(float(mean_abs_true_test)) and float(mean_abs_true_test) > 0:
        _sync_result_data_field("mean_abs_true_test", float(mean_abs_true_test))

    save_last_results_json({"meta": snap_meta, "results": snap_results})

    # Update session_state only when running under Streamlit to avoid bare-mode warnings.
    try:
        import streamlit as st
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
        except Exception:
            get_script_run_ctx = None

        ctx = get_script_run_ctx() if get_script_run_ctx is not None else None
        if get_script_run_ctx is not None and ctx is None:
            return results

        st.session_state["last_results"] = snap_results
        st.session_state["last_meta"] = snap_meta
        st.session_state["last_results_source"] = "fresh" if not bool((snap_results.get("data") or {}).get("degraded", False)) else "degraded"
    except Exception:
        pass

    return results
