from __future__ import annotations

import pandas as pd
from typing import Dict, Any, Tuple, Optional
import os
from pathlib import Path
import numpy as np
import random

from utils.schemas import PipelineRunModel
from utils.feature_pipeline import save_feature_contract_if_any
from services.pipeline_reporting import write_pipeline_reports
from services.pipeline_results import (
    baseline_degraded_results,
    looks_like_required_core_error,
    normalize_results_for_app,
)
from services.pipeline_snapshot import (
    build_snapshot_meta,
    enrich_snapshot_payload,
)
from services.artifact_paths import resolve_run_dir_from_artifacts
from services.pipeline_dense import (
    attach_ts_and_rename,
    normalize_dense,
    standardize_dense_df,
    to_dense_df,
)
from services.pipeline_series import (
    assign_missing_split_timestamps,
    build_training_true_series,
)
from services.pipeline_residuals import apply_registry_residual_modeling
from services.pipeline_errors import build_error_payload, infer_error_stage_and_action
from services.pipeline_metrics import (
    calc_metrics,
    update_baseline_metrics,
    update_dense_metrics,
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
            data_block["backtest_metrics"] = calc_metrics(
                np.asarray(bt.get("y_true")),
                np.asarray(bt.get("y_pred")),
            )

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
            run_dir = resolve_run_dir_from_artifacts(
                artifacts,
                default_dir=str(Path(__file__).resolve().parents[1] / "artifacts"),
            )
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
            assign_missing_split_timestamps(
                data_blk,
                _df_input,
                time_col=time_col,
                val_true=val_true,
                test_true=test_true,
            )
        except Exception as _e:
            print(f"[pipeline] warn: failed to infer timestamps: {_e}")

        val_dense = to_dense_df(val_true, val_pred)
        test_dense = to_dense_df(test_true, test_pred)
        data_blk["val_dense"] = val_dense
        data_blk["test_dense"] = test_dense
        _progress(0.88, "metrics + residual modeling")

        val_dense, test_dense = apply_registry_residual_modeling(
            config=config,
            data_blk=data_blk,
            artifacts=artifacts,
            val_dense=val_dense,
            test_dense=test_dense,
            df_input=_df_input,
            time_col=time_col,
            value_col=value_col,
        )

        # 标准化 + 改名（保留 y_true/yhat）
        val_ts  = data_blk.get("val_timestamps")
        test_ts = data_blk.get("test_timestamps")
        val_dense_std = standardize_dense_df(normalize_dense(val_dense, time_col), time_col)
        test_dense_std = standardize_dense_df(normalize_dense(test_dense, time_col), time_col)
        val_dense_std = attach_ts_and_rename(val_dense_std, val_ts, "val", time_col)
        test_dense_std = attach_ts_and_rename(test_dense_std, test_ts, "test", time_col)

        data_blk["val_dense"]  = val_dense_std
        data_blk["test_dense"] = test_dense_std

        metrics_blk = config.setdefault("metrics", {})
        val_metrics, test_metrics = update_dense_metrics(data_blk, metrics_blk, val_dense_std, test_dense_std)

        # Baseline metrics (naive / seasonal)
        try:
            v_len = int(len(np.asarray(val_true).ravel())) if val_true is not None else 0
            te_len = int(len(np.asarray(test_true).ravel())) if test_true is not None else 0
            update_baseline_metrics(data_blk, metrics_blk, _df_input[value_col], v_len, te_len, config)
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
        train_true = build_training_true_series(data_blk, scaler=scaler, config=config, value_col=value_col)

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

    val_dense = standardize_dense_df(normalize_dense(val_dense, time_col), time_col)
    test_dense = standardize_dense_df(normalize_dense(test_dense, time_col), time_col)

    split_info = (data_blk.get('split') or {})
    train_len = split_info.get('train_len'); val_len = split_info.get('val_len'); test_len = split_info.get('test_len')

    metrics_blk = config.setdefault("metrics", {})
    val_metrics, test_metrics = update_dense_metrics(data_blk, metrics_blk, val_dense, test_dense)

    try:
        if isinstance(data_blk.get("dataframe"), pd.DataFrame):
            y_all_source = data_blk["dataframe"][value_col]
        else:
            y_all_source = config.get("dataframe")[value_col]  # type: ignore[index]
        v_len = int(len(val_dense)) if isinstance(val_dense, pd.DataFrame) else 0
        te_len = int(len(test_dense)) if isinstance(test_dense, pd.DataFrame) else 0
        update_baseline_metrics(data_blk, metrics_blk, y_all_source, v_len, te_len, config)
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
    train_true = build_training_true_series(
        data_blk,
        scaler=scaler,
        config=config,
        value_col=value_col,
        emit_warning=True,
    )

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
        safe_artifacts_from_config,
        save_last_results_json,
        strip_heavy_inplace,
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

    # Make raw df discoverable by pipeline (both old/new keys)
    config["dataframe"] = df.copy()
    config.setdefault("data", {})
    config["data"]["dataframe"] = df.copy()
    # Persist UI-selected feature candidates for downstream trainers (non-Informer models rely on this).
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
            err_stage, err_action = infer_error_stage_and_action(e, default_stage="train")
            results = build_error_payload(e, stage=err_stage, action=err_action, artifacts=config.get("artifacts", {}))
            config.setdefault("data", {})["last_error"] = {
                "error_stage": err_stage,
                "error_type": type(e).__name__,
                "action": err_action,
                "message": str(e),
            }

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
        write_pipeline_reports(config, results, model_name)
    except Exception:
        pass

    snap_meta = build_snapshot_meta(
        config=config,
        uploaded_name=uploaded_name,
        model_name=model_name,
        time_col=time_col,
        value_col=value_col,
    )
    snap_results = cacheable_results(results)
    enrich_snapshot_payload(
        df=df,
        config=config,
        results=results,
        snap_results=snap_results,
        time_col=time_col,
        value_col=value_col,
    )

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
