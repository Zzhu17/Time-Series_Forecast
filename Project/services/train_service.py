from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import shutil
import os
import time
import math

import pandas as pd

from configs.config import load_yaml_config
from services.registry import register_model
from services.pipeline_loader import load_pipeline_module
from services.snapshot import cacheable_results
from services.training_payloads import normalize_training_payload
from utils.metrics import observe_degrade, observe_task


def _training_params_summary(normalized: Dict[str, Any], *, task_id: str) -> Dict[str, Any]:
    return {
        "run_id": task_id,
        "model_name": normalized.get("model_name"),
        "model_alias": normalized.get("model_alias"),
        "time_col": normalized.get("time_col"),
        "value_col": normalized.get("value_col"),
        "feature_cols": list(normalized.get("feature_cols") or []),
        "device": normalized.get("device", "cpu"),
        "allow_degrade": bool(normalized.get("allow_degrade", False)),
        "residual_modeling": normalized.get("residual_modeling"),
    }


def _artifact_dir_for_task(task_id: str) -> Path:
    project_dir = Path(__file__).resolve().parents[2]
    art_root = project_dir / "artifacts" / "runs"
    art_dir = art_root / task_id
    art_dir.mkdir(parents=True, exist_ok=True)
    return art_dir


def _artifact_root() -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts" / "runs"


def _safe_float(val: Any) -> Optional[float]:
    try:
        f = float(val)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


def _extract_primary_nrmse(metrics: Dict[str, Any]) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    for key in ("test", "validation"):
        blk = metrics.get(key)
        if isinstance(blk, dict):
            score = _safe_float(blk.get("nrmse"))
            if score is not None:
                return score
    return _safe_float(metrics.get("nrmse"))


def evaluate_training_gate(*, metrics: Dict[str, Any], degraded: bool) -> Dict[str, Any]:
    threshold = _safe_float(os.getenv("TRAINING_GATE_MAX_NRMSE", "1.0"))
    if threshold is None:
        threshold = 1.0
    nrmse = _extract_primary_nrmse(metrics)
    checks = {
        "not_degraded": not bool(degraded),
        "nrmse_available": nrmse is not None,
        "nrmse_within_threshold": (nrmse is not None and nrmse <= threshold),
    }
    passed = all(checks.values())
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "passed": passed,
        "failed_checks": failed,
        "thresholds": {"max_nrmse": threshold},
        "observed": {"nrmse": nrmse, "degraded": bool(degraded)},
    }


def _purge_old_runs(current_run_id: str, keep: int = 1) -> None:
    if keep <= 0:
        return
    runs_root = _artifact_root()
    if not runs_root.exists():
        return
    run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    if not run_dirs:
        return
    keep_ids = {current_run_id}
    if keep > 1:
        extras = [p for p in sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True) if p.name != current_run_id]
        for p in extras[: max(0, keep - 1)]:
            keep_ids.add(p.name)
    for p in run_dirs:
        if p.name in keep_ids:
            continue
        shutil.rmtree(p, ignore_errors=True)


def _write_latest_report(run_id: str, artifacts: Dict[str, Any]) -> None:
    project_dir = Path(__file__).resolve().parents[2]
    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = artifacts.get("run_dir") if isinstance(artifacts, dict) else None
    if not run_dir:
        model_path = artifacts.get("model_path") if isinstance(artifacts, dict) else None
        if isinstance(model_path, str) and model_path:
            run_dir = str(Path(model_path).parent)
    meta = {
        "run_dir": run_dir,
        "leaderboard": artifacts.get("leaderboard_path") if isinstance(artifacts, dict) else None,
        "report": artifacts.get("report_path") if isinstance(artifacts, dict) else None,
    }
    (output_dir / "latest_report.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _persist_training_params(task_id: str, training_params: Dict[str, Any]) -> str:
    run_dir = _artifact_dir_for_task(task_id)
    out_path = run_dir / "training_params.json"
    out_path.write_text(json.dumps(training_params, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def _model_filename(model_name: str) -> str:
    key = str(model_name).lower()
    if key == "xgboost":
        return "xgboost_model.json"
    if key == "randomforest":
        return "random_forest.pkl"
    if key == "arima":
        return "arima_model.pkl"
    if key == "prophet":
        return "prophet_model.pkl"
    if key == "lstm":
        return "lstm_model.pth"
    if key == "informer":
        return "informer_model.pth"
    return "model.bin"

def _default_post_calibration() -> Dict[str, Any]:
    return {
        "enabled": True,
        "a_clip": [0.8, 1.2],
        "b_clip_ratio": 0.1,
        "ridge": 1e-6,
        "mape_guard_rel": 1.02,
    }


def _default_target_transform(model_name: str) -> Dict[str, Any]:
    return {
        "enabled": model_name.lower() in ("informer", "lstm"),
        "method": "log1p",
    }


def _metric_value(metrics_block: Dict[str, Any], metric_name: str) -> Optional[float]:
    if not isinstance(metrics_block, dict):
        return None
    target = metric_name.lower()
    for key, val in metrics_block.items():
        if str(key).lower() != target:
            continue
        try:
            num = float(val)
            if num == num:
                return num
        except Exception:
            return None
    return None


def _evaluate_quality_gate(config: Dict[str, Any], metrics: Dict[str, Any]) -> Optional[str]:
    gate_cfg = (config or {}).get("quality_gate")
    if not isinstance(gate_cfg, dict) or not bool(gate_cfg.get("enabled", True)):
        return None
    test_metrics = metrics.get("test") if isinstance(metrics.get("test"), dict) else {}
    baseline_metrics = metrics.get("baseline") if isinstance(metrics.get("baseline"), dict) else {}
    reasons: List[str] = []

    required_metrics = gate_cfg.get("required_metrics")
    if isinstance(required_metrics, dict):
        for metric_name, threshold in required_metrics.items():
            val = _metric_value(test_metrics, str(metric_name))
            if val is None:
                reasons.append(f"missing test metric: {metric_name}")
                continue
            try:
                limit = float(threshold)
            except Exception:
                reasons.append(f"invalid threshold for {metric_name}")
                continue
            if val > limit:
                reasons.append(f"{metric_name}={val:.6g} > {limit:.6g}")

    baseline_cfg = gate_cfg.get("baseline")
    if isinstance(baseline_cfg, dict) and bool(baseline_cfg.get("enabled", True)):
        max_degradation = baseline_cfg.get("max_degradation")
        if isinstance(max_degradation, dict):
            for metric_name, rel_tol in max_degradation.items():
                test_val = _metric_value(test_metrics, str(metric_name))
                base_val = _metric_value(baseline_metrics, str(metric_name))
                if test_val is None or base_val is None:
                    reasons.append(f"missing baseline compare metric: {metric_name}")
                    continue
                try:
                    tol = float(rel_tol)
                except Exception:
                    reasons.append(f"invalid baseline threshold for {metric_name}")
                    continue
                allow_upper = base_val * (1.0 + tol)
                if test_val > allow_upper:
                    reasons.append(
                        f"{metric_name} degraded: test={test_val:.6g}, baseline={base_val:.6g}, tol={tol:.2%}"
                    )

    if reasons:
        return "; ".join(reasons)
    return None


def build_training_config(
    *,
    df: pd.DataFrame,
    task_id: str,
    model_name: str,
    model_alias: Optional[str] = None,
    time_col: str,
    value_col: str,
    feature_cols: List[str],
    residual_modeling: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    dtype: str = "float32",
) -> Dict[str, Any]:
    try:
        base_cfg = load_yaml_config()
    except Exception:
        base_cfg = {}
    config = dict(base_cfg) if isinstance(base_cfg, dict) else {}
    config["run_id"] = task_id

    config.setdefault("model", {})
    config["model"]["name"] = model_name
    config["model_type"] = model_name
    if model_alias:
        config["model_alias"] = model_alias

    default_cfg = config.setdefault("default", {})
    default_cfg["time_col"] = time_col
    default_cfg["value_col"] = value_col
    default_cfg.setdefault("device", device)
    default_cfg.setdefault("dtype", dtype)
    config["device"] = device

    config.setdefault("visualization", {"pipeline_plot": False, "build_continuous": False})
    config.setdefault("prediction", {"rolling": {"enabled": False}})
    config.setdefault("callbacks", {})
    config.setdefault("data", {})
    config["data"]["dataframe"] = df.copy()

    art_dir = _artifact_dir_for_task(task_id)
    model_file = _model_filename(model_name)
    config["artifacts"] = {
        "run_id": task_id,
        "run_dir": str(art_dir),
        "model_path": str(art_dir / model_file),
        "scaler_path": str(art_dir / "scaler.pkl"),
        "residual_model_path": str(art_dir / "residual_model.pkl"),
        "y_scaler_path": str(art_dir / "value_scaler.pkl"),
        "feature_cols_path": str(art_dir / "feature_cols.json"),
        "feature_report_path": str(art_dir / "feature_report.json"),
        "xgboost_model_path": str(art_dir / "xgboost_model.json"),
        "xgboost_residual_model_path": str(art_dir / "xgboost_residual_model.json"),
    }

    config.setdefault("target_transform", _default_target_transform(model_name))
    config.setdefault("post_calibration", _default_post_calibration())

    model_cfg = config.setdefault("model_config", {})
    if isinstance(model_cfg, dict):
        inf_cfg = model_cfg.setdefault("Informer", {})
        if isinstance(inf_cfg, dict):
            inf_cfg.setdefault("auto_feature_cols", True)
            inf_cfg.setdefault("lock_feature_order", True)
            inf_cfg["feature_cols"] = list(feature_cols)

    if residual_modeling:
        config["residual_modeling"] = residual_modeling
        if model_name.lower() == "informer":
            config.setdefault("model_config", {}).setdefault("Informer", {})["use_residual"] = False
            config.setdefault("post_calibration", {})["enabled"] = False

    return config


def run_training_task(payload: Dict[str, Any], *, task_id: str, emit_metrics: bool = True) -> Dict[str, Any]:
    start_ts = time.time()
    status = "success"
    model_name = "unknown"
    try:
        df, normalized, contract_report = normalize_training_payload(
            payload,
            auto_select_features=True,
        )
        model_name = normalized["model_name"]
        model_alias = normalized.get("model_alias") if isinstance(normalized.get("model_alias"), str) else None
        time_col = normalized["time_col"]
        value_col = normalized["value_col"]
        feature_cols = normalized["feature_cols"]
        residual_modeling = normalized.get("residual_modeling")
        residual_modeling = residual_modeling if isinstance(residual_modeling, dict) else None
        device = normalized.get("device") if isinstance(normalized.get("device"), str) else "cpu"
        config = build_training_config(
            df=df,
            task_id=task_id,
            model_name=model_name,
            model_alias=model_alias,
            time_col=time_col,
            value_col=value_col,
            feature_cols=feature_cols,
            residual_modeling=residual_modeling,
            device=device,
        )

        pipeline_mod = load_pipeline_module()
        results = pipeline_mod.run_pipeline_and_update_state(
            df=df.copy(),
            config=config,
            feature_cols=feature_cols,
            uploaded_name=normalized.get("uploaded_name"),
            model_name=model_name,
            time_col=time_col,
            value_col=value_col,
            allow_degrade=bool(normalized.get("allow_degrade", False)),
            progress_cb=None,
        )

        metrics = results.get("metrics", {}) if isinstance(results, dict) else {}
        artifacts = results.get("artifacts", {}) if isinstance(results, dict) else {}
        artifacts = artifacts if isinstance(artifacts, dict) else {}
        data = results.get("data", {}) if isinstance(results, dict) else {}
        degraded = bool(data.get("degraded", False))
        degraded_reason = data.get("degraded_reason")
        training_params = _training_params_summary(normalized, task_id=task_id)
        artifacts["training_params"] = training_params
        try:
            run_dir = artifacts.get("run_dir")
            if isinstance(run_dir, str) and run_dir:
                p = Path(run_dir) / "training_params.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(training_params, ensure_ascii=False, indent=2), encoding="utf-8")
                artifacts["training_params_path"] = str(p)
        except Exception:
            pass
        gate_failed_reason = _evaluate_quality_gate(config, metrics)
        model_stage = "archived" if gate_failed_reason else "candidate"
        fallback_model = data.get("degraded_mode") if degraded else None
        if degraded and emit_metrics:
            observe_degrade(model=model_alias or model_name, reason=degraded_reason)

        params = {
            "task_id": task_id,
            "model_name": model_name,
            "model_alias": model_alias,
            "time_col": time_col,
            "value_col": value_col,
            "feature_cols": feature_cols,
            "residual_modeling": residual_modeling,
            "contract_report": contract_report,
            "training_params": training_params,
        }
        trainer_params = artifacts.get(f"{str(model_name).lower()}_params") if isinstance(artifacts, dict) else None
        params["training_params"] = trainer_params if isinstance(trainer_params, dict) else {}
        gate = evaluate_training_gate(metrics=metrics if isinstance(metrics, dict) else {}, degraded=degraded)
        params["quality_gate"] = gate
        params["gate_passed"] = bool(gate.get("passed", False))
        if isinstance(artifacts, dict):
            artifacts["training_params_path"] = _persist_training_params(task_id, params["training_params"])
            artifacts["quality_gate"] = gate
        model_record = register_model(
            name=str(model_alias or model_name),
            version=task_id,
            stage="candidate" if bool(gate.get("passed", False)) else "archived",
            stage=model_stage,
            params=params,
            metrics=metrics,
            artifacts=artifacts,
        )

        try:
            _write_latest_report(task_id, artifacts if isinstance(artifacts, dict) else {})
            keep_runs = int(os.getenv("ARTIFACT_RETENTION_RUNS", "1") or 1)
            _purge_old_runs(task_id, keep=keep_runs)
        except Exception:
            pass

        snap_results = cacheable_results(results) if isinstance(results, dict) else {}

        return {
            "run_id": task_id,
            "metrics": metrics,
            "artifacts": artifacts,
            "degraded": degraded,
            "degraded_reason": degraded_reason,
            "gate_failed_reason": gate_failed_reason,
            "fallback_model": fallback_model,
            "model_record": model_record,
            "results": results,
            "cacheable_results": snap_results,
            "training_params": training_params,
        }
    except Exception:
        status = "failed"
        raise
    finally:
        if emit_metrics:
            if status == "success":
                try:
                    if "degraded" in locals() and bool(degraded):
                        observe_degrade(stage="train", model=model_name, reason=str(degraded_reason or "degraded"))
                except Exception:
                    pass
            observe_task(
                task_type="train",
                model=model_name,
                duration=time.time() - start_ts,
                status=status,
            )
