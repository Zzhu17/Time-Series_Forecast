from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import shutil
import os
import time

import pandas as pd

from configs.config import load_yaml_config
from services.registry import register_model
from services.pipeline_loader import load_pipeline_module
from services.snapshot import cacheable_results
from services.training_payloads import normalize_training_payload
from utils.metrics import observe_task


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
        model_record = register_model(
            name=str(model_alias or model_name),
            version=task_id,
            stage="candidate",
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
            observe_task(
                task_type="train",
                model=model_name,
                duration=time.time() - start_ts,
                status=status,
            )
