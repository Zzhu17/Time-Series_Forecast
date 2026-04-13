from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

from services.artifact_paths import resolve_run_dir_from_artifacts


def write_pipeline_reports(config: Dict[str, Any], results: Dict[str, Any], model_name: str) -> None:
    try:
        from evaluation.report import build_leaderboard, write_leaderboard_csv, write_report_html
    except Exception:
        return

    artifacts = config.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        return
    run_dir = resolve_run_dir_from_artifacts(artifacts)
    if not run_dir:
        return

    leaderboard_path = Path(run_dir) / "leaderboard.csv"
    report_path = Path(run_dir) / "report.html"
    metrics = results.get("metrics", {}) if isinstance(results.get("metrics"), dict) else {}
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
        dataset_id=str(artifacts.get("dataset_id") or ""),
        metrics=metrics,
        baseline_metrics=base_metrics if isinstance(base_metrics, dict) else {},
        drift=drift if isinstance(drift, dict) else None,
        leaderboard_path=str(leaderboard_path),
        artifacts=artifacts,
    )

    result_data = results.get("data")
    if not isinstance(result_data, dict):
        result_data = {}
        results["data"] = result_data
    result_data["leaderboard"] = df_lb.to_dict(orient="records")
    result_data["leaderboard_path"] = str(leaderboard_path)
    result_data["report_path"] = str(report_path)
    artifacts["leaderboard_path"] = str(leaderboard_path)
    artifacts["report_path"] = str(report_path)
    config["artifacts"] = artifacts
