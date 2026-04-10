#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


METRICS = ("MAPE", "RMSE", "MAE")


@dataclass
class MetricTrend:
    metric: str
    values: List[float]
    consecutive_degradation: int


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _load_recent_models(db_path: Path, limit: int) -> List[Dict[str, Any]]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, name, version, params, metrics, created_at FROM models ORDER BY datetime(created_at) DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    out: List[Dict[str, Any]] = []
    for row in rows:
        params_raw = row["params"]
        metrics_raw = row["metrics"]
        try:
            params = json.loads(params_raw) if params_raw else {}
        except Exception:
            params = {}
        try:
            metrics = json.loads(metrics_raw) if metrics_raw else {}
        except Exception:
            metrics = {}
        out.append(
            {
                "id": row["id"],
                "name": row["name"],
                "version": row["version"],
                "created_at": row["created_at"],
                "params": params if isinstance(params, dict) else {},
                "metrics": metrics if isinstance(metrics, dict) else {},
            }
        )
    return out


def _metric_values(recent_models: List[Dict[str, Any]], metric: str) -> List[float]:
    values: List[float] = []
    metric_l = metric.lower()
    for rec in reversed(recent_models):
        test_metrics = rec.get("metrics", {}).get("test", {})
        if not isinstance(test_metrics, dict):
            continue
        value = _to_float(test_metrics.get(metric))
        if value is None:
            value = _to_float(test_metrics.get(metric_l))
        if value is not None:
            values.append(value)
    return values


def _degrade_streak(values: List[float]) -> int:
    if len(values) < 2:
        return 0
    streak = 0
    for idx in range(len(values) - 1, 0, -1):
        if values[idx] > values[idx - 1]:
            streak += 1
        else:
            break
    return streak


def _gate_decision(latest: Dict[str, Any], thresholds: Dict[str, Optional[float]]) -> Dict[str, Any]:
    params = latest.get("params", {})
    if isinstance(params, dict) and isinstance(params.get("gate_passed"), bool):
        return {
            "gate_passed": bool(params["gate_passed"]),
            "source": "model_params.gate_passed",
            "failed_metrics": [],
        }

    failed_metrics: List[str] = []
    test_metrics = latest.get("metrics", {}).get("test", {})
    for metric in METRICS:
        threshold = thresholds.get(metric)
        if threshold is None:
            continue
        observed = None
        if isinstance(test_metrics, dict):
            observed = _to_float(test_metrics.get(metric))
            if observed is None:
                observed = _to_float(test_metrics.get(metric.lower()))
        if observed is None:
            failed_metrics.append(f"{metric}:missing")
            continue
        if observed > threshold:
            failed_metrics.append(f"{metric}:{observed:.6g}>{threshold:.6g}")

    return {
        "gate_passed": len(failed_metrics) == 0,
        "source": "thresholds",
        "failed_metrics": failed_metrics,
    }


def _risk_level(gate_passed: bool, trend_warning: bool, strongest_streak: int, streak_threshold: int) -> str:
    if not gate_passed:
        return "high"
    if trend_warning and strongest_streak >= streak_threshold + 1:
        return "high"
    if trend_warning:
        return "medium"
    return "low"


def build_report(
    recent_models: List[Dict[str, Any]],
    history_size: int,
    streak_threshold: int,
    thresholds: Dict[str, Optional[float]],
) -> Dict[str, Any]:
    if not recent_models:
        return {
            "history_size": history_size,
            "records_found": 0,
            "gate_passed": True,
            "trend_status": "insufficient_data",
            "risk_level": "medium",
            "warning": "no model records found",
            "metrics": {},
        }

    trend_details: Dict[str, Dict[str, Any]] = {}
    strongest_streak = 0
    trend_warning = False
    for metric in METRICS:
        values = _metric_values(recent_models, metric)
        streak = _degrade_streak(values)
        strongest_streak = max(strongest_streak, streak)
        if streak >= streak_threshold:
            trend_warning = True
        trend_details[metric] = {
            "values": values,
            "consecutive_degradation": streak,
            "threshold": streak_threshold,
            "warning": streak >= streak_threshold,
        }

    gate = _gate_decision(recent_models[0], thresholds)
    trend_status = "warning_degrading" if trend_warning else "stable"
    report = {
        "history_size": history_size,
        "records_found": len(recent_models),
        "latest_record": {
            "id": recent_models[0].get("id"),
            "name": recent_models[0].get("name"),
            "version": recent_models[0].get("version"),
            "created_at": recent_models[0].get("created_at"),
        },
        "gate_passed": gate["gate_passed"],
        "gate_source": gate["source"],
        "gate_failed_metrics": gate["failed_metrics"],
        "trend_status": trend_status,
        "risk_level": _risk_level(gate["gate_passed"], trend_warning, strongest_streak, streak_threshold),
        "metrics": trend_details,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute quality gate and trend status from recent training results.")
    parser.add_argument("--db-path", default="Project/output/tasks.db")
    parser.add_argument("--history-size", type=int, default=int(os.getenv("QUALITY_GATE_HISTORY_SIZE", "5")))
    parser.add_argument(
        "--degrade-streak-threshold",
        type=int,
        default=int(os.getenv("QUALITY_GATE_DEGRADE_STREAK", "3")),
    )
    parser.add_argument("--mape-threshold", type=float, default=_to_float(os.getenv("QUALITY_GATE_MAPE_THRESHOLD")))
    parser.add_argument("--rmse-threshold", type=float, default=_to_float(os.getenv("QUALITY_GATE_RMSE_THRESHOLD")))
    parser.add_argument("--mae-threshold", type=float, default=_to_float(os.getenv("QUALITY_GATE_MAE_THRESHOLD")))
    parser.add_argument("--output", default="quality_gate_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = Path(args.db_path)
    history_size = max(1, int(args.history_size))
    streak_threshold = max(1, int(args.degrade_streak_threshold))
    recent_models = _load_recent_models(db_path, history_size)
    report = build_report(
        recent_models=recent_models,
        history_size=history_size,
        streak_threshold=streak_threshold,
        thresholds={"MAPE": args.mape_threshold, "RMSE": args.rmse_threshold, "MAE": args.mae_threshold},
    )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if report.get("trend_status") == "warning_degrading":
        print(
            "::warning::Quality gate trend warning: consecutive degradation reached threshold "
            f"({streak_threshold})."
        )
    print(f"gate_passed={report.get('gate_passed')} trend_status={report.get('trend_status')} risk_level={report.get('risk_level')}")
    if report.get("records_found", 0) == 0:
        print("::warning::No training model records found; quality gate treated as soft-pass.")
        return 0
    return 0 if bool(report.get("gate_passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
