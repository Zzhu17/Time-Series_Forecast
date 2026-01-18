from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Optional

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from evaluation.report import build_leaderboard, write_leaderboard_csv, write_report_html


def _load_registry_record(registry_path: Path, run_id: str) -> Optional[Dict[str, Any]]:
    if not registry_path.exists():
        return None
    try:
        data = json.loads(registry_path.read_text())
    except Exception:
        return None
    records = data if isinstance(data, list) else [data]
    for record in reversed(records):
        if isinstance(record, dict) and record.get("run_id") == run_id:
            return record
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish leaderboard/report artifacts.")
    parser.add_argument("--run-dir", required=True, help="Run artifacts directory.")
    parser.add_argument("--output-dir", default="Project/output", help="Output directory for latest files.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    leaderboard = run_dir / "leaderboard.csv"
    report = run_dir / "report.html"
    run_id = run_dir.name
    registry_path = output_dir / "run_registry.json"
    record = _load_registry_record(registry_path, run_id)

    if not leaderboard.exists() or not report.exists():
        metrics = record.get("metrics") if isinstance(record, dict) else {}
        metrics = metrics if isinstance(metrics, dict) else {}
        base_metrics = metrics.get("baseline") if isinstance(metrics, dict) else {}
        drift = metrics.get("drift") if isinstance(metrics, dict) else None
        artifacts = record.get("artifacts") if isinstance(record, dict) else {}
        artifacts = artifacts if isinstance(artifacts, dict) else {}
        model_name = str(record.get("model_name") if isinstance(record, dict) else "") or run_id
        dataset_id = str(artifacts.get("dataset_id") or "")

        if not leaderboard.exists():
            df_lb = build_leaderboard(
                model_name=model_name,
                metrics=metrics if isinstance(metrics, dict) else {},
                baseline_metrics=base_metrics if isinstance(base_metrics, dict) else {},
            )
            write_leaderboard_csv(df_lb, leaderboard)

        if not report.exists():
            write_report_html(
                path=report,
                model_name=model_name,
                dataset_id=dataset_id,
                metrics=metrics if isinstance(metrics, dict) else {},
                baseline_metrics=base_metrics if isinstance(base_metrics, dict) else {},
                drift=drift if isinstance(drift, dict) else None,
                leaderboard_path=str(leaderboard),
                artifacts=artifacts if isinstance(artifacts, dict) else {},
            )

    if leaderboard.exists():
        shutil.copy2(leaderboard, output_dir / "leaderboard_latest.csv")
    if report.exists():
        shutil.copy2(report, output_dir / "report_latest.html")

    meta = {
        "run_dir": str(run_dir),
        "leaderboard": str(output_dir / "leaderboard_latest.csv") if leaderboard.exists() else None,
        "report": str(output_dir / "report_latest.html") if report.exists() else None,
    }
    (output_dir / "latest_report.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
