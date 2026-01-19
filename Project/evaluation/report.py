from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json

import numpy as np
import pandas as pd


def build_leaderboard(
    *,
    model_name: str,
    metrics: Dict[str, Dict[str, float]] | None,
    baseline_metrics: Dict[str, Any] | None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metrics = metrics or {}
    baseline_metrics = baseline_metrics or {}

    def _add(model: str, split: str, m: Dict[str, Any] | None):
        if not isinstance(m, dict):
            return
        rows.append(
            {
                "model": model,
                "split": split,
                "rmse": m.get("rmse"),
                "mape": m.get("mape"),
                "nrmse": m.get("nrmse"),
                "smape": m.get("smape"),
            }
        )

    _add(model_name, "val", metrics.get("validation"))
    _add(model_name, "test", metrics.get("test"))

    naive = baseline_metrics.get("naive") if isinstance(baseline_metrics, dict) else None
    seasonal = baseline_metrics.get("seasonal") if isinstance(baseline_metrics, dict) else None
    if isinstance(naive, dict):
        _add("baseline_naive", "val", naive.get("val"))
        _add("baseline_naive", "test", naive.get("test"))
    if isinstance(seasonal, dict):
        _add("baseline_seasonal", "val", seasonal.get("val"))
        _add("baseline_seasonal", "test", seasonal.get("test"))

    return pd.DataFrame(rows)


def write_leaderboard_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_report_html(
    *,
    path: Path,
    model_name: str,
    dataset_id: str | None,
    metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    drift: Dict[str, Any] | None,
    leaderboard_path: str,
    artifacts: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def _coerce_json(value: Any) -> Any:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        if isinstance(value, pd.Series):
            return value.tolist()
        if isinstance(value, dict):
            return {k: _coerce_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_coerce_json(v) for v in value]
        return value

    payload = {
        "model": model_name,
        "dataset_id": dataset_id,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "drift": drift,
        "leaderboard": leaderboard_path,
        "artifacts": artifacts,
    }
    payload = _coerce_json(payload)
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Forecast Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 24px; background: #faf7f2; color: #222; }}
    h1 {{ margin-bottom: 6px; }}
    pre {{ background: #fff; padding: 12px; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>Forecast Report</h1>
  <p><strong>Model:</strong> {model_name}</p>
  <p><strong>Dataset ID:</strong> {dataset_id or '--'}</p>
  <p><strong>Leaderboard:</strong> {leaderboard_path}</p>
  <h2>Summary</h2>
  <pre>{json.dumps(payload, ensure_ascii=False, indent=2)}</pre>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")
