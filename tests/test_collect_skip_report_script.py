from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "collect_skip_report.py"


def test_collect_skip_report_outputs_summary_and_metrics(tmp_path: Path):
    log_file = tmp_path / "pytest.log"
    report_file = tmp_path / "skip-report.txt"
    metrics_file = tmp_path / "skip-metrics.json"

    log_file.write_text(
        "\n".join(
            [
                "SKIPPED tests/test_a.py:10: TEST_MATRIX_OPTIONAL_DEP_MISSING: torch",
                "SKIPPED tests/test_b.py:20: TEST_MATRIX_OPTIONAL_DEP_MISSING: torch",
                "SKIPPED tests/test_c.py:30: TEST_MATRIX_PLATFORM_SKIP: win32-only",
                "3 skipped in 1.23s",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(log_file), str(report_file), str(metrics_file)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    report = report_file.read_text(encoding="utf-8")
    assert "summary_skipped=3" in report
    assert "skip_lines=3" in report
    assert "2 | TEST_MATRIX_OPTIONAL_DEP_MISSING: torch" in report

    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert metrics["summary_skipped"] == 3
    assert metrics["skip_lines"] == 3
    assert metrics["reason_counts"]["TEST_MATRIX_OPTIONAL_DEP_MISSING: torch"] == 2
