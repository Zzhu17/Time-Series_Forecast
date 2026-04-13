#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: collect_skip_report.py <log_file> <report_file> <metrics_file>", file=sys.stderr)
        return 2

    log_path = Path(sys.argv[1])
    report_path = Path(sys.argv[2])
    metrics_path = Path(sys.argv[3])

    text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    skip_lines = [line.strip() for line in text.splitlines() if line.startswith("SKIPPED ")]

    reasons: list[str] = []
    for line in skip_lines:
        match = re.match(r"^SKIPPED .*?:\d+:\s*(.*)$", line)
        reasons.append(match.group(1).strip() if match else line)

    reason_counts = Counter(reasons)

    summary_skipped = 0
    for match in re.finditer(r"(\d+)\s+skipped", text):
        summary_skipped = max(summary_skipped, int(match.group(1)))
    if summary_skipped == 0:
        summary_skipped = len(skip_lines)

    top_reasons = reason_counts.most_common(10)

    report_lines = [
        "# Skip summary",
        f"summary_skipped={summary_skipped}",
        f"skip_lines={len(skip_lines)}",
        "",
        "## Top skip reasons",
    ]
    if top_reasons:
        for reason, count in top_reasons:
            report_lines.append(f"- {count} | {reason}")
    else:
        report_lines.append("- 0 | No skips reported.")

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    metrics = {
        "summary_skipped": summary_skipped,
        "skip_lines": len(skip_lines),
        "reason_counts": dict(reason_counts),
        "top_reasons": [{"reason": reason, "count": count} for reason, count in top_reasons],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
