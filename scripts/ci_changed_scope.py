#!/usr/bin/env python3
"""Classify changed paths for CI strategy selection."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable

CRITICAL_PREFIXES = (
    "Project/services/",
    "Project/configs/",
)
CRITICAL_EXACT = {
    "Project/models/registry.py",
}
MODEL_PREFIXES = (
    "models/",
    "training/",
    "Project/models/",
    "Project/training/",
)
DOC_PREFIXES = (
    "docs/",
    "doc/",
)
DOC_SUFFIXES = (
    ".md",
    ".rst",
)


def git_changed_files(base: str, head: str) -> list[str]:
    cmd = ["git", "diff", "--name-only", f"{base}..{head}"]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def is_doc_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith(DOC_PREFIXES) or lowered.endswith(DOC_SUFFIXES)


def starts_with_any(path: str, prefixes: Iterable[str]) -> bool:
    return any(path.startswith(prefix) for prefix in prefixes)


def classify(changed_files: list[str], event_name: str) -> dict[str, object]:
    if event_name == "schedule":
        return {
            "event_name": event_name,
            "changed_files": changed_files,
            "docs_only": False,
            "force_full": True,
            "force_models": True,
            "run_minimal": False,
            "run_lint_docs": True,
            "strategy_source": "nightly-schedule-full-regression",
            "reason": "Nightly run always executes full + models regression.",
        }

    critical_hits = [
        p
        for p in changed_files
        if starts_with_any(p, CRITICAL_PREFIXES) or p in CRITICAL_EXACT
    ]
    model_hits = [p for p in changed_files if starts_with_any(p, MODEL_PREFIXES)]
    docs_only = bool(changed_files) and all(is_doc_path(p) for p in changed_files)

    if critical_hits:
        return {
            "event_name": event_name,
            "changed_files": changed_files,
            "docs_only": False,
            "force_full": True,
            "force_models": True,
            "run_minimal": True,
            "run_lint_docs": True,
            "strategy_source": "critical-path-full-regression",
            "reason": "Critical path changed: " + ", ".join(critical_hits),
        }

    if model_hits:
        return {
            "event_name": event_name,
            "changed_files": changed_files,
            "docs_only": False,
            "force_full": True,
            "force_models": True,
            "run_minimal": True,
            "run_lint_docs": True,
            "strategy_source": "model-training-path-full-regression",
            "reason": "Model/training path changed: " + ", ".join(model_hits),
        }

    if docs_only:
        return {
            "event_name": event_name,
            "changed_files": changed_files,
            "docs_only": True,
            "force_full": False,
            "force_models": False,
            "run_minimal": False,
            "run_lint_docs": True,
            "strategy_source": "docs-only-fast-path",
            "reason": "Only documentation files changed.",
        }

    return {
        "event_name": event_name,
        "changed_files": changed_files,
        "docs_only": False,
        "force_full": False,
        "force_models": False,
        "run_minimal": True,
        "run_lint_docs": True,
        "strategy_source": "default-minimal-matrix",
        "reason": "No critical/model-only triggers matched.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--event-name", default="pull_request")
    parser.add_argument(
        "--changed-file",
        action="append",
        default=[],
        help="Manually provide changed file path (can be repeated).",
    )
    parser.add_argument(
        "--write-github-output",
        action="store_true",
        help="Write scalar outputs to the file at $GITHUB_OUTPUT.",
    )
    args = parser.parse_args()

    changed_files = list(args.changed_file)
    if not changed_files and args.base and args.head:
        changed_files = git_changed_files(args.base, args.head)

    result = classify(changed_files=changed_files, event_name=args.event_name)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.write_github_output:
        output_raw = os.environ.get("GITHUB_OUTPUT")
        if not output_raw:
            raise RuntimeError("--write-github-output set but GITHUB_OUTPUT is empty")
        output_file = Path(output_raw)
        with output_file.open("a", encoding="utf-8") as fh:
            for key in (
                "docs_only",
                "force_full",
                "force_models",
                "run_minimal",
                "run_lint_docs",
                "strategy_source",
                "reason",
            ):
                value = result[key]
                fh.write(f"{key}={str(value).lower() if isinstance(value, bool) else value}\n")
            fh.write("changed_files_json<<EOF\n")
            fh.write(json.dumps(result["changed_files"], ensure_ascii=False))
            fh.write("\nEOF\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
