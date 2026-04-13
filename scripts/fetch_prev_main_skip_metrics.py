#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

OUT_FILE = Path("skip-metrics-main-prev.json")


def write_fallback() -> None:
    OUT_FILE.write_text("{}\n", encoding="utf-8")


def get_json(url: str, headers: dict[str, str]) -> dict:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_bytes(url: str, headers: dict[str, str]) -> bytes:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def main() -> int:
    owner_repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    token = os.environ.get("GITHUB_TOKEN", "")

    if not owner_repo or not token:
        write_fallback()
        print("[skip-metrics] missing repository or token; writing fallback")
        return 0

    api = f"https://api.github.com/repos/{owner_repo}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    try:
        runs = get_json(f"{api}/actions/workflows/ci.yml/runs?branch=main&status=success&per_page=10", headers)
        candidate = next((run for run in runs.get("workflow_runs", []) if str(run.get("id")) != str(run_id)), None)
        if not candidate:
            write_fallback()
            print("[skip-metrics] no previous successful main run found")
            return 0

        artifacts = get_json(f"{api}/actions/runs/{candidate['id']}/artifacts", headers)
        target = next((a for a in artifacts.get("artifacts", []) if a.get("name") == "minimal-skip-metrics"), None)
        if not target:
            write_fallback()
            print("[skip-metrics] minimal-skip-metrics artifact not found")
            return 0

        with zipfile.ZipFile(io.BytesIO(download_bytes(target["archive_download_url"], headers))) as zf:
            json_members = [name for name in zf.namelist() if name.endswith(".json")]
            if not json_members:
                write_fallback()
                print("[skip-metrics] artifact zip has no json payload")
                return 0
            OUT_FILE.write_text(zf.read(json_members[0]).decode("utf-8"), encoding="utf-8")

        print("[skip-metrics] restored previous main skip metrics")
        return 0
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 404):
            write_fallback()
            print(f"[skip-metrics] non-fatal HTTP {exc.code}; writing fallback")
            return 0
        raise
    except Exception as exc:  # pragma: no cover - best effort path
        write_fallback()
        print(f"[skip-metrics] non-fatal error: {exc}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
