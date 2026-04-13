#!/usr/bin/env bash
set -euo pipefail

scenario="${1:-unknown}"
timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
bundle_root="failure_bundle"
bundle_dir="${bundle_root}/${timestamp}"

mkdir -p "${bundle_dir}"

copy_if_exists() {
  local src="$1"
  local dest_name="$2"
  if [ -f "$src" ]; then
    cp "$src" "${bundle_dir}/${dest_name}"
  else
    printf "missing: %s\n" "$src" > "${bundle_dir}/${dest_name}"
  fi
}

copy_if_exists "pytest-minimal.log" "pytest.log"
if grep -q "^missing:" "${bundle_dir}/pytest.log"; then
  copy_if_exists "pytest-full.log" "pytest.log"
fi

copy_if_exists "skip-reasons-minimal.txt" "skip-reasons.txt"

python -m pip freeze > "${bundle_dir}/pip-freeze.txt" || echo "pip freeze failed" > "${bundle_dir}/pip-freeze.txt"

{
  echo "scenario=${scenario}"
  echo "job=${GITHUB_JOB:-local}"
  echo "run_id=${GITHUB_RUN_ID:-local}"
  echo "sha=${GITHUB_SHA:-local}"
} > "${bundle_dir}/test-matrix-scenario.txt"

if [ -f "docs/repo/MODEL_READINESS_REPORT.md" ]; then
  cp "docs/repo/MODEL_READINESS_REPORT.md" "${bundle_dir}/quality-gate-report.md"
elif [ -f "MODEL_READINESS_REPORT.md" ]; then
  cp "MODEL_READINESS_REPORT.md" "${bundle_dir}/quality-gate-report.md"
elif [ -f "Project/artifacts/latest/report.json" ]; then
  cp "Project/artifacts/latest/report.json" "${bundle_dir}/quality-gate-report.json"
else
  echo "quality gate report not found" > "${bundle_dir}/quality-gate-report.txt"
fi

if [ ! -f "${bundle_root}/index.md" ]; then
  {
    echo "# Failure Bundle Index"
    echo
    echo "| Timestamp (UTC) | Scenario | Path |"
    echo "|---|---|---|"
  } > "${bundle_root}/index.md"
fi

echo "| ${timestamp} | ${scenario} | ${timestamp}/ |" >> "${bundle_root}/index.md"

{
  echo "# Failure bundle ${timestamp}"
  echo
  echo "- scenario: ${scenario}"
  echo "- created_at_utc: ${timestamp}"
  echo
  echo "## Included files"
  ls -1 "${bundle_dir}" | sed 's/^/- /'
} > "${bundle_dir}/README.md"


echo "FAILURE_BUNDLE_DIR=${bundle_dir}" >> "${GITHUB_ENV:-/dev/null}" || true
echo "failure bundle created at ${bundle_dir}"
