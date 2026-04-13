#!/usr/bin/env bash
set -euo pipefail

mode="minimal"
enforce_skip_policy="0"

emit_scenario_meta() {
  local scenario="$1"
  {
    echo "scenario=${scenario}"
    echo "timestamp_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  } > test-matrix-scenario.txt
}

usage() {
  cat <<'USAGE'
Usage: scripts/run_test_matrix.sh [minimal|full|models] [--enforce-skip-policy]

  minimal  安装 requirements-ci 场景，允许 skip 并导出 skip 摘要
  full     全量依赖场景，禁止非平台 skip（配合 --enforce-skip-policy 强制）
  models   仅跑重依赖模型链路（make test-models）
USAGE
}

parse_args() {
  while (($# > 0)); do
    case "$1" in
      minimal|full|models)
        mode="$1"
        ;;
      --enforce-skip-policy)
        enforce_skip_policy="1"
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
    shift
  done
}

collect_skip_report() {
  local log_file="$1"
  local report_file="$2"
  local metrics_file="$3"

  python scripts/collect_skip_report.py "$log_file" "$report_file" "$metrics_file"
}

enforce_full_skip_policy() {
  local metrics_file="$1"
  local bad
  bad=$(python - "$metrics_file" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1], encoding="utf-8"))
bad = 0
for reason, count in metrics.get("reason_counts", {}).items():
  if not reason.startswith("TEST_MATRIX_PLATFORM_SKIP:"):
    bad += int(count)
print(bad)
PY
)

  if [ "$bad" -gt 0 ]; then
    echo "Found non-platform skips in full matrix: ${bad}" >&2
    cat "$metrics_file" >&2
    exit 1
  fi
}

run_minimal() {
  emit_scenario_meta "minimal"
  ./scripts/check_test_env.sh --strict
  set -o pipefail
  PYTHONPATH=Project pytest -q -rs tests | tee pytest-minimal.log
  collect_skip_report "pytest-minimal.log" "skip-reasons-minimal.txt" "skip-metrics-minimal.json"
}

run_full() {
  emit_scenario_meta "full"
  ./scripts/check_test_env.sh --strict
  set -o pipefail
  PYTHONPATH=Project pytest -q -rs tests | tee pytest-full.log
  collect_skip_report "pytest-full.log" "skip-reasons-full.txt" "skip-metrics-full.json"
  if [ "$enforce_skip_policy" = "1" ]; then
    enforce_full_skip_policy "skip-metrics-full.json"
  fi
}

run_models() {
  emit_scenario_meta "models"
  ./scripts/check_test_env.sh --strict
  make test-models
}

parse_args "$@"

case "$mode" in
  minimal)
    run_minimal
    ;;
  full)
    run_full
    ;;
  models)
    run_models
    ;;
esac
