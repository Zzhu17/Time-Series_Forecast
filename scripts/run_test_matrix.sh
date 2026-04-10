#!/usr/bin/env bash
set -euo pipefail

mode="${1:-minimal}"

emit_scenario_meta() {
  local scenario="$1"
  {
    echo "scenario=${scenario}"
    echo "timestamp_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  } > test-matrix-scenario.txt
}

usage() {
  cat <<'USAGE'
Usage: scripts/run_test_matrix.sh [minimal|full|models]

  minimal  安装 requirements-ci 场景，允许 skip 并导出 skip 摘要
  full     全量依赖场景，禁止非平台 skip
  models   仅跑重依赖模型链路（make test-models）
USAGE
}

run_minimal() {
  emit_scenario_meta "minimal"
  ./scripts/check_test_env.sh --strict
  set -o pipefail
  PYTHONPATH=Project pytest -q -rs tests | tee pytest-minimal.log
  {
    echo "# Minimal skip summary"
    grep -E '^SKIPPED ' pytest-minimal.log || echo "No skips reported."
  } > skip-reasons-minimal.txt
}

run_full() {
  emit_scenario_meta "full"
  ./scripts/check_test_env.sh --strict
  set -o pipefail
  PYTHONPATH=Project pytest -q -rs tests | tee pytest-full.log
  if grep -qE '^SKIPPED ' pytest-full.log; then
    bad_skips=$(grep -E '^SKIPPED ' pytest-full.log | grep -vc 'TEST_MATRIX_PLATFORM_SKIP:' || true)
    if [ "${bad_skips}" -gt 0 ]; then
      echo "Found non-platform skips in full matrix: ${bad_skips}" >&2
      grep -E '^SKIPPED ' pytest-full.log >&2
      exit 1
    fi
  fi
}

run_models() {
  emit_scenario_meta "models"
  ./scripts/check_test_env.sh --strict
  make test-models
}

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
  -h|--help)
    usage
    ;;
  *)
    echo "Unknown mode: $mode" >&2
    usage >&2
    exit 2
    ;;
esac
