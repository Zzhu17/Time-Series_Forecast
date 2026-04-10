#!/usr/bin/env bash
set -euo pipefail

mode="strict"

usage() {
  cat <<'USAGE'
Usage: scripts/check_test_env.sh [--strict|--soft]

  --strict  Missing dependencies cause a non-zero exit (default; CI-friendly)
  --soft    Missing dependencies are reported as warnings but script exits 0
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --strict)
      mode="strict"
      ;;
    --soft)
      mode="soft"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[check_test_env] Unknown option: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python}"
REQUIRED_MODULES="${CHECK_TEST_ENV_REQUIRED_MODULES:-pytest}"

which_python="$(command -v "${PYTHON_BIN}" || true)"
python_version="$("${PYTHON_BIN}" -V 2>&1 || true)"
sys_executable="$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)' 2>/dev/null || true)"

echo "[check_test_env] mode=${mode}"
echo "[check_test_env] which python: ${which_python:-<not found>}"
echo "[check_test_env] python -V: ${python_version:-<unavailable>}"
echo "[check_test_env] sys.executable: ${sys_executable:-<unavailable>}"

if [ -z "$which_python" ]; then
  echo "[check_test_env] python interpreter not found: ${PYTHON_BIN}" >&2
  [ "$mode" = "soft" ] && exit 0
  exit 1
fi

missing_modules=()
for module in ${REQUIRED_MODULES}; do
  if ! "${PYTHON_BIN}" -c "import ${module}" >/dev/null 2>&1; then
    missing_modules+=("${module}")
  fi
done

if [ ${#missing_modules[@]} -gt 0 ]; then
  prefix="[check_test_env][ERROR]"
  if [ "$mode" = "soft" ]; then
    prefix="[check_test_env][WARN]"
  fi

  echo "${prefix} Missing required test dependencies: ${missing_modules[*]}" >&2
  echo "${prefix} Interpreter used for probing: ${sys_executable:-${which_python}}" >&2
  echo "${prefix} Install minimal test dependencies with one of:" >&2
  echo "  pip install -r requirements-dev.txt" >&2
  echo "  pip install -r requirements-ci.txt" >&2

  if [ "$mode" = "strict" ]; then
    exit 1
  fi
fi

echo "[check_test_env] OK: required modules (${REQUIRED_MODULES}) are importable in ${sys_executable:-${which_python}}"
