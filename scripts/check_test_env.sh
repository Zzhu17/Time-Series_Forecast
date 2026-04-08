#!/usr/bin/env bash
set -euo pipefail

missing_modules=()

for module in pytest httpx fastapi; do
  if ! python -c "import ${module}" >/dev/null 2>&1; then
    missing_modules+=("${module}")
  fi
done

if [ ${#missing_modules[@]} -gt 0 ]; then
  echo "[check_test_env] Missing required test dependencies: ${missing_modules[*]}" >&2
  echo "[check_test_env] Install minimal test dependencies with one of:" >&2
  echo "  pip install -r requirements-dev.txt" >&2
  echo "  pip install -r requirements-ci.txt" >&2
  exit 1
fi

echo "[check_test_env] OK: pytest/httpx/fastapi are importable."
