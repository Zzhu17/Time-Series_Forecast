#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/ci_install.sh [--profile minimal|full] [--lock-file path] [--ci-extra-file path]

Installs CI dependencies via a single entry point.
- If lock file exists, install from lock first (default: requirements.lock).
- CI extras are installed from requirements-ci.txt.
- Produces freeze snapshots and verifies install idempotency across 3 reruns.
USAGE
}

profile="minimal"
lock_file="requirements.lock"
ci_extra_file="requirements-ci.txt"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile)
      profile="$2"
      shift 2
      ;;
    --lock-file)
      lock_file="$2"
      shift 2
      ;;
    --ci-extra-file)
      ci_extra_file="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ci_install] Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$profile" in
  minimal|full)
    ;;
  *)
    echo "[ci_install] Unsupported profile: $profile" >&2
    exit 2
    ;;
esac

mkdir -p ci-artifacts

echo "[ci_install] profile=${profile}"
python -m pip install --upgrade pip wheel

install_cmd=(python -m pip install)

if [ -f "$lock_file" ]; then
  echo "[ci_install] Installing lock file: $lock_file"
  "${install_cmd[@]}" -r "$lock_file"
else
  echo "[ci_install] Lock file not found ($lock_file), fallback to Project/requirements.txt"
  "${install_cmd[@]}" -r Project/requirements.txt
fi

if [ -f "$ci_extra_file" ]; then
  echo "[ci_install] Installing CI extras: $ci_extra_file"
  "${install_cmd[@]}" -r "$ci_extra_file"
fi

if [ "$profile" = "full" ]; then
  echo "[ci_install] full profile selected (lock + CI extras already applied)"
fi

base_freeze="ci-artifacts/pip-freeze-initial.txt"
python -m pip freeze | tee "$base_freeze" >/dev/null

baseline_hash=""
for run in 1 2 3; do
  echo "[ci_install] idempotency check run ${run}/3"
  if [ -f "$lock_file" ]; then
    "${install_cmd[@]}" -r "$lock_file" >/dev/null
  else
    "${install_cmd[@]}" -r Project/requirements.txt >/dev/null
  fi
  if [ -f "$ci_extra_file" ]; then
    "${install_cmd[@]}" -r "$ci_extra_file" >/dev/null
  fi

  snap="ci-artifacts/pip-freeze-rerun-${run}.txt"
  python -m pip freeze > "$snap"
  hash_val="$(sha256sum "$snap" | awk '{print $1}')"
  echo "[ci_install] rerun ${run} freeze hash: ${hash_val}"

  if [ -z "$baseline_hash" ]; then
    baseline_hash="$hash_val"
  elif [ "$baseline_hash" != "$hash_val" ]; then
    echo "[ci_install] ERROR: dependency snapshot changed at rerun ${run}" >&2
    exit 1
  fi
done

echo "[ci_install] OK: dependency install is stable across 3 reruns"
