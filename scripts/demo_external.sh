#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/Project"
export MPLCONFIGDIR="$ROOT_DIR/Project/tmp"
export TMPDIR="$ROOT_DIR/Project/tmp"

mkdir -p "$ROOT_DIR/Project/tmp"

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
MODEL_NAME="${MODEL_NAME:-baseline}"
HORIZON="${HORIZON:-24}"
DATA_FILE="Project/Data/sample_timeseries.csv"

uvicorn api.app:app --host "$API_HOST" --port "$API_PORT" > "$ROOT_DIR/Project/tmp/demo_api.log" 2>&1 &
API_PID=$!

cleanup() {
  kill "$API_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

sleep 2

curl -s "http://$API_HOST:$API_PORT/health" || true

curl -s -X POST \
  -F "file=@${DATA_FILE}" \
  -F "model_name=$MODEL_NAME" \
  -F "time_col=date" \
  -F "value_col=value" \
  -F "horizon=$HORIZON" \
  "http://$API_HOST:$API_PORT/train_file_sync" \
  | tee "$ROOT_DIR/Project/tmp/demo_last_run.json"

curl -s "http://$API_HOST:$API_PORT/artifacts/latest" \
  | tee "$ROOT_DIR/Project/tmp/demo_latest.json"

echo "Demo outputs:"
echo "  API log: $ROOT_DIR/Project/tmp/demo_api.log"
echo "  Run payload: $ROOT_DIR/Project/tmp/demo_last_run.json"
echo "  Latest artifacts: $ROOT_DIR/Project/tmp/demo_latest.json"
