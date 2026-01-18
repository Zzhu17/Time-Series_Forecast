#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

API_BASE="${API_BASE:-http://localhost:8000}"
DATA_FILE="Project/Data/sample_timeseries.csv"

mkdir -p Project/tmp

echo "==> Health check"
curl -sf "${API_BASE}/health" >/dev/null

echo "==> Train (sync)"
TRAIN_RESP="$(curl -s -X POST \
  -F "file=@${DATA_FILE}" \
  -F "model_name=baseline" \
  -F "time_col=date" \
  -F "value_col=value" \
  -F "horizon=24" \
  "${API_BASE}/train_file_sync")"

echo "${TRAIN_RESP}" > Project/tmp/smoke_train.json

RUN_ID="$(python - <<'PY'
import json, sys
data=json.loads(open('Project/tmp/smoke_train.json','r',encoding='utf-8').read())
print(data.get('run_id',''))
PY
)"

if [ -z "${RUN_ID}" ]; then
  echo "Missing run_id in train response" >&2
  exit 1
fi

echo "==> Publish latest"
RUN_DIR=""
if [ -d "artifacts/runs/${RUN_ID}" ]; then
  RUN_DIR="artifacts/runs/${RUN_ID}"
elif [ -d "Project/artifacts/runs/${RUN_ID}" ]; then
  RUN_DIR="Project/artifacts/runs/${RUN_ID}"
fi

if [ -n "${RUN_DIR}" ]; then
  python Project/spark_jobs/publish_leaderboard_report.py \
    --run-dir "${RUN_DIR}" \
    --output-dir Project/output >/dev/null
else
  echo "Skip publish_latest: run_dir not found on host (docker volume may not include artifacts)."
fi

echo "==> /artifacts/latest"
LATEST_RESP="$(curl -s "${API_BASE}/artifacts/latest")"
echo "${LATEST_RESP}" > Project/tmp/smoke_latest.json

python - <<'PY'
import json
data=json.loads(open('Project/tmp/smoke_latest.json','r',encoding='utf-8').read())
assert data.get('run_id'), 'latest missing run_id'
assert data.get('data',{}).get('leaderboard_path') is not None, 'latest missing leaderboard_path'
if data.get('data',{}).get('report_path') is None:
    print('warn: latest report_path is null')
PY

echo "==> /models availability"
MODELS_RESP="$(curl -s "${API_BASE}/models")"
echo "${MODELS_RESP}" > Project/tmp/smoke_models.json

python - <<'PY'
import json
models=json.loads(open('Project/tmp/smoke_models.json','r',encoding='utf-8').read())
assert isinstance(models, list) and models, 'models list empty'
for m in models:
    if 'available' not in m:
        raise SystemExit('model missing availability flag')
print('models_ok')
PY

echo "Smoke tests OK"
