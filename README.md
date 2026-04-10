# Time-Series_Forecast

Streamlit-based playground for time-series forecasting.

## Quickstart

```bash
pip install -r Project/requirements.txt
PYTHONPATH=Project uvicorn api.app:app --reload --port 8000
streamlit run Project/app.py
```

The Streamlit UI calls the FastAPI service; keep the API running at the URL shown in the sidebar (default `http://localhost:8000`).

## Config

- Base config: `Project/configs/configs.yaml`
- Optional environment overrides: `Project/configs/configs.{env}.yaml`
- Select env via `TSF_ENV` (e.g., `dev`, `staging`, `prod`)

## 当前训练主链路

当前训练入口已经切换到 service/pipeline 主链路，`Project/training/train.py` 仅保留为弃用占位：

1. `services.train_service.run_training_task`
2. `services.pipeline.run_train_predict_pipeline`
3. `models.registry.TRAINER_REGISTRY`

### PR Checklist（文档一致性）

- [ ] 若变更训练入口/调度链路，已同步更新 `README.md`、`docs/architecture.md` 与 `Project/configs/configs.yaml` 的相关说明。
- [ ] 若变更 `Project/models/registry.py`，已完成 registry 变更检查（`SUPPORTED_MODELS`/catalog 一致、capability 语义无冲突、hybrid 命名遵循 `<residual_model>+<base_model>`）。

### 模型注册命名约定（Hybrid）

- hybrid 模型统一使用小写且以 `+` 连接：`<residual_model>+<base_model>`。
- 示例：`xgboost+informer`、`xgboost+lstm`。
- `SUPPORTED_MODELS` 作为单一元数据源；catalog 信息由其派生，避免双处维护导致漂移。

## Repo hygiene (GitHub-friendly)

This repo generates training outputs locally (models, scalers, plots, snapshots). They are intentionally ignored by git:

- `Project/artifacts/`
- `Project/output/`

If you previously committed large files (e.g. `Project/artifacts/informer_model.pth`), they were removed from tracking and added to `.gitignore`.

## Development

- Install lightweight dev tools: `pip install -r requirements-dev.txt`
- 测试依赖最小集合（可导入即可运行测试收集）：`pytest`、`httpx`、`fastapi`
  - 快速校验命令：`./scripts/check_test_env.sh`
- Run fast tests (no heavy deps required): `PYTHONPATH=Project pytest -q tests`
- CI runs on GitHub Actions (`.github/workflows/ci.yml`) using `requirements-ci.txt`.
- Lint/type check (optional locally): `ruff check .` and `PYTHONPATH=Project mypy Project tests` (configured in `pyproject.toml`)
- Coverage config lives in `coverage.toml` (use `pytest --cov`).
- Pre-commit hooks: `pre-commit install` (config: `.pre-commit-config.yaml`)
- Common tasks: `make test`, `make lint`, `make format`, `make run-api`, `make run-ui`

## Docs

Local docs are in `docs/` (served by MkDocs via `mkdocs.yml`):

```bash
mkdocs serve
```

## Packaging

This repo includes minimal package metadata in `pyproject.toml` and uses `VERSION` for SemVer.

## Versioning

- SemVer is tracked in `VERSION`.
- Release tags must match the file (e.g., `v0.3.0`).

## API (FastAPI)

- Launch API locally:
  ```bash
  PYTHONPATH=Project uvicorn api.app:app --reload --port 8000
  ```
- If a built frontend exists at `Project/frontend/dist`, it is served at `/ui` to avoid shadowing API routes.
- Optional auth: set `TSF_API_TOKEN` (or `API_TOKEN`) in the API environment; Streamlit will send it if provided in the sidebar.
- Endpoints:
  - `GET /health`
  - `GET /health/live` (liveness)
  - `GET /health/ready` (readiness: DB/config/storage)
  - `GET /metrics` (Prometheus metrics)
  - `GET /models` (currently lists available model keys)
  - `POST /predict_online_file` (Informer online rolling inference; accepts CSV upload)
  - `POST /predict` (supports registry-based `model_id`/`model_version` for Informer/XGBoost; baseline persistence fallback is used if no artifacts)
- `POST /train` (creates async task; uses SQLite/`DATABASE_URL` for task metadata; currently uses XGBoost if artifacts exist else baseline)
- `POST /train_file` (CSV upload → async training task; feature_cols must exist or be recomputable)
  - `POST /train_file_sync` (CSV upload → sync train+predict; returns metrics/plot_data)
  - `POST /train_file_streamlit` (CSV upload → sync train+predict; Streamlit payload)
  - `GET /tasks/{id}`, `GET /tasks` (task status)
  - `POST /models/register`, `POST /models/{id}/promote`, `GET /models/registry`, `GET /models/production`
  - Feature contract: `feature_cols` must be present in data or recomputable (lag/rolling/time features). Missing core features return 400.
  - Example payload:
    ```json
    {
      "model_name": "baseline",
      "time_col": "date",
      "value_col": "value",
      "horizon": 3,
      "rows": [
        {"date": "2024-01-01", "value": 1.0},
        {"date": "2024-01-02", "value": 2.0}
      ]
    }
    ```
- Database for tasks:
  - Default: SQLite at `Project/output/tasks.db` (env files use `tasks_dev.db` / `tasks_staging.db` / `tasks_prod.db`)
  - Override: set `DATABASE_URL` (e.g., `postgresql://user:pass@host:5432/dbname`) and ensure driver installed (`psycopg[binary]` for Postgres)
- Logging: API/tasks emit JSON logs (logger name `ts-forecast`) with trace_id/task_id/duration where applicable.
- Metrics: see `monitoring/alerts.yaml` for basic alert rules.
- Training quality gate:
  - Every training run now performs a gate check before registry write.
  - Default threshold: `test.nrmse <= 1.0` (override via `TRAINING_GATE_MAX_NRMSE`).
  - Degraded runs or missing metrics fail the gate and are stored as `archived` instead of `candidate`.
- Traceability:
  - Every run persists `training_params.json` under `artifacts/runs/<run_id>/`.
  - Registry params include `training_params` + `quality_gate` so catalog/registry/trainer semantics stay aligned.
- Async tasks: set `CELERY_ENABLED=1` plus Redis URLs (`CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`) to enqueue training via Celery.
  - Docker default uses `redis://redis:6379/0`; local runs should use `redis://localhost:6379/0`.

## Docker (recommended)

```bash
cp .env.example .env
docker compose up --build
```

This starts API + Streamlit UI + React UI + Redis + a Celery worker (async training queue).
To disable the queue and run training in-process, set `CELERY_ENABLED=0` in your env file.

API: `http://localhost:8000`  
UI: `http://localhost:8501`
React UI (served by API): `http://localhost:8000/ui`

### Staging / Prod (compose overrides)

```bash
# staging
docker compose --env-file .env.staging -f docker-compose.yml -f docker-compose.staging.yml up --build

# prod
docker compose --env-file .env.prod -f docker-compose.yml -f docker-compose.prod.yml up --build
```

Staging ports: API `8001`, UI `8502`, Redis `6380`  
Prod ports: API `8002`, UI `8503`, Redis `6381`, React (Nginx) `8080`

## Observability Demo (Prometheus + Grafana)

Run the full ops stack:

```bash
docker compose -f docker-compose.ops.yml up --build
```

Endpoints:
- API: `http://localhost:8000`
- UI: `http://localhost:8501`
- Prometheus: `http://localhost:9090`
- Alertmanager: `http://localhost:9093`
- Grafana: `http://localhost:3000` (user: admin / pass: admin)

Grafana dashboard:
- Folder: `TS Forecast`
- Dashboard: `TS Forecast - Ops`

Prometheus alerts:
- Rules live at `monitoring/alerts.yaml`
- Alertmanager config: `monitoring/alertmanager.yml` (webhook -> `/alerts`)
- Includes `DegradeRateHigh` threshold alert based on `degrade_events_total`.

## Terraform (IaC skeleton)

Terraform starter files live in `terraform/aws`. They define provider requirements and a minimal S3 bucket placeholder.
You will need an AWS account + credentials configured (`AWS_PROFILE` or `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`) to run `terraform init/plan/apply`.

### Optional: completely remove large files from git history

If you want to shrink the remote repository (history rewrite required):

```bash
pip install git-filter-repo
git filter-repo --path Project/artifacts/ --invert-paths
git push --force --all
git push --force --tags
```

Alternatively, use Git LFS for `*.pth` / `*.pkl` if you need to version model artifacts.

## React UI (Customer)

```bash
cd Project/frontend
npm install
npm run dev
```

Set `VITE_API_URL` if the API is not on `http://localhost:8000`.

Build for production and serve from FastAPI:

```bash
cd Project/frontend
npm run build
cd ../..  # back to repo root
PYTHONPATH=Project uvicorn api.app:app --reload --port 8000
```

If you want to stay inside `Project/frontend`, use:

```bash
PYTHONPATH=.. uvicorn api.app:app --reload --port 8000
```

Open `http://localhost:8000/ui`.

### Production (Nginx static)

The prod compose file includes a dedicated Nginx container serving the React build:

```
docker compose --env-file .env.prod -f docker-compose.yml -f docker-compose.prod.yml up --build
```

Open `http://localhost:8080`.

## Design Decisions / Failure Modes

- Optional dependencies (Celery/Prophet/ARIMA/torch) are lazy-imported to keep the API running with minimal installs.
- When required features are missing, prediction can degrade to a naive baseline instead of hard-failing.
- Training artifacts are stored per run (`Project/artifacts/runs/<run_id>/`) to keep runs reproducible.
- Data preprocessing is deterministic (cleaning + optional resample + outlier clip) and produces `processed.parquet` plus `data_profile.json`.
- Drift checks compare validation/test residual distributions and surface a lightweight drift signal.
- Hybrid models (LSTM/Informer + XGBoost residual) reuse the same residual modeling hook to keep the stack explainable.
- Airflow runs use hour-level `run_id` to avoid collisions and keep partitions traceable.

Failure modes to watch:
- Missing optional deps (torch/pmdarima/prophet/xgboost) will mark models as unavailable and/or trigger fallback.
- Very long series can slow ARIMA auto-search; use fixed-order or cap `max_train_rows`.
- Feature leakage (future columns) will inflate metrics; use the feature contract rules to block them.
- Residual modeling can overfit if residuals are noisy; disable or tighten the residual config.

## CLI Demo

Run the full pipeline from a local file:

```bash
PYTHONPATH=Project python Project/cli/run_pipeline.py \
  --data Project/Data/sample_timeseries.csv \
  --model xgboost \
  --time-col date \
  --value-col value \
  --horizon 24 \
  --feature-cols auto
```

## External Demo Script

Single-command demo that starts the API, runs a sync train, and prints the latest artifacts:

```bash
bash scripts/demo_external.sh
```

Override model/horizon:

```bash
MODEL_NAME=baseline HORIZON=24 bash scripts/demo_external.sh
```

## Smoke Tests

Assuming API is already running:

```bash
bash scripts/test_smoke.sh
```

The CLI writes a summary to `Project/output/cli_last_run.json` and prints it to stdout.

## Airflow + Spark (DE Pipeline)

Data layers:
- Bronze: `Project/Data/bronze/source=<source>/dt=<YYYY-MM-DD>/`
- Silver: `Project/Data/silver/ds=<YYYY-MM-DD>/`
- Gold: `Project/Data/gold/ds=<YYYY-MM-DD>/`
- Predictions: `Project/Data/predictions/ds=<YYYY-MM-DD>/`

Spark jobs (see `Project/spark_jobs/`):
- `extract_to_bronze.py`
- `dq_check_bronze.py`
- `spark_clean_to_silver.py`
- `spark_features_to_gold.py`
- `train_and_register_model.py`
- `batch_predict_and_store.py`
- `publish_leaderboard_report.py`

Airflow DAG:
- `Project/airflow/dags/forecast_pipeline.py` (uses env overrides like `TSF_RAW_PATH`, `TSF_MODEL_NAME`, `TSF_TIME_COL`)
