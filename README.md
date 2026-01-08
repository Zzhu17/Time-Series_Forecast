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

## Repo hygiene (GitHub-friendly)

This repo generates training outputs locally (models, scalers, plots, snapshots). They are intentionally ignored by git:

- `Project/artifacts/`
- `Project/output/`

If you previously committed large files (e.g. `Project/artifacts/informer_model.pth`), they were removed from tracking and added to `.gitignore`.

## Development

- Install lightweight dev tools: `pip install -r requirements-dev.txt`
- Run fast tests (no heavy deps required): `PYTHONPATH=Project pytest -q tests`
- CI runs on GitHub Actions (`.github/workflows/ci.yml`) using `requirements-ci.txt`.
- Lint/type check (optional locally): `ruff check .` and `PYTHONPATH=Project mypy Project tests` (configured in `pyproject.toml`)

## Versioning

- SemVer is tracked in `VERSION`.
- Release tags must match the file (e.g., `v0.3.0`).

## API (FastAPI)

- Launch API locally:
  ```bash
  PYTHONPATH=Project uvicorn api.app:app --reload --port 8000
  ```
- If a built frontend exists at `Project/Universal Time-Series Forecast/dist`, it is served at `/ui` to avoid shadowing API routes.
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
- Async tasks: set `CELERY_ENABLED=1` plus Redis URLs (`CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`) to enqueue training via Celery.
  - Docker default uses `redis://redis:6379/0`; local runs should use `redis://localhost:6379/0`.

## Docker (recommended)

```bash
cp .env.example .env
docker compose up --build
```

This starts API + UI + Redis + a Celery worker (async training queue).
To disable the queue and run training in-process, set `CELERY_ENABLED=0` in your env file.

API: `http://localhost:8000`  
UI: `http://localhost:8501`

### Staging / Prod (compose overrides)

```bash
# staging
docker compose --env-file .env.staging -f docker-compose.yml -f docker-compose.staging.yml up --build

# prod
docker compose --env-file .env.prod -f docker-compose.yml -f docker-compose.prod.yml up --build
```

Staging ports: API `8001`, UI `8502`, Redis `6380`  
Prod ports: API `8002`, UI `8503`, Redis `6381`

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
