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
- CI runs on GitHub Actions (`.github/workflows/ci.yml`) with the same command set.
- Lint/type check (optional locally): `ruff check .` and `PYTHONPATH=Project mypy Project tests` (configured in `pyproject.toml`)

## API (FastAPI)

- Launch API locally:
  ```bash
  PYTHONPATH=Project uvicorn api.app:app --reload --port 8000
  ```
- If a built frontend exists at `Project/Universal Time-Series Forecast/dist`, it is served at `/ui` to avoid shadowing API routes.
- Optional auth: set `TSF_API_TOKEN` (or `API_TOKEN`) in the API environment; Streamlit will send it if provided in the sidebar.
- Endpoints:
  - `GET /health`
  - `GET /models` (currently lists available model keys)
  - `POST /predict_online_file` (Informer online rolling inference; accepts CSV upload)
  - `POST /predict` (supports registry-based `model_id`/`model_version` for Informer/XGBoost; baseline persistence fallback is used if no artifacts)
  - `POST /train` (creates async task; uses SQLite/`DATABASE_URL` for task metadata; currently uses XGBoost if artifacts exist else baseline)
  - `GET /tasks/{id}`, `GET /tasks` (task status)
  - `POST /models/register`, `POST /models/{id}/promote`, `GET /models/registry`, `GET /models/production`
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
  - Default: SQLite at `Project/output/tasks.db`
  - Override: set `DATABASE_URL` (e.g., `postgresql://user:pass@host:5432/dbname`) and ensure driver installed (`psycopg[binary]` for Postgres)
- Logging: API/tasks emit JSON logs (logger name `ts-forecast`) with trace_id/task_id/duration where applicable.

## Docker (recommended)

```bash
cp .env.example .env
docker compose up --build
```

API: `http://localhost:8000`  
UI: `http://localhost:8501`

### Optional: completely remove large files from git history

If you want to shrink the remote repository (history rewrite required):

```bash
pip install git-filter-repo
git filter-repo --path Project/artifacts/ --invert-paths
git push --force --all
git push --force --tags
```

Alternatively, use Git LFS for `*.pth` / `*.pkl` if you need to version model artifacts.
