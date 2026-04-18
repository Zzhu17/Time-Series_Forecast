# Time-Series_Forecast

Production-oriented time-series forecasting platform built around FastAPI, Streamlit, React, async workers, CI, and AWS-ready infrastructure.

## Overview

This repository provides:

- training and prediction pipelines for multiple forecasting models
- a FastAPI service layer for training, inference, registry, and task APIs
- Streamlit and React user interfaces
- async task execution with Celery + Redis
- Docker, monitoring, and Terraform-based deployment assets

## Quick Start

### Local API + Streamlit

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r Project/requirements.txt
cp env/.env.dev.example .env
PYTHONPATH=Project uvicorn api.app:app --reload --port 8000
streamlit run Project/app.py
```

### Common Commands

```bash
make test
make lint
make format
make run-api
make run-ui
make docker-up
```

### Docker

```bash
cp env/.env.dev.example .env
docker compose --project-directory . -f infra/compose/docker-compose.yml up --build
```

## Documentation Map

### Project Docs

- [Documentation Index](docs/index.md)
- [Architecture](docs/architecture.md)
- [API Reference](docs/api.md)
- [Development Guide](docs/development.md)
- [Release Runbook](docs/release-runbook.md)
- [Degrade Runbook](docs/runbook-degrade.md)
- [Test Matrix Dependencies](docs/test-matrix-deps.md)
- [CI / SLO Notes](docs/ci-slo.md)

### Repository Docs

- [Repo Docs Index](docs/repo/index.md)
- [Changelog](docs/repo/CHANGELOG.md)
- [Contributing](docs/repo/CONTRIBUTING.md)
- [Security](docs/repo/SECURITY.md)
- [Code of Conduct](docs/repo/CODE_OF_CONDUCT.md)

## Runtime Entry Points

### Application

- API: `PYTHONPATH=Project uvicorn api.app:app --reload --port 8000`
- Streamlit UI: `streamlit run Project/app.py`
- React UI (dev): `cd Project/frontend && npm install && npm run dev`

### Compose Environments

- Base stack: `infra/compose/docker-compose.yml`
- Staging override: `infra/compose/docker-compose.staging.yml`
- Production override: `infra/compose/docker-compose.prod.yml`
- Ops stack: `infra/compose/docker-compose.ops.yml`

## Security Defaults

- `staging` / `prod` now require `TSF_API_TOKEN`
- CORS is environment-driven through `TSF_CORS_ALLOW_ORIGINS`
- raw client IP logging is disabled by default
- the ops stack requires `GF_SECURITY_ADMIN_PASSWORD`
- Terraform admin CIDRs must be explicitly set and cannot be `0.0.0.0/0`
- Terraform now expects a pre-created SSM SecureString for the API token instead of storing the token in Terraform state

### Infrastructure Assets

- Environment templates: `env/`
- Dockerfiles and nginx config: `infra/docker/`
- Monitoring config: `infra/monitoring/`
- Terraform: `terraform/aws/`

## Repository Structure

```text
Project/           application code
tests/             automated tests
scripts/           repo and CI helper scripts
docs/              project and repository documentation
env/               checked-in environment templates
infra/compose/     Docker Compose entry points
infra/docker/      Dockerfiles and nginx config
infra/monitoring/  Prometheus, Alertmanager, Grafana provisioning
terraform/aws/     AWS infrastructure baseline
.github/           GitHub workflow and community files
```

## Current Training Path

The main training flow is:

1. `services.train_service.run_training_task`
2. `services.pipeline.run_train_predict_pipeline`
3. `models.registry.TRAINER_REGISTRY`

`Project/training/train.py` remains as a legacy placeholder, not the primary orchestration entry.

## Development Notes

- Python baseline: `3.10+`
- Lightweight dev dependencies: `requirements-dev.txt`
- CI dependencies: `requirements-ci.txt`
- Config base: `Project/configs/configs.yaml`
- Optional env overrides: `Project/configs/configs.{env}.yaml`
- Active environment: `TSF_ENV`

## Release / Deployment Notes

- Versioning source: `VERSION`
- Python project metadata: `pyproject.toml`
- MkDocs config: `mkdocs.yml`
- Terraform bootstrap and production deployment details live under `docs/` and `terraform/aws/`
