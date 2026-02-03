# Architecture

## Overview
The system is a time-series forecasting platform with:
- Streamlit UI for exploration
- FastAPI backend for inference and training
- Optional Celery + Redis for async training
- Model artifacts stored under `Project/artifacts/`

## Data Flow
1) User uploads CSV or sends JSON rows.
2) Preprocessing builds features (lags, rolling, time features).
3) Model trains or infers, returning predictions and metrics.

## Components
- API: `Project/api`
- UI (Streamlit): `Project/app.py`
- React UI: `Project/frontend`
- Training pipeline: `Project/training`
- Feature pipeline: `Project/utils`
