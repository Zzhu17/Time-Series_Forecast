# API

Base URL: `http://localhost:8000`

## Health
- `GET /health`
- `GET /health/live`
- `GET /health/ready`

## Observability
- `GET /metrics`
- `GET /metrics/degrade_metric`
- `GET /metrics/degrade_summary?window_minutes=60&limit=5`

## Models
- `GET /models`
- `GET /models/registry`
- `GET /models/production`
- `POST /models/register`
- `POST /models/{id}/promote`

## Prediction
- `POST /predict`
- `POST /predict_online_file`

## Training
- `POST /train`
- `POST /train_file`
- `POST /train_file_sync`
- `POST /train_file_streamlit`
- `GET /tasks`
- `GET /tasks/{id}`
