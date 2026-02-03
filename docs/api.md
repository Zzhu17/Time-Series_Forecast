# API

Base URL: `http://localhost:8000`

## Health
- `GET /health`
- `GET /health/live`
- `GET /health/ready`

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
