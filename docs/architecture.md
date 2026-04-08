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
3) Trainer dispatches by model registry key (`TRAINER_REGISTRY` / `FORECASTER_REGISTRY`).
4) Model trains or infers, returning predictions, metrics, and normalized `training_params`.
5) Training quality gate evaluates degraded flag + NRMSE threshold before model registration.
6) Artifacts and trace metadata (`training_params.json`, gate result) are persisted under `Project/artifacts/runs/<run_id>/`.

## Components
- API: `Project/api`
- UI (Streamlit): `Project/app.py`
- React UI: `Project/frontend`
- Training pipeline: `Project/training`
- Feature pipeline: `Project/utils`

## 当前训练主链路

当前训练执行路径如下（`Project/training/train.py` 已弃用，仅作兼容占位）：

1. `services.train_service.run_training_task`：接收训练任务并标准化上下文。
2. `services.pipeline.run_train_predict_pipeline`：执行统一 train/predict pipeline。
3. `models.registry.TRAINER_REGISTRY`：按模型键选择 trainer/adaptor 实现。
