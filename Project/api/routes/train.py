from __future__ import annotations

import time
import uuid
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from utils.logging_utils import log_json, setup_json_logger
from schemas.api import TaskResponse, TrainRequest
from services.request_utils import (
    auto_feature_cols,
    clean_dataframe_for_json,
    ensure_required_columns,
    ensure_target_in_features,
    parse_feature_cols,
    parse_residual_modeling,
    read_csv_upload,
)
from services.train_service import run_training_task
from jobs.tasks import get_task, submit_train_task
from utils.schemas import PipelineRunModel

LOGGER = setup_json_logger()
router = APIRouter()


def _ensure_model_selected(model_name: str) -> str:
    if not isinstance(model_name, str) or not model_name.strip():
        raise HTTPException(status_code=400, detail="请选择 model")
    cleaned = model_name.strip()
    if cleaned.lower() in ("none", "null"):
        raise HTTPException(status_code=400, detail="请选择 model")
    return cleaned


@router.post("/train", response_model=TaskResponse)
def train(req: TrainRequest):
    _ensure_model_selected(req.model_name)
    trace_id = str(uuid.uuid4())
    start_ts = time.time()
    PipelineRunModel(
        time_col=req.time_col,
        value_col=req.value_col,
        model_name=req.model_name,
        feature_cols=req.feature_cols or [],
        residual_modeling=req.residual_modeling,
    )
    task_id = submit_train_task(req.dict())
    rec = get_task(task_id)
    if rec is None:
        raise HTTPException(status_code=500, detail="failed to create task")
    log_json(
        LOGGER,
        "train_task_created",
        trace_id=trace_id,
        task_id=task_id,
        model=req.model_name,
        duration_ms=int((time.time() - start_ts) * 1000),
    )
    return rec


@router.post("/train_file", response_model=TaskResponse)
async def train_from_file(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    time_col: str = Form(...),
    value_col: str = Form(...),
    horizon: int = Form(1),
    feature_cols: Optional[str] = Form(None),
    residual_modeling: Optional[str] = Form(None),
):
    """
    Accept a CSV upload and create a training task. feature_cols/residual_modeling are JSON strings or comma lists.
    """
    model_name = _ensure_model_selected(model_name)
    df = read_csv_upload(file)
    ensure_required_columns(df, time_col, value_col)
    df = clean_dataframe_for_json(df)

    fc_list = parse_feature_cols(feature_cols)
    rm_cfg = parse_residual_modeling(residual_modeling)

    fc_list = ensure_target_in_features(fc_list, value_col)
    payload = {
        "model_name": model_name,
        "time_col": time_col,
        "value_col": value_col,
        "horizon": int(horizon),
        "rows": df.to_dict(orient="records"),
        "feature_cols": fc_list,
        "residual_modeling": rm_cfg,
    }
    PipelineRunModel(
        time_col=time_col,
        value_col=value_col,
        model_name=model_name,
        feature_cols=fc_list or [],
        residual_modeling=rm_cfg,
    )
    task_id = submit_train_task(payload)
    rec = get_task(task_id)
    if rec is None:
        raise HTTPException(status_code=500, detail="failed to create task")
    return rec


@router.post("/train_file_sync")
async def train_file_sync(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    time_col: str = Form(...),
    value_col: str = Form(...),
    horizon: int = Form(24),
    feature_cols: Optional[str] = Form(None),
    residual_modeling: Optional[str] = Form(None),
    allow_degrade: bool = Form(False),
    device: str = Form("cpu"),
):
    """
    Synchronous train+predict endpoint:
    - Reads CSV
    - Auto-selects numeric feature columns (filters missing/correlation) when feature_cols not provided
    - Runs pipeline and returns metrics + plot_data
    """
    model_name = _ensure_model_selected(model_name)
    df = read_csv_upload(file)
    ensure_required_columns(df, time_col, value_col)
    df = clean_dataframe_for_json(df)

    fc_list = parse_feature_cols(feature_cols)
    rm_cfg = parse_residual_modeling(residual_modeling)

    if not fc_list:
        fc_list = auto_feature_cols(df.copy(), time_col, value_col)
    fc_list = ensure_target_in_features(fc_list, value_col)

    PipelineRunModel(
        time_col=time_col,
        value_col=value_col,
        model_name=model_name,
        feature_cols=fc_list or [],
        residual_modeling=rm_cfg,
    )

    payload = {
        "model_name": model_name,
        "time_col": time_col,
        "value_col": value_col,
        "horizon": int(horizon),
        "rows": df.to_dict(orient="records"),
        "feature_cols": fc_list,
        "residual_modeling": rm_cfg,
        "allow_degrade": bool(allow_degrade),
        "device": device,
        "uploaded_name": getattr(file, "filename", None),
    }
    task_id = str(uuid.uuid4())
    try:
        result = run_training_task(payload, task_id=task_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    payload_out = result.get("cacheable_results", {}) if isinstance(result, dict) else {}
    if not isinstance(payload_out, dict):
        payload_out = {}
    payload_out["feature_cols"] = fc_list
    payload_out["task_model"] = model_name
    payload_out["model_record"] = result.get("model_record") if isinstance(result, dict) else None
    return payload_out


@router.post("/train_file_streamlit")
async def train_file_streamlit(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    time_col: str = Form(...),
    value_col: str = Form(...),
    horizon: int = Form(24),
    feature_cols: Optional[str] = Form(None),
    residual_modeling: Optional[str] = Form(None),
    allow_degrade: bool = Form(False),
    device: str = Form("cpu"),
):
    """
    Streamlit-like sync endpoint: builds the same config as app.py, runs pipeline, and returns cacheable_results
    (metrics + plot_data + degraded flags) for frontend rendering.
    """
    model_name = _ensure_model_selected(model_name)
    df = read_csv_upload(file)
    ensure_required_columns(df, time_col, value_col)
    df = clean_dataframe_for_json(df)

    fc_list = parse_feature_cols(feature_cols)
    rm_cfg = parse_residual_modeling(residual_modeling)
    if not fc_list:
        fc_list = auto_feature_cols(df.copy(), time_col, value_col)
    fc_list = ensure_target_in_features(fc_list, value_col)

    PipelineRunModel(
        time_col=time_col,
        value_col=value_col,
        model_name=model_name,
        feature_cols=fc_list or [],
        residual_modeling=rm_cfg,
    )

    payload = {
        "model_name": model_name,
        "time_col": time_col,
        "value_col": value_col,
        "horizon": int(horizon),
        "rows": df.to_dict(orient="records"),
        "feature_cols": fc_list,
        "residual_modeling": rm_cfg,
        "allow_degrade": bool(allow_degrade),
        "device": device,
        "uploaded_name": getattr(file, "filename", None),
    }
    task_id = str(uuid.uuid4())
    try:
        result = run_training_task(payload, task_id=task_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    payload_out = result.get("cacheable_results", {}) if isinstance(result, dict) else {}
    if not isinstance(payload_out, dict):
        payload_out = {}
    payload_out["feature_cols"] = fc_list
    payload_out["task_model"] = model_name
    payload_out["model_record"] = result.get("model_record") if isinstance(result, dict) else None
    return payload_out
