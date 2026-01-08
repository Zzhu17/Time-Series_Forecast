from __future__ import annotations

import time
import uuid
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from utils.logging_utils import log_json, setup_json_logger
from schemas.api import TaskResponse, TrainRequest
from services.request_utils import (
    clean_dataframe_for_json,
    parse_feature_cols,
    parse_residual_modeling,
    read_csv_upload,
)
from services.training_payloads import prepare_training_payload
from services.train_service import run_training_task
from jobs.tasks import get_task, submit_train_task

LOGGER = setup_json_logger()
router = APIRouter()


def _payload_or_400(payload: dict, df=None) -> dict:
    try:
        normalized, _report = prepare_training_payload(payload, df_override=df, auto_select_features=True)
        return normalized
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/train", response_model=TaskResponse)
def train(req: TrainRequest):
    trace_id = str(uuid.uuid4())
    start_ts = time.time()
    rows = [r.data for r in req.rows]
    payload = {
        "model_name": req.model_name,
        "time_col": req.time_col,
        "value_col": req.value_col,
        "horizon": req.horizon,
        "rows": rows,
        "feature_cols": req.feature_cols,
        "residual_modeling": req.residual_modeling,
        "allow_degrade": bool(getattr(req, "allow_degrade", False)),
        "device": getattr(req, "device", "cpu"),
    }
    payload = _payload_or_400(payload)
    task_id = submit_train_task(payload)
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
    df = read_csv_upload(file)
    df = clean_dataframe_for_json(df)

    fc_list = parse_feature_cols(feature_cols)
    rm_cfg = parse_residual_modeling(residual_modeling)

    payload = {
        "model_name": model_name,
        "time_col": time_col,
        "value_col": value_col,
        "horizon": int(horizon),
        "rows": df.to_dict(orient="records"),
        "feature_cols": fc_list,
        "residual_modeling": rm_cfg,
        "uploaded_name": getattr(file, "filename", None),
    }
    payload = _payload_or_400(payload, df=df)
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
    df = read_csv_upload(file)
    df = clean_dataframe_for_json(df)

    fc_list = parse_feature_cols(feature_cols)
    rm_cfg = parse_residual_modeling(residual_modeling)

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
    payload = _payload_or_400(payload, df=df)
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
    df = read_csv_upload(file)
    df = clean_dataframe_for_json(df)

    fc_list = parse_feature_cols(feature_cols)
    rm_cfg = parse_residual_modeling(residual_modeling)

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
    payload = _payload_or_400(payload, df=df)
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
