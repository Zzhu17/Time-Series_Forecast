from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from utils.logging_utils import log_json, setup_json_logger
from schemas.api import PredictRequest, PredictResponse
from services.predict_service import PredictionNotFoundError, run_prediction
from services.online_predict_service import run_online_predict
from services.request_utils import clean_dataframe_for_json, ensure_required_columns, read_csv_upload
from utils.metrics import observe_predict

LOGGER = setup_json_logger()
router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    trace_id = str(uuid.uuid4())
    req_start = time.time()
    payload = req.dict()
    payload["rows"] = [r.data for r in req.rows]

    try:
        result = run_prediction(payload)
    except PredictionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    resp = PredictResponse(
        status=result.get("status", "ok"),
        degraded=bool(result.get("degraded", False)),
        reason=result.get("reason"),
        predictions=result.get("predictions", []),
        used_model=result.get("used_model", req.model_name),
    )
    log_json(
        LOGGER,
        "predict",
        trace_id=trace_id,
        model=resp.used_model,
        degraded=resp.degraded,
        reason=resp.reason,
        duration_ms=int((time.time() - req_start) * 1000),
    )
    return resp


@router.post("/predict_online_file")
async def predict_online_file(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    time_col: str = Form(...),
    value_col: str = Form(...),
    horizon_days: int = Form(1),
    step_mode: str = Form("Block step (= horizon)"),
    allow_degrade: bool = Form(False),
    device: str = Form("cpu"),
    model_id: str = Form(""),
    model_version: str = Form(""),
):
    start_ts = time.time()
    df = read_csv_upload(file)
    ensure_required_columns(df, time_col, value_col)
    df = clean_dataframe_for_json(df)

    horizon_steps = int(24 * int(horizon_days))
    step = None if step_mode.startswith("Block") else 1

    try:
        merged, dblk = run_online_predict(
            df=df,
            model_name=model_name,
            time_col=time_col,
            value_col=value_col,
            horizon_steps=horizon_steps,
            step=step,
            allow_degrade=bool(allow_degrade),
            device=device,
            model_id=model_id or None,
            model_version=model_version or None,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        return {
            "status": "ok",
            "predictions": merged.astype(float).tolist(),
            "degraded": bool(dblk.get("degraded", False)) if isinstance(dblk, dict) else False,
            "degraded_reason": dblk.get("degraded_reason") if isinstance(dblk, dict) else None,
            "degraded_mode": dblk.get("degraded_mode") if isinstance(dblk, dict) else None,
            "missing_required_core": dblk.get("missing_required_core") if isinstance(dblk, dict) else None,
            "dropped_optional_features": dblk.get("dropped_optional_features") if isinstance(dblk, dict) else None,
        }
    finally:
        observe_predict(model=model_name, duration=time.time() - start_ts)
