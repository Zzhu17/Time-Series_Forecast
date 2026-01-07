from __future__ import annotations

import time
import uuid

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from utils.logging_utils import log_json, setup_json_logger
from services.predict_utils import baseline_predict, predict_with_xgboost
from schemas.api import PredictRequest, PredictResponse
from services.predict_service import predict_from_registry
from services.online_predict_service import run_online_predict
from services.request_utils import clean_dataframe_for_json, ensure_required_columns, read_csv_upload
from utils.schemas import PipelineRunModel

LOGGER = setup_json_logger()
router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    trace_id = str(uuid.uuid4())
    req_start = time.time()
    PipelineRunModel(
        time_col=req.time_col,
        value_col=req.value_col,
        model_name=req.model_name,
        feature_cols=req.feature_cols or [],
        residual_modeling=req.residual_modeling,
    )

    data_rows = [r.data for r in req.rows]
    df = pd.DataFrame(data_rows)

    model = req.model_name.lower()
    model_id = getattr(req, "model_id", None)
    model_version = getattr(req, "model_version", None)
    allow_degrade = bool(getattr(req, "allow_degrade", False))
    if model == "baseline":
        preds = baseline_predict(df, req.value_col, req.horizon)
        log_json(
            LOGGER,
            "predict",
            trace_id=trace_id,
            model=model,
            degraded=False,
            duration_ms=int((time.time() - req_start) * 1000),
        )
        return PredictResponse(status="ok", degraded=False, predictions=preds.tolist(), used_model="baseline")

    # Prefer registry-backed prediction when model_id/version provided or when model has registry artifacts.
    if model_id or model_version or model in ("xgboost", "informer", "randomforest", "lstm", "arima", "prophet"):
        try:
            preds, degraded, used_model, reason = predict_from_registry(
                df=df,
                model_name=model,
                horizon=req.horizon,
                time_col=req.time_col,
                value_col=req.value_col,
                allow_degrade=allow_degrade,
                model_id=model_id,
                model_version=model_version,
            )
        except Exception as e:
            if model_id or model_version:
                raise HTTPException(status_code=404, detail=str(e)) from e
            # fallback to default xgboost artifacts if registry not found
            if model == "xgboost":
                try:
                    preds, degraded, used_model, reason = predict_with_xgboost(
                        df,
                        time_col=req.time_col,
                        value_col=req.value_col,
                        horizon=req.horizon,
                        baseline_fallback=True,
                    )
                except Exception as inner:
                    raise HTTPException(status_code=400, detail=str(inner)) from inner
            else:
                # fallback baseline for unsupported models
                preds = baseline_predict(df, req.value_col, req.horizon)
                degraded = True
                used_model = f"{model}->baseline"
                reason = "model_not_available"

        resp = PredictResponse(
            status="ok",
            degraded=degraded,
            reason=reason or None,
            predictions=preds.tolist(),
            used_model=used_model,
        )
        log_json(
            LOGGER,
            "predict",
            trace_id=trace_id,
            model=used_model,
            degraded=degraded,
            reason=reason,
            duration_ms=int((time.time() - req_start) * 1000),
        )
        return resp

    try:
        preds = baseline_predict(df, req.value_col, req.horizon)
        resp = PredictResponse(
            status="ok",
            degraded=True,
            reason="heavy model loading not implemented; returned baseline",
            predictions=preds.tolist(),
            used_model=f"{model}->baseline",
        )
        log_json(
            LOGGER,
            "predict",
            trace_id=trace_id,
            model=model,
            degraded=True,
            reason=f"{model} not wired",
            duration_ms=int((time.time() - req_start) * 1000),
        )
        return resp
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


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

    return {
        "status": "ok",
        "predictions": merged.astype(float).tolist(),
        "degraded": bool(dblk.get("degraded", False)) if isinstance(dblk, dict) else False,
        "degraded_reason": dblk.get("degraded_reason") if isinstance(dblk, dict) else None,
        "degraded_mode": dblk.get("degraded_mode") if isinstance(dblk, dict) else None,
        "missing_required_core": dblk.get("missing_required_core") if isinstance(dblk, dict) else None,
        "dropped_optional_features": dblk.get("dropped_optional_features") if isinstance(dblk, dict) else None,
    }
