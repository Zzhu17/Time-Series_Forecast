from __future__ import annotations

import time
import uuid
from typing import Any, List, Optional
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

from utils.schemas import PipelineRunModel
from server.xgb_loader import XGBPredictor
from server.tasks import submit_train_task, get_task, list_tasks
from server.registry import register_model, promote_model, get_model as get_model_rec, list_models as list_models_rec, latest_production
from server.logging_utils import setup_json_logger, log_json
from services.pipeline_loader import load_pipeline_module
from services.snapshot import cacheable_results
from ui.model_config import load_xgboost_hparams_from_configs_yaml

LOGGER = setup_json_logger()
app = FastAPI(title="TS Forecast API", version="0.1.0")

# CORS for React frontend (adjust origins if you want to lock down)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: serve built frontend (if present) from Universal Time-Series Forecast/dist
try:
    frontend_dist = "Universal Time-Series Forecast/dist"
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
except Exception:
    pass


# ------------------------
# Pydantic Schemas (API)
# ------------------------
class Row(BaseModel):
    __root__: dict

    @property
    def data(self) -> dict:
        return self.__root__


class PredictRequest(BaseModel):
    model_name: str = Field(..., description="Model to use (e.g., informer/lstm/xgboost/baseline).")
    time_col: str = Field(..., description="Timestamp column name in the rows.")
    value_col: str = Field(..., description="Target column name.")
    rows: List[Row] = Field(..., description="List of records (dict-like).")
    horizon: int = Field(1, description="Forecast horizon (points).")
    feature_cols: Optional[List[str]] = Field(None, description="Optional feature column ordering.")
    residual_modeling: Optional[dict] = Field(None, description="Residual options (matches app config).")

    @validator("model_name", "time_col", "value_col")
    def _non_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("must be a non-empty string")
        return v.strip()

    @validator("rows")
    def _non_empty_rows(cls, v: List[Row]):
        if not v:
            raise ValueError("rows must not be empty")
        return v

    @validator("horizon")
    def _horizon_pos(cls, v: int):
        if v <= 0:
            raise ValueError("horizon must be > 0")
        return v

    class Config:
        extra = "ignore"


class PredictResponse(BaseModel):
    status: str
    degraded: bool = False
    reason: Optional[str] = None
    predictions: List[float]
    used_model: str


class TrainRequest(PredictRequest):
    pass


class TaskResponse(BaseModel):
    id: str
    status: str
    model_name: str
    params: Optional[dict] = None
    metrics: Optional[dict] = None
    artifacts: Optional[dict] = None
    error: Optional[str] = None
    degraded: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ModelRegisterRequest(BaseModel):
    name: str
    version: Optional[str] = None
    stage: str = Field(default="candidate", description="candidate/production/archived")
    params: Optional[dict] = None
    metrics: Optional[dict] = None
    artifacts: Optional[dict] = None

    @validator("name")
    def _non_empty_name(cls, v: str):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("name must be non-empty")
        return v.strip()

    @validator("stage")
    def _valid_stage(cls, v: str):
        if v not in ("candidate", "production", "archived"):
            raise ValueError("stage must be candidate/production/archived")
        return v


class ModelResponse(BaseModel):
    id: str
    name: str
    version: Optional[str] = None
    stage: str
    params: Optional[dict] = None
    metrics: Optional[dict] = None
    artifacts: Optional[dict] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    promoted_at: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    description: str


# ------------------------
# Helpers
# ------------------------
def _baseline_predict(df: pd.DataFrame, value_col: str, horizon: int) -> List[float]:
    if value_col not in df.columns:
        raise KeyError(f"Missing target column '{value_col}' in rows.")
    y = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if len(y) == 0:
        raise ValueError("No numeric values found for target column.")
    last = float(y.iloc[-1])
    return [last for _ in range(horizon)]


def _auto_feature_cols(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    *,
    miss_thresh: float = 0.4,
    corr_thresh: float = 0.05,
) -> List[str]:
    """
    Auto-select numeric-like feature columns:
    - drop time/target
    - drop missing rate above threshold
    - drop low correlation vs target
    Fallback: single-var (target only) if none qualify.
    """
    num_cols = []
    for c in df.columns:
        if c in (time_col, value_col):
            continue
        try:
            col = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            continue
        miss = float(col.isna().mean()) if len(col) else 1.0
        if miss > miss_thresh:
            continue
        num_cols.append((c, miss, col))

    tgt = pd.to_numeric(df[value_col], errors="coerce")
    feats = []
    for name, miss, col in num_cols:
        try:
            corr = tgt.corr(col)
        except Exception:
            corr = None
        if corr is None or pd.isna(corr) or abs(float(corr)) < corr_thresh:
            continue
        feats.append(name)

    return feats if feats else [value_col]


def _numeric_like_features(df: pd.DataFrame, time_col: str, value_col: str) -> List[str]:
    """
    Collect numeric-like columns (match Streamlit auto feature behavior): target + all numeric-like (not time_col).
    """
    out = [value_col]
    for c in df.columns:
        if c in (time_col, value_col):
            continue
        try:
            col = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            continue
        notna = float(col.notna().mean()) if len(col) else 0.0
        if notna > 0.01:  # at least some numeric content
            out.append(c)
    return out


def _build_streamlit_like_config(
    *,
    df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
    residual_cfg: dict | None,
    device_choice: str,
    time_col: str,
    value_col: str,
):
    art_dir = Path(__file__).resolve().parents[1] / "artifacts"
    config = {
        "model": {"name": model_name},
        "default": {
            "time_col": time_col,
            "value_col": value_col,
            "device": device_choice,
            "dtype": "float32",
        },
        "visualization": {
            "pipeline_plot": False,
            "build_continuous": False,
        },
        "target_transform": {
            "enabled": model_name.lower() in ("informer", "lstm"),
            "method": "log1p",
        },
        "post_calibration": {
            "enabled": True,
            "a_clip": [0.8, 1.2],
            "b_clip_ratio": 0.1,
            "ridge": 1e-6,
            "mape_guard_rel": 1.02,
        },
        "model_config": {
            "Informer": {
                "seq_len": 96,
                "label_len": 48,
                "pred_len": 8,
                "auto_feature_cols": True,
                "lock_feature_order": True,
                "feature_cols": feature_cols,
                "feature_selection": {
                    "missing_rate_threshold": 0.4,
                    "low_variance_threshold": 1e-8,
                    "redundant_corr_threshold": 0.95,
                    "max_features": None,
                    "leakage_name_patterns": ["label", "target", "future", "t+", "lead", "yhat", "predict"],
                    "safe_default_cols": ["month", "day_of_month", "day_of_week", "hour", "day_of_year"],
                    "required_core_cols": [],
                    "repairable_core_cols": ["month", "day_of_month", "day_of_week", "hour", "day_of_year"],
                    "core_cols": [],
                },
            }
        },
        "artifacts": {
            "model_path": str(art_dir / "informer_model.pth"),
            "scaler_path": str(art_dir / "scaler.pkl"),
            "residual_model_path": str(art_dir / "residual_model.pkl"),
            "y_scaler_path": str(art_dir / "value_scaler.pkl"),
            "feature_cols_path": str(art_dir / "feature_cols.json"),
            "feature_report_path": str(art_dir / "feature_report.json"),
        },
        "callbacks": {},
        "device": device_choice,
        "data": {"dataframe": df.copy()},
        "model_type": model_name,
    }

    # Residual modeling
    if residual_cfg:
        config["residual_modeling"] = residual_cfg
        if model_name.lower() == "informer":
            config.setdefault("model_config", {}).setdefault("Informer", {})["use_residual"] = False
            config.setdefault("post_calibration", {})["enabled"] = False

    # XGBoost hyperparameters and artifact paths
    res_choice = residual_cfg.get("model_type") if isinstance(residual_cfg, dict) else None
    if model_name.lower() == "xgboost" or str(res_choice).lower() == "xgboost":
        try:
            xgb_hp = load_xgboost_hparams_from_configs_yaml()
            if isinstance(xgb_hp, dict) and xgb_hp:
                config.setdefault("model_config", {})["XGBoost"] = xgb_hp
        except Exception:
            pass
    try:
        if model_name.lower() == "xgboost":
            config.setdefault("artifacts", {})["xgboost_model_path"] = str(art_dir / "xgboost_model.json")
    except Exception:
        pass
    try:
        if str(res_choice).lower() == "xgboost":
            config.setdefault("artifacts", {})["xgboost_residual_model_path"] = str(art_dir / "xgboost_residual_model.json")
    except Exception:
        pass

    return config


# ------------------------
# Routes
# ------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models", response_model=List[ModelInfo])
def list_models():
    return [
        ModelInfo(name="baseline", description="Naive last-value persistence."),
        ModelInfo(name="informer", description="Heavy model; requires artifacts (not loaded by default)."),
        ModelInfo(name="lstm", description="Heavy model; requires artifacts (not loaded by default)."),
        ModelInfo(name="xgboost", description="Requires trained artifacts."),
    ]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    trace_id = str(uuid.uuid4())
    req_start = time.time()
    # Validate minimal pipeline schema; reuses existing schema class.
    PipelineRunModel(
        time_col=req.time_col,
        value_col=req.value_col,
        model_name=req.model_name,
        feature_cols=req.feature_cols or [],
        residual_modeling=req.residual_modeling,
    )

    # Convert rows to DataFrame
    data_rows = [r.data for r in req.rows]
    df = pd.DataFrame(data_rows)

    # For now: provide a safe baseline predictor; heavy models require artifact loading (not implemented here).
    model = req.model_name.lower()
    if model == "baseline":
        preds = _baseline_predict(df, req.value_col, req.horizon)
        log_json(
            LOGGER,
            "predict",
            trace_id=trace_id,
            model=model,
            degraded=False,
            duration_ms=int((time.time() - req_start) * 1000),
        )
        return PredictResponse(status="ok", degraded=False, predictions=preds, used_model="baseline")

    if model == "xgboost":
        # Expect artifacts at default locations; degrade to baseline if missing
        model_path = "Project/artifacts/xgboost_model.json"
        contract_path = "Project/artifacts/feature_cols.json"
        target_transform = None  # could be loaded from artifacts if saved
        try:
            predictor = XGBPredictor(
                model_path=model_path,
                feature_contract_path=contract_path,
                target_transform=target_transform,
                time_col=req.time_col,
                value_col=req.value_col,
            )
            preds, meta, degraded, reason = predictor.predict(df, horizon=req.horizon)
            resp = PredictResponse(
                status="ok",
                degraded=bool(degraded),
                reason=reason,
                predictions=preds.tolist(),
                used_model="xgboost",
            )
            log_json(
                LOGGER,
                "predict",
                trace_id=trace_id,
                model="xgboost",
                degraded=bool(degraded),
                reason=reason,
                duration_ms=int((time.time() - req_start) * 1000),
            )
            return resp
        except Exception as e:
            # fallback baseline
            try:
                preds = _baseline_predict(df, req.value_col, req.horizon)
                resp = PredictResponse(
                    status="ok",
                    degraded=True,
                    reason=f"xgboost failed: {e}; baseline fallback",
                    predictions=preds,
                    used_model="xgboost->baseline",
                )
                log_json(
                    LOGGER,
                    "predict",
                    trace_id=trace_id,
                    model="xgboost",
                    degraded=True,
                    reason=str(e),
                    duration_ms=int((time.time() - req_start) * 1000),
                )
                return resp
            except Exception as inner:
                raise HTTPException(status_code=400, detail=f"xgboost failed: {e}; baseline failed: {inner}") from inner

    # Other heavy models not wired yet -> degrade with baseline
    try:
        preds = _baseline_predict(df, req.value_col, req.horizon)
        resp = PredictResponse(
            status="ok",
            degraded=True,
            reason="heavy model loading not implemented; returned baseline",
            predictions=preds,
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


@app.post("/train", response_model=TaskResponse)
def train(req: TrainRequest):
    trace_id = str(uuid.uuid4())
    start_ts = time.time()
    # Reuse schema validation
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


@app.post("/train_file", response_model=TaskResponse)
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
    import json as _json

    # Read CSV into rows
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to read csv: {e}") from e

    # Validate required columns
    missing_cols = [c for c in (time_col, value_col) if c not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"CSV missing columns: {missing_cols}")

    # Replace non-finite values with None for JSON safety
    df = df.replace([pd.NA, pd.NaT, float("inf"), float("-inf")], pd.NA)
    df = df.where(pd.notna(df), None)

    fc_list = None
    if feature_cols:
        try:
            fc_list = _json.loads(feature_cols)
        except Exception:
            fc_list = [c.strip() for c in feature_cols.split(",") if c.strip()]
    rm_cfg = None
    if residual_modeling:
        try:
            rm_cfg = _json.loads(residual_modeling)
        except Exception:
            rm_cfg = None

    payload = {
        "model_name": model_name,
        "time_col": time_col,
        "value_col": value_col,
        "horizon": int(horizon),
        "rows": df.to_dict(orient="records"),
        "feature_cols": fc_list,
        "residual_modeling": rm_cfg,
    }
    # Reuse validation
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


@app.post("/train_file_sync")
async def train_file_sync(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    time_col: str = Form(...),
    value_col: str = Form(...),
    horizon: int = Form(24),
    feature_cols: Optional[str] = Form(None),
    residual_modeling: Optional[str] = Form(None),
    allow_degrade: bool = Form(False),
):
    """
    Synchronous train+predict endpoint:
    - Reads CSV
    - Auto-selects numeric feature columns (filters missing/correlation) when feature_cols not provided
    - Runs pipeline and returns metrics + plot_data
    """
    import json as _json

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to read csv: {e}") from e

    missing_cols = [c for c in (time_col, value_col) if c not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"CSV missing columns: {missing_cols}")

    # Clean for downstream JSON safety
    df = df.replace([pd.NA, pd.NaT, float("inf"), float("-inf")], pd.NA)
    df = df.where(pd.notna(df), None)

    fc_list = None
    if feature_cols:
        try:
            fc_list = _json.loads(feature_cols)
        except Exception:
            fc_list = [c.strip() for c in feature_cols.split(",") if c.strip()]
    rm_cfg = None
    if residual_modeling:
        try:
            rm_cfg = _json.loads(residual_modeling)
        except Exception:
            rm_cfg = None

    if not fc_list:
        fc_list = _auto_feature_cols(df.copy(), time_col, value_col)

    # Validate basic schema
    PipelineRunModel(
        time_col=time_col,
        value_col=value_col,
        model_name=model_name,
        feature_cols=fc_list or [],
        residual_modeling=rm_cfg,
    )

    art_dir = Path(__file__).resolve().parents[1] / "artifacts"
    config = {
        "model": {"name": model_name},
        "model_type": model_name,
        "default": {
            "time_col": time_col,
            "value_col": value_col,
            "device": "cpu",
            "dtype": "float32",
        },
        "visualization": {"pipeline_plot": False, "build_continuous": False},
        "prediction": {"rolling": {"enabled": False}},
        "data": {"dataframe": df.copy()},
        "callbacks": {},
        "artifacts": {
            "model_path": str(art_dir / "model.pth"),
            "scaler_path": str(art_dir / "scaler.pkl"),
            "residual_model_path": str(art_dir / "residual_model.pkl"),
            "y_scaler_path": str(art_dir / "value_scaler.pkl"),
            "feature_cols_path": str(art_dir / "feature_cols.json"),
            "feature_report_path": str(art_dir / "feature_report.json"),
        },
    }

    try:
        pipeline_mod = load_pipeline_module()
        results = pipeline_mod.run_pipeline_and_update_state(
            df=df.copy(),
            config=config,
            feature_cols=fc_list,
            uploaded_name=getattr(file, "filename", None),
            model_name=model_name,
            time_col=time_col,
            value_col=value_col,
            allow_degrade=bool(allow_degrade),
            progress_cb=None,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if results.get("status") not in ("ok", "success"):
        raise HTTPException(status_code=400, detail=results.get("message", "pipeline failed"))

    data_blob = results.get("data", {}) or {}
    return {
        "status": "ok",
        "metrics": results.get("metrics", {}),
        "plot_data": data_blob.get("plot_data"),
        "split": data_blob.get("split"),
        "feature_cols": fc_list,
        "degraded": bool(data_blob.get("degraded", False)),
        "degraded_reason": data_blob.get("degraded_reason"),
        "task_model": model_name,
    }


@app.post("/train_file_streamlit")
async def train_file_streamlit(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    time_col: str = Form(...),
    value_col: str = Form(...),
    horizon: int = Form(24),
    feature_cols: Optional[str] = Form(None),
    residual_modeling: Optional[str] = Form(None),
    device: str = Form("cpu"),
    allow_degrade: bool = Form(False),
):
    """
    Streamlit-like sync endpoint: builds the same config as app.py, runs pipeline, and returns cacheable_results
    (metrics + plot_data + degraded flags) for frontend rendering.
    """
    import json as _json

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to read csv: {e}") from e

    missing_cols = [c for c in (time_col, value_col) if c not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"CSV missing columns: {missing_cols}")

    df = df.replace([pd.NA, pd.NaT, float("inf"), float("-inf")], pd.NA)
    df = df.where(pd.notna(df), None)

    fc_list = None
    if feature_cols:
        try:
            fc_list = _json.loads(feature_cols)
        except Exception:
            fc_list = [c.strip() for c in feature_cols.split(",") if c.strip()]
    rm_cfg = None
    if residual_modeling:
        try:
            rm_cfg = _json.loads(residual_modeling)
        except Exception:
            rm_cfg = None
    if not fc_list:
        fc_list = _numeric_like_features(df.copy(), time_col, value_col)

    PipelineRunModel(
        time_col=time_col,
        value_col=value_col,
        model_name=model_name,
        feature_cols=fc_list or [],
        residual_modeling=rm_cfg,
    )

    config = _build_streamlit_like_config(
        df=df.copy(),
        feature_cols=fc_list,
        model_name=model_name,
        residual_cfg=rm_cfg,
        device_choice=device,
        time_col=time_col,
        value_col=value_col,
    )

    try:
        pipeline_mod = load_pipeline_module()
        results = pipeline_mod.run_pipeline_and_update_state(
            df=df.copy(),
            config=config,
            feature_cols=fc_list,
            uploaded_name=getattr(file, "filename", None),
            model_name=model_name,
            time_col=time_col,
            value_col=value_col,
            allow_degrade=bool(allow_degrade),
            progress_cb=None,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if results.get("status") not in ("ok", "success"):
        raise HTTPException(status_code=400, detail=results.get("message", "pipeline failed"))

    payload = cacheable_results(results)
    # Add convenience fields for frontend
    payload["feature_cols"] = fc_list
    payload["task_model"] = model_name
    return payload


@app.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task_status(task_id: str):
    rec = get_task(task_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="task not found")
    return rec


@app.get("/tasks", response_model=List[TaskResponse])
def get_task_list(limit: int = 20, offset: int = 0):
    return list_tasks(limit=limit, offset=offset)


@app.post("/models/register", response_model=ModelResponse)
def register_model_api(req: ModelRegisterRequest):
    rec = register_model(
        name=req.name,
        version=req.version,
        stage=req.stage,
        params=req.params,
        metrics=req.metrics,
        artifacts=req.artifacts,
    )
    return rec


@app.post("/models/{model_id}/promote", response_model=ModelResponse)
def promote_model_api(model_id: str, stage: str = "production"):
    rec = promote_model(model_id, stage=stage)
    if rec is None:
        raise HTTPException(status_code=404, detail="model not found")
    return rec


@app.get("/models/registry", response_model=List[ModelResponse])
def list_models_api(limit: int = 50, offset: int = 0):
    return list_models_rec(limit=limit, offset=offset)


@app.get("/models/production", response_model=ModelResponse)
def get_latest_production_model():
    rec = latest_production()
    if rec is None:
        raise HTTPException(status_code=404, detail="no production model")
    return rec
