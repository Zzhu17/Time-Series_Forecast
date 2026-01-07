from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from configs.config import load_yaml_config
from services.registry import get_model, latest_model_for_name
from services.predict_helpers import load_feature_cols, load_pickle, prepare_feature_frame
from utils.target_transform import inverse_transform_array


def _ensure_model_name(model_name: Any) -> str:
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("请选择 model")
    cleaned = model_name.strip()
    if cleaned.lower() in ("none", "null"):
        raise ValueError("请选择 model")
    return cleaned


def _build_predict_config(
    *,
    model_record: Dict[str, Any],
    model_name: str,
    time_col: str,
    value_col: str,
    device: str,
    allow_degrade: bool,
) -> Dict[str, Any]:
    try:
        base_cfg = load_yaml_config()
    except Exception:
        base_cfg = {}
    config = dict(base_cfg) if isinstance(base_cfg, dict) else {}
    config.setdefault("model", {})
    config["model"]["name"] = model_name
    config["model_type"] = model_name

    default_cfg = config.setdefault("default", {})
    default_cfg["time_col"] = time_col
    default_cfg["value_col"] = value_col
    default_cfg.setdefault("device", device)
    default_cfg.setdefault("dtype", "float32")
    config["device"] = device

    artifacts = model_record.get("artifacts") if isinstance(model_record, dict) else None
    if not isinstance(artifacts, dict) or not artifacts.get("model_path"):
        raise RuntimeError("Model artifacts not found for prediction.")
    config["artifacts"] = artifacts

    config.setdefault("model_config", {})
    inf_cfg = config["model_config"].setdefault("Informer", {})
    if isinstance(inf_cfg, dict):
        params = model_record.get("params") if isinstance(model_record, dict) else None
        if isinstance(params, dict):
            feature_cols = params.get("feature_cols")
            if isinstance(feature_cols, list) and feature_cols:
                inf_cfg["feature_cols"] = feature_cols
        inf_cfg.setdefault("feature_cols", [value_col])

    config.setdefault("prediction", {})
    config["prediction"].setdefault("degrade", {})["enabled"] = bool(allow_degrade)
    return config


def _baseline_rolling(df: pd.DataFrame, value_col: str) -> np.ndarray:
    y = pd.to_numeric(df[value_col], errors="coerce")
    return y.shift(1).to_numpy(dtype=float)


def _inverse_target_with_scaler(
    arr: np.ndarray,
    scaler: Any,
    feature_cols: list[str],
    value_col: str,
) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if scaler is None or not hasattr(scaler, "inverse_transform"):
        return arr
    try:
        idx = feature_cols.index(value_col)
    except Exception:
        idx = 0
    wide = np.zeros((len(arr), len(feature_cols)), dtype=np.float32)
    wide[:, idx] = arr
    try:
        inv = scaler.inverse_transform(wide)
        return np.asarray(inv[:, idx], dtype=np.float32).reshape(-1)
    except Exception:
        return arr


def _load_lstm_model(model_path: str):
    try:
        import torch
        from models.lstm import lstm_model
    except Exception as e:
        raise RuntimeError(f"lstm dependencies missing: {e}") from e

    state = torch.load(model_path, map_location="cpu")
    w = state.get("lstm.weight_ih_l0")
    if w is None:
        raise ValueError("lstm state_dict missing weight_ih_l0")
    input_size = int(w.shape[1])
    hidden_size = int(w.shape[0] // 4)
    num_layers = len([k for k in state.keys() if k.startswith("lstm.weight_ih_l")])
    model = lstm_model(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.0,
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _load_xgb_model(model_path: str):
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:
        raise RuntimeError(f"xgboost not installed: {e}") from e
    mdl = xgb.XGBRegressor()
    mdl.load_model(model_path)
    return mdl


def _rolling_predict_tree(
    df: pd.DataFrame,
    *,
    model: Any,
    feature_cols: list[str],
    time_col: str,
    value_col: str,
) -> np.ndarray:
    feat_df = prepare_feature_frame(
        df,
        feature_cols=feature_cols,
        time_col=time_col,
        value_col=value_col,
        tail_rows=1,
        allow_nan=True,
    )
    X = feat_df[feature_cols].to_numpy(dtype=np.float32)
    mask = np.isfinite(X).all(axis=1)
    preds = np.full(len(df), np.nan, dtype=float)
    if mask.any():
        preds[mask] = model.predict(X[mask])
    return preds


def run_online_predict(
    *,
    df: pd.DataFrame,
    model_name: str,
    time_col: str,
    value_col: str,
    horizon_steps: int,
    step: int | None,
    allow_degrade: bool,
    device: str,
    model_id: str | None = None,
    model_version: str | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    model_name = _ensure_model_name(model_name)
    model_key = model_name.lower()

    rec = None
    if model_id:
        rec = get_model(model_id)
    if rec is None and model_version:
        rec = _find_model_by_version(model_name, model_version)
    if rec is None:
        rec = latest_model_for_name(model_name)
    if rec is None:
        raise RuntimeError(f"No model found for name '{model_name}'. Train a model first.")

    if model_key == "informer":
        config = _build_predict_config(
            model_record=rec,
            model_name=model_name,
            time_col=time_col,
            value_col=value_col,
            device=device,
            allow_degrade=allow_degrade,
        )
        try:
            from models.informer.predict import InformerPredictor
        except Exception as e:
            raise RuntimeError(f"InformerPredictor unavailable: {e}") from e
        predictor = InformerPredictor(config)
        merged = predictor.rolling_predict(df.copy(), horizon=horizon_steps, step=step, mode="overwrite")
        dblk = config.get("data", {}) if isinstance(config, dict) else {}
        return np.asarray(merged).reshape(-1), dblk

    artifacts = rec.get("artifacts") if isinstance(rec, dict) else None
    if not isinstance(artifacts, dict):
        raise RuntimeError("Model artifacts missing.")

    degraded = False
    reason = ""
    degraded_mode = None
    if int(horizon_steps) != 1 or (step not in (None, 1)):
        degraded = True
        reason = "non_informer_one_step_only"
        degraded_mode = "limited"

    try:
        if model_key == "xgboost":
            model_path = artifacts.get("xgboost_model_path") or artifacts.get("model_path")
            if not isinstance(model_path, str) or not model_path:
                raise ValueError("xgboost model_path missing")
            feature_cols = [c for c in load_feature_cols(artifacts) if c != value_col]
            if not feature_cols:
                raise ValueError("xgboost feature_cols missing")
            model = _load_xgb_model(model_path)
            preds = _rolling_predict_tree(
                df,
                model=model,
                feature_cols=feature_cols,
                time_col=time_col,
                value_col=value_col,
            )
        elif model_key == "randomforest":
            model_path = artifacts.get("model_path")
            if not isinstance(model_path, str) or not model_path:
                raise ValueError("randomforest model_path missing")
            feature_cols = [c for c in load_feature_cols(artifacts) if c != value_col]
            if not feature_cols:
                raise ValueError("randomforest feature_cols missing")
            model = load_pickle(model_path)
            preds = _rolling_predict_tree(
                df,
                model=model,
                feature_cols=feature_cols,
                time_col=time_col,
                value_col=value_col,
            )
        elif model_key == "lstm":
            model_path = artifacts.get("model_path")
            if not isinstance(model_path, str) or not model_path:
                raise ValueError("lstm model_path missing")
            feature_cols = load_feature_cols(artifacts)
            if not feature_cols:
                raise ValueError("lstm feature_cols missing")
            try:
                import torch
            except Exception as e:
                raise RuntimeError(f"lstm torch unavailable: {e}") from e
            seq_len = 10
            best_params = artifacts.get("best_params") or {}
            if isinstance(best_params, dict):
                try:
                    seq_len = int(best_params.get("seq_len", seq_len))
                except Exception:
                    pass
            seq_len = max(1, int(seq_len))
            feat_df = prepare_feature_frame(
                df,
                feature_cols=feature_cols,
                time_col=time_col,
                value_col=value_col,
                tail_rows=seq_len,
                allow_nan=True,
            )
            X_all = feat_df[feature_cols].to_numpy(dtype=np.float32)
            scaler = None
            scaler_path = artifacts.get("scaler_path")
            if isinstance(scaler_path, str) and scaler_path:
                try:
                    scaler = load_pickle(scaler_path)
                except Exception:
                    scaler = None
            if scaler is not None:
                try:
                    X_all = scaler.transform(X_all.astype(np.float32))
                except Exception:
                    pass

            model = _load_lstm_model(model_path)
            preds_sc = np.full(len(df), np.nan, dtype=np.float32)
            for i in range(seq_len, len(X_all)):
                window = X_all[i - seq_len : i]
                if not np.isfinite(window).all():
                    continue
                inp = np.expand_dims(window, axis=0)
                pred_sc = model(
                    torch.tensor(inp, dtype=torch.float32)
                ).detach().cpu().numpy().reshape(-1)[0]
                preds_sc[i] = pred_sc

            preds = preds_sc.astype(float)
            mask = np.isfinite(preds_sc)
            if mask.any():
                preds[mask] = _inverse_target_with_scaler(preds_sc[mask], scaler, feature_cols, value_col)
                tt = artifacts.get("target_transform") if isinstance(artifacts, dict) else None
                if tt:
                    try:
                        preds[mask] = inverse_transform_array(preds[mask], tt)
                    except Exception:
                        pass
        elif model_key == "arima":
            model_path = artifacts.get("model_path")
            if not isinstance(model_path, str) or not model_path:
                raise ValueError("arima model_path missing")
            model = load_pickle(model_path)
            y = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
            if y.size == 0:
                raise ValueError("empty series")
            y_ffill = pd.Series(y).ffill().to_numpy(dtype=float)
            preds = np.full(len(y_ffill), np.nan, dtype=float)
            if hasattr(model, "update"):
                if np.isfinite(y_ffill[0]):
                    try:
                        model.update(np.asarray([y_ffill[0]], dtype=float))
                    except Exception:
                        try:
                            model.update(y_ffill[0])
                        except Exception:
                            pass
                for i in range(1, len(y_ffill)):
                    try:
                        preds[i] = float(np.asarray(model.predict(n_periods=1)).ravel()[-1])
                    except TypeError:
                        preds[i] = float(np.asarray(model.predict(1)).ravel()[-1])
                    if np.isfinite(y_ffill[i]):
                        try:
                            model.update(np.asarray([y_ffill[i]], dtype=float))
                        except Exception:
                            try:
                                model.update(y_ffill[i])
                            except Exception:
                                pass
                if np.isnan(y).any():
                    degraded = True
                    reason = reason or "arima_nan_filled"
                    degraded_mode = degraded_mode or "limited"
            else:
                preds[1:] = y_ffill[:-1]
                degraded = True
                reason = reason or "arima_no_update"
                degraded_mode = degraded_mode or "limited"
        elif model_key == "prophet":
            model_path = artifacts.get("model_path")
            if not isinstance(model_path, str) or not model_path:
                raise ValueError("prophet model_path missing")
            if time_col not in df.columns:
                raise ValueError("missing time_col for prophet")
            model = load_pickle(model_path)
            ds = pd.to_datetime(df[time_col], errors="coerce")
            if ds.isna().all():
                raise ValueError("invalid timestamps for prophet")
            freq = pd.infer_freq(ds.dropna())
            if freq is None:
                diffs = ds.dropna().diff().dropna()
                if diffs.empty:
                    freq = "D"
                else:
                    freq = diffs.mode().iloc[0]
            if ds.isna().any():
                start = ds.dropna().iloc[0]
                ds = pd.date_range(start=start, periods=len(ds), freq=freq)
            fcst = model.predict(pd.DataFrame({"ds": ds}))
            preds = fcst["yhat"].to_numpy(dtype=float)
            degraded = True
            reason = reason or "prophet_in_sample"
            degraded_mode = degraded_mode or "in_sample"
        else:
            raise ValueError(f"unsupported model '{model_name}'")
    except Exception as e:
        if allow_degrade:
            preds = _baseline_rolling(df, value_col)
            degraded = True
            reason = reason or f"{model_key}_fallback: {e}"
            degraded_mode = "baseline"
        else:
            raise

    dblk = {
        "degraded": bool(degraded),
        "degraded_reason": reason or None,
        "degraded_mode": degraded_mode,
    }
    return np.asarray(preds).reshape(-1), dblk


def _find_model_by_version(model_name: str, model_version: str) -> Dict[str, Any] | None:
    from services.registry import list_models

    records = list_models(limit=200, offset=0)
    for rec in records:
        if str(rec.get("name", "")).lower() != model_name.lower():
            continue
        if str(rec.get("version", "")).lower() == str(model_version).lower():
            return rec
    return None
