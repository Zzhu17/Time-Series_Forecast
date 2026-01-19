from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from functools import lru_cache
import os
import time

import numpy as np
import pandas as pd

from configs.config import load_yaml_config
from services.predict_utils import baseline_predict, predict_with_xgboost
from services.prediction_payloads import normalize_prediction_payload
from services.registry import get_model, latest_model_for_name, list_models
from services.predict_helpers import load_pickle, prepare_feature_frame, load_json_file
from services.xgb_loader import XGBPredictor
from utils.feature_contract import (
    ensure_calendar_features,
    is_recomputable_name,
    recompute_feature_column,
    safe_time_features,
)
from utils.feature_pipeline import align_predict_df
from utils.feature_selection import load_feature_contract
from utils.metrics import observe_predict
from utils.target_transform import inverse_transform_array
from models.registry import FORECASTER_REGISTRY


def _find_model_record(
    *,
    model_name: str,
    model_id: Optional[str],
    model_version: Optional[str],
) -> Optional[Dict[str, Any]]:
    if model_id:
        return get_model(model_id)
    if model_version:
        records = list_models(limit=200, offset=0)
        for rec in records:
            if str(rec.get("name", "")).lower() != model_name.lower():
                continue
            if str(rec.get("version", "")).lower() == str(model_version).lower():
                return rec
        return None
    return latest_model_for_name(model_name)


class PredictionNotFoundError(Exception):
    pass


def _file_mtime(path: str) -> float:
    try:
        return float(os.path.getmtime(path))
    except Exception:
        return 0.0


@lru_cache(maxsize=16)
def _load_pickle_cached(path: str, mtime: float):
    return load_pickle(path)


@lru_cache(maxsize=16)
def _load_json_cached(path: str, mtime: float):
    return load_json_file(path)


@lru_cache(maxsize=4)
def _xgb_predictor_cached(model_path: str, contract_path: str, mtime_model: float, mtime_contract: float, time_col: str, value_col: str):
    return XGBPredictor(
        model_path=model_path,
        feature_contract_path=contract_path or None,
        target_transform=None,
        time_col=time_col,
        value_col=value_col,
    )


def _fallback_feature_contract(
    feature_cols: list[str],
    *,
    time_col: str,
    value_col: str,
) -> Dict[str, Any]:
    required_core: list[str] = []
    repairable_core: list[str] = []
    for c in feature_cols:
        if c == time_col:
            continue
        if c in safe_time_features() or is_recomputable_name(c):
            repairable_core.append(c)
        else:
            required_core.append(c)
    if value_col not in required_core and value_col not in repairable_core and value_col in feature_cols:
        required_core = [value_col] + required_core
    return {
        "feature_cols": list(feature_cols),
        "required_core_cols": list(required_core),
        "repairable_core_cols": list(repairable_core),
        "core_cols": list(required_core),
    }


def _load_feature_contract(artifacts: Dict[str, Any] | None) -> Optional[Dict[str, Any]]:
    if not isinstance(artifacts, dict):
        return None
    path = artifacts.get("feature_cols_path")
    if isinstance(path, str) and path:
        contract = load_feature_contract(path)
        if isinstance(contract, dict):
            return contract
    return None


def _load_feature_cols_cached(artifacts: Optional[Dict[str, Any]]) -> list[str]:
    if isinstance(artifacts, dict):
        cols = artifacts.get("feature_cols")
        if isinstance(cols, (list, tuple)) and cols:
            return [str(c) for c in cols if str(c).strip()]
        path = artifacts.get("feature_cols_path")
    else:
        path = None
    if isinstance(path, str) and path:
        payload = _load_json_cached(path, _file_mtime(path))
        if isinstance(payload, (list, tuple)):
            return [str(c) for c in payload if str(c).strip()]
        if isinstance(payload, dict):
            inner = payload.get("feature_cols")
            if isinstance(inner, (list, tuple)):
                return [str(c) for c in inner if str(c).strip()]
    return []


def _infer_future_index(df: pd.DataFrame, *, time_col: str, horizon: int) -> Optional[pd.DatetimeIndex]:
    if horizon <= 0 or time_col not in df.columns:
        return None
    ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    ts = ts.dropna()
    if ts.empty:
        return None
    try:
        ts = ts.dt.tz_localize(None)
    except Exception:
        pass
    ts = ts.sort_values()
    freq = None
    try:
        freq = pd.infer_freq(ts)
    except Exception:
        freq = None
    if not freq:
        diffs = ts.diff().dropna()
        if diffs.empty:
            return None
        try:
            freq = diffs.mode().iloc[0]
        except Exception:
            freq = None
        if freq is None or (hasattr(freq, "value") and freq.value <= 0):
            try:
                freq = diffs.median()
            except Exception:
                freq = None
    if not freq:
        return None
    try:
        return pd.date_range(start=ts.iloc[-1], periods=horizon + 1, freq=freq)[1:]
    except Exception:
        return None


def _build_residual_feature_frame(
    *,
    df: pd.DataFrame,
    preds: np.ndarray,
    time_col: str,
    value_col: str,
    feature_cols: list[str],
) -> Optional[pd.DataFrame]:
    if not feature_cols:
        return None
    preds = np.asarray(preds, dtype=float).reshape(-1)
    horizon = int(len(preds))
    if horizon <= 0:
        return None
    future_index = _infer_future_index(df, time_col=time_col, horizon=horizon)
    if future_index is None:
        return None

    df_hist = df.copy()
    df_hist[time_col] = pd.to_datetime(df_hist[time_col], errors="coerce", utc=True)
    try:
        df_hist[time_col] = df_hist[time_col].dt.tz_localize(None)
    except Exception:
        pass
    df_hist = df_hist.dropna(subset=[time_col]).sort_values(time_col)
    if df_hist.empty:
        return None
    if value_col in df_hist.columns:
        df_hist[value_col] = pd.to_numeric(df_hist[value_col], errors="coerce")

    future_df = pd.DataFrame({time_col: future_index, value_col: preds})
    df_aug = pd.concat([df_hist[[time_col, value_col]], future_df], ignore_index=True)

    work = df_aug.copy()
    if any(c in safe_time_features() for c in feature_cols):
        try:
            work = ensure_calendar_features(work, time_col=time_col)
        except Exception:
            return None

    for c in feature_cols:
        if c in ("yhat", time_col):
            continue
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
        elif is_recomputable_name(c):
            try:
                work[c] = recompute_feature_column(work, c, value_col=value_col, time_col=time_col)
            except Exception:
                return None
        else:
            return None

    tail = work.tail(horizon)
    if tail.empty or len(tail) != horizon:
        return None
    X = pd.DataFrame({"yhat": preds})
    for c in feature_cols:
        if c == "yhat":
            continue
        if c not in tail.columns:
            return None
        X[c] = pd.to_numeric(tail[c], errors="coerce")
    if not np.isfinite(X.to_numpy(dtype=np.float64)).all():
        return None
    return X


def _apply_xgboost_residual(
    *,
    df: pd.DataFrame,
    preds: np.ndarray,
    time_col: str,
    value_col: str,
    artifacts: Dict[str, Any],
    residual_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, bool, str]:
    model_path = artifacts.get("xgboost_residual_model_path")
    if not isinstance(model_path, str) or not model_path:
        return preds, False, "residual_model_path_missing"
    feature_cols = artifacts.get("residual_feature_cols")
    if not isinstance(feature_cols, list) or not feature_cols:
        feature_cols = residual_cfg.get("feature_cols") if isinstance(residual_cfg, dict) else None
    if not isinstance(feature_cols, list) or not feature_cols:
        base = []
        lags = residual_cfg.get("lags") if isinstance(residual_cfg, dict) else None
        rolls = residual_cfg.get("rolling_windows") if isinstance(residual_cfg, dict) else None
        diffs = residual_cfg.get("diffs") if isinstance(residual_cfg, dict) else None
        base = ["month", "day_of_month", "day_of_week", "hour", "day_of_year"]
        if isinstance(lags, list):
            base += [f"lag_{int(k)}" for k in lags if int(k) > 0]
        if isinstance(rolls, list):
            for w in rolls:
                wi = int(w)
                if wi > 0:
                    base += [f"rolling_mean_{wi}", f"rolling_std_{wi}"]
        if isinstance(diffs, list):
            base += [f"diff_{int(k)}" for k in diffs if int(k) > 0]
        feature_cols = ["yhat"] + base
    feature_cols = ["yhat"] + [c for c in feature_cols if c != "yhat"]

    X = _build_residual_feature_frame(
        df=df,
        preds=preds,
        time_col=time_col,
        value_col=value_col,
        feature_cols=feature_cols,
    )
    if X is None:
        return preds, False, "residual_features_unavailable"
    try:
        import xgboost as xgb  # type: ignore
    except Exception:
        return preds, False, "xgboost_not_installed"

    try:
        mdl = xgb.XGBRegressor()
        mdl.load_model(model_path)
        try:
            num_features = mdl.get_booster().num_features()
            if num_features and int(num_features) != int(X.shape[1]):
                return preds, False, "residual_feature_mismatch"
        except Exception:
            pass
        res_hat = mdl.predict(X.to_numpy(dtype=np.float32)).astype(np.float64, copy=False).reshape(-1)
        if len(res_hat) != len(preds):
            return preds, False, "residual_length_mismatch"
        return (np.asarray(preds, dtype=float).reshape(-1) + res_hat), True, ""
    except Exception:
        return preds, False, "residual_model_failed"


def _build_informer_config(
    *,
    model_record: Dict[str, Any],
    time_col: str,
    value_col: str,
    allow_degrade: bool,
) -> Dict[str, Any]:
    try:
        base_cfg = load_yaml_config()
    except Exception:
        base_cfg = {}
    config = dict(base_cfg) if isinstance(base_cfg, dict) else {}
    default_cfg = config.setdefault("default", {})
    default_cfg["time_col"] = time_col
    default_cfg["value_col"] = value_col
    config["device"] = default_cfg.get("device", "cpu")

    artifacts = model_record.get("artifacts") if isinstance(model_record, dict) else None
    if not isinstance(artifacts, dict) or not artifacts.get("model_path"):
        raise ValueError("Missing model artifacts for Informer prediction.")
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


def _adjust_horizon(preds: np.ndarray, horizon: int) -> np.ndarray:
    preds = np.asarray(preds).reshape(-1)
    if horizon <= 0:
        return np.array([], dtype=float)
    if len(preds) >= horizon:
        return preds[:horizon]
    if len(preds) == 0:
        return preds
    pad = np.full(horizon - len(preds), float(preds[-1]), dtype=float)
    return np.concatenate([preds, pad], axis=0)


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


def _align_df_for_contract(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    contract: Optional[Dict[str, Any]],
    time_col: str,
    value_col: str,
    tail_rows: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    contract_payload = contract or _fallback_feature_contract(
        feature_cols,
        time_col=time_col,
        value_col=value_col,
    )
    aligned, report, usable_cols = align_predict_df(
        df,
        contract=contract_payload,
        time_col=time_col,
        value_col=value_col,
        tail_rows=tail_rows,
    )
    if list(usable_cols) != list(feature_cols):
        dropped = sorted(set(feature_cols) - set(usable_cols))
        raise ValueError(f"optional features dropped: {dropped}")
    return aligned, report


def predict_from_registry(
    *,
    df: pd.DataFrame,
    model_name: str,
    horizon: int,
    time_col: str,
    value_col: str,
    allow_degrade: bool,
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
    residual_modeling: Optional[Dict[str, Any]] = None,
    model_alias: Optional[str] = None,
) -> Tuple[np.ndarray, bool, str, str]:
    lookup_name = model_alias or model_name
    record = _find_model_record(
        model_name=lookup_name,
        model_id=model_id,
        model_version=model_version,
    )
    if record is None and model_alias:
        record = _find_model_record(
            model_name=model_name,
            model_id=model_id,
            model_version=model_version,
        )
    if record is None:
        raise ValueError("model not found in registry")

    model_key = model_name.lower()
    artifacts = record.get("artifacts") if isinstance(record, dict) else {}
    params = record.get("params") if isinstance(record, dict) else {}
    residual_cfg = residual_modeling if isinstance(residual_modeling, dict) else None
    if residual_cfg is None and isinstance(params, dict):
        residual_cfg = params.get("residual_modeling") if isinstance(params.get("residual_modeling"), dict) else None
    alias = model_alias
    if alias is None and isinstance(params, dict):
        alias = params.get("model_alias") if isinstance(params.get("model_alias"), str) else None
    if alias is None and isinstance(record, dict):
        alias = record.get("name") if isinstance(record.get("name"), str) else None

    def _fallback(err: Exception, key: str) -> Tuple[np.ndarray, bool, str, str]:
        if allow_degrade:
            preds = baseline_predict(df, value_col, horizon)
            return preds, True, f"{key}->baseline", str(err)
        raise err

    if model_key == "xgboost":
        try:
            model_path = None
            if isinstance(artifacts, dict):
                model_path = artifacts.get("xgboost_model_path") or artifacts.get("model_path")
                contract_path = artifacts.get("feature_cols_path")
            else:
                contract_path = None
            mp = str(model_path) if model_path else ""
            cp = str(contract_path) if contract_path else ""
            predictor = _xgb_predictor_cached(mp, cp, _file_mtime(mp), _file_mtime(cp), time_col, value_col)
            preds, _meta, degraded, reason = predictor.predict(df, horizon=horizon)
            return preds, bool(degraded), "xgboost", reason or ""
        except Exception as e:
            return _fallback(e, "xgboost")

    if model_key == "randomforest":
        try:
            model_path = artifacts.get("model_path") if isinstance(artifacts, dict) else None
            if not isinstance(model_path, str) or not model_path:
                raise ValueError("randomforest model_path missing")
            feature_cols = _load_feature_cols_cached(artifacts)
            if not feature_cols:
                raise ValueError("randomforest feature_cols missing")
            contract = _load_feature_contract(artifacts if isinstance(artifacts, dict) else None)
            aligned_df, _rep = _align_df_for_contract(
                df,
                feature_cols=feature_cols,
                contract=contract,
                time_col=time_col,
                value_col=value_col,
                tail_rows=1,
            )
            feat_df = prepare_feature_frame(
                aligned_df,
                feature_cols=feature_cols,
                time_col=time_col,
                value_col=value_col,
                tail_rows=1,
                tail_only=True,
            )
            X = feat_df[feature_cols].tail(1).to_numpy(dtype=np.float32)
            model = _load_pickle_cached(model_path, _file_mtime(model_path))
            pred_one = np.asarray(model.predict(X), dtype=float).reshape(-1)
            preds = np.repeat(pred_one[-1], max(1, horizon)).astype(float)
            degraded = horizon > 1
            reason = "multi_step_not_supported" if degraded else ""
            return preds, degraded, "randomforest", reason
        except Exception as e:
            return _fallback(e, "randomforest")

    if model_key == "lstm":
        try:
            model_path = artifacts.get("model_path") if isinstance(artifacts, dict) else None
            if not isinstance(model_path, str) or not model_path:
                raise ValueError("lstm model_path missing")
            feature_cols = _load_feature_cols_cached(artifacts)
            if not feature_cols:
                raise ValueError("lstm feature_cols missing")
            seq_len = 10
            if isinstance(artifacts, dict):
                best_params = artifacts.get("best_params") or {}
                try:
                    seq_len = int(best_params.get("seq_len", seq_len))
                except Exception:
                    pass
            seq_len = max(1, int(seq_len))
            contract = _load_feature_contract(artifacts if isinstance(artifacts, dict) else None)
            aligned_df, _rep = _align_df_for_contract(
                df,
                feature_cols=feature_cols,
                contract=contract,
                time_col=time_col,
                value_col=value_col,
                tail_rows=seq_len,
            )
            feat_df = prepare_feature_frame(
                aligned_df,
                feature_cols=feature_cols,
                time_col=time_col,
                value_col=value_col,
                tail_rows=seq_len,
                tail_only=True,
            )
            X_all = feat_df[feature_cols].to_numpy(dtype=np.float32)
            scaler = None
            if isinstance(artifacts, dict):
                scaler_path = artifacts.get("scaler_path")
                if isinstance(scaler_path, str) and scaler_path:
                    try:
                        scaler = _load_pickle_cached(scaler_path, _file_mtime(scaler_path))
                    except Exception:
                        scaler = None
            if scaler is not None:
                try:
                    X_all = scaler.transform(X_all.astype(np.float32))
                except Exception:
                    pass

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

            if X_all.shape[0] < 1:
                raise ValueError("not enough rows for lstm inference")
            seq_len = min(seq_len, X_all.shape[0])
            history = X_all[-seq_len:].copy()
            preds_sc = []
            idx = feature_cols.index(value_col) if value_col in feature_cols else 0
            for _ in range(max(1, horizon)):
                inp = torch.tensor(history[np.newaxis, :, :], dtype=torch.float32)
                pred_sc = model(inp).detach().cpu().numpy().reshape(-1)[0]
                preds_sc.append(pred_sc)
                next_row = history[-1].copy()
                next_row[idx] = pred_sc
                if history.shape[0] >= seq_len:
                    history = np.vstack([history[1:], next_row])
                else:
                    history = np.vstack([history, next_row])

            preds_sc = np.asarray(preds_sc, dtype=np.float32)
            preds = _inverse_target_with_scaler(preds_sc, scaler, feature_cols, value_col)
            tt = artifacts.get("target_transform") if isinstance(artifacts, dict) else None
            if tt:
                try:
                    preds = inverse_transform_array(preds, tt)
                except Exception:
                    pass
            if residual_cfg and isinstance(artifacts, dict):
                model_type = str(residual_cfg.get("model_type", "xgboost")).lower()
                if model_type not in ("xgboost", "xgb"):
                    return preds, False, "lstm", ""
                preds, applied, reason = _apply_xgboost_residual(
                    df=df,
                    preds=preds,
                    time_col=time_col,
                    value_col=value_col,
                    artifacts=artifacts,
                    residual_cfg=residual_cfg,
                )
                if applied:
                    return preds, False, alias or "lstm", ""
                if reason:
                    return preds, False, "lstm", f"residual_skipped:{reason}"
            return preds, False, "lstm", ""
        except Exception as e:
            return _fallback(e, "lstm")

    if model_key == "arima":
        try:
            model_path = artifacts.get("model_path") if isinstance(artifacts, dict) else None
            if not isinstance(model_path, str) or not model_path:
                raise ValueError("arima model_path missing")
            model = _load_pickle_cached(model_path, _file_mtime(model_path))
            if hasattr(model, "predict"):
                try:
                    preds = model.predict(n_periods=max(1, horizon))
                except TypeError:
                    preds = model.predict(max(1, horizon))
            elif hasattr(model, "forecast"):
                preds = model.forecast(max(1, horizon))
            else:
                raise ValueError("arima model has no predict/forecast")
            return np.asarray(preds, dtype=float).reshape(-1), False, "arima", ""
        except Exception as e:
            return _fallback(e, "arima")

    if model_key == "prophet":
        try:
            model_path = artifacts.get("model_path") if isinstance(artifacts, dict) else None
            if not isinstance(model_path, str) or not model_path:
                raise ValueError("prophet model_path missing")
            model = _load_pickle_cached(model_path, _file_mtime(model_path))
            if time_col not in df.columns:
                raise ValueError("missing time_col for prophet")
            ts = pd.to_datetime(df[time_col], errors="coerce")
            ts = ts.dropna()
            if ts.empty:
                raise ValueError("no valid timestamps for prophet")
            freq = pd.infer_freq(ts)
            if freq is None:
                diffs = ts.diff().dropna()
                if diffs.empty:
                    freq = "D"
                else:
                    freq = diffs.mode().iloc[0]
            future = pd.date_range(start=ts.iloc[-1], periods=max(1, horizon) + 1, freq=freq)[1:]
            future_df = pd.DataFrame({"ds": future})
            forecast = model.predict(future_df)
            preds = forecast["yhat"].to_numpy(dtype=float).reshape(-1)
            return preds, False, "prophet", ""
        except Exception as e:
            return _fallback(e, "prophet")

    if model_key == "informer":
        try:
            from models.informer.predict import InformerPredictor
        except Exception as e:
            raise RuntimeError(f"InformerPredictor unavailable: {e}") from e
        config = _build_informer_config(
            model_record=record,
            time_col=time_col,
            value_col=value_col,
            allow_degrade=allow_degrade,
        )
        predictor = InformerPredictor(config)
        preds = predictor.predict(df.copy())
        preds = _adjust_horizon(preds, horizon)
        dblk = config.get("data", {}) if isinstance(config, dict) else {}
        degraded = bool(dblk.get("degraded", False)) if isinstance(dblk, dict) else False
        reason = dblk.get("degraded_reason") if isinstance(dblk, dict) else ""
        if residual_cfg and isinstance(artifacts, dict):
            model_type = str(residual_cfg.get("model_type", "xgboost")).lower()
            if model_type not in ("xgboost", "xgb"):
                return preds, degraded, "informer", reason or ""
            preds, applied, res_reason = _apply_xgboost_residual(
                df=df,
                preds=preds,
                time_col=time_col,
                value_col=value_col,
                artifacts=artifacts,
                residual_cfg=residual_cfg,
            )
            if applied:
                return preds, degraded, alias or "informer", reason or ""
            if res_reason:
                return preds, degraded, "informer", f"{reason or ''}|residual_skipped:{res_reason}".strip("|")
        return preds, degraded, "informer", reason or ""

    # Unsupported model types fall back to baseline
    preds = baseline_predict(df, value_col, horizon)
    return preds, True, f"{model_key}->baseline", "model_not_supported"


def run_prediction(payload: Dict[str, Any]) -> Dict[str, Any]:
    start_ts = time.time()
    model_name = "unknown"
    try:
        df, normalized, contract_report = normalize_prediction_payload(payload)

        model_name = normalized["model_name"]
        model = model_name.lower()
        horizon = int(normalized.get("horizon", 1))
        time_col = normalized["time_col"]
        value_col = normalized["value_col"]
        model_id = normalized.get("model_id")
        model_version = normalized.get("model_version")
        allow_degrade = bool(normalized.get("allow_degrade", False))
        residual_modeling = normalized.get("residual_modeling")
        residual_modeling = residual_modeling if isinstance(residual_modeling, dict) else None
        model_alias = normalized.get("model_alias") if isinstance(normalized.get("model_alias"), str) else None

        forecaster_factory = FORECASTER_REGISTRY.get(model)
        if forecaster_factory is not None and not model_alias and not residual_modeling and not model_id and not model_version:
            try:
                forecaster = forecaster_factory()
                if bool(getattr(forecaster, "supports_predict", False)):
                    preds = forecaster.predict(
                        df,
                        horizon,
                        {"default": {"time_col": time_col, "value_col": value_col}},
                    )
                    return {
                        "status": "ok",
                        "predictions": np.asarray(preds, dtype=float).reshape(-1).tolist(),
                        "degraded": False,
                        "reason": None,
                        "used_model": model,
                        "contract_report": contract_report,
                    }
            except Exception:
                pass

        if model == "baseline":
            preds = baseline_predict(df, value_col, horizon)
            return {
                "status": "ok",
                "predictions": preds.tolist(),
                "degraded": False,
                "reason": None,
                "used_model": "baseline",
                "contract_report": contract_report,
            }

        if model_id or model_version or model in ("xgboost", "informer", "randomforest", "lstm", "arima", "prophet"):
            try:
                preds, degraded, used_model, reason = predict_from_registry(
                    df=df,
                    model_name=model,
                    horizon=horizon,
                    time_col=time_col,
                    value_col=value_col,
                    allow_degrade=allow_degrade,
                    model_id=model_id,
                    model_version=model_version,
                    residual_modeling=residual_modeling,
                    model_alias=model_alias,
                )
            except Exception as e:
                if model_id or model_version:
                    raise PredictionNotFoundError(str(e)) from e
                if model == "xgboost":
                    preds, degraded, used_model, reason = predict_with_xgboost(
                        df,
                        time_col=time_col,
                        value_col=value_col,
                        horizon=horizon,
                        baseline_fallback=True,
                    )
                else:
                    preds = baseline_predict(df, value_col, horizon)
                    degraded = True
                    used_model = f"{model}->baseline"
                    reason = "model_not_available"

            return {
                "status": "ok",
                "predictions": preds.tolist(),
                "degraded": bool(degraded),
                "reason": reason or None,
                "used_model": used_model,
                "contract_report": contract_report,
            }

        preds = baseline_predict(df, value_col, horizon)
        return {
            "status": "ok",
            "predictions": preds.tolist(),
            "degraded": True,
            "reason": "model_not_supported",
            "used_model": f"{model}->baseline",
            "contract_report": contract_report,
        }
    finally:
        observe_predict(model=model_name, duration=time.time() - start_ts)
