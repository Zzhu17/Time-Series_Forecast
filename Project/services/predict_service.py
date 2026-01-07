from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from configs.config import load_yaml_config
from services.predict_utils import baseline_predict
from services.registry import get_model, latest_model_for_name, list_models
from services.predict_helpers import load_feature_cols, load_pickle, prepare_feature_frame
from services.xgb_loader import XGBPredictor
from utils.target_transform import inverse_transform_array


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
) -> Tuple[np.ndarray, bool, str, str]:
    record = _find_model_record(
        model_name=model_name,
        model_id=model_id,
        model_version=model_version,
    )
    if record is None:
        raise ValueError("model not found in registry")

    model_key = model_name.lower()
    artifacts = record.get("artifacts") if isinstance(record, dict) else {}

    def _fallback(err: Exception, key: str) -> Tuple[np.ndarray, bool, str, str]:
        if allow_degrade:
            preds = baseline_predict(df, value_col, horizon)
            return preds, True, f"{key}->baseline", str(err)
        raise err

    if model_key == "xgboost":
        model_path = None
        if isinstance(artifacts, dict):
            model_path = artifacts.get("xgboost_model_path") or artifacts.get("model_path")
            contract_path = artifacts.get("feature_cols_path")
        else:
            contract_path = None
        predictor = XGBPredictor(
            model_path=str(model_path) if model_path else "",
            feature_contract_path=str(contract_path) if contract_path else None,
            target_transform=None,
            time_col=time_col,
            value_col=value_col,
        )
        preds, _meta, degraded, reason = predictor.predict(df, horizon=horizon)
        return preds, bool(degraded), "xgboost", reason or ""

    if model_key == "randomforest":
        try:
            model_path = artifacts.get("model_path") if isinstance(artifacts, dict) else None
            if not isinstance(model_path, str) or not model_path:
                raise ValueError("randomforest model_path missing")
            feature_cols = load_feature_cols(artifacts)
            if not feature_cols:
                raise ValueError("randomforest feature_cols missing")
            feat_df = prepare_feature_frame(
                df,
                feature_cols=feature_cols,
                time_col=time_col,
                value_col=value_col,
                tail_rows=1,
                tail_only=True,
            )
            X = feat_df[feature_cols].tail(1).to_numpy(dtype=np.float32)
            model = load_pickle(model_path)
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
            feature_cols = load_feature_cols(artifacts)
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
            feat_df = prepare_feature_frame(
                df,
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
                        scaler = load_pickle(scaler_path)
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
            return preds, False, "lstm", ""
        except Exception as e:
            return _fallback(e, "lstm")

    if model_key == "arima":
        try:
            model_path = artifacts.get("model_path") if isinstance(artifacts, dict) else None
            if not isinstance(model_path, str) or not model_path:
                raise ValueError("arima model_path missing")
            model = load_pickle(model_path)
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
            model = load_pickle(model_path)
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
        return preds, degraded, "informer", reason or ""

    # Unsupported model types fall back to baseline
    preds = baseline_predict(df, value_col, horizon)
    return preds, True, f"{model_key}->baseline", "model_not_supported"
