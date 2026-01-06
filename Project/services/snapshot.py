from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]  # .../Project
DEFAULT_SNAPSHOT_PATH = PROJECT_DIR / "output" / "last_results.json"
LOGGER = logging.getLogger(__name__)


def as_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, str):
            xs = x.strip()
            if xs.isdigit() or (xs.startswith("-") and xs[1:].isdigit()):
                return int(xs)
            return int(float(xs))
        if x is not None:
            return int(x)
    except Exception:
        pass
    return default


def safe_jsonify(obj: Any, *, max_depth: int = 4, max_items: int = 50):
    if max_depth <= 0:
        return repr(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        out = []
        for i, v in enumerate(obj):
            if i >= max_items:
                out.append(f"...(+{len(obj) - max_items} more)")
                break
            out.append(safe_jsonify(v, max_depth=max_depth - 1, max_items=max_items))
        return out
    if isinstance(obj, dict):
        out = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                out["..."] = f"(+{len(obj) - max_items} more keys)"
                break
            try:
                ks = str(k)
            except Exception:
                ks = repr(k)
            out[ks] = safe_jsonify(v, max_depth=max_depth - 1, max_items=max_items)
        return out
    try:
        if hasattr(obj, "tolist"):
            return safe_jsonify(obj.tolist(), max_depth=max_depth - 1, max_items=max_items)
    except Exception:
        pass
    return f"<{type(obj).__name__}>"


def _coerce_plot_blob(blob: Any):
    """Convert a stringified plot_data blob back to a dict, otherwise return as-is."""
    if isinstance(blob, str):
        try:
            import json as _json, ast

            try:
                return _json.loads(blob)
            except Exception:
                return ast.literal_eval(blob)
        except Exception:
            return None
    return blob


def reset_snapshot(path: str = os.path.join("output", "last_results.json")) -> None:
    """Remove the last_results snapshot so a new run starts cleanly."""
    try:
        p = Path(path) if isinstance(path, (str, Path)) else DEFAULT_SNAPSHOT_PATH
        if not p.is_absolute():
            p = PROJECT_DIR / p
        p.unlink(missing_ok=True)
    except Exception:
        LOGGER.warning("Failed to reset snapshot at %s", path, exc_info=True)


def cacheable_results(results: dict) -> dict:
    if not isinstance(results, dict):
        return {"status": "error", "message": "non-dict results"}

    data = results.get("data", {}) if isinstance(results.get("data", {}), dict) else {}
    metrics = results.get("metrics", {}) if isinstance(results.get("metrics", {}), dict) else {}
    arts = results.get("artifacts", {}) if isinstance(results.get("artifacts", {}), dict) else {}

    data_keep = {}
    for k in (
        "split",
        "plot_data",
        "mean_abs_true_val",
        "mean_abs_true_test",
        "degraded",
        "degraded_mode",
        "degraded_reason",
        "degraded_error",
        "missing_required_core",
        "dropped_optional_features",
        "val_plot_png",
        "val_plot_html",
        "test_plot_png",
        "test_plot_html",
    ):
        if k in data:
            data_keep[k] = data.get(k)

    arts_keep = {}
    for k in (
        "model_path",
        "scaler_path",
        "residual_model_path",
        "y_scaler_path",
        "feature_cols_path",
        "feature_report_path",
        "target_transform",
        "feature_missing_report",
        "feature_cols",
        "target_idx",
    ):
        if k in arts:
            arts_keep[k] = arts.get(k)

    return {
        "status": results.get("status", "ok"),
        "message": results.get("message"),
        "metrics": metrics,
        "data": data_keep,
        "artifacts": arts_keep,
    }


def save_last_results_json(payload: dict, path: str = os.path.join("output", "last_results.json")) -> None:
    try:
        p = Path(path) if isinstance(path, (str, Path)) else DEFAULT_SNAPSHOT_PATH
        if not p.is_absolute():
            p = PROJECT_DIR / p
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(safe_jsonify(payload), f, ensure_ascii=False, indent=2)
    except Exception:
        LOGGER.error("Failed to save snapshot to %s", path, exc_info=True)


def load_last_results_json(path: str = os.path.join("output", "last_results.json")) -> Optional[dict]:
    try:
        p = Path(path) if isinstance(path, (str, Path)) else DEFAULT_SNAPSHOT_PATH
        if not p.is_absolute():
            p = PROJECT_DIR / p
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return None

        # Repair stringified plot_data if present.
        res = obj.get("results")
        if isinstance(res, dict):
            data = res.get("data")
            if isinstance(data, dict):
                pd_blob = data.get("plot_data")
                if isinstance(pd_blob, dict):
                    data["plot_data"] = {k: _coerce_plot_blob(v) for k, v in pd_blob.items()}
                    res["data"] = data
                    obj["results"] = res
        return obj
    except Exception:
        LOGGER.warning("Failed to load snapshot from %s", path, exc_info=True)
        return None


def pack_plot_series(ts, y_true, y_pred, *, max_n: int = 4000) -> Optional[dict]:
    try:
        yt = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy(dtype=float).reshape(-1)
        yp = pd.to_numeric(pd.Series(y_pred), errors="coerce").to_numpy(dtype=float).reshape(-1)
        L = int(min(len(yt), len(yp)))
        if L <= 0:
            return None
        yt = yt[:L]
        yp = yp[:L]
        if ts is None:
            xs = pd.date_range(start=pd.Timestamp.today().normalize(), periods=L, freq="D").to_numpy()
        else:
            xs0 = pd.to_datetime(ts, errors="coerce", utc=True)
            try:
                xs0 = xs0.dt.tz_localize(None)
            except Exception:
                try:
                    xs0 = xs0.tz_localize(None)  # type: ignore[attr-defined]
                except Exception:
                    pass
            xs0 = pd.to_datetime(xs0, errors="coerce")
            try:
                xs = xs0.to_numpy()
            except Exception:
                xs = np.asarray(xs0)
            xs = np.asarray(xs).reshape(-1)[:L]
            if xs.size != L or bool(np.asarray(pd.isna(xs)).all()):
                xs = pd.date_range(start=pd.Timestamp.today().normalize(), periods=L, freq="D").to_numpy()

        m = np.isfinite(yt) & np.isfinite(yp) & (~pd.isna(xs))
        m = np.asarray(m, dtype=bool)
        xs = np.asarray(xs)[m]
        yt = np.asarray(yt)[m]
        yp = np.asarray(yp)[m]
        if xs.size == 0:
            return None
        if xs.size > max_n:
            xs = xs[-max_n:]
            yt = yt[-max_n:]
            yp = yp[-max_n:]
        ts_list = pd.to_datetime(xs, errors="coerce").astype(str).tolist()
        return {"ts": ts_list, "true": yt.astype(float).tolist(), "pred": yp.astype(float).tolist()}
    except Exception:
        return None


def safe_artifacts_from_config(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        return {}
    arts = cfg.get("artifacts") or {}
    if not isinstance(arts, dict):
        return {}
    safe_keys = (
        "model_path",
        "scaler_path",
        "residual_model_path",
        "y_scaler_path",
        "feature_cols_path",
        "feature_report_path",
        "target_transform",
        "target_transform_applied",
        "feature_missing_report",
        "feature_cols",
        "target_idx",
        "randomforest_params",
        "best_params",
        "rf_best_params",
    )
    out = {}
    for k in safe_keys:
        if k not in arts:
            continue
        v = arts.get(k)
        if v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)) and len(v) <= 500:
            out[k] = list(v)
        elif isinstance(v, dict) and len(v) <= 200:
            out[k] = safe_jsonify(v, max_depth=4, max_items=200)
    return out


def strip_heavy_inplace(cfg: dict) -> None:
    try:
        arts = cfg.get("artifacts") if isinstance(cfg, dict) else None
        if isinstance(arts, dict):
            for k in (
                "scaler",
                "value_scaler",
                "y_scaler",
                "final_model",
                "model",
                "torch_model",
                "sk_model",
            ):
                if k in arts:
                    arts[k] = None
    except Exception:
        LOGGER.debug("Failed to strip heavy artifacts from config", exc_info=True)
