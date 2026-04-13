from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def apply_registry_residual_modeling(
    *,
    config: Dict[str, Any],
    data_blk: Dict[str, Any],
    artifacts: Dict[str, Any],
    val_dense: Any,
    test_dense: Any,
    df_input: Any,
    time_col: str,
    value_col: str,
) -> Tuple[Any, Any]:
    try:
        rm_cfg = (config.get("residual_modeling") or {})
        already_applied = bool(data_blk.get("residual_applied"))
        rm_enabled = bool(rm_cfg.get("enabled", False)) and not already_applied
        if rm_enabled and isinstance(val_dense, pd.DataFrame) and not val_dense.empty:
            model_type = str(rm_cfg.get("model_type", "LinearRegression")).strip().lower()

            if model_type in ("xgboost", "xgb"):
                try:
                    from models.xgboost import build_xgboost_regressor
                except Exception as imp_e:
                    print(f"[pipeline][registry-route] residual modeling skipped (xgboost missing): {imp_e}")
                    build_xgboost_regressor = None  # type: ignore[assignment]

                if build_xgboost_regressor is not None:
                    try:
                        def _as_np1(s: pd.Series) -> np.ndarray:
                            return np.asarray(
                                pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64),
                                dtype=np.float64,
                            ).reshape(-1)

                        yhat_val = _as_np1(val_dense["yhat"])
                        ytrue_val = _as_np1(val_dense["y_true"])
                        y_res_val = (ytrue_val - yhat_val).reshape(-1)

                        df_src = df_input if isinstance(df_input, pd.DataFrame) else data_blk.get("dataframe")
                        if not isinstance(df_src, pd.DataFrame) or df_src.empty:
                            raise ValueError("missing dataframe")

                        v_len = int(len(val_dense))
                        te_len = int(len(test_dense)) if isinstance(test_dense, pd.DataFrame) else 0
                        t_len = max(0, int(len(df_src)) - v_len - te_len)

                        lags = rm_cfg.get("lags") or [1, 2, 3, 6, 12, 24]
                        rolls = rm_cfg.get("rolling_windows") or [6, 12, 24, 48]
                        diffs = rm_cfg.get("diffs") or [1, 24]
                        base_features = rm_cfg.get("feature_cols")
                        if not isinstance(base_features, list) or not any(isinstance(x, str) and x.strip() for x in base_features):
                            base_features = ["month", "day_of_month", "day_of_week", "hour", "day_of_year"]
                            base_features += [f"lag_{int(k)}" for k in lags if int(k) > 0]
                            for w in rolls:
                                wi = int(w)
                                if wi > 0:
                                    base_features += [f"rolling_mean_{wi}", f"rolling_std_{wi}"]
                            base_features += [f"diff_{int(k)}" for k in diffs if int(k) > 0]

                        from utils.feature_contract import ensure_calendar_features, is_recomputable_name, recompute_feature_column

                        feat_df = df_src.copy()
                        try:
                            if time_col in feat_df.columns:
                                feat_df = ensure_calendar_features(feat_df, time_col=time_col)
                        except Exception:
                            pass

                        computed_cols: List[str] = []
                        for c in base_features:
                            if not isinstance(c, str) or not c.strip() or c == time_col:
                                continue
                            if c in feat_df.columns:
                                try:
                                    feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce")
                                    computed_cols.append(c)
                                except Exception:
                                    continue
                            elif is_recomputable_name(c):
                                try:
                                    feat_df[c] = recompute_feature_column(feat_df, c, value_col=value_col, time_col=time_col)
                                    computed_cols.append(c)
                                except Exception:
                                    continue

                        feat_val = feat_df.iloc[t_len : t_len + v_len].reset_index(drop=True)
                        feat_test = (
                            feat_df.iloc[t_len + v_len : t_len + v_len + te_len].reset_index(drop=True) if te_len > 0 else None
                        )

                        Xv = pd.DataFrame({"yhat": yhat_val[: len(feat_val)]})
                        for c in computed_cols:
                            if c in feat_val.columns:
                                Xv[c] = pd.to_numeric(feat_val[c], errors="coerce")

                        Xt = None
                        if isinstance(test_dense, pd.DataFrame) and not test_dense.empty and feat_test is not None:
                            yhat_test = _as_np1(test_dense["yhat"])
                            Xt = pd.DataFrame({"yhat": yhat_test[: len(feat_test)]})
                            for c in computed_cols:
                                if c in feat_test.columns:
                                    Xt[c] = pd.to_numeric(feat_test[c], errors="coerce")

                        y_res_val = y_res_val[: len(Xv)]
                        train_mask = np.isfinite(y_res_val) & np.isfinite(Xv["yhat"].to_numpy(dtype=np.float64))
                        n_all = int(np.sum(train_mask))
                        if n_all < 20:
                            raise ValueError("too few valid rows")

                        Xv_fit = Xv.to_numpy(dtype=np.float32)[train_mask]
                        yv_fit = y_res_val.astype(np.float32, copy=False)[train_mask]

                        try:
                            es_rounds = int(
                                (rm_cfg.get("early_stopping_rounds") if isinstance(rm_cfg, dict) else None)
                                or ((config.get("model_config") or {}).get("XGBoost", {}) or {}).get("early_stopping_rounds", 0)
                                or 0
                            )
                        except Exception:
                            es_rounds = 0

                        split = int(max(10, min(n_all - 10, int(n_all * 0.8))))
                        Xtr, ytr = Xv_fit[:split], yv_fit[:split]
                        Xev, yev = Xv_fit[split:], yv_fit[split:]

                        mdl = build_xgboost_regressor(config)
                        eval_set = [(Xev, yev)] if (int(es_rounds) > 0 and Xev.size and np.isfinite(yev).any()) else []

                        import inspect

                        fit_kwargs: Dict[str, Any] = {}
                        try:
                            sig = inspect.signature(mdl.fit)
                            fit_params = sig.parameters
                        except Exception:
                            fit_params = {}
                        if eval_set and "eval_set" in fit_params:
                            fit_kwargs["eval_set"] = eval_set
                        if "verbose" in fit_params:
                            fit_kwargs["verbose"] = False
                        if eval_set and int(es_rounds) > 0:
                            es = max(1, int(es_rounds))
                            if "early_stopping_rounds" in fit_params:
                                fit_kwargs["early_stopping_rounds"] = es
                            elif "callbacks" in fit_params:
                                try:
                                    import xgboost as xgb  # type: ignore

                                    fit_kwargs["callbacks"] = [xgb.callback.EarlyStopping(rounds=es, save_best=True)]
                                except Exception:
                                    pass
                        try:
                            mdl.fit(Xtr, ytr, **fit_kwargs)
                        except TypeError:
                            minimal: Dict[str, Any] = {}
                            if eval_set and "eval_set" in fit_params:
                                minimal["eval_set"] = eval_set
                            if "verbose" in fit_params:
                                minimal["verbose"] = False
                            mdl.fit(Xtr, ytr, **minimal)

                        res_hat_val = mdl.predict(Xv.to_numpy(dtype=np.float32)).astype(np.float64, copy=False).reshape(-1)
                        lv = int(min(len(val_dense), len(res_hat_val)))
                        if lv > 0:
                            val_dense = val_dense.copy()
                            y0 = _as_np1(val_dense["yhat"])
                            col_i = int(list(val_dense.columns).index("yhat"))
                            val_dense.iloc[:lv, col_i] = y0[:lv] + res_hat_val[:lv]

                        if isinstance(test_dense, pd.DataFrame) and not test_dense.empty and Xt is not None:
                            res_hat_test = mdl.predict(Xt.to_numpy(dtype=np.float32)).astype(np.float64, copy=False).reshape(-1)
                            lt = int(min(len(test_dense), len(res_hat_test)))
                            if lt > 0:
                                test_dense = test_dense.copy()
                                y0t = _as_np1(test_dense["yhat"])
                                col_it = int(list(test_dense.columns).index("yhat"))
                                test_dense.iloc[:lt, col_it] = y0t[:lt] + res_hat_test[:lt]

                        try:
                            path = (config.get("artifacts") or {}).get("xgboost_residual_model_path")
                            if isinstance(path, str) and path:
                                mdl.save_model(path)
                                artifacts["xgboost_residual_model_path"] = path
                        except Exception:
                            pass

                        data_blk["residual_applied"] = True
                        artifacts["residual_model_type"] = "xgboost"
                        residual_features = ["yhat"] + list(computed_cols)
                        data_blk["residual_report"] = {
                            "model_type": "xgboost",
                            "features": residual_features,
                            "early_stopping_rounds": int(es_rounds or 0),
                            "n_train_rows": int(n_all),
                        }
                        artifacts["residual_feature_cols"] = residual_features
                        print("[pipeline][registry-route] residual modeling applied (xgboost).")
                    except Exception as fit_e:
                        print(f"[pipeline][registry-route] residual modeling skipped (xgboost failed): {fit_e}")

            else:
                try:
                    from sklearn.linear_model import Lasso, LinearRegression, Ridge

                    model_cls = LinearRegression
                    if model_type == "ridge":
                        model_cls = Ridge
                    elif model_type == "lasso":
                        model_cls = Lasso
                except Exception as imp_e:
                    print(f"[pipeline][registry-route] residual model import failed: {imp_e}")
                    model_cls = None

                if model_cls is not None:
                    x_val_res = np.asarray(val_dense[["yhat"]].to_numpy(dtype=np.float64), dtype=np.float64)
                    y_val_res = (
                        np.asarray(val_dense["y_true"].to_numpy(dtype=np.float64), dtype=np.float64)
                        - np.asarray(val_dense["yhat"].to_numpy(dtype=np.float64), dtype=np.float64)
                    ).reshape(-1)

                    try:
                        res_mdl = model_cls()
                        res_mdl.fit(x_val_res, y_val_res)

                        val_dense = val_dense.copy()
                        val_yhat = np.asarray(val_dense["yhat"].to_numpy(dtype=np.float64), dtype=np.float64)
                        val_dense["yhat"] = val_yhat + res_mdl.predict(x_val_res)

                        if isinstance(test_dense, pd.DataFrame) and not test_dense.empty:
                            x_test_res = np.asarray(test_dense[["yhat"]].to_numpy(dtype=np.float64), dtype=np.float64)
                            test_dense = test_dense.copy()
                            test_yhat = np.asarray(test_dense["yhat"].to_numpy(dtype=np.float64), dtype=np.float64)
                            test_dense["yhat"] = test_yhat + res_mdl.predict(x_test_res)

                        artifacts["residual_model"] = res_mdl
                        artifacts["residual_model_type"] = model_type
                        data_blk["residual_applied"] = True
                        print("[pipeline][registry-route] residual modeling applied.")
                    except Exception as fit_e:
                        print(f"[pipeline][registry-route] residual modeling skipped (fit failed): {fit_e}")
    except Exception as e:
        print(f"[pipeline][registry-route] residual modeling skipped: {e}")

    return val_dense, test_dense
