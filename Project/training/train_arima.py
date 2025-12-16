# train_arima.py  — 强化版（仅本文件改动）
# 依赖：metrics.rmse / utils.array_utils.clean_and_unify_arrays / models.arima.build_arima_model

import numpy as np
from sklearn.metrics import mean_absolute_error
from utils.array_utils import clean_and_unify_arrays
from models.arima import build_arima_model

# 统一度量：使用全局 metrics.rmse（避免各处口径不一致）
try:
    from evaluation.metrics import compute_rmse as _rmse
except Exception:
    # 极端兜底：如 metrics.rmse 不存在时，用 sklearn 的 root_mean_squared_error / mean_squared_error
    try:
        from sklearn.metrics import root_mean_squared_error as _rmse  # sklearn >= 1.4
    except Exception:
        from sklearn.metrics import mean_squared_error as _mse
        def _rmse(y_true, y_pred):
            return _mse(y_true, y_pred, squared=False)

# 可选：用于残差自相关诊断
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
except Exception:
    acorr_ljungbox = None


def _arima_runtime_cfg(config):
    """
    读取 ARIMA 相关的运行时配置（滚动、切分、诊断），带安全默认值。
    """
    cfg = dict(config or {})
    # 顶层 default
    default_blk = cfg.get("default") or {}
    value_col = cfg.get("value_col", default_blk.get("value_col", "value"))

    mcfg = (cfg.get("model_config") or {}).get("ARIMA", {}) or {}
    roll = (mcfg.get("rolling") or {})
    trn  = (mcfg.get("train") or {})
    diag = (mcfg.get("diagnostics") or {})

    # 全局 split 优先生效；缺省再回 ARIMA.train
    sp = cfg.get("split") or {}
    r_train = float(sp.get("train", 0.6))
    r_val   = float(sp.get("val",   trn.get("val_ratio", 0.2)))
    r_test  = float(sp.get("test",  trn.get("test_ratio", 0.2)))

    return {
        "value_col": value_col,
        "rolling_enabled": bool(roll.get("enabled", True)),
        "refit_every": int(roll.get("refit_every", 3)),          # 默认更积极：3
        "forecast_h": int(roll.get("forecast_h", 1)),            # 必须 1 步
        "use_refit_on_exception": bool(roll.get("use_refit_on_exception", True)),
        "ljungbox_alpha": float(diag.get("ljungbox_alpha", 0.05)),
        "save_plots": bool(diag.get("save_plots", False)),
        "plot_path": str(diag.get("plot_path", "artifacts/arima_diagnostics")),
        "r_train": r_train, "r_val": r_val, "r_test": r_test,
    }


def _rolling_forecast(base_hist, future_true, config, initial_model=None):
    """
    逐步预测：每步预测 forecast_h（默认=1），并在 refit_every 间隔重拟合。
    避免一次性长步 forecast 退化成“直线/台阶”。
    """
    rcfg = _arima_runtime_cfg(config)
    forecast_h = rcfg["forecast_h"]
    refit_every = rcfg["refit_every"]
    use_refit_on_exception = rcfg["use_refit_on_exception"]

    preds = []
    hist = np.asarray(base_hist, dtype=float).ravel()
    local_model = initial_model

    for t in range(len(future_true)):
        # 周期重拟合或首次拟合
        if (t == 0) or (t % refit_every == 0) or (local_model is None):
            local_model = build_arima_model(hist, config)
            # 打印本次模型阶数（便于观察是否仍退化到 (0,1,1)）
            try:
                print(f"[ARIMA][Fit] order= {getattr(local_model,'order',None)} "
                      f"seasonal_order= {getattr(local_model,'seasonal_order',None)}")
            except Exception:
                pass

        try:
            yhat = local_model.predict(n_periods=forecast_h)
            preds.append(float(np.asarray(yhat, dtype=float).ravel()[-1]))
        except Exception:
            # 出错时按需立即重拟合一次
            if use_refit_on_exception:
                local_model = build_arima_model(hist, config)
                yhat = local_model.predict(n_periods=forecast_h)
                preds.append(float(np.asarray(yhat, dtype=float).ravel()[-1]))
            else:
                # 兜底：用上一时刻值
                last = float(hist[-1]) if hist.size else 0.0
                preds.append(last)

        # 将“当前已观测的真值”并回历史（不泄漏未来）
        hist = np.concatenate([hist, [float(future_true[t])]], axis=0)

    return np.asarray(preds, dtype=float)


def train_arima_model(df, config):
    """
    强化版训练：SARIMA + auto_arima(智能范围) + 逐点滚动 + 周期重拟合 + 偏置/方差校准 + Ljung–Box 诊断
    返回签名保持不变：
        val_true, val_forecast, test_true, test_forecast, final_model, test_forecast_df, best_params
    """
    # === 配置与取列 ===
    rcfg = _arima_runtime_cfg(config)
    value_col = rcfg["value_col"]

    # === 读取序列，规整为 1D float，去掉 NaN/inf ===
    series_all = np.asarray(df[value_col], dtype=float).ravel()
    if series_all.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, None, None, None
    mask = np.isfinite(series_all)
    series = series_all[mask]
    if series.size < 5:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, None, None, None

    # === 切分（内联 6:2:2 或 configs 中的比例）===
    n_total = int(len(series))
    n_train = max(1, int(round(n_total * rcfg["r_train"])))
    n_val   = max(0, int(round(n_total * rcfg["r_val"])))
    n_test  = max(0, n_total - n_train - n_val)
    y_train = series[:n_train]
    y_val   = series[n_train:n_train + n_val]
    y_test  = series[n_train + n_val:n_train + n_val + n_test]
    print(f"[ARIMA][Split] n_train={len(y_train)} n_val={len(y_val)} n_test={len(y_test)}")

    # === 初次拟合（仅用训练段） ===
    base_model = build_arima_model(y_train, config)
    try:
        print(f"[ARIMA][Model] order={getattr(base_model,'order',None)} "
              f"seasonal_order={getattr(base_model,'seasonal_order',None)} "
              f"with_intercept={getattr(base_model,'with_intercept',None)}")
    except Exception as e:
        print(f"[ARIMA][Model] info unavailable: {e}")

    # === 验证/测试：逐点滚动 + 周期重拟合 ===
    if rcfg["rolling_enabled"]:
        y_val_pred  = _rolling_forecast(y_train, y_val,  config, initial_model=base_model) if y_val.size else np.array([], float)
        hist_tv     = np.concatenate([y_train, y_val], axis=0)
        y_test_pred = _rolling_forecast(hist_tv, y_test, config, initial_model=None)       if y_test.size else np.array([], float)
    else:
        # 仅用于调试：一次性静态预测（不建议）
        y_val_pred  = base_model.predict(n_periods=len(y_val)).astype(float).ravel()  if y_val.size  else np.array([], float)
        y_test_pred = base_model.predict(n_periods=len(y_test)).astype(float).ravel() if y_test.size else np.array([], float)

    # === 对齐（各自分段内对齐，避免错配） ===
    val_true,  val_forecast,  _ = clean_and_unify_arrays(y_val,  y_val_pred)
    test_true, test_forecast, _ = clean_and_unify_arrays(y_test, y_test_pred)

    # === 偏置校准（Bias） ===
    try:
        pred_cfg = (config.get("prediction") or {})
        roll_cfg = (pred_cfg.get("rolling") or {})
        do_calib = bool(roll_cfg.get("calibrate", True))
        mcfg     = (config.get("model_config") or {}).get("ARIMA", {}) or {}
        m        = int(mcfg.get("seasonal_period", 7))
        if do_calib and val_true.size and val_forecast.size:
            K = int(max(7, min(2 * max(1, m), val_true.size)))  # K≈2*s
            bias = float((val_true[-K:] - val_forecast[-K:]).mean())
            val_forecast  = (val_forecast  + bias).astype(float)
            test_forecast = (test_forecast + bias).astype(float) if test_forecast.size else test_forecast
            print(f"[ARIMA][Calibrate] use K={K}, bias={bias:.4f}")
    except Exception as e:
        print(f"[ARIMA][Calibrate] skip: {e}")

    # === 方差校准（Variance） ===
    try:
        pred_cfg = (config.get("prediction") or {})
        roll_cfg = (pred_cfg.get("rolling") or {})
        var_cfg  = (roll_cfg.get("variance_calibration") or {})
        if bool(var_cfg.get("enabled", True)) and val_true.size and val_forecast.size:
            mcfg   = (config.get("model_config") or {}).get("ARIMA", {}) or {}
            m      = int(mcfg.get("seasonal_period", 7))
            k_mult = float(var_cfg.get("k_mult", 2.0))
            K_var  = int(max(7, min(int(max(1, k_mult) * max(1, m)), val_true.size)))
            if K_var > 5 and val_true.size >= K_var and val_forecast.size >= K_var:
                std_true = float(np.std(val_true[-K_var:], ddof=1))
                std_pred = float(np.std(val_forecast[-K_var:], ddof=1))
                if std_pred > 0:
                    lb = float(var_cfg.get("ratio_lb", 0.7))
                    ub = float(var_cfg.get("ratio_ub", 1.5))
                    if lb > ub:
                        lb, ub = ub, lb
                    r = std_true / std_pred
                    r = max(lb, min(ub, r))
                    mu_v = float(np.mean(val_forecast[-K_var:]))
                    val_forecast  = (mu_v + (val_forecast  - mu_v) * r).astype(float)
                    if test_forecast.size:
                        mu_t = float(np.mean(test_forecast[:min(len(test_forecast), K_var)]))
                        test_forecast = (mu_t + (test_forecast - mu_t) * r).astype(float)
                    print(f"[ARIMA][VarCalib] K={K_var}, std_true={std_true:.4f}, std_pred={std_pred:.4f}, "
                          f"ratio={r:.3f}, bounds=[{lb:.2f},{ub:.2f}]")
            else:
                print(f"[ARIMA][VarCalib] skip: insufficient window (K={K_var}, len(val)={len(val_true)})")
        else:
            print("[ARIMA][VarCalib] disabled or no data")
    except Exception as e:
        print(f"[ARIMA][VarCalib] skip: {e}")

    # === 指标 ===
    try:
        if val_true.size and val_forecast.size:
            rmse_val = _rmse(val_true, val_forecast)
            mae_val  = mean_absolute_error(val_true, val_forecast)
            print(f"[ARIMA][VAL]  RMSE={rmse_val:.4f}  MAE={mae_val:.4f}")
    except Exception:
        pass
    try:
        if test_true.size and test_forecast.size:
            rmse_test = _rmse(test_true, test_forecast)
            mae_test  = mean_absolute_error(test_true, test_forecast)
            print(f"[ARIMA][TEST] RMSE={rmse_test:.4f}  MAE={mae_test:.4f}")
    except Exception:
        pass

    # === Ljung–Box 诊断（验证段残差） ===
    try:
        if acorr_ljungbox is not None and val_true.size > 3 and val_true.size == val_forecast.size:
            resid_val = (val_true - val_forecast)
            lag = max(1, min(10, int(len(resid_val) // 4)))  # 经验：min(10, len/4)
            lb = acorr_ljungbox(resid_val, lags=[lag], return_df=True)
            pval = float(lb["lb_pvalue"].iloc[-1])
            alpha = rcfg["ljungbox_alpha"]
            conclusion = "OK(未见显著自相关)" if pval > alpha else "Not OK(残差可能自相关)"
            print(f"[ARIMA][Ljung-Box] lag={lag}, p={pval:.4f}, alpha={alpha:.2f} => {conclusion}")
        else:
            print("[ARIMA][Ljung-Box] 跳过（statsmodels 未安装或样本太少）")
    except Exception as e:
        print(f"[ARIMA][Ljung-Box] 诊断失败：{e}")

    # === 输出保持原签名 ===
    final_model = base_model
    test_forecast_df = None
    try:
        best_params = getattr(final_model, "get_params", lambda: None)()
    except Exception:
        best_params = None

    return val_true, val_forecast, test_true, test_forecast, final_model, test_forecast_df, best_params