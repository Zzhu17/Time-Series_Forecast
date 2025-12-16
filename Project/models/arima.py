from typing import Any
import numpy as np
from pmdarima import auto_arima
from pmdarima.arima.utils import ndiffs, nsdiffs


def _get_arima_cfg(config: dict) -> dict:
  """
  从全局 config 读取 ARIMA/SARIMA 配置（保持兼容：若缺失则用安全默认值）
  """
  cfg = dict(config or {})
  mcfg = (cfg.get("model_config") or {}).get("ARIMA", {}) or {}
  # 基本项
  use_seasonal = bool(mcfg.get("use_seasonal", True))
  seasonal_period = int(mcfg.get("seasonal_period", 7))
  # 搜索范围
  max_p = int(mcfg.get("max_p", 5))
  max_d = int(mcfg.get("max_d", 2))
  max_q = int(mcfg.get("max_q", 5))
  max_P = int(mcfg.get("max_P", 2))
  max_D = int(mcfg.get("max_D", 1))
  max_Q = int(mcfg.get("max_Q", 2))
  stepwise = bool(mcfg.get("stepwise", True))
  seasonal_test = str(mcfg.get("seasonal_test", "ocsb"))
  enforce_stationarity = bool(mcfg.get("enforce_stationarity", True))
  enforce_invertibility = bool(mcfg.get("enforce_invertibility", True))
  return {
      "use_seasonal": use_seasonal,
      "seasonal_period": seasonal_period,
      "max_p": max_p, "max_d": max_d, "max_q": max_q,
      "max_P": max_P, "max_D": max_D, "max_Q": max_Q,
      "stepwise": stepwise,
      "seasonal_test": seasonal_test,
      "enforce_stationarity": enforce_stationarity,
      "enforce_invertibility": enforce_invertibility,
  }


def build_auto_arima(series: Any):
  """
  朴素 baseline：不开季节搜索，仅用于 plain 模式或回退。
  """
  return auto_arima(
      series,
      seasonal=False,
      stepwise=True,
      trace=False,              # 关闭冗长日志
      suppress_warnings=True,
      error_action='ignore',
      information_criterion="aic",
  )


def build_arima_model(y_train: Any, config: dict) -> Any:
  """
  强化版：动态估计 d/D，按样本规模与季节性自适应设置 max_p/q/P/Q，
  并在命中边界时进行一次或两次“渐进扩容”（progressive widening）。
  同时加入：
    - 小样本时自动禁用季节项（样本不足 2*s）
    - 自适应截距/漂移（d>=1 且 D==0 时 with_intercept=True）
    - 小样本采用 AICc，大样本采用 AIC
    - Box-Cox 方差稳定化（由 configs 开关）
  函数名保持不变。
  """
  c = _get_arima_cfg(config)

  # 模式开关：plain -> 直接使用朴素 auto_arima；smart -> 使用增强版（默认）
  mcfg = (config.get("model_config") or {}).get("ARIMA", {}) or {}
  if str(mcfg.get("mode", "smart")).lower() == "plain":
      return build_auto_arima(y_train)

  y_train = np.asarray(y_train, dtype=float).ravel()
  n = int(y_train.size)
  n = max(n, 1)

  # 读取 ARIMA 段的可选开关（如 boxcox）
  use_boxcox = bool(mcfg.get("boxcox", False))

  # 1) 基于统计检验估计 d, D（上限依旧由配置约束）
  try:
      est_d = ndiffs(y_train, test='adf', max_d=c["max_d"])
  except Exception:
      est_d = min(1, c["max_d"])
  try:
      est_D = nsdiffs(
          y_train,
          m=c["seasonal_period"],
          test='ocsb',
          max_D=c["max_D"]
      ) if c["use_seasonal"] and c["seasonal_period"] > 1 else 0
  except Exception:
      est_D = 0
  # 防御：不超过配置上限
  d_cap = int(min(est_d, c["max_d"]))
  D_cap = int(min(est_D, c["max_D"]))

  # 2) 按样本规模与季节性给出初始搜索上限（不会改写 configs，只在本函数内部生效）
  #    小样本避免过高阶；大样本适度放宽
  if n < 200:
      p0 = q0 = min(2, c["max_p"])
      P0 = Q0 = min(1, c["max_P"])
  elif n < 1000:
      p0 = q0 = min(5, c["max_p"])
      P0 = Q0 = min(2, c["max_P"])
  else:
      p0 = q0 = min(7, c["max_p"])
      P0 = Q0 = min(3, c["max_P"])

  # 针对季节性周期的额外限制（避免过拟合）
  use_seasonal = bool(c["use_seasonal"])
  s = int(c["seasonal_period"])
  if not use_seasonal or s <= 1:
      P0 = Q0 = 0

  # —— 小样本自动禁用季节项（样本长度不足 2*s）——
  if use_seasonal and s > 1 and n < 2 * s:
      use_seasonal = False
      D_cap = 0
      P0 = Q0 = 0

  # 确保参数量不超负荷：粗略限制 (p+q+P+Q) 不超过 n/10 且不超过 12
  budget = max(2, min(12, n // 10))
  while (p0 + q0 + P0 + Q0) > budget and (p0 > 1 or q0 > 1):
      if p0 >= q0 and p0 > 1:
          p0 -= 1
      elif q0 > 1:
          q0 -= 1

  # 最终用于 auto_arima 的搜索上限（取内部上限与配置上限的更小值）
  max_p = int(min(p0, c["max_p"]))
  max_q = int(min(q0, c["max_q"]))
  max_P = int(min(P0, c["max_P"]))
  max_Q = int(min(Q0, c["max_Q"]))

  # —— 自适应截距/漂移：有非季节差分且无季节差分时允许截距，避免直线回均值 —— 
  with_ic = (d_cap >= 1 and D_cap == 0)

  # —— 小样本用 AICc，大样本用 AIC —— 
  ic = "aicc" if n < 200 else "aic"

  def _fit_with_bounds(mp, md, mq, mP, mD, mQ):
      return auto_arima(
          y_train,
          seasonal=use_seasonal,
          m=s,
          start_p=0, start_q=0, start_P=0, start_Q=0,
          max_p=mp, max_d=md, max_q=mq,
          max_P=mP, max_D=mD, max_Q=mQ,
          stepwise=c["stepwise"],
          seasonal_test=c["seasonal_test"],
          enforce_stationarity=c["enforce_stationarity"],
          enforce_invertibility=c["enforce_invertibility"],
          suppress_warnings=True,
          error_action="ignore",
          trace=False,
          information_criterion=ic,
          d=d_cap, D=D_cap,
          with_intercept="auto" if with_ic else "False", 
          boxcox=use_boxcox        
      )

  # 3) 先用自适应的范围拟合
  model = _fit_with_bounds(max_p, d_cap, max_q, max_P, D_cap, max_Q)

  # 4) 渐进扩容：如果选出来的模型在任何一个维度上命中了上限，适度扩 1 并再试一次（最多两轮）
  def _hit_boundary(m, mp, mq, mP, mQ):
      try:
          (p, d, q) = m.order
          (P, D, Q, s_) = m.seasonal_order
          return (p >= mp) or (q >= mq) or (P >= mP) or (Q >= mQ)
      except Exception:
          return False

  rounds = 0
  while _hit_boundary(model, max_p, max_q, max_P, max_Q) and rounds < 2:
      # 温和放宽（不超过硬上限）
      max_p = min(max_p + 1, c["max_p"])
      max_q = min(max_q + 1, c["max_q"])
      if use_seasonal:
          max_P = min(max_P + 1, c["max_P"])
          max_Q = min(max_Q + 1, c["max_Q"])
      model = _fit_with_bounds(max_p, d_cap, max_q, max_P, D_cap, max_Q)
      rounds += 1

  try:
      print("[ARIMA][Fit] order=", model.order, "seasonal_order=", model.seasonal_order)
  except Exception:
      pass

  return model
