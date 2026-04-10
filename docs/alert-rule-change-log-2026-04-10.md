# Alert Rule 变更说明（2026-04-10）

## 背景
为完成 degrade 告警“可触发 -> 可恢复确认”的闭环，本次对告警规则做了两项增补。

## 变更内容

### 1) 新增 `FallbackErrorSpike`
- **目的**：对 `reason="fallback_error"` 的集中爆发进行专项告警，避免主告警被稀释。
- **表达式**：`sum(increase(tsf_degrade_total{reason="fallback_error"}[10m])) >= 5`
- **触发窗口**：持续 10 分钟。
- **等级**：`critical`。

### 2) 新增 `DegradeRateRecoveredInfo`
- **目的**：在主 degrade 速率回落后提供“恢复态”信号，帮助值班完成关闭确认。
- **表达式**：`sum(rate(tsf_degrade_total[10m])) < 0.02`
- **触发窗口**：持续 15 分钟。
- **等级**：`info`。

## 预期收益
- 降低 fallback 失败场景的漏报概率。
- 提供标准化恢复信号，减少“恢复但未关闭”或“过早关闭”的操作偏差。

## 风险与注意事项
- `DegradeRateRecoveredInfo` 为信息告警，不应单独作为恢复凭据；需结合 runbook 第 7 节执行关闭验收。
- 若业务基线流量显著波动，建议按环境（staging/prod）微调阈值。
