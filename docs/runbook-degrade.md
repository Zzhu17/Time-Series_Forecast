# Degrade 告警 Runbook

> 目标：把 degrade 从“可观测”提升到“可处置”。

## 1. 告警触发后的第一响应（0~5 分钟）

1. 在 Grafana 查看：
   - `Degrade Rate (5m/1h)` 是否持续上升。
   - `Top Degrade Reasons (1h)` 是否出现单一原因占比过高。
   - `Degrade Distribution by Model (1h)` 是否集中在某个模型。
2. 在 API 查看最近窗口摘要：
   - `GET /metrics/degrade_summary?window_minutes=60&limit=10`
3. 记录当前时间窗口、主导 reason、主导 model，作为排障起点。

## 2. 数据契约排查（Data Contract）

重点检查是否存在特征缺失、字段类型漂移、时间列异常：

1. 检查最新输入数据 schema 与训练时契约是否一致。
2. 核对关键字段：
   - `time_col` 是否可解析、是否单调递增。
   - `value_col` 是否存在大量空值/异常值。
3. 若 reason 包含 `feature_contract_fallback`：
   - 优先恢复标准特征输入链路。
   - 临时保底：启用已验证的最小特征集，避免持续失败。

## 3. 依赖与环境排查（Dependency）

1. 检查在线依赖健康状态：
   - 数据库/缓存/对象存储连接是否异常。
   - 上游特征服务是否超时或返回空。
2. 检查发布变更：
   - 最近 24h 是否升级了模型运行时或依赖包。
   - 是否存在配置漂移（环境变量、配置文件）。
3. 如出现大量超时或异常：
   - 先降流或切换到稳定版本配置，防止 degrade 扩散。

## 4. 模型可用性排查（Model Availability）

1. 检查主模型是否可加载：
   - 模型文件是否存在、版本是否匹配、加载是否报错。
2. 检查 reason：
   - `model_not_available` / `model_not_supported`：优先修复模型注册与路由。
   - `multi_step_not_supported` / `non_informer_one_step_only`：校验请求参数与模型能力是否匹配。
3. 若单模型异常明显：
   - 将流量切到次优稳定模型（按业务SLO优先）。

## 5. Fallback 链路排查

1. 确认 fallback 是否按预期触发且可返回结果。
2. 检查 fallback 质量是否低于业务最低阈值：
   - 若 fallback 可用但质量显著下降，需升级为事故处理。
3. 如 fallback 本身报错（`fallback_error`）：
   - 立即修复 fallback 依赖或回退到更基础策略（如 naive/baseline）。

## 6. 处置策略与回滚建议

1. **短期止血**：
   - 切换流量到稳定模型。
   - 降级高风险请求参数（如超大 horizon）。
2. **中期修复**：
   - 补齐数据契约校验并在入口强校验失败快返。
   - 为高频 reason 增加专项告警和自动化诊断。
3. **长期治理**：
   - 将 top reason 纳入周报，跟踪占比和MTTR。
   - 对高频模型建立可用性 SLO 与发布门禁。

## 7. 关闭告警前的验收标准

满足以下条件可考虑关闭：

1. `Degrade Rate (5m)` 回落并稳定在阈值以下。
2. 主导 reason 占比恢复正常（无单点原因持续放大）。
3. 主模型链路恢复，fallback 仅偶发触发。
4. 已补充事故记录（时间线、根因、修复与预防动作）。
