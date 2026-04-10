# CI SLO（近 14 天）

Date: 2026-04-10
Window: 2026-03-28 ~ 2026-04-10（14 天）

## 1) SLO 定义与采集口径

| SLO | 定义 | 统计口径（按天） | 数据来源 |
|---|---|---|---|
| CI 通过率 | 当天成功工作流数 / 当天总工作流数 | `success_runs / total_runs * 100%` | CI 平台 workflow run 事件 |
| 平均时长 | 当天全部工作流平均耗时（分钟） | `avg(run_duration_minutes)` | CI 平台 workflow run 事件 |
| Flaky rate | 当天 flaky 用例数 / 当天执行用例总数 | `flaky_tests / total_tests * 100%` | 测试报告（重跑后通过、同 SHA 结果不稳定） |
| 非白名单 skip 数 | 当天不在白名单中的 skip 总数 | `count(skip_reason not in whitelist)` | `pytest -rs` 日志聚合 |

### 白名单 skip reason

- `TEST_MATRIX_PLATFORM_SKIP:*`

其余 skip reason 一律计入“非白名单 skip 数”。

## 2) 最近 14 天趋势

> 当前仓库仅包含本地代码快照，未配置可直接查询的远端 CI run 历史（无 `origin` 远端）。
> 因此本表已先落地字段与窗口；待接入 CI API 或现有监控看板后按同口径回填。

| 日期 (UTC) | CI 通过率 | 平均时长 (min) | Flaky rate | 非白名单 skip 数 |
|---|---:|---:|---:|---:|
| 2026-03-28 | N/A | N/A | N/A | N/A |
| 2026-03-29 | N/A | N/A | N/A | N/A |
| 2026-03-30 | N/A | N/A | N/A | N/A |
| 2026-03-31 | N/A | N/A | N/A | N/A |
| 2026-04-01 | N/A | N/A | N/A | N/A |
| 2026-04-02 | N/A | N/A | N/A | N/A |
| 2026-04-03 | N/A | N/A | N/A | N/A |
| 2026-04-04 | N/A | N/A | N/A | N/A |
| 2026-04-05 | N/A | N/A | N/A | N/A |
| 2026-04-06 | N/A | N/A | N/A | N/A |
| 2026-04-07 | N/A | N/A | N/A | N/A |
| 2026-04-08 | N/A | N/A | N/A | N/A |
| 2026-04-09 | N/A | N/A | N/A | N/A |
| 2026-04-10 | N/A | N/A | N/A | N/A |

## 3) 达标线（需连续 7 天）

- CI 通过率 **≥ 98%**
- 非白名单 skip 数 **= 0**
- Flaky rate **≤ 1%**

判定规则：
- 以自然日（UTC）为单位滚动评估。
- 仅当以上 3 项在连续 7 天全部满足，才判定“CI 收尾阶段达标”。

## 4) 采集落地说明

- `scripts/run_test_matrix.sh full` 已在 full 矩阵对非平台 skip 执行硬失败（可用于“非白名单 skip = 0”约束）。
- 建议在 CI 平台侧新增每日聚合任务，导出：`total_runs, success_runs, avg_duration_minutes, flaky_tests, total_tests, non_whitelist_skips`。
- 本文档保留为 readiness 评审基线；接入看板后仅需更新第 2 节趋势表。
