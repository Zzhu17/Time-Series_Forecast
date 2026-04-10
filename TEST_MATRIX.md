# TEST_MATRIX

Date: 2026-04-10
Owner: Platform + Modeling

## Day 1（稳定性）完成项

1. **环境脚本统一**：新增 `scripts/run_test_matrix.sh`，统一 minimal/full/models 三个执行入口，全部前置 `scripts/check_test_env.sh --strict`。
2. **skip 矩阵拆分**：
   - `minimal`：允许 skip，并强制归档 `skip-reasons-minimal.txt`。
   - `full`：仅允许 `TEST_MATRIX_PLATFORM_SKIP:*`。
   - `models`：重依赖模型链路单独执行，不与 full skip 规则混淆。

## 依赖矩阵

- `minimal`: `requirements-ci.txt`
- `full`: `Project/requirements.txt` + `requirements-ci.txt`

详见：`docs/test-matrix-deps.md`。

## 执行矩阵与命令（统一入口）

| Scenario | Command | Skip Policy | Artifact |
|---|---|---|---|
| minimal / lint + type + tests | `ruff check . && PYTHONPATH=Project mypy Project tests && scripts/run_test_matrix.sh minimal` | 允许 skip，必须有 reason | `skip-reasons-minimal.txt` |
| full / full tests | `scripts/run_test_matrix.sh full` | 仅允许 `TEST_MATRIX_PLATFORM_SKIP:*` | `pytest-full.log` |
| full / model focused | `scripts/run_test_matrix.sh models` | 目标 0 skip（依赖缺失应在环境层解决） | CI log |

## 统计口径（passed / failed / skipped）

- 统计值由 CI 运行结果回填。
- skip 判定由 `scripts/run_test_matrix.sh` 与 workflow 双重约束。
- 出现非策略 skip 时直接 fail，阻断合并。
