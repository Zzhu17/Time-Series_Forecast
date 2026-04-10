# TEST_MATRIX

Date: 2026-04-10

## 依赖矩阵

- `minimal`: `requirements-ci.txt`
- `full`: `Project/requirements.txt` + `requirements-ci.txt`

详见：`docs/test-matrix-deps.md`。

## 按场景统计（passed / failed / skipped）

| Scenario | Command | Passed | Failed | Skipped | Notes |
|---|---|---:|---:|---:|---|
| minimal / lint + type + tests | `ruff check . && PYTHONPATH=Project mypy Project tests && PYTHONPATH=Project pytest -q -rs tests` | - | - | - | 允许 skip，必须输出并归档 skip reason。 |
| full / full tests | `PYTHONPATH=Project pytest -q -rs tests` | - | - | **目标 0~1** | 仅允许 `TEST_MATRIX_PLATFORM_SKIP:*` 类型 skip。 |
| full / model focused | `make test-models` | - | - | **目标 0** | 覆盖重依赖模型集成链路。 |

> 统计值由 CI 运行结果回填；规则由 workflow 强制执行。
