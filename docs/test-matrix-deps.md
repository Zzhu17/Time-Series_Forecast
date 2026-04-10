# Test Matrix Dependencies

本文档定义 CI 的两套依赖与 skip 策略，用于收敛 optional dependency 导致的测试噪声。

## 1) minimal 依赖（`test-minimal`）

- 安装文件：`requirements-ci.txt`
- 目标：快速反馈，覆盖 lint / type / 核心测试
- skip 策略：**允许 skip**，但必须记录 skip 原因（用于后续治理）
- 统一 skip reason 前缀：
  - `TEST_MATRIX_OPTIONAL_DEP_MISSING: <package>`

## 2) full 依赖（`test-full`）

- 安装文件：
  - `Project/requirements.txt`
  - `requirements-ci.txt`
- 目标：重依赖完整验证（包含 torch / prophet / optuna / xgboost 等）
- skip 策略：
  - 默认要求 **0 skip**
  - 仅允许平台限制类 skip（需显式标注）：
    - `TEST_MATRIX_PLATFORM_SKIP: <reason>`

## 3) 统计口径

建议从 `pytest -rs` 输出中按 reason 前缀聚合：

- `TEST_MATRIX_OPTIONAL_DEP_MISSING`
- `TEST_MATRIX_PLATFORM_SKIP`
- 其他（需人工排查并规范化）

目标：`test-full` 场景下 skip 收敛到 **0~1**（仅平台限制项）。
