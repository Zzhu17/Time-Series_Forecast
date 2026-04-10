# MODEL_READINESS_REPORT

Date: 2026-04-08

## Scope
- Test dependency & collection hardening for API tests.
- Model test template added for standardized 7-tuple contract checks.
- Registry/catalog consistency uplift via shared `MODEL_CATALOG` source.
- Training params unified into runtime output + artifact persistence.
- API sync train endpoints now passthrough training parameter summary.
- Prophet rolling CV support added (config-gated).
- LSTM/Informer stability uplift (seed metadata + smoke-mode knobs + structured informer error).

## Readiness Summary
- Baseline/API contract tests are runnable in minimal envs where `httpx` may be absent (tests skip instead of collect-fail).
- Training pipeline now exposes `training_params` in:
  - returned task payload,
  - `artifacts.training_params`,
  - `training_params.json` file under run artifact dir (when available).
- Model catalog and runtime registry are aligned on a single shared definition.

## 跨模型参数可比性（Cross-model Parameter Comparability）
- 已定义并落地统一训练参数 schema，核心字段固定为：
  - `model`
  - `split`
  - `core_hparams`
  - `runtime`
  - `data_signature`
  - `trainer_version`
- 各模型 adaptor/trainer（ARIMA、Prophet、Informer、LSTM、RandomForest、XGBoost）均已映射到上述同名字段，语义一致：
  - `split`：统一为 `train_len/val_len/test_len`
  - `core_hparams`：模型核心超参（可用于 ablation 对比）
  - `runtime`：训练状态与运行时信息（如 `fit_status`、`seed`）
  - `data_signature`：数据规模与关键信号（如列名、特征列）
- 新增 schema 验证测试：每个模型训练后均执行统一 schema 校验，确保产物可直接进入自动 leaderboard、ablation、回归对比流程，无需手工对齐字段。

## Risks / Follow-ups
- Heavy model families (Prophet / Torch / XGBoost / ARIMA optional deps) still depend on environment package availability.
- Prophet rolling CV can be computationally expensive; should remain disabled by default in production.
- Consider adding explicit API schema field for `training_params` in typed response models.
