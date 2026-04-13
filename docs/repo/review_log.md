# Review Log

## 主执行链路（第一阶段相关）

1. `Project/cli/run_pipeline.py`
   读取 CSV/Parquet，解析命令行参数，构造训练配置。
2. `Project/services/training_payloads.py`
   负责把外部请求规范化为统一训练 payload，并生成 feature contract report。
3. `Project/services/train_service.py`
   负责加载基础配置、生成 run 级 artifacts 路径、调用 pipeline、执行质量门禁并注册模型记录。
4. `Project/services/pipeline_loader.py`
   负责按需 reload `services.pipeline`，避免长生命周期进程缓存旧模块。
5. `Project/services/pipeline.py`
   负责主流程编排：数据预处理、trainer dispatch、结果标准化、降级路径、snapshot/leaderboard/report 写回。
6. `Project/models/registry.py`
   提供 trainer/forecaster 的懒加载注册表。
7. `Project/training/adaptor/informer_adaptor.py`
   把 Informer 训练入口适配为统一 7 元组输出。
8. `Project/models/informer/train.py`
   执行 Informer 训练、滚动预测、校准、残差建模和 artifacts 写回。

---

文件名：`Project/cli/run_pipeline.py`
测试覆盖状态：第二级
职责描述：CLI 入口，读取本地数据并调用统一训练/预测主链路。
目前最大的冗余点：CLI 自己做 `feature_cols` 自动推断和 payload 组装，这与 `training_payloads`/`train_service` 已有逻辑重复；结果汇总直接原样 JSON 序列化，也把下游内部对象暴露给入口层。
哪些逻辑应该保留：参数解析、文件读取、运行 ID 生成、将结果写到 CLI 输出文件。
哪些逻辑应该合并或删除：本地重复的 `feature_cols` 推断与配置拼装应收敛到 service 层统一入口；结果序列化应复用 snapshot/safe-json 逻辑，而不是直接 `json.dump(results)`。
哪些逻辑应该延后到别的阶段执行：任何模型相关默认决策都不应在 CLI 层做，应延后到训练配置规范化阶段。
依赖关系：影响 `services.train_service`、`services.training_payloads`、`services.snapshot`、CLI 文档说明。
建议的行动类型：保守精简

文件名：`Project/services/training_payloads.py`
测试覆盖状态：第一级
职责描述：把外部训练请求标准化为受 schema 约束的训练 payload，并生成特征契约报告。
目前最大的冗余点：自动特征推断职责与 CLI/其他调用入口重复；当前文件同时处理 schema 校验、特征默认推断、契约报告三类决策。
哪些逻辑应该保留：`TrainingPayload` 校验后统一输出 `normalized payload + contract_report` 这条边界。
哪些逻辑应该合并或删除：自动特征推断触发条件需要成为单一可信入口，避免 CLI、API、service 各自做一遍；契约报告构建可继续保留，但默认值来源应收敛到配置层。
哪些逻辑应该延后到别的阶段执行：与运行环境绑定的默认配置解析不应散落在不同调用方，应延后到统一 config resolution 阶段。
依赖关系：影响 CLI、API 训练入口、`services.train_service`、feature contract 校验。
建议的行动类型：结构重构

文件名：`Project/services/train_service.py`
测试覆盖状态：第一级
职责描述：组装训练运行时配置，调用 pipeline，并在训练结束后执行质量门禁和模型注册。
目前最大的冗余点：同一个文件同时负责配置组装、artifacts 路径、质量门禁、持久化、模型注册与训练任务编排，职责过多；部分默认值（如后校准、目标变换）与配置文件形成双重来源。
哪些逻辑应该保留：训练任务统一入口、质量门禁判定、训练参数持久化、模型注册。
哪些逻辑应该合并或删除：`build_training_config` 中的默认策略应与 YAML 配置收敛成单一来源；latest report / old run cleanup / gate suggestion 可拆成独立 helper 模块，降低训练主入口负担。
哪些逻辑应该延后到别的阶段执行：报告发布和目录清理不应与训练主成功路径强耦合，可延后到后处理阶段。
依赖关系：影响 API/CLI 训练入口、模型注册表、artifacts 目录结构、质量门禁测试。
建议的行动类型：结构重构

文件名：`Project/services/pipeline_loader.py`
测试覆盖状态：第二级
职责描述：为长生命周期运行环境提供 `services.pipeline` 的 reload 包装。
目前最大的冗余点：功能单一但未被更高层接口隐藏，调用方需要知道 reload 语义；缺少正式单测验证 reload 行为。
哪些逻辑应该保留：模块 reload 封装本身。
哪些逻辑应该合并或删除：可将 reload 语义进一步隐藏在 service 层，不让更多调用方直接感知动态 import。
哪些逻辑应该延后到别的阶段执行：无。
依赖关系：影响 `services.train_service` 和任何直接依赖动态加载的入口。
建议的行动类型：保守精简

文件名：`Project/services/pipeline.py`
测试覆盖状态：第二级
职责描述：统一编排训练、预测、降级、结果标准化与产物回写的主流程文件。
目前最大的冗余点：文件过大且职责过宽，既做 orchestration，又做数据标准化、异常映射、绘图序列组装、leaderboard/report 写回、降级结果构造；还存在 registry 调度之外的 Informer 直连回退，说明入口未完全收敛。
哪些逻辑应该保留：trainer dispatch、主流程状态组织、统一结果结构、降级路径、最小必要的 pipeline 编排。
哪些逻辑应该合并或删除：结果标准化 helper、错误 payload 构造、dense dataframe 归一化、leaderboard/report 写回应拆出独立模块；Informer 直连回退如果已被 registry 覆盖，应评估删除或明确保留条件。
哪些逻辑应该延后到别的阶段执行：report/leaderboard 生成、连续绘图 payload 拼装、部分 UI/Streamlit 定制输出应延后到后处理阶段，而不是压在主训练路径中。
依赖关系：影响几乎所有训练/预测入口、snapshot、report、feature pipeline、registry、Informer trainer。
建议的行动类型：保守精简

文件名：`Project/models/registry.py`
测试覆盖状态：第一级
职责描述：维护模型、trainer、forecaster 的懒加载注册表及对外模型元数据。
目前最大的冗余点：`MODEL_REGISTRY`、`TRAINER_REGISTRY`、`FORECASTER_REGISTRY` 与 `SUPPORTED_MODELS`/`MODEL_CATALOG` 存在多份并行定义，新增模型时需要维护多处。
哪些逻辑应该保留：懒加载代理、统一的产品级模型元数据。
哪些逻辑应该合并或删除：可继续把多张 registry 表收敛到单一声明源，再派生出不同视图，减少新增模型时的重复编辑。
哪些逻辑应该延后到别的阶段执行：无。
依赖关系：影响训练/预测 dispatch、模型列表 API、catalog contract 测试。
建议的行动类型：结构重构

文件名：`Project/training/adaptor/informer_adaptor.py`
测试覆盖状态：第一级
职责描述：把 Informer 训练结果适配成统一 7 元组格式，供 registry 和通用流程消费。
目前最大的冗余点：adapter 内重复承担了结果解析、split 回推、training_params 生成和 smoke config 注入；这些职责一部分与 pipeline 的结果标准化重叠。
哪些逻辑应该保留：Informer 与统一 trainer 接口之间的适配边界。
哪些逻辑应该合并或删除：`result_df`/`data_blk` 双路径解析可抽成公共 helper；训练参数组装应尽量走统一 builder，避免 adapter 单独推导 split/epochs。
哪些逻辑应该延后到别的阶段执行：smoke-only 参数覆盖不应在适配器里散落，可延后到更高层测试配置装配阶段。
依赖关系：影响 Informer 训练入口、registry 调度、Informer 相关测试。
建议的行动类型：结构重构

文件名：`Project/models/informer/train.py`
测试覆盖状态：第二级
职责描述：执行 Informer 训练并在同一文件内完成验证/测试滚动预测、校准、残差建模和部分 artifacts 写回。
目前最大的冗余点：单文件承担了训练循环、特征筛选、目标变换、滚动预测、校准、残差修正、baseline 统计和 artifacts 回写，属于典型“研究脚本式全栈文件”；第一阶段要关心的 debug/calibrate/residual 默认路径也都落在这里。
哪些逻辑应该保留：核心训练循环、窗口准备、模型前向、必要的早停与指标计算。
哪些逻辑应该合并或删除：校准、残差建模、滚动整段预测和基线统计应从训练主循环中解耦；日志/print 输出应统一接入配置化 logging，而不是散落在训练细节里。
哪些逻辑应该延后到别的阶段执行：校准、残差补偿、最终长序列后处理应延后到训练完成后的独立后处理阶段。
依赖关系：影响 Informer 训练表现、rolling predict、residual modeling、artifacts 写回与多项下游结果字段。
建议的行动类型：保守精简

---

## 当前优先级（只基于已审主链路）

1. `Project/cli/run_pipeline.py`
   原因：影响范围小，且存在明确的稳定性问题（CLI 结果序列化直接依赖下游对象可 JSON 化）。
2. `Project/services/training_payloads.py`
   原因：有正式测试覆盖，且 feature inference/default 决策重复，适合先收敛为单一入口。
3. `Project/services/train_service.py`
   原因：有正式测试覆盖，但职责较多，适合在 payload/default 策略收敛后再拆分。
4. `Project/training/adaptor/informer_adaptor.py`
   原因：有正式测试覆盖，可在不动主训练文件的前提下先消除一部分结果解析重复。
5. `Project/services/pipeline.py`
   原因：主收益最大，但仅有第二级覆盖，本轮只能做保守精简，不适合直接做结构重构。
6. `Project/models/informer/train.py`
   原因：耦合最深，且只有第二级覆盖，必须保持保守。
