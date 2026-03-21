# 实验记录

## 用途

本文件用于统一记录本地实验，确保模型改动、指标和结论可以持续对比，而不是散落在终端日志或临时备注里。

## 记录规则

1. 每个配置运行单独占一行。
2. 至少记录模型名、核心结构改动、最佳验证 AUC 与时延。
3. 每组实验后补一条简短结论。
4. 如果实验之间不可直接比较，需要说明原因。

## 汇总表

| 编号 | 配置                                        | 模型                           | 核心改动                                                     |   参数量 | 最佳验证 AUC | 平均时延（毫秒/样本） | P95 时延（毫秒/样本） | 结论                            |
| ---- | ------------------------------------------- | ------------------------------ | ------------------------------------------------------------ | -------: | -----------: | --------------------: | --------------------: | ------------------------------- |
| E001 | configs/baseline.yaml                       | baseline                       | 扁平哈希历史事件 token + 候选感知 attention pooling          |      n/a |       0.6891 |                0.1172 |                0.3262 | 初始基线已经具备竞争力          |
| E002 | configs/ucasim_v1.yaml                      | ucasim_v1                      | recent history + memory + 多层候选感知交互 block             | 22431649 |       0.5222 |                0.2091 |                0.7539 | 结构明显弱于简单基线            |
| E003 | configs/ucasim_v1.yaml                      | ucasim_v1                      | 在 UCASIM v1 上加入分解事件编码                              | 22432033 |       0.4450 |                0.1969 |                0.6821 | 输入升级没有救回当前骨干        |
| E004 | configs/decomposed_baseline.yaml            | decomposed_baseline            | 分解事件编码 + 轻量候选感知汇聚                              | 19473889 |       0.7088 |                0.1229 |                0.3405 | 分解输入基线成为新的强参考      |
| E005 | configs/decomposed_dual_path.yaml           | decomposed_dual_path           | 在分解输入上增加 recent summary 与 compressed memory summary | 19550689 |       0.7046 |                0.1286 |                0.3546 | 方向有效，但仍不如 E004         |
| E006 | configs/creatorwyx_din_adapter.yaml         | creatorwyx_din_adapter         | 将 creatorwyx DIN 目标注意力适配到当前 parquet 数据流        | 19501731 |       0.7769 |                0.1131 |                0.3403 | 当前最强主线                    |
| E007 | configs/creatorwyx_grouped_din_adapter.yaml | creatorwyx_grouped_din_adapter | 按 action/content/item 三组分别做 DIN 汇聚后再融合           | 19668586 |       0.7446 |                0.1331 |                0.3652 | 归纳偏置更清晰，但暂时不如 E006 |

## 详细记录

### E001

- 输出目录：outputs/baseline/summary.json
- 观察：简单的候选感知汇聚在样例数据上已经有不错表现。
- 结论：继续保留，作为后续所有结构的参考下界。

### E002

- 输出目录：outputs/ucasim_v1/summary.json
- 观察：更复杂的统一骨干显著增加了时延，同时 AUC 明显下降。
- 结论：当前 UCASIM v1 block 设计还不适合作为主线。

### E003

- 输出目录：outputs/ucasim_v1/summary.json
- 观察：分解事件输入在表示层面有直觉优势，但放在当前 UCASIM v1 骨干上仍学不到有效交互。
- 结论：核心问题主要在骨干，而不是分解输入思路本身。

### E004

- 输出目录：outputs/decomposed_baseline/summary.json
- 观察：分解事件编码在几乎不明显增加时延的前提下提升了 AUC。
- 结论：分解输入应成为后续结构试验的基础版本。

### E005

- 输出目录：outputs/decomposed_dual_path/summary.json
- 观察：recent summary 与 compressed memory summary 没有破坏性能，但也没有超越更简单的分解输入基线。
- 结论：双路径摘要方向成立，但当前聚合设计还不值得提升为主线。

### E006

- 配置：configs/creatorwyx_din_adapter.yaml
- 输出目录：outputs/creatorwyx_din_adapter/summary.json
- 观察：外部仓库原始数据管线依赖 Tenrec CSV 与自定义 mocked 字段，不能直接复用；但将其核心 DIN 目标注意力迁移到当前 parquet 流程后，取得了迄今最强结果。
- 结论：DIN 风格目标注意力是当前任务上更合适的归纳偏置，应成为新的主线参考。

### E007

- 配置：configs/creatorwyx_grouped_din_adapter.yaml
- 输出目录：outputs/creatorwyx_grouped_din_adapter/summary.json
- 观察：把历史拆成 action_seq、content_seq、item_seq 三路分别做 DIN 汇聚是可行的，但当前融合方式仍弱于单路 DIN adapter。
- 结论：分组历史是值得继续做的方向，但还需要更强的分支融合或更显式的事件类型建模。

## 当前主线

当前主线配置：configs/creatorwyx_din_adapter.yaml

选择理由：

1. 当前最佳验证 AUC。
2. 平均时延略低于之前的分解输入基线。
3. 它验证了 DIN 风格目标注意力优于当前 UCASIM block 设计。
4. 它运行在当前 parquet 数据流上，后续扩展成本低。
5. E007 说明仅仅拆分历史分支还不够，单路 DIN adapter 仍是最稳的对照系。

## 下一步实验

1. 在 creatorwyx_din_adapter 上进一步显式拆分 item-like、action-like、timestamp-like 事件组件。
2. 基于特征工程产出的字段字典，决定哪些字段需要单独建模、压缩或分组。
3. 尝试将 grouped branch 的信息以 gated residual 的方式回灌到单路 DIN adapter，而不是直接替代主干。

## 模板

后续实验请按以下格式追加：

### EXXX

- 配置：
- 输出目录：
- 核心改动：
- 最佳验证 AUC：
- 平均时延（毫秒/样本）：
- P95 时延（毫秒/样本）：
- 观察：
- 结论：
