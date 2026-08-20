---
icon: lucide/folder-open
---

# 实验包总览

本目录记录仓库中所有可运行的实验包。它们共享同一套 PCVR 训练、评估、推理、checkpoint sidecar 和 bundle 打包流程；差异集中在输入 tokenization、交互主干和默认训练策略上。

如果把仓库看成一个统一推荐系统实验台，那么 `src/taac2026` 是稳定运行时，`experiments/` 是可替换的研究假设。每个实验包都应回答四个问题：

1. 它要验证什么建模假设。
2. 它如何组织用户静态特征、候选物品特征和行为序列。
3. 它的训练默认值是否会影响公平对照。
4. 它的输出和 checkpoint 是否仍满足共享 PCVR 契约。

## 一、怎么选

| 目标                 | 从这里开始                                  | 适合回答的问题                                                                      |
| -------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------- |
| 需要一个干净参照     | [Baseline](baseline.md)                     | HyFormer 原始路线在当前 runtime 下的最低复杂度表现。                                |
| 想做用户-物品交互    | [InterFormer](interformer.md)               | 用户上下文、物品候选和序列行为是否应在双分支中交替融合。                            |
| 想做统一 Transformer | [OneTrans](onetrans.md)                     | 序列 token 与非序列 token 进入同一 causal 主干后，逐层压缩是否有效。                |
| 想试 TokenFormer     | [TokenFormer](tokenformer.md)               | BFTS 分层注意力和 NLIR 门控能否缓解统一 token 流里的序列坍塌传播。                  |
| 想试分布感知统一流   | [Symbiosis](symbiosis.md)                   | 缺失、风险、序列 memory、metadata mask 和 candidate readout 如何服务线上泛化。      |
| 想试最终阶段主模型   | [DualQ](dualq.md)                           | DualQ、时间对齐多域序列、pair 加权残差 tokenizer 与全局时间 token 的组合。          |
| 想复现工业赛冠军结构 | [QueryFormer](queryformer.md)               | 四类 Query attention、逐字段 DCNv2 与多列独立 embedding 能否带来可复现增益。        |
| 想知道线上机器       | [Host Device Info](host-device-info.md)     | 线上 CPU/GPU/CUDA/Python/网络/依赖源到底是什么状态。                                |
| 想看线上数据分布     | [Online Dataset EDA](online-dataset-eda.md) | train 和 infer 数据的 schema、缺失率、基数、序列长度、dense 分布是否漂移。          |

## 二、模型谱系

当前模型实验大致分为四类。

**参照。** Baseline 用最少默认增强保留 HyFormer 的干净零刻度。Cache、数据增强、Muon 和 accelerator backend 属于共享训练或性能变量，应通过类型化配置和 benchmark 独立比较，不再用另一个模型包承载。

**异构交互。** InterFormer 仍把非序列上下文和序列上下文视作不同类型的信息流，只是在 block 内加强两者交互。QueryFormer 进一步显式区分用户、候选广告与各行为域，并通过双向 query 生成和检索建立信息回路。它们适合做“分支结构是否比全量拼接更稳”的对照。

**统一 token 流。** OneTrans、TokenFormer 和 Symbiosis 都把多域特征与行为序列放到更统一的 token 空间。它们的差别在于如何控制长序列计算，以及是否显式表达 token 来源、缺失与风险。

**竞赛方案迁移。** DualQ 和 QueryFormer 分别承载学术赛道统一模块方案与工业赛冠军公开结构。它们保留明确来源和差异说明，用于和论文式实验形成现实方案对照。

**表示诊断。** 有效秩计算属于共享 modeling 能力，不再由诊断专用模型拥有。Symbiosis 在启用训练诊断时记录输入、输出 token 的 effective rank，其他实验可复用同一 `masked_effective_rank` primitive。

## 三、统一运行方式

模型实验本地训练：

```bash
bash run.sh train \
  --experiment experiments/baseline \
  --run-dir outputs/baseline_smoke
```

评估同一个 run 目录：

```bash
bash run.sh val \
  --experiment experiments/baseline \
  --run-dir outputs/baseline_smoke
```

推理需要 checkpoint 和结果目录：

```bash
bash run.sh infer \
  --experiment experiments/baseline \
  --checkpoint outputs/baseline_smoke \
  --result-dir outputs/baseline_infer
```

线上上传物由独立打包命令生成：

```bash
uv run taac-package-train --experiment experiments/baseline --output-dir outputs/bundles/baseline_training
uv run taac-package-infer --experiment experiments/baseline --output-dir outputs/bundles/baseline_inference
```

维护类实验也复用 `run.sh train` 入口，但不一定支持推理 bundle。`host_device_info` 不需要数据集；`online_dataset_eda` 需要平台或本地显式提供 parquet/schema。

## 四、包结构契约

普通 PCVR 模型实验通常只有两个必需文件：

```text
experiments/<name>/
├── __init__.py
└── model.py
```

`__init__.py` 声明 `EXPERIMENT`、实验名、模型类和默认训练配置。`model.py` 实现模型类，至少满足：

- `forward(inputs)` 返回 `(B,)` 或 `(B, action_num)` logits。
- `predict(inputs)` 返回 `(logits, embeddings)`。
- `num_ns` 能表达非序列 token 数，供 runtime 日志和契约测试使用。
- sparse embedding 参数能通过 `EmbeddingParameterMixin` 与 dense 参数分组。
- checkpoint sidecar 中的 `schema.json` 和 `train_config.json` 能重建模型。

维护工具包更轻：

```text
experiments/<tool>/
├── __init__.py
└── runner.py
```

它们通常导出 `FunctionExperiment`，可以没有模型类和 checkpoint sidecar；实验包用接收结构化 request 的函数声明所支持的操作，共享 CLI 和 bundle runtime 负责调用。

## 五、改实验时先看哪里

- 实验入口：`experiments/<name>/__init__.py`
- 模型实现：`experiments/<name>/model.py`
- 实验发现与装载：`src/taac2026/application/experiments/`
- 模型输入契约：`src/taac2026/infrastructure/modeling/model_contract.py`
- 新增实验流程：[新增实验包](../guide/contributing.md)
- 测试选择：[测试](../guide/testing.md)

不要从 `docs/archive/files/...` 推断当前契约；那里是历史快照。新增或修改实验后，优先跑实验包 contract tests，再根据是否动到数据、checkpoint、bundle 或 accelerator 扩大验证范围。
