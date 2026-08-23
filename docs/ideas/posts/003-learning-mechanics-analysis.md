---
icon: lucide/lightbulb
---

# Learning Mechanics 分析方法：从缩放律、学习量子到低秩动力学

:material-calendar: 2026-04-26 · :material-tag: 学习力学, Scaling Law, Quanta, 低秩更新, 谱分析

## 原文章出处

- **标题**：终于，学界找到了深度学习的「牛顿定律」
- **来源**：机器之心公众号
- **链接**：<https://mp.weixin.qq.com/s/v3XujOLco3fMJuEzQKer0w>
- **说明**：对论文 *There Will Be a Scientific Theory of Deep Learning* 的中文综述，提炼了 Learning Mechanics 的五条主线。

- **标题**：On neural scaling and the quanta hypothesis
- **作者**：Eric J. Michaud
- **链接**：<https://learningmechanics.pub/quanta/>
- **日期**：2026-04-23

- **标题**：Deep linear networks are a surprisingly useful toy model of weight-space dynamics
- **作者**：Mark Rhee, Dhruva Karkada, Jamie Simon
- **链接**：<https://learningmechanics.pub/deep-linear-nets/>
- **日期**：2026-04-23

<!-- more -->

## AI 解读

这组文章关心的不是再发明一个更复杂的模型，而是建立一套解释和诊断训练过程的方法。对 TAAC 2026 来说，价值在于把“模型为什么涨分、什么时候涨分、靠哪些样本或特征涨分”变成可观测对象，服务统一模块创新奖和 Scaling Law 创新奖。

### 方法 1：用可解 toy model 解释真实训练现象

Learning Mechanics 主张先找能精确分析的简化系统，再把其中的机制迁移回真实模型。深度线性网络是典型例子：虽然端到端函数仍是线性的，但参数空间优化是非凸的，能展示分阶段学习、低秩更新、鞍点到鞍点的轨迹和隐式低秩偏置。

可借鉴的分析动作：

- 对模型权重、embedding 表、MLP/attention 投影矩阵做 SVD，观察训练中奇异值是否按顺序增长。
- 记录每个 checkpoint 的权重增量 `delta W` 的有效秩、top singular value 占比、谱熵。
- 比较小初始化、默认初始化、不同深度下 loss 曲线是否更接近“台阶式”或“平滑式”。
- 把复杂模型中出现的低秩学习现象先在一个小型矩阵分解或线性化推荐任务上复现。

### 方法 2：用谱分解定位“先学什么、后学什么”

深度线性网络文章的核心分析路径是：写出梯度流方程，白化输入，旋转到输入输出协方差的 SVD 基，假设权重先快速对齐，再把矩阵动力学解耦成逐个奇异模式的标量动力学。结论是大奇异值模式更早学到，小奇异值模式更晚学到。

迁移到本项目时，不需要完整解析解，也可以做经验版谱分析：

- 对非序列特征、序列特征、用户侧 dense 特征、物品侧特征分别构造统计矩阵，估计主方向。
- 检查模型 embedding 或中间表征是否先对齐高方差、高频或高互信息方向。
- 在训练中跟踪不同特征组的梯度范数、激活范数、表示漂移量。
- 对比 baseline、hyformer、symbiosis 等实验包是否学习相同主方向，只是速度不同。

### 方法 3：把平滑缩放律拆成很多离散学习单元

Quanta hypothesis 试图解释一个张力：总体 loss 随参数、数据、step 呈平滑幂律下降，但很多能力在局部看像突然出现。它的解释是，模型学习了大量离散或近似离散的“学习量子”，每个量子只影响一小部分样本，整体 loss 把这些小相变平均掉了。

可借鉴的分析动作：

- 不只看整体 AUC/logloss，还记录 per-sample、per-user-segment、per-feature-group、per-sequence-domain 的学习曲线。
- 统计哪些样本在某个训练阶段 loss 突然下降，哪些样本始终缓慢改善，哪些样本出现 inverse scaling。
- 把样本按 loss 曲线形状聚类，识别“早学会”“中期学会”“晚学会”“不稳定”的样本族。
- 估计样本族频率是否长尾分布，并观察模型规模、训练步数、数据量增加时是否按频率顺序解锁。

### 方法 4：用梯度相似性发现潜在机制或样本簇

Quanta 文章中，一个具体实验是对模型预测正确且低 loss 的 token 计算 loss 关于参数的梯度，再用梯度余弦相似度和谱聚类寻找相似机制。直觉是：如果模型在不同样本上调用了类似机制，这些样本的梯度方向也会相似。

迁移到 PCVR 推荐任务，可以把 token 换成样本或样本中的目标 item：

- 从 validation 或 held-out batch 中抽样，计算单样本 loss gradient。
- 只保留最后几层、embedding projection 或特定模块的梯度，控制显存和计算量。
- 对梯度向量做随机投影或 PCA 降维，再计算 cosine similarity。
- 谱聚类后分析每个簇的 schema 特征、序列长度、行为域、label_type、item/user 高频程度。
- 输出簇级别 AUC/logloss 和训练阶段变化，判断模型是否存在“技能簇”。

### 方法 5：把 emergence 和 metric artifact 分开看

文章提醒，很多“涌现能力”可能是指标造成的视觉跳变，例如 accuracy 会把概率分布的微小变化变成 0/1 翻转。推荐任务中也有类似问题：AUC、top-k 命中、logloss、校准误差对模型变化的敏感性不同。

可借鉴的分析动作：

- 同时记录 AUC、logloss、分桶 calibration、top-k/rank proxy，避免只看单一指标。
- 对相同 checkpoint 输出概率分布，观察是整体置信度提高，还是正负样本排序真正拉开。
- 对看似“突然变好”的模型配置，检查 per-sample loss 是否早已有隐藏进展。
- 对关键实验跑多 seed，区分真实相变和随机种子噪声。

### 方法 6：研究参数、数据、step 的联合缩放

Quanta 文章讨论了参数缩放、数据缩放、step 缩放和联合缩放之间的差异，并指出真实模型里“大模型学习效率更高”会让简单的瓶颈模型失效。本项目正好有 Scaling Law 创新奖背景，适合把这条路线变成可复核实验。

可借鉴的分析动作：

- 以 `N` 表示模型参数量，`D` 表示训练样本量或数据比例，`S` 表示训练 step，记录 `L(N,D,S)`。
- 分别扫模型宽度、层数、embedding 维度、训练数据比例、训练步数。
- 拟合 `L = E + A N^-alpha_N`、`L = E + B D^-alpha_D`、`L = E + C S^-alpha_S`。
- 进一步比较 Chinchilla 风格可加形式和 Quanta 风格全局指数形式哪一个更贴近本赛题。
- 记录等 loss 曲线，观察增加模型规模是否能减少达到同等 loss 所需 step。

### 方法 7：比较不同架构的表征收敛

Learning Mechanics 的另一个主线是普适行为：不同架构和数据集可能学到相似表征。对本项目来说，可以用它判断多个实验包是不是在学同一类结构，只是表达方式不同。

可借鉴的分析动作：

- 在相同 validation slice 上抽取 baseline、symbiosis、hyformer、interformer 等模型的中间表征。
- 用 CKA、RSA、线性 probe 或 nearest-neighbor overlap 比较表示相似度。
- 对序列域、非序列域、用户 dense、物品特征分别比较表征收敛程度。
- 如果某个模型 AUC 高但表征与其他模型差异大，优先分析它学到的新增结构。

## 我们的看法

Learning Mechanics 对本项目最直接的价值不是创造另一个模型包，而是解释总体指标背后的样本分化。第一版实现选择验证集逐样本 checkpoint trace，而不是直接做全参数 SVD：前者复用现有 checkpoint 和数据划分，成本较低，也能为后续梯度聚类与表征分析提供固定样本索引。

### 已实现：固定验证样本 learning trace

`taac-analysis-learning-trace` 会发现一个训练 run 下所有 `global_step*/model.safetensors`，读取 checkpoint 的 `train_config.json`，严格重建当时的验证划分，并让同一批样本依次通过所有 checkpoint。它输出每个 checkpoint 的 AUC/LogLoss、逐样本概率与 BCE、首次持续学会的 step、遗忘次数、学习类别，以及类别级序列长度和缺失率画像。

这里的“学会”使用 `opposite_class_median` 相对排序规则：正样本分数高于当期负样本中位数，或负样本分数低于当期正样本中位数。它比固定 `0.5` 阈值更适合低正例率 PCVR，但仍是分析 proxy，不应解释成线上分类阈值。

```bash
uv run taac-analysis-learning-trace \
  --experiment experiments/tokenformer \
  --run-dir outputs/learning_mechanics/tokenformer_seed42 \
  --dataset-path outputs/sample_data/demo_1000.parquet \
  --schema-path docs/archive/files/schema/sample_1000_raw.schema.json \
  --batch-size 32 \
  --device cuda \
  --no-amp
```

run 目录必须至少包含两个 step checkpoint。训练时用 `--data.eval_every_n_steps` 控制 checkpoint 观察间隔；如果它不小于 `--optimizer.max_steps`，通常只会得到最终 checkpoint，分析入口会直接拒绝这种输入。

输出位于 `<run-dir>/learning_trace/`：

| 文件 | 内容 |
| --- | --- |
| `learning_trace.json` | checkpoint 指标、类别计数、结构画像、数据和判定口径 |
| `learning_trace_samples.jsonl` | 逐样本 score、BCE、learned states、遗忘次数、类别和结构特征 |
| `learning_trace.svg` | AUC/LogLoss、类别 BCE 轨迹和正负样本类别分布 |
| `learning_trace.log` | checkpoint 加载与验证划分重建日志 |

学习类别定义如下：

- `early`：在观测区间前 1/3 内进入学会状态，并保持到最后。
- `late`：更晚进入学会状态，并保持到最后。
- `unstable`：至少发生一次从学会到未学会的遗忘事件。
- `unlearned`：截至最后一个 checkpoint 仍未持续学会，且没有先学会再遗忘。

### 2026-08-23 TokenFormer 首次验收

在 NVIDIA A30 上使用 `demo_1000.parquet`、`timestamp_auto` 80/20 划分、seed 42，训练 TokenFormer 100 step，每 20 step 保存一次 checkpoint。201 个固定验证样本中有 22 个正样本。

| Step | AUC | LogLoss |
| ---: | ---: | ---: |
| 20 | 0.7590 | 0.3120 |
| 40 | **0.7862** | 0.3057 |
| 60 | 0.7821 | 0.2683 |
| 80 | 0.7765 | **0.2634** |
| 100 | 0.7676 | 0.2690 |

样本分类为 `early=181`、`late=0`、`unstable=12`、`unlearned=8`。稳定早学组的 BCE 从 step 20 到 100 平均改善约 `0.124`；unstable 和 unlearned 组却分别平均恶化约 `0.471`、`1.016`。AUC 在 step 40 达峰后回落，而总体 LogLoss 继续改善到 step 80，说明大多数样本的概率拟合收益掩盖了少量样本的排序遗忘。这次只有 5 个、且最早为 step 20 的观测点，因此 `late=0` 不能证明模型没有晚学样本；更密集的早期 checkpoint 是下一次实验需要补的口径。

原始 checkpoint 和生成结果保存在本地 `outputs/learning_mechanics_20260823/tokenformer_seed42/`，属于可再生输出，不提交仓库。

### 从诊断到干预：增加序列保留量

首次验收暴露了一个可操作的信号：seed 42 的 unstable 样本平均总序列长度约为 `688`，高于 early 样本的 `632`，而 TokenFormer 默认每个序列域只取最近 `64` 条。于是做了最小干预，把 `seq_top_k` 从 `64` 提高到 `96`，其余训练设置保持一致，并用 seed 17、42、97 做配对复验。

| Step | Baseline AUC | `top_k=96` AUC | 配对差值 | Baseline LogLoss | `top_k=96` LogLoss | 配对差值 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.7562 ± 0.0051 | 0.7637 ± 0.0045 | **+0.00745** | 0.3108 ± 0.0017 | 0.3107 ± 0.0012 | -0.00013 |
| 40 | 0.7773 ± 0.0083 | 0.7818 ± 0.0028 | **+0.00449** | 0.3024 ± 0.0031 | 0.3022 ± 0.0036 | -0.00023 |
| 60 | 0.7771 ± 0.0052 | 0.7832 ± 0.0016 | **+0.00609** | 0.2710 ± 0.0024 | 0.2704 ± 0.0016 | -0.00058 |
| 80 | 0.7756 ± 0.0048 | 0.7850 ± 0.0009 | **+0.00940** | 0.2652 ± 0.0039 | 0.2627 ± 0.0037 | **-0.00250** |
| 100 | 0.7722 ± 0.0068 | 0.7779 ± 0.0073 | **+0.00567** | 0.2695 ± 0.0029 | 0.2679 ± 0.0052 | -0.00155 |

表中是三个 seed 的均值 ± 样本标准差。所有观测 step 的平均 AUC 都提高，尤其第 80 步三个 seed 均为正增益；每个 run 的最佳 AUC 均值从 `0.7793` 提高到 `0.7853`，最佳 LogLoss 均值从 `0.2652` 降到 `0.2627`。这支持“增加行为上下文能够减缓后期排序遗忘”的假设。

但样本分类给出了一条重要反证：三个 seed 合计的 unlearned 从 `24` 降至 `18`，unstable 却从 `29` 增至 `37`。对 baseline 中原本 `53` 个 problematic 样本做同样本配对，`top_k=96` 只把 `6` 个迁移到 early/late，不过三个 seed 的这组样本最终 BCE 都下降，降幅分别为 `0.0694`、`0.0318`、`0.0375`。因此当前结论是：该改动提高了整体指标和难样本概率质量，但没有消除样本级波动。

这次样本规模只有 201、正样本 22，三个 seed 只能作为方向性证据。A30 训练遥测显示峰值已分配显存从 `1705 MiB` 增至 `1887 MiB`，约增加 `10.7%`；并行运行使用了不同 GPU，耗时不能作严格对比。暂不修改 TokenFormer 默认值；下一步应在正式数据切片上复验，并以同卡串行基准测吞吐、峰值显存和按序列长度分桶的指标，再决定收益是否覆盖线上算力成本。

## 实施清单

- [x] 先实现 `analysis trace`：从训练 checkpoint 和 validation slice 导出结构化轨迹。
- [ ] 再实现 `analysis spectra`：权重和权重增量 SVD，输出低秩指标。
- [ ] 再实现 `analysis scaling-fit`：读取多个 run summary，拟合简单幂律。
- [x] `analysis sample-curves` 已并入 trace：保存固定样本跨 checkpoint 的 score、loss 和 learned states。
- [ ] 再实现 `analysis gradient-clusters`：小样本单样本梯度相似性聚类。
- [ ] 最后实现 `analysis representation-similarity`：跨实验包表征收敛分析。
