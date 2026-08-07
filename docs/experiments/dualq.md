---
icon: lucide/git-branch
---

# DualQ

## 摘要

DualQ 是 TAAC2026 学术赛道第二轮 Top-17（统一模块创新奖）方案的
`round2best` 变体（来源：<https://github.com/zzhlkw-ai/TAAC2026>，MIT
License）在共享 PCVR runtime 上的迁移。它把「多域行为序列 + 用户静态特征 + 候选
物品 + pair 交互特征」统一翻译成 token 流，再交给同构 HyFormer 主干。

**迁移范围：模型结构已迁移，数据特征尚未迁移。** 本包完成了模型结构层面的完整迁
移（DualQ、多域序列 attention、pair tokenizer、时间特征、全局时间 token），但依赖
训练集 profile 的数据侧特征（id 频次 bucket、item dense 统计、item time dense、
OOF CTR/CVR 编码）是来源仓库的 dataset 侧逻辑，不在本包内。这些特征要么在后续阶段
单独实现，要么显式声明为不支持（见「六、迁移清单」）。

## 一、为什么需要 DualQ

来源仓库 `round2best` 是最终阶段主模型，吸收了两条线：`round1best` 的 HyFormer
序列建模（时间对齐 interleave、多视图序列扩展）和 `rankup` 的高有效秩表征（NS
tokenizer、dual-Q 查询拆分）。迁移目标是让这套结构在共享 runtime 上可训练、可评估、
可推理，并保持 checkpoint sidecar 契约。

## 二、实验入口

入口位于 `experiments/dualq/__init__.py`。

| 项目                | 默认值                                                 |
| ------------------- | ------------------------------------------------------ |
| 实验名              | `pcvr_dualq`                                      |
| 模型类              | `PCVRDualQ`                                       |
| 配置扩展            | `DualQModelConfig`（dualq 开关）+ 训练默认值 |
| batch size          | `576`                                                  |
| split / sampling    | `timestamp_auto` / `step_random`                       |
| 序列上限            | `seq_a:256,seq_b:256,seq_c:512,seq_d:512`              |
| `d_model / emb_dim` | `192 / 64`                                             |
| block / head        | `2 / 4`                                                |
| seq encoder         | `swiglu`，`seq_causal=False`                           |
| 优化器 / lr         | `adamw`（dense）/ `4.83e-4`，sparse lr `0.05`          |
| EMA                 | `0.999 @ 1500`（开启）                                 |
| AMP / compile       | BF16 AMP 开启，compile 开启                            |
| loss                | BCE                                                    |
| validation          | 每 5000 步评估，early stopping 监控 AUC                |

## 三、建模决策

### 3.1 多域序列扩展与时间对齐 interleave

- 每个序列域先按 `seq_interest_ratios`（默认 `1.0,0.7`）截断出「全量 + 近 70%」多
  个视图；截断后各视图内部按时间倒序排列。
- `use_time_aligned_interleave=True` 时不再生成多视图，而是把所有域的事件按
  request 时间差升序合并成**一条**时间对齐序列（`num_sequences=1`），全局按
  event 时间统一排序。

### 3.2 DualQ 查询拆分

`num_queries`（默认 6）按 `user_q_tokens + item_q_tokens`（4 + 2）拆成用户侧与
物品侧两组查询 token，分别对序列 token 流做多视图注意力，再拼接。

### 3.3 NS tokenizer 与 pair 特征

- 用户/物品稀疏特征按 singleton group 组织，`rankmixer` tokenizer 在每个 group
  内做符号化 softmax 混合；pair fids（默认 `62,63,64,65,66,89,90,91`）在模型构造
  时按 schema 偏移被**从用户侧剥离**，不再进入 NS group。
- pair dense 值作为权重参与 `CrossRankMixerNSTokenizer` 的加权残差池化：fid
  `< 89` 用 L1 归一化权重，fid `89-91` 用带符号 softmax 权重（`use_weighted_residual`
  门控初值 `-4`）。
- item dense 特征按 schema 字段 tokenize（`ItemDenseTokenizer`，fid 129 存在时拆
  body/stat）；用户侧 fid 61（`user_emb_dim=256`）与 fid 87
  （`user_seq_block_dim * user_seq_num = 320`）按精确维度拆分。

### 3.4 时间特征（全部从规范时间戳派生）

模型只读 `PCVRModelInput`，所有时间特征在 forward 内从 event `timestamps` 与
`request_timestamp` 计算：

- gap bucket：相邻事件间隔绝对值映射到 `BUCKET_BOUNDARIES` 桶（0 为填充/首个）。
- ts_float（8 维）：`log1p(diff_days)`、域缩放天数、`log1p(diff_hours)`、hour 的
  cos/sin、星期 cos/sin、到下一事件间隔（`seq_c`/`seq_d` 按来源语义缩放）。
- ts_stat（6 维）：`log1p(max/min/mean diff)` 与 `≤900s/≤3600s/≤86400s` 事件计数。
- 全局时间 token：由 `request_timestamp` 按 UTC+8 派生 hour/day-of-week/weekend
  vocab embedding，拼到序列流末尾。

### 3.5 消融开关

`use_time_gap_domain_gates`、`use_fid87_token_residual`、`use_time_decay_summary`、
`use_global_time_token`、`use_seq_gap_buckets` 均为类型化布尔开关，走共享 CLI、
写入 checkpoint sidecar、并从 sidecar 重建。

## 四、统一运行方式

```bash
# 训练（本地 CPU 冒烟）
bash run.sh train \
  --experiment experiments/dualq \
  --run-dir outputs/dualq_smoke \
  --schema-path docs/archive/files/schema/sample_1000_raw.schema.json \
  --optimizer.device cpu \
  --data.num_workers 0 \
  --data.batch_size 8 \
  --optimizer.max_steps 1 \
  --runtime.no_compile \
  --runtime.no_amp

# 评估 / 推理复用同一 run 目录（同一 schema.json + train_config.json）
bash run.sh val --experiment experiments/dualq --run-dir outputs/dualq_smoke
bash run.sh infer \
  --experiment experiments/dualq \
  --checkpoint outputs/dualq_smoke \
  --result-dir outputs/dualq_infer
```

实验包不携带任何 hooks、CLI parser 或 checkpoint loader；这些都由框架拥有。

## 五、与来源的差异

- `use_item_time_query_token` 不支持：它依赖来源仓库 dataset 侧的 profile 合成
  列（item time dense），本包未迁移这些列，因此显式报错而非静默忽略。
- `use_din` 未迁移：DIN 动态兴趣路径在来源仓库最终配置中未启用。
- `MultiSeqQueryGenerator` / `DINQBias` 组件未引入（来源中未被最终配置使用）。
- `seq_causal=False`、`use_rope=False`：与来源 `round2best` run.sh 保持一致。

## 六、迁移清单

| 特性                   | 状态               | 说明                                                                              |
| ---------------------- | ------------------ | --------------------------------------------------------------------------------- |
| DualQ query tokens     | 已迁移             | `user_q_tokens=4` / `item_q_tokens=2`，构造时校验 `num_queries == 4 + 2`          |
| 多域序列 attention     | 已迁移             | 多视图截断 + `use_time_aligned_interleave` 时间对齐合并                           |
| pair feature tokenizer | 已迁移             | pair fids 从用户侧剥离，`CrossRankMixerNSTokenizer` 加权残差                      |
| item dense grouping    | 已迁移             | 按 schema 字段 tokenize，fid 129 body/stat 拆分                                   |
| time gap / time stats  | 已迁移             | 从规范时间戳在模型内派生（gap bucket / ts_float / ts_stat）                       |
| global time token      | 已迁移             | 从 request 时间戳按 UTC+8 派生 hour/dow/weekend                                   |
| train-set profile 特征 | 未迁移（第二阶段） | id 频次 bucket、item dense 统计、item time dense 需数据集侧单独实现，不能静默跳过 |
| OOF CTR/CVR 编码       | 未迁移（第二阶段） | item OOF 目标编码依赖离线训练集，显式推迟                                         |

**结论：模型结构已迁移，数据特征尚未迁移。** 当前 checkpoint sidecar 完整记录模型
配置；一旦第二阶段实现 profile 特征，需要扩展 `DualQModelConfig` 与
schema 派生逻辑，并升级 sidecar 版本校验。

## 七、测试

```bash
uv run pytest tests/unit/experiments/test_dualq_model.py -q
uv run pytest tests/contract/experiments/test_dualq.py -q
uv run pytest tests/contract/experiments/ -q
```
