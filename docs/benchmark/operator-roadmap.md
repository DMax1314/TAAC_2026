---
icon: lucide/cpu
---

# Triton / TileLang 算子路线图

本页记录 PCVR runtime 里 Triton / TileLang 算子的开发缺口和建议推进顺序。截至 2026-05-20，源码里的 FlashAttention 只声明 `torch` / `tilelang`，RMSNorm 声明并实现 `torch` / `tilelang` / `triton`。后续新增算子应先保留 torch reference，再接入 accelerator backend、数值校验、GPU 单测和 benchmark 文档。

## 当前后端矩阵

| 算子                       | Torch reference        | TileLang | Triton | 备注                                                   |
| -------------------------- | ---------------------- | -------- | ------ | ------------------------------------------------------ |
| FlashAttention             | 已接入                 | 已接入   | 缺失   | `FlashAttentionBackend = Literal["torch", "tilelang"]` |
| RMSNorm                    | 已接入                 | 已接入   | 已接入 | benchmark 支持 `torch,tilelang,triton`                 |
| Embedding bag mean         | 已接入                 | 已接入   | 已接入 | 另有 forward-only `cuembed` 对照                       |
| LayerNorm                  | PyTorch `nn.LayerNorm` | 缺失     | 已接入 | 当前覆盖 last-dim affine LayerNorm；TileLang 待补      |
| SwiGLU / activation fusion | PyTorch eager          | 缺失     | 缺失   | 多个实验用 `F.silu(gate) * value`                      |
| BCE loss                   | PyTorch eager          | 缺失     | 缺失   | 训练热路径使用 `F.binary_cross_entropy_with_logits`    |

## 缺失后端

| #   | 算子                            | 当前状态                                            | 优先级 | 说明                                                                                                                                                                     |
| --- | ------------------------------- | --------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | FlashAttention Triton backend   | `FlashAttentionBackend` 仅支持 `torch` / `tilelang` | P0     | 在 `flash_attention.py` 中补齐 Triton forward、training forward、backward preprocess 和 backward kernels，并同步 runtime backend literal、CLI 参数、GPU 单测与 benchmark |
| 2   | Gated Delta Rule Triton backend | 当前为 TileLang kernel 族，无 Triton 备选           | P1     | `fused_fwd`、`fused_bwd`、`kkt_solve`、`prepare_h` 等子算子需要逐项 Triton 化，并维持 TileLang reference 对照                                                            |

## 缺算子

| #   | 算子                     | 当前用法                                                      | 优先级 | 说明                                                                                                |
| --- | ------------------------ | ------------------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------------- |
| 3   | LayerNorm                | 全项目大量使用 `nn.LayerNorm`                                 | P0     | Triton fwd + bwd 已接入；后续补 TileLang backend，并逐步替换高频模型路径                            |
| 4   | Fused SwiGLU             | baseline/tokenformer/symbiosis 使用纯 PyTorch 激活乘法 | P0     | 融合 gate、SiLU 和 value 乘法，减少中间张量分配；Linear 融合可作为后续阶段                          |
| 5   | Fused BCE loss           | 训练热路径使用 `F.binary_cross_entropy_with_logits`           | P1     | 融合 logits 到 loss/reduction，减少显存往返，需覆盖 sample weight 或 reduction 策略后再替换训练路径 |
| 6   | Fused GELU               | baseline 使用 `F.gelu()`                                      | P2     | 单算子收益较小，更适合作为 Linear + GELU fusion 的子目标                                            |
| 7   | Fused SiLU               | tokenizer 等多处使用 `nn.SiLU()`                              | P2     | 单独加速收益有限，优先服务 Linear + SiLU 或 SwiGLU fusion                                           |
| 8   | L2Norm                   | `tensor_ops.py` 当前依赖 torch/compile 路径                   | P2     | GDR 内部使用，TileLang 化可提升 chunk 内吞吐                                                        |
| 9   | Fused Dropout + Residual | attention/FFN 后常见 `dropout(x) + residual`                  | P1     | 融合 add + dropout，减少一次显存读写；需要保证训练随机性和 eval fallback 行为                       |

## 已有 TODO 标记

| #   | 位置                                                                                              | 内容                                           | 优先级 | 状态                                                                                                 |
| --- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------- |
| 10  | `src/taac2026/infrastructure/accelerators/chunking.py`                                            | `prepare_chunk_indices` 已接入 tilelang kernel | P1     | ✅ 已完成（保留 torch fallback，两路径语义一致）                                                      |
| 11  | `src/taac2026/infrastructure/accelerators/attention/kernels/gated_delta_rule/context_parallel.py` | `_calc_cp_seqs` 保留 Python 实现               | P1     | ⏸ 评估后不 kernel 化：决策启发式本质在 Python，函数被 `tensor_cache` 缓存且输出变长，kernel 化收益≈0 |

## 已解决的历史问题

| #   | 问题                                                                                | 处理                                                                                                                     |
| --- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 12  | GDR kernel 族使用已移除的 `T.gemm_v1` API，tilelang 0.1.12 下完全不可编译           | `kkt_solve` / `prepare_h` / `fused_fwd` / `fused_bwd` 共 35 处迁移为 `T.gemm`，sm_90a target 下 6 个 kernel 全部编译通过 |
| 13  | `prepare_h` 加载 A 时未对最后一个 chunk 越界行做 mask，`X = A^T @ K` 会污染全部输出 | V/A 加载段增加边界检查，越界行置 0                                                                                       |

## 真实模型训练 step profiling（2026-08，A30）

曾用当时的 Baseline+ 实验（真实数据管线 + 真实模型 + BCE/Muon/Adagrad 全链路）做
torch profiler 采样，优化前每 step 的 GPU
self 时间约 168ms，优化后约 120ms（-29%），step 耗时中位数 309ms → 268ms，训练
loss/AUC 曲线完全一致。Baseline+ 已不再作为独立模型维护；这些数字保留为历史优化证据，
不能和当前 Baseline 的绝对耗时直接比较。

| #   | 优化项                                                                                                                                                                                                                                             | 量化效果                                                                                                                                                                        |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 14  | `Muon` 的 1D/0D AdamW 分支 foreach 化 + 2D 参数按形状分组做 batched Newton-Schulz（`bmm`）                                                                                                                                                         | `Muon.step` GPU 94ms → 40ms（-57%），数学与逐参数实现完全等价                                                                                                                   |
| 15  | 全部 `nn.Embedding` 开启 `sparse=True`；`FeatureEmbeddingBank` 的 bag-mean 包 `SparseEmbeddingBagMean`（forward 保留多后端加速器，backward 构造 COO 稀疏梯度）                                                                                     | embedding backward 278ms → 28ms（-90%）；每 step dense 梯度从 ~2.2GB 降到 ~MB 级（H20 19.6GiB 显存下避免 OOM 的关键）                                                           |
| 16  | 自定义 `PCVRSparseAdagrad`（`src/taac2026/infrastructure/optimization/sparse_adagrad.py`）：用 `scatter_reduce` 合并重复行 + `index_add`/`index_select` 更新，完全绕开 torch sparse 原语（`coalesce`/`sparse_mask`/`_make_sparse`/invariant 检查） | 消除 ~165 次/step 的 `coalesce` kernel（~78ms）与 ~640 次/step 的 sparse 张量构造；数值与 `torch.optim.Adagrad` 逐位一致（1e-8 级）；AMP（fp16 GradScaler）下回退 torch Adagrad |
| 21  | Adagrad 行合并双策略：大表（vocab > 100K）改排序分段（`unique` + `index_add_`，全部操作保持在 nnz 规模），小表保留全表 scatter+any（固定 kernel 开销占优）                                                                                         | 165 参数模拟训练场景 GPU step 58ms → 41ms（-30%）；大表单独 3.4x；与旧实现及 torch Adagrad 数学完全一致（1e-8 级）                                                              |
| 17  | `clip_grad_norms_with_sparse`：torch 2.13 的 `clip_grad_norm_` 已移除 sparse 分支，对 SparseCUDA 直接抛 `NotImplementedError`                                                                                                                      | 兼容稀疏梯度裁剪（`_values()` 范数），旧语义不变                                                                                                                                |

复现命令（默认 tilelang 后端即可，无需 `--rms_norm_backend torch`）：

```bash
uv run python tools/profile_train_step.py \
  --experiment experiments/baseline --optimizer.device cuda --optimizer.max_steps 30 \
  --optimizer.dense_optimizer_type muon --model.rms_norm_backend tilelang \
  --dataset-path outputs/sample_data/demo_1000.parquet \
  --schema-path docs/archive/files/schema/sample_1000_raw.schema.json
```

剩余热点（按收益排序）：Muon NS 迭代的 `ns_steps=5`（~40ms/step，调小需先验证训练
质量）、DataLoader CPU 等待（~170ms/step，本地小数据集固有）。原始 roadmap 中的
Fused SwiGLU / BCE 在真实 profile 中不是热点（`mm`/`addmm` 合计仅 ~30ms/step），
优先级低于优化器路径。Adagrad 行合并已通过双策略优化（见 #21）。

### 附带发现

| #   | 问题                                                                                                                           | 现状                                                                                                                                                                              |
| --- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 18  | 历史 Baseline+ 路径暴露 `RMSNorm(d_model*6)`=384 非 2 的幂时 TileLang 后端拒绝训练 | ✅ 已完成：共享 RMSNorm 对非 2 幂 dim 自动 pad 到安全形状（2 的幂或 128 的倍数，`effective_cols` 保持真实除数）；当时已做 GPU smoke 3 步验证 |
| 19  | tilelang rms_norm kernel cache key 含 `rows`，序列 token 数逐 batch 变化导致每 step 重编译                                     | ✅ 已完成：kernel 改用动态行数（`T.dynamic`）编译，cache key 去掉 `rows`；同一 `(cols, dtype, eps, block_rows)` 只编译一次（GPU 测试断言 fwd/bwd cache 各 1），triton 后端同步支持 |
| 20  | torch 2.13 `nn.Embedding(sparse=True)` 的 backward 返回 uncoalesced 稀疏梯度，`clip_grad_norm_` 已移除 sparse 分支             | 已用 `clip_grad_norms_with_sparse` 兼容（见 #17）                                                                                                                                 |

## 新算子提案

| #   | 算子                         | 来源/动机                                                     | 优先级 |
| --- | ---------------------------- | ------------------------------------------------------------- | ------ |
| 12  | SiLU Attention Triton kernel | 公开 UniRec 方案已有实现，本仓库尚未接入共享 runtime | P1     |
| 13  | Fused Linear + Activation    | Linear + SiLU / GELU 融合，减少 kernel launch 开销            | P1     |
| 14  | Fused Scale + Bias + Add     | RMSNorm 后 affine 和 residual add 融合                        | P2     |
| 15  | Top-K / Top-P sampling       | 推理时 logits 到概率/采样的 fused 路径                        | P2     |

## 建议开发顺序

1. 优化器热路径（已部分完成，见上表 #14-17）：真实 profile 显示 Muon/Adagrad 占训练
   step GPU 时间 ~60%。剩余项：Muon NS 迭代 fused kernel（`ns_steps` 迭代的 gemm 链）、
   自定义 Adagrad 的 `scatter_reduce`+`any` 合并步骤融合。
2. FlashAttention Triton backend，先补齐文档曾经宣称但源码缺失的 backend。
3. LayerNorm Triton / TileLang，优先覆盖高频模型路径（tilelang rms_norm 的非 2 幂
   dim 已通过 pad 解决、动态行数复用已解决，见附带发现 #18/#19）。
4. Fused SwiGLU，让 baseline、tokenformer、symbiosis 等实验受益（真实 profile
   中 `mm`/`addmm` 合计 ~30ms/step，收益有限但多实验受益）。
5. Fused BCE loss，训练 loss 热路径（真实 profile 中占比极小，仅在校验损失前顺手做）。
6. Gated Delta Rule Triton backend，作为 TileLang kernel 族的可替代实现。

## 接入验收清单

新增或替换算子时，至少补齐以下内容：

| 类别            | 要求                                                                             |
| --------------- | -------------------------------------------------------------------------------- |
| Runtime surface | 明确 backend literal、fallback 规则、dtype/device/shape 限制和错误消息           |
| Reference       | 保留 torch reference，并在单测里做数值误差对照                                   |
| Autograd        | 训练路径算子必须覆盖 backward；如果只支持 inference，需要在 API 和文档中显式说明 |
| Tests           | 补充 CPU-safe fallback 单测、CUDA GPU 单测，以及已有 benchmark CLI 的参数覆盖    |
| Docs            | 在 `docs/benchmark/` 新增或更新页面，记录命令、支持状态、误差口径和最近验收观察  |

推荐从窄口径验证开始：

```bash
uv run pytest tests/unit/infrastructure/accelerators -q
uv run pytest tests/gpu/infrastructure/accelerators -q
uv run taac-benchmark-pcvr-tilelang-ops --operator <operator> --device cuda --backends torch,tilelang,triton
```

如果本地没有 CUDA，GPU 单测和 accelerator benchmark 可以记录为未运行，但不能据此宣称 Triton 或 TileLang backend 已验收。
