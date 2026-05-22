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
| 4   | Fused SwiGLU             | baseline/tokenformer/rankup/symbiosis 使用纯 PyTorch 激活乘法 | P0     | 融合 gate、SiLU 和 value 乘法，减少中间张量分配；Linear 融合可作为后续阶段                          |
| 5   | Fused BCE loss           | 训练热路径使用 `F.binary_cross_entropy_with_logits`           | P1     | 融合 logits 到 loss/reduction，减少显存往返，需覆盖 sample weight 或 reduction 策略后再替换训练路径 |
| 6   | Fused GELU               | baseline 使用 `F.gelu()`                                      | P2     | 单算子收益较小，更适合作为 Linear + GELU fusion 的子目标                                            |
| 7   | Fused SiLU               | tokenizer 等多处使用 `nn.SiLU()`                              | P2     | 单独加速收益有限，优先服务 Linear + SiLU 或 SwiGLU fusion                                           |
| 8   | L2Norm                   | `tensor_ops.py` 当前依赖 torch/compile 路径                   | P2     | GDR 内部使用，TileLang 化可提升 chunk 内吞吐                                                        |
| 9   | Fused Dropout + Residual | attention/FFN 后常见 `dropout(x) + residual`                  | P1     | 融合 add + dropout，减少一次显存读写；需要保证训练随机性和 eval fallback 行为                       |

## 已有 TODO 标记

| #   | 位置                                                                                              | 内容                                                 | 优先级 |
| --- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ------ |
| 10  | `src/taac2026/infrastructure/accelerators/chunking.py`                                            | `prepare_chunk_indices` 注释 `TODO: tilelang kernel` | P1     |
| 11  | `src/taac2026/infrastructure/accelerators/attention/kernels/gated_delta_rule/context_parallel.py` | `_calc_cp_seqs` 注释 `TODO: tilelang kernel`         | P1     |

## 新算子提案

| #   | 算子                         | 来源/动机                                                     | 优先级 |
| --- | ---------------------------- | ------------------------------------------------------------- | ------ |
| 12  | SiLU Attention Triton kernel | UniRec 上游仓库已有实现，本仓库文档提到但尚未接入共享 runtime | P1     |
| 13  | Fused Linear + Activation    | Linear + SiLU / GELU 融合，减少 kernel launch 开销            | P1     |
| 14  | Fused Scale + Bias + Add     | RMSNorm 后 affine 和 residual add 融合                        | P2     |
| 15  | Top-K / Top-P sampling       | 推理时 logits 到概率/采样的 fused 路径                        | P2     |

## 建议开发顺序

1. FlashAttention Triton backend，先补齐文档曾经宣称但源码缺失的 backend。
2. LayerNorm Triton / TileLang，优先覆盖高频模型路径。
3. Fused SwiGLU，让 baseline、tokenformer、rankup、symbiosis 等实验同时受益。
4. `prepare_chunk_indices` 和 `_calc_cp_seqs` 的 TileLang TODO，解除 GDR 性能瓶颈。
5. Fused BCE loss，聚焦训练 step 热路径。
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
