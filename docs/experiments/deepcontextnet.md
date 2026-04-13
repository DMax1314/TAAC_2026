# DeepContextNet

**上下文感知深度建模**

## 概述

DeepContextNet 采用 HSTU 风格的序列建模与 Muon 优化器路线，强调上下文感知的深度特征交互。是当前实验包中注意力头数最多、学习率最保守的包。

## 模型架构

- 4 层 Transformer，**8 头**注意力
- Embedding 维度 128
- Recent sequence length 32（最长）
- Batch size 32（最小，匹配更大模型的显存需求）

## 默认配置

| 参数              | 值   |
| ----------------- | ---- |
| `embedding_dim`   | 128  |
| `num_layers`      | 4    |
| `num_heads`       | 8    |
| `epochs`          | 10   |
| `batch_size`      | 32   |
| `learning_rate`   | 2e-4 |
| `pairwise_weight` | 0.0  |

## 快速运行

```bash
uv run taac-train --experiment config/gen/deepcontextnet
uv run taac-evaluate single --experiment config/gen/deepcontextnet
```

## 输出目录

```
outputs/gen/deepcontextnet/
```

## 来源

[suyanli220/TAAC-2026-Baseline-Tencent-Advertisement-Contest](https://github.com/suyanli220/TAAC-2026-Baseline-Tencent-Advertisement-Contest)
