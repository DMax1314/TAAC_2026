# UniRec

**多阶段融合**

## 概述

UniRec 强调 unified tokenization、混合 attention mask 和 scaling law 探索，支持 2 卡 DDP 训练。独特之处在于使用独立的特征交叉层、序列层、静态层和融合层组成多阶段处理管道。

## 模型架构

- 4 层 Transformer，4 头注意力
- Embedding 维度 128
- **feature_cross_layers = 1**：特征交叉层
- **sequence_layers = 1**（通过 static_layers=3 配置）：序列建模层
- **fusion_layers = 1**：融合层
- 2 个 memory slots
- 8 个行为分段
- Pairwise ranking loss（weight=0.25）

## 默认配置

| 参数                   | 值   |
| ---------------------- | ---- |
| `embedding_dim`        | 128  |
| `num_layers`           | 4    |
| `num_heads`            | 4    |
| `segment_count`        | 8    |
| `memory_slots`         | 2    |
| `feature_cross_layers` | 1    |
| `static_layers`        | 3    |
| `fusion_layers`        | 1    |
| `epochs`               | 10   |
| `batch_size`           | 64   |
| `learning_rate`        | 1e-3 |
| `pairwise_weight`      | 0.25 |

## 快速运行

```bash
uv run taac-train --experiment config/gen/unirec
uv run taac-evaluate single --experiment config/gen/unirec
```

## 输出目录

```
outputs/gen/unirec/
```

## 来源

[hojiahao/TAAC2026](https://github.com/hojiahao/TAAC2026)
