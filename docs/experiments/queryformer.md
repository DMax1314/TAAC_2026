---
icon: lucide/search-code
---

# QueryFormer

QueryFormer 是 TAAC × KDD Cup 2026 工业赛道冠军方案的 clean-room 实验实现。冠军团队公开的技术文章给出了模型结构、消融和训练策略，但没有发布可移植源码；因此本包只把公开且能够形成明确契约的部分接到共享 PCVR runtime，不声称与冠军私有实现逐行或逐参数一致。

- 原始文章：[我用 DeepSeek 网页版拿下 KDD Cup 冠军！](https://mp.weixin.qq.com/s/yG6PnTw4U6OYynNLSpvBYg)
- 可访问转载：[搜狐时间线](https://timeline.sohu.com/news/Ld7vJbDw4I)
- 文章报告工业赛道测试集 AUC `0.83254`。

## 已集成的冠军结构

| 公开方案组件          | 当前实现                                                                                                       |
| --------------------- | -------------------------------------------------------------------------------------------------------------- |
| 分组 Tokenization     | sparse 字段按 `PCVRNSConfig` 的用户/广告语义组产生 token；dense 字段按 schema 原始边界拆分，每个字段独立经过全秩 DCNv2 后产生一个 token。 |
| QuerySelfAttn         | 用户 token 内部、广告 token 内部分别做 self-attention。                                                        |
| QueryCrossAttn        | 用户与广告 token 双向 cross-attention，让序列检索前的 query 同时感知两侧上下文。                               |
| SeqQueryCrossAttn     | 每个序列域先产生初始 query，再主动读取更新后的用户与广告 token；关闭该模块时使用显式 MLP query 作为消融基线。  |
| QuerySeqCrossAttn     | 上一步得到的动态 query 再读取对应行为序列，形成候选广告相关的序列表示。                                        |
| 多列 Embedding Matrix | 默认 `H=4`；每列拥有完全独立的 sparse、sequence embedding、DCNv2 和 attention 参数，最后融合列表示。           |
| 高基数特征            | 超过 `emb_skip_threshold` 的 sparse/sequence id 使用共享 runtime 的 hash compression，不再静默丢弃。           |

文章中四个 Attention 的单项消融下降分别为 `0.00020`、`0.00029`、`0.00023` 和 `0.00038`；其中 SeqQueryCrossAttn 的贡献最大。多列实验最终选择 `H=4`。这些数字用于确定默认开关，不是本仓库的复现实验结果。

## 默认训练策略

| 配置             | 默认值                                            |
| ---------------- | ------------------------------------------------- |
| dense optimizer  | `muon`                                            |
| sparse optimizer | 共享 runtime 的 Adagrad                           |
| scheduler        | cosine                                           |
| EMA              | 开启，`decay=0.999`                               |
| precision        | BF16 AMP                                          |
| compile          | 开启                                              |
| loss             | BCE                                               |

公开文章使用的是 MuonPlus。当前 runtime 只提供经过验证的 `muon`，所以默认值采用最接近且已有完整 checkpoint/恢复契约的实现，没有把两者写成同一个算法。梯度裁剪由共享 trainer 固定执行。

## 运行

```bash
bash run.sh train \
  --experiment experiments/queryformer \
  --run-dir outputs/queryformer
```

常用消融参数：

```bash
--model.num_embedding_columns 1
--model.no_use_query_self_attention
--model.no_use_query_cross_attention
--model.no_use_query_seq_cross_attention
--model.no_use_seq_query_cross_attention
```

`QueryFormerModelConfig` 会进入 `train_config.json`，评估和推理会从 checkpoint sidecar 重建同一列数、DCNv2 深度和 Attention 开关。

## 多列执行方式

实现把列作为显式的 `H` 维，而不是在 Python 中逐列调用模型：

- sparse 和 sequence embedding 把每列映射到同一张物理表中的不相交词表区间，一次 lookup 产生 `[B, H, ...]`；各列参数和梯度仍完全独立，padding row 0 是唯一共享且固定为零的行。
- DCNv2、Linear 和 LayerNorm 的参数都带独立的 `H` 维，通过 batched matmul 计算。
- Attention 在独立 Q/K/V 投影后把 `B × H` 合并成 batch，一次调用 PyTorch SDPA；CUDA 可选择 fused SDPA kernel。
- `runtime.compile` 只编译 dense DCNv2/Attention/readout 主干；不同词表尺寸的 sparse lookup 保持 eager，避免 Inductor 为每张表反复重编译，并保留 COO sparse backward。

这里没有整模型 `torch.func.vmap` 路径。当前 sparse embedding 的反向梯度是 COO tensor，PyTorch 2.13 在整列 `vmap` backward 合并稀疏梯度时会失败；直接采用列批处理参数布局可以避开该限制，也不会在每个 forward 临时复制超大 embedding 表。

列批处理不减少 `H` 倍参数量或理论 FLOPs，并可能提高峰值显存；它优化的是 GPU 并行度和 kernel launch 数。真实性能仍应在目标 batch size、序列长度和线上 schema 上分别比较 `H=1` 与 `H=4`。
