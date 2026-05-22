---
icon: lucide/activity
---

# LayerNorm Benchmark

## 支持记录

- CLI operator：`layer_norm`。
- Torch reference：`torch.nn.functional.layer_norm(x, (cols,), weight, bias, eps)`。
- Triton backend 支持 `float16`、`bfloat16`、`float32` CUDA tensor，当前覆盖 last-dim LayerNorm、1D affine weight 和 1D bias。
- Triton inference path 使用 forward-only kernel，不写回 backward 需要的 `mean` / `inv_std`；训练 path 仍使用保存中间量的 autograd kernel。
- TileLang backend 尚未接入；显式传 `tilelang` 会在 benchmark 中报告 unsupported。
- benchmark 只测算子本体，不代表完整模型 step time。
- 主要源码：`src/taac2026/infrastructure/accelerators/normalization/layer_norm.py` 和 `src/taac2026/infrastructure/accelerators/normalization/kernels/triton.py`。

## 推荐命令

```bash
uv run taac-benchmark-pcvr-tilelang-ops \
  --operator layer_norm \
  --device cuda \
  --rows 8192 \
  --cols 128 \
  --dtype float16 \
  --backends torch,triton \
  --steps 100 \
  --warmup-steps 20 \
  --repeats 5 \
  > outputs/benchmarks/layer_norm_accelerators.json
```

## 读数要点

- `cols` 是归一化维度，当前 Triton kernel 按 last dimension 编译 specialization。
- `block_rows` 会影响训练 path 的行块配置；当前 Triton inference path 使用 one-row-per-program forward-only kernel。
- LayerNorm backward 同时产生 input、weight 和 bias 梯度，训练收益不能只看 forward-only。
- 如果 `triton` backend 报 unsupported，先确认 CUDA、Triton 安装以及 dtype / device 是否符合限制。

## 最近验收观察

最近一次本地验收口径：H800、CUDA 12.6、PyTorch 2.7.1+cu126、`rows=8192`、`cols=128`、`float16`、`torch.no_grad()` forward-only、参数保持 `requires_grad=True`。

| 口径 | torch ms | triton ms | triton / torch | max abs error |
| ---- | -------: | --------: | -------------: | ------------: |
| `taac-benchmark-pcvr-tilelang-ops` / `do_bench` | 0.0171 | 0.0111 | 1.54x | 0.001953 |
| 预编译 callable 的 Python 循环 + CUDA event | 0.0182 | 0.0458 | 0.40x | 0.003906 |

解释：`do_bench` 更接近纯 kernel 稳态时间；Python 循环口径会暴露 Triton 小 kernel 的 launch / 调度开销。当前 standalone Triton LayerNorm 已有正确的 forward-only inference path，但不建议无脑默认替换模型里的 `nn.LayerNorm`。更有价值的下一步是做 `LayerNorm + Linear`、`LayerNorm + residual` 或训练 backward 的融合口径，让 Triton kernel 把 launch 开销摊到更多工作上。
