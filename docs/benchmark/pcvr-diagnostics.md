---
icon: lucide/activity
---

# PCVR Smoke Diagnostics

本地只有 `demo_1000.parquet` 时，AUC 只能作为 smoke signal。更可靠的图应该看运行成本、预测行为、模型间差异和 seed 稳定性。

## 2026-08-20 模型保留检查

为了判断 TokenFormer 是否还有独立维护价值，在 commit `41e1d38` 的工作树上对 7 个现存模型做了同口径的 3-seed smoke。环境为 NVIDIA A30、driver `595.71.05`、PyTorch `2.13.0+cu132`；数据为本地 `demo_1000.parquet`，使用 `timestamp_auto` 划分、`valid_ratio=0.2`、seed `17/42/97`，每次训练 20 step。统一使用 AdamW、`lr=1e-4`、torch Flash/RMS backend，关闭 AMP、compile、EMA 和模型私有数据增强；模型自身结构宽度保持默认值。

下表 AUC/LogLoss 来自训练过程中未参与优化的 201 行留出集，时间和显存是 20 step 训练均值。独立 `val` 跑遍 1000 行，只用于预测相关性，不用于质量排名。

| 模型 | AUC（mean ± sample sd） | LogLoss（mean ± sample sd） | 参数量 | 训练时间 | CUDA peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.7318 ± 0.0176 | 0.3198 ± 0.0053 | 160.9M | 8.96s | 1766 MiB |
| InterFormer | 0.7110 ± 0.0294 | 0.3485 ± 0.0070 | 159.2M | 10.09s | 1728 MiB |
| OneTrans | 0.7240 ± 0.0103 | 0.3395 ± 0.0076 | 544.7M | 17.54s | 4332 MiB |
| **TokenFormer** | **0.7422 ± 0.0020** | **0.3112 ± 0.0017** | **159.0M** | **8.71s** | **1705 MiB** |
| Symbiosis | 0.7721 ± 0.0067 | 0.3185 ± 0.0019 | 85.0M | 10.48s | 1280 MiB |
| DualQ | 0.7591 ± 0.0103 | 0.2842 ± 0.0041 | 546.7M | 17.69s | 5248 MiB |
| QueryFormer | 0.7546 ± 0.0043 | 0.3175 ± 0.0034 | 2190.8M | 46.79s | 17681 MiB |

**结论：保留 TokenFormer。** 它不是本次最高 AUC 的模型，但在 3 个 seed 上方差最小，LogLoss 仅次于 DualQ，并处于最低延迟、最低显存的一档。把每个模型的 3-seed 逐样本预测先取均值后，TokenFormer 与 DualQ、QueryFormer 的 Spearman 相关系数分别为 `0.840`、`0.741`：排序信号并非完全独立，但也不能视为同一模型的等价缩放。它以约 1/3 和 1/14 的参数量提供接近的质量，仍然是有价值的高效结构对照。这个 1000 行、20 step 结果只能支持仓库保留决策，不能外推为正式榜单结论。

原始 run、逐样本预测和生成图保存在本地 `outputs/model_selection_20260820/`；该目录是可再生输出，不提交到仓库。

## 生成方式

先发现当前仓库支持的 PCVR 模型实验包。`host_device_info`、`online_dataset_eda` 这类 maintenance/EDA 包不会产出 PCVR 预测文件，不纳入这组图。

```bash
uv run python - <<'PY'
from pathlib import Path

from taac2026.application.experiments.registry import load_experiment_package

for child in sorted(Path("experiments").iterdir()):
    if not child.is_dir() or child.name.startswith("__"):
        continue
    experiment = load_experiment_package(f"experiments/{child.name}")
    if experiment.kind == "pcvr":
        print(child.name)
PY
```

然后分别训练、评估和推理这些实验包。下面命令会对所有 PCVR 模型包生成 `outputs/smoke/<name>_seed42`；本地 demo smoke 显式使用 torch 后端和 `--no-compile`，避免把 TileLang/NVCC 或 `torch.compile` 的一次性编译成本混进诊断图。

```bash
export CUDA_VISIBLE_DEVICES=0
SCHEMA="outputs/perf/pcvr_synthetic_300x/schema.json"

for exp in baseline interformer onetrans tokenformer symbiosis dualq queryformer; do
  run_dir="outputs/smoke/${exp}_seed42"
  bash run.sh train \
    --experiment "experiments/${exp}" \
    --run-dir "$run_dir" \
    --schema-path "$SCHEMA" \
    --optimizer.seed 42 \
    --data.num_workers 0 \
    --model.flash_attention_backend torch \
    --model.rms_norm_backend torch \
    --runtime.no_compile

  checkpoint="$(find "$run_dir" -mindepth 2 -maxdepth 2 -name model.safetensors | sort | tail -n 1)"

  bash run.sh val \
    --experiment "experiments/${exp}" \
    --run-dir "$run_dir" \
    --schema-path "$SCHEMA" \
    --num-workers 0 \
    --no-compile

  bash run.sh infer \
    --experiment "experiments/${exp}" \
    --schema-path "$SCHEMA" \
    --checkpoint "$checkpoint" \
    --result-dir "$run_dir" \
    --num-workers 0 \
    --no-compile
done
```

然后把多个 run 目录交给诊断绘图命令：

```bash
uv run taac-plot-pcvr-diagnostics \
  --run baseline=outputs/smoke/baseline_seed42 \
  --run interformer=outputs/smoke/interformer_seed42 \
  --run onetrans=outputs/smoke/onetrans_seed42 \
  --run tokenformer=outputs/smoke/tokenformer_seed42 \
  --run symbiosis=outputs/smoke/symbiosis_seed42 \
  --run dualq=outputs/smoke/dualq_seed42 \
  --run queryformer=outputs/smoke/queryformer_seed42 \
  --group-by label \
  --output-dir figures/pcvr_diagnostics
```

默认命令只在每个 run 目录都已经有 `evaluation.json` 和 `validation_predictions.jsonl` 时生成图；如果这些文件缺失，CLI 会直接报错并给出需要先跑的 `bash run.sh val ...` 命令。完整机器可读摘要写入 `pcvr_diagnostics_summary.json`，终端只打印精简报告；需要把完整 JSON 打到 stdout 时加 `--json`。

输出目录会包含：

| 文件                               | 含义                                                      |
| ---------------------------------- | --------------------------------------------------------- |
| `pcvr_runtime_resources.svg`       | 模型级平均耗时、吞吐、参数量和 CPU / CUDA 峰值资源占用    |
| `pcvr_prediction_distribution.svg` | seed-mean 预测的类条件分布和正负样本均值间隔              |
| `pcvr_prediction_correlation.svg`  | 模型间 Spearman 相关性与模型内 seed 一致性                |
| `pcvr_sample_disagreement.svg`     | 模型级 seed-mean 的样本分歧和 top disagreement 样本       |
| `pcvr_stability.svg`               | AUC、LogLoss、逐样本 seed drift 和评估耗时稳定性           |
| `pcvr_diagnostics_summary.json`    | 绘图所用 run、metrics、telemetry 和图路径摘要             |

## 输入约定

每个 `--run` 可以是目录，也可以是 `label=目录`。目录下优先读取这些文件：

- `evaluation.json`
- `validation_predictions.jsonl`
- `training_summary.json`
- `training_telemetry.json`
- `evaluation_telemetry.json`
- `inference_telemetry.json`

`bash run.sh train` 会写 `training_summary.json` 和 `training_telemetry.json`。`bash run.sh val` 会写 `evaluation.json`、`validation_predictions.jsonl` 和 `evaluation_telemetry.json`。`bash run.sh infer` 会在 result dir 写 `predictions.json` 和 `inference_telemetry.json`。

稳定性图和 summary 中的质量指标优先读取 `training_summary.json` 的 `validation_metrics`，其 `metric_source` 为 `training_validation`。只有训练摘要没有留出集指标时，才回退到 `evaluation.json.metrics`，并标记为 `evaluation`。图中的 `±` 是同组 run 的 sample SD，误差线表示 min-max；`Prediction Seed Drift` 是同一模型对每个样本跨 seed 计算 sample SD 后再取样本均值。

预测分布、相关性和分歧图始终来自 `validation_predictions.jsonl`。同组有多个 seed 时，先按样本键对齐并取 seed 均值，再比较模型；这样模型差异不会和 seed 噪声混在同一个 21-run 矩阵里。相关性使用 Spearman 而不是 Pearson，因为这里关心的是推荐排序信号，而不是概率刻度是否相同。应确认所有 `val` 命令使用相同的数据范围和样本键。

资源图同样先按组取平均。只有所有模型组都有 inference telemetry 时才显示 inference 列和 inference tradeoff；否则使用 evaluation efficiency，不把缺失值伪装成 0，也不拿不完整的子集做横向比较。

如果你只是想检查路径或预览占位图，可以加 `--allow-partial`，但这种输出不应该用于分析：

```bash
uv run taac-plot-pcvr-diagnostics \
  --run baseline=outputs/smoke/baseline_seed42 \
  --output-dir figures/pcvr_diagnostics \
  --allow-partial
```

## 稳定性分组

默认按 `evaluation.json` 里的 `experiment_name` 分组。多 seed 跑法可以这样：

```bash
uv run taac-plot-pcvr-diagnostics \
  --run baseline_seed1=outputs/smoke/baseline_seed1 \
  --run baseline_seed2=outputs/smoke/baseline_seed2 \
  --run tokenformer_seed1=outputs/smoke/tokenformer_seed1 \
  --run tokenformer_seed2=outputs/smoke/tokenformer_seed2 \
  --group-by label-prefix \
  --output-dir figures/pcvr_diagnostics
```

`--group-by label-prefix` 会把 `baseline_seed1`、`baseline_seed2` 归到 `baseline`。分组同时控制资源汇总、预测聚合、相关性、分歧和稳定性，不只影响稳定性图。如果想把每个 run 当成独立模型比较，用 `--group-by label`。

## 解读原则

- `runtime_resources` 回答“质量接近时，哪个模型更快、更小、更省显存”；不同硬件或运行参数不能放在同一张图里横比。
- `prediction_distribution` 比较正负样本的 seed-mean 概率形状和均值间隔；它能发现输出塌缩或异常长尾，但不能替代校准曲线。
- `prediction_correlation` 的左图判断模型是否提供不同排序，右图判断同一模型换 seed 后排序是否稳定。
- `sample_disagreement` 在模型级均值上定位意见分歧最大的样本，适合回到原始特征做 case study；它不是误差归因。
- `stability` 同时报告质量指标的 seed 波动和逐样本 prediction drift；demo1000 下应优先看稳定性，不追逐单次最高 AUC。

这些图仍然不是正式 leaderboard 结论。它们的定位是本地 smoke benchmark 的工程诊断面板。
