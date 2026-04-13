# 快速开始

## 前置要求

- Python ≥ 3.12（推荐 3.13）
- [uv](https://docs.astral.sh/uv/) 包管理器
- GPU（可选，CPU 也可运行但较慢）

## 安装

```bash
# 安装 Python
uv python install 3.13

# 克隆仓库并同步依赖
git clone https://github.com/Puiching-Memory/TAAC_2026.git
cd TAAC_2026
uv sync --locked
```

!!! warning "关于 PyPI 镜像"
    仓库在 `pyproject.toml` 里固定了 canonical PyPI 作为默认索引，以保证 `uv.lock` 在本机和 CI 间一致。不要额外传 `--default-index` 或 `--index-url` 指向国内镜像，否则 `uv` 会判定 `uv.lock` 需要更新。  
    如果只是想加速下载，优先使用系统代理或透明代理。

## 下载样例数据

实验包默认使用 HuggingFace 上的样例数据集 [`TAAC2026/data_sample_1000`](https://huggingface.co/datasets/TAAC2026/data_sample_1000)。框架会自动从本地 HuggingFace 缓存中解析数据路径。

```bash
# 下载数据到仓库根下的 data/ 目录（作为 HF 缓存根）
uv run python -c "
from datasets import load_dataset
load_dataset('TAAC2026/data_sample_1000', cache_dir='data')
"
```

下载完成后，实验包会自动在 `data/datasets--TAAC2026--data_sample_1000/` 下解析 parquet 文件，优先使用 `refs/main` 指向的快照，无需手动指定版本哈希。

## 训练第一个模型

```bash
uv run taac-train --experiment config/gen/baseline
```

训练完成后，产物会写入 `outputs/gen/baseline/`：

```
outputs/gen/baseline/
├── best.pt                 # 最佳 checkpoint（按 val AUC 选择）
├── summary.json            # 训练摘要（指标、超参数、耗时）
├── training_curves.json    # 逐 epoch 训练曲线
└── profiling/              # 模型参数量、FLOPs、推理延迟
```

## 评估

```bash
# 评估默认输出目录中的 best.pt
uv run taac-evaluate single --experiment config/gen/baseline
```

评估报告包含：AUC、PR-AUC、Brier Score、LogLoss、GAUC，以及推理延迟（ms/sample）。

## 超参数搜索

```bash
# 使用 Optuna 搜索 20 个 trial，自动按 GPU 显存并发调度
uv run taac-search --experiment config/gen/baseline --trials 20
```

搜索默认约束：

- 模型参数量 ≤ 3 GiB
- 验证集端到端推理总时长 ≤ 180 秒

→ 详细搜索配置见 [超参数搜索](guide/search.md)

## 运行其他实验包

所有实验包使用相同的 CLI 接口，只需替换 `--experiment` 路径：

```bash
# 训练 InterFormer
uv run taac-train --experiment config/gen/interformer

# 训练 OneTrans
uv run taac-train --experiment config/gen/onetrans

# 训练 HyFormer
uv run taac-train --experiment config/gen/hyformer
```

→ 完整实验包列表见 [实验包总览](experiments/index.md)

## 跑测试

```bash
# 完整回归
uv run pytest tests -q

# 快速单元测试
uv run pytest -m unit -q
```

→ 详细测试指南见 [测试](guide/testing.md)

## CLI 命令速查

| 命令                          | 用途               |
| ----------------------------- | ------------------ |
| `taac-train`                  | 训练实验包         |
| `taac-evaluate`               | 评估 checkpoint    |
| `taac-search`                 | Optuna 超参数搜索  |
| `taac-plot-model-performance` | 生成性能对比图     |
| `taac-clean-pycache`          | 清理 `__pycache__` |
