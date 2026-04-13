# 开发指南：新增实验包

## 概述

新增一个实验包只需在 `config/gen/` 下创建一个目录，实现四个构建函数并导出 `EXPERIMENT` 对象。

## 目录结构

```
config/gen/my_experiment/
├── __init__.py    # 导出 EXPERIMENT
├── data.py        # build_data_pipeline
├── model.py       # build_model_component（模型架构）
└── utils.py       # build_loss_stack / build_optimizer_component
```

## 第 1 步：创建 `__init__.py`

```python
from __future__ import annotations

from pathlib import Path

from taac2026.domain.config import DataConfig, ModelConfig, TrainConfig
from taac2026.domain.experiment import ExperimentSpec

from .data import build_data_pipeline
from .model import build_model_component
from .utils import build_loss_stack, build_optimizer_component

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "data" / "datasets--TAAC2026--data_sample_1000"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "gen" / "my_experiment"

EXPERIMENT = ExperimentSpec(
    name="my_experiment",
    data=DataConfig(
        dataset_path=str(DEFAULT_DATASET),
        max_seq_len=32,
        max_feature_tokens=16,
        max_event_features=4,
        stream_batch_rows=256,
        val_ratio=0.2,
        label_action_type=2,
        dense_feature_dim=16,
    ),
    model=ModelConfig(
        name="my_experiment",
        vocab_size=131072,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        # ... 根据你的架构设计调整
    ),
    train=TrainConfig(
        epochs=10,
        batch_size=64,
        learning_rate=1e-3,
        output_dir=str(DEFAULT_OUTPUT_DIR),
    ),
    build_data_pipeline=build_data_pipeline,
    build_model_component=build_model_component,
    build_loss_stack=build_loss_stack,
    build_optimizer_component=build_optimizer_component,
    switches={"logging": True, "visualization": True},
)
```

## 第 2 步：实现 `data.py`

```python
def build_data_pipeline(data_config, train_config):
    """返回 (train_loader, val_loader, DataStats)"""
    ...
```

可以参考 `config/gen/baseline/data.py` 的标准实现。大多数实验包可以直接复用 baseline 的数据管道。

## 第 3 步：实现 `model.py`

```python
import torch.nn as nn

def build_model_component(data_config, model_config, dense_dim):
    """返回 nn.Module"""
    return MyModel(data_config, model_config, dense_dim)
```

模型接收一个 `BatchTensors` 数据类作为输入，输出一个 logit 张量。

## 第 4 步：实现 `utils.py`

```python
def build_loss_stack(model, train_config, pos_weight):
    """返回 (loss_fn, auxiliary_loss)"""
    ...

def build_optimizer_component(model, train_config):
    """返回 optimizer"""
    ...
```

!!! warning "不允许跨包导入"
    `build_loss_stack` 和 `build_optimizer_component` 必须在你自己的 `utils.py` 中实现，不能从其他实验包导入。

## 第 5 步：验证

```bash
# 检查实验包能否正确加载和前向传播
uv run pytest tests/test_experiment_packages.py -q

# 运行训练
uv run taac-train --experiment config/gen/my_experiment

# 运行评估
uv run taac-evaluate single --experiment config/gen/my_experiment
```

## 第 6 步：更新测试

确保 `tests/test_experiment_packages.py` 覆盖了新实验包。该测试会自动扫描 `config/gen/` 下的所有包并验证：

- `EXPERIMENT` 对象正确导出
- 数据管道构建成功
- 模型前向传播无报错
- `build_loss_stack` 和 `build_optimizer_component` 解析到包自身的模块

## 数据集路径约定

默认数据集路径指向 HuggingFace 缓存结构：

```
data/datasets--TAAC2026--data_sample_1000/
```

框架会自动解析 parquet 文件，优先使用 `refs/main` 指向的快照。不要在默认配置中硬编码具体的快照哈希。
