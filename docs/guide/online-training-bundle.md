---
icon: lucide/package
---

# 线上训练打包

当线上平台要求你手动上传代码和启动脚本时，推荐先用 `taac-package-train` 生成单个 zip。
这个命令会把指定实验包和训练所需的最小运行时裁剪出来，避免把 tests、文档、搜索 CLI、其他实验包一并上传。

## 适用场景

- 线上训练环境只接受单个压缩包上传
- 你希望每次只发布一个实验包，而不是整个仓库
- 平台提供统一的数据挂载路径和 bash 入口

## 生成 zip

```bash
# 使用默认命名
uv run taac-package-train --experiment config/baseline

# 自定义 zip 文件名
uv run taac-package-train --experiment config/interformer --bundle-name interformer-round1

# 覆盖已有产物
uv run taac-package-train --experiment config/interformer --output /tmp/interformer-online.zip --force
```

默认输出目录：

```text
outputs/training_bundles/
```

例如 baseline 会生成：

```text
outputs/training_bundles/baseline-train-bundle.zip
```

## 压缩包结构

生成的 zip 顶层包含：

```text
baseline-train-bundle.zip
├── README.md
├── bundle_manifest.json
├── run.sh
└── runtime_payload.tar.gz
```

其中：

- `run.sh` 负责解压 payload、安装依赖并执行训练
- `runtime_payload.tar.gz` 里包含最小训练源码和目标实验包
- `bundle_manifest.json` 记录实验路径、环境变量名和 payload 统计信息

其中 `runtime_payload.tar.gz` 解压后还会包含 `pyproject.toml` 与 `uv.lock`，用于保持与仓库一致的锁定依赖解析。

payload 内部大致结构如下：

```text
project/
├── config/
│   └── baseline/
├── pyproject.toml
├── uv.lock
└── src/
    └── taac2026/
        ├── __init__.py
        ├── application/
        │   ├── __init__.py
        │   └── training/
        ├── domain/
        └── infrastructure/
```

这里不会包含：

- `tests/`
- `docs/`
- `site/`
- 其他 `config/<name>/`
- 搜索和评估 CLI

## 线上运行方式

解压 zip 后，至少设置数据路径再执行 `run.sh`：

```bash
export TAAC_DATASET_PATH=/path/to/train.parquet
bash run.sh
```

如果你要显式打开运行时优化开关：

```bash
export TAAC_DATASET_PATH=/path/to/train.parquet
export TAAC_OUTPUT_DIR=/path/to/output
bash run.sh --compile --amp --amp-dtype bfloat16
```

`run.sh` 内部执行的是：

```bash
uv sync --locked --extra "${TAAC_CUDA_PROFILE:-cuda126}"
uv run taac-train --experiment ./config/baseline --dataset-path "$TAAC_DATASET_PATH" --run-dir "$TAAC_OUTPUT_DIR"
```

也就是说，bundle 本质上仍然复用了仓库里的训练 CLI，只是把输入裁成了最小可上传运行时。

## 环境变量

| 变量 | 是否必填 | 作用 |
| ---- | -------- | ---- |
| `TAAC_DATASET_PATH` | 是 | 线上数据路径，支持 parquet 文件或数据缓存目录 |
| `TAAC_OUTPUT_DIR` | 否 | 覆盖训练产物输出目录，默认写到 zip 同级 `outputs/` |
| `TAAC_BUNDLE_WORKDIR` | 否 | 控制 payload 的解压目录 |
| `TAAC_CUDA_PROFILE` | 否 | 选择 `cpu` / `cuda126`，默认 `cuda126` |
| `TAAC_ENABLE_TE` | 否 | 设为 `1` 时安装 `transformer-engine` 额外依赖 |
| `TAAC_FORCE_EXTRACT` | 否 | 设为 `1` 时强制重新解压 `runtime_payload.tar.gz` |

## 与直接训练的区别

直接在仓库里训练时，实验包通常使用仓库内默认的数据缓存路径：

```bash
uv run taac-train --experiment config/baseline
```

而 bundle 需要显式告诉运行时数据在哪，所以训练 CLI 新增了 `--dataset-path` 覆盖项：

```bash
uv run taac-train --experiment config/baseline --dataset-path /path/to/train.parquet
```

这使得：

- 本地仓库训练仍可继续用默认数据路径
- 线上训练 bundle 可以在外部挂载数据目录上直接运行

## 常见问题

### `TAAC_DATASET_PATH` 未设置

`run.sh` 会直接退出并返回非零状态。先设置环境变量再运行：

```bash
export TAAC_DATASET_PATH=/path/to/train.parquet
bash run.sh
```

### 已有旧 payload，但想重新展开

```bash
export TAAC_DATASET_PATH=/path/to/train.parquet
export TAAC_FORCE_EXTRACT=1
bash run.sh
```

### 线上环境需要额外的 Transformer Engine

```bash
export TAAC_DATASET_PATH=/path/to/train.parquet
export TAAC_ENABLE_TE=1
bash run.sh
```

### 想检查 bundle 实际包含了什么

可以直接查看 `bundle_manifest.json`，或者本地解压 `runtime_payload.tar.gz` 验证结构。

## 相关命令

```bash
# 本地训练
uv run taac-train --experiment config/baseline

# 线上训练打包
uv run taac-package-train --experiment config/baseline

# 直接覆盖数据路径
uv run taac-train --experiment config/baseline --dataset-path /path/to/train.parquet
```