---
icon: lucide/chart-column
---

# 数据集分析与可视化

本分区提供 TAAC 2026 数据集的自动化探索性分析，以及从上届竞赛论文中提炼的数据层面经验。

当前 EDA CLI 已区分两类输入：

- `test`：带 `label_type` / `label_action_type` 的 sample 或离线测试数据，默认写出 `docs/assets/figures/eda/` 图表资产。
- `online`：线上正式 parquet 数据，默认写出 `outputs/reports/online_dataset_eda.json` 和 `outputs/reports/online_dataset_eda_charts/`，不会覆盖文档图表。单文件工具改为“编辑脚本顶部常量后直接运行”，默认按小 batch 做全量流式扫描；如果只想做局部抽样，请直接修改脚本内的 `ONLINE_EDA_MAX_ROWS` 或 `ONLINE_EDA_SAMPLE_PERCENT`。

若你在本地重跑 sample/test 流程，建议显式传入 `demo_1000.parquet` 与同目录 `schema.json`，并把 `docs/assets/figures/eda/` 下刷新后的 JSON 一并提交：

```bash
uv run taac-dataset-eda \
    --dataset-path data/sample_1000_raw/demo_1000.parquet \
    --schema-path data/sample_1000_raw/schema.json

uv run taac-dataset-eda \
    --dataset-path path/to/demo_1000.parquet \
    --schema-path path/to/schema.json \
    --output outputs/reports/dataset_eda.json

# 先编辑 tools/run_online_dataset_eda.sh 顶部的 ONLINE_EDA_* 配置，再直接运行
bash tools/run_online_dataset_eda.sh

uv run taac-bench-report --output outputs/reports/benchmark_report.json  # 写出占位 benchmark 报告
```

## 文档索引

| 文档                                 | 说明                                           |
| ------------------------------------ | ---------------------------------------------- |
| [数据集 EDA 报告](dataset-eda.md)    | 本届数据集的自动化分析报告（含图表）           |
| [性能基准](benchmarks.md)            | 当前 benchmark 占位报告入口说明                 |
| [评估指标分析](evaluation.md)        | 评估协议解读与指标优化方向                     |
| [TAAC 2025 论文洞察](taac2025-insights.md) | TAAC 2025 论文关键经验提炼 |

## 当前状态

当前仓库维护的稳定入口是 CLI 和 `tools/run_online_dataset_eda.sh`。Python 级 EDA 逻辑以 CLI 为准，不再单独承诺额外的公共 API。

## 快速背景

TAAC（腾讯广告算法挑战赛）是针对工业广告场景的**全模态生成推荐**竞赛。核心任务是：给定用户的全模态广告互动历史序列，预测用户下一个最可能交互的广告。

**与经典推荐的关键区别**：

- 生成式范式：自回归序列预测，而非判别式重排序
- 全模态输入：协作 ID + 文本嵌入 + 视觉嵌入
- 异构行为：曝光 / 点击 / 转化，需区分行为类型
- 工业级规模：百万到千万级用户序列
