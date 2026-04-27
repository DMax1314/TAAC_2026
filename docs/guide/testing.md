---
icon: lucide/clipboard-list
---

# 测试

## 测试分层

| 层级              | 标记                  | 特点                         | 示例                                                 |
| ----------------- | --------------------- | ---------------------------- | ---------------------------------------------------- |
| **Unit**          | `-m unit`             | 纯逻辑、轻依赖、快速         | 指标计算、CLI 参数解析、契约校验                     |
| **Integration**   | `-m integration`      | 跨模块闭环、需要数据         | 训练→评估→搜索完整流程                               |
| **CPU Benchmark** | `-m benchmark_cpu`    | CPU-only 基准，自动 CI 执行  | `tests/benchmarks/cpu/bench_ffn_forward.py`          |
| **GPU Benchmark** | `-m benchmark_gpu`    | GPU-only 基准，仅本地 CUDA CLI 执行 | `tests/benchmarks/gpu/bench_transformer_backends.py` |
| **Property**      | 归入 unit             | 基于 Hypothesis 的性质测试   | 指标边界、稳定哈希                                   |
| **Fault**         | 归入 unit/integration | 失败路径和异常分支           | 坏 JSON、缺失 EXPERIMENT、worker 崩溃                |

分类通过目录实现自动标记：`tests/unit/`、`tests/integration/`、`tests/gpu/`、`tests/benchmarks/cpu/` 和 `tests/benchmarks/gpu/` 会分别路由到对应阶段，不需要在每条用例上手写 `@pytest.mark`。

## 常用命令

```bash
# 同步环境
uv sync --locked --extra cpu

# 如果要跑 integration / gpu / benchmark_gpu
uv sync --locked --extra cuda126

# 完整回归
uv run pytest tests -q

# 只跑快速单元测试
uv run pytest -m unit -q

# 只跑集成测试
uv run pytest -m integration -q

# 自动 CI 上的 CPU benchmark
uv run pytest tests/benchmarks/cpu -m benchmark_cpu -v

# GPU benchmark（CLI only）
uv run pytest tests/benchmarks/gpu -m benchmark_gpu -v
```

## Coverage 收集

```bash
uv run --with coverage coverage erase
uv run --with coverage coverage run -m pytest -m unit -q
uv run --with coverage coverage run --append -m pytest -m integration -q
uv run --with coverage coverage report
```

Coverage 统计范围与门槛由 `pyproject.toml` 控制：

| 配置     | 值                                                                                            |
| -------- | --------------------------------------------------------------------------------------------- |
| 统计范围 | `src/taac2026/domain`、`src/taac2026/application/search`、`src/taac2026/application/training` |
| 分支覆盖 | 开启                                                                                          |
| 最低门槛 | 70%                                                                                           |

本地完整 coverage 仍建议按上面的 unit + integration 方式采集。快速 CI 会自动再跑一轮 `benchmark_cpu` 的 CPU-safe 子集，但 coverage 门槛仍只对 `-m unit` 的 CPU-safe 子集执行：`src/taac2026/domain/*`、`src/taac2026/application/training/__init__.py`、`src/taac2026/application/training/cli.py`、`src/taac2026/application/training/runtime_optimization.py`。`integration`、`gpu` 与 `benchmark_gpu` 则保留在本地 CUDA CLI 入口中。

## 模块改动后的最小复核

| 改动范围                                          | 跑哪些测试                                                                                                                                                            |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `domain/metrics.py`                               | `tests/unit/test_metrics.py tests/unit/test_property_based.py`                                                                                                        |
| `infrastructure/experiments/payload.py` 或 loader | `tests/unit/test_payload.py`                                                                                                                                          |
| `application/training/`                           | `tests/integration/test_profiling_unit.py tests/integration/test_profiling.py tests/unit/test_runtime_optimization.py tests/integration/test_training_recovery.py`    |
| `application/search/`                             | `tests/integration/test_search_trial.py tests/integration/test_search_worker.py tests/integration/test_search_worker_integration.py tests/integration/test_search.py` |
| 数据读取 / batch 组装                             | `tests/integration/test_data_pipeline.py tests/integration/test_runtime_integration.py`                                                                               |

!!! tip "快速复核模板"
    ```bash
    uv run pytest tests/unit/test_metrics.py tests/unit/test_property_based.py -q
    ```

## 当前回归覆盖点

| 测试文件                            | 覆盖内容                                   |
| ----------------------------------- | ------------------------------------------ |
| `test_experiment_packages.py`       | 全部 10 个实验包的前向传播和数据管道所有权 |
| `test_training_recovery.py`         | Checkpoint 恢复、训练曲线一致性            |
| `test_search_worker_integration.py` | 多 trial 派发与结果收敛                    |
| `test_property_based.py`            | 指标边界条件（空输入、NaN）                |
| `test_payload.py`                   | 缺 section、坏字段、导入失败               |
| `test_search_worker.py`             | Worker 丢结果、坏 JSON、trial 异常         |

## 编写新测试

1. 新文件加入后，放进对应阶段目录；benchmark 需要进一步区分到 `tests/benchmarks/cpu/` 或 `tests/benchmarks/gpu/`；未分类的测试如果仍留在 `tests/` 根目录，收集阶段会直接失败
2. 命令统一使用 `uv run pytest ...`
3. Property 测试控制样本数，保持稳定和快速
4. 影响训练输出物时，检查 `best.pt`、`summary.json`、`training_curves.json`、`profiling/` 的兼容性
5. 影响搜索运行时时，至少覆盖 success、fail、pruned 三类状态
