---
icon: lucide/database-zap
---

# Cache Policies Benchmark

> **2026-08-02 更新**：下文 2026-05-07 的测评在 symbiosis v3（`6047913`）之前测得。该提交为序列转换引入了逐元素 Python 去重/统计循环，使基础 batch 转换慢了约一个数量级；`batch_converter.py` 已于 2026-08-02 将 `_sequence_stats_and_dedup` 向量化（64 位哈希分组 + 精确校验），demo 口径下 convert 从约 `1.6s` 降至约 `0.2s`。因此下文表格中的**绝对吞吐数字已被取代**（当前 `none` 单 worker 约 `936 rows/s`，`lru` 约 `1721 rows/s`、1.84x），cache 相对收益结论方向不变，建议按文末命令重测。

> **2026-08-02 更新（OPT trace 生命周期）**：训练循环此前只在数据集构建时配置一次 OPT trace（`cyclic=False`），多轮 sweep 训练中首个 sweep 之后 trace 不再指向真实访问序列。现在 trainer 在每轮 sweep 边界重新调用 `PCVRStepDataset.refresh_opt_trace(start_step)`，把 trace 窗口重新对准下一个 sweep，同时**保留缓存内容**（key universe 不变时不再清空 slot，避免每轮重新预热）。离线模拟（120 keys、容量 64、3 轮 205 步）中，sweep 1/2 命中率由不重建的 `57%/56%` 提升到重建的 `76%/74%`，逼近全量 trace 的 `81%/79%`；真实 demo 数据端到端验证同样从 `25%/32%` 提升到 `40%/45%`。native OPT 驱逐逻辑与 Belady 最优逐点一致（离线对照 `0.512` vs `0.512`），命中率差异全部来自 trace 窗口覆盖范围。

> **2026-08-02 更新（30x 复测）**：用 30 倍合成数据（`/tmp/pcvr_synth_30x`，30,000 行 / 30 row groups，口径见下方"2026-08-02 复测"小节）重测当前代码。w0 下 `none` 801 rows/s，`lru` 1304 rows/s（1.63x、命中 46.8%），`opt` 1488 rows/s（1.86x、命中 54.1%）；w4 下 `none` 1642 rows/s，`lru` 2916 rows/s（1.78x），`opt` 2889 rows/s（1.76x、命中 52.7%，吞吐略低于 `lru` 属 shared cache 同步成本）。训练多轮场景中 trace 重建收益显著：第三轮 sweep 命中率 82.8% vs 不重建 62.1%（+20.7pp）。下方"2026-08-02 复测"小节为完整数字。

## 当前结论

### 2026-08-02 复测（30x 合成数据）

口径：`/tmp/pcvr_synth_30x/demo_30000.parquet`（30,000 行 / 30 row groups / 约 1.2 GB）+ 对应 `schema.json`；`batch_size=256`、`cache_batches=64`、`max_batches=200`（实测一轮 sweep 116 steps）、`warmup_batches=5`、`buffer_batches=1`、`sampling_strategy=step_random`、`torch_threads=4`、默认 shuffle。预加载 native extension 后计时。

| workers | mode   | rows/s | batches/s | vs `none` | hit rate | cache impl             | trace |
| ------: | ------ | -----: | --------: | --------: | -------: | ---------------------- | ----: |
|       0 | `none` |    801 |       3.2 |     1.00x |     0.0% | `PCVRMemoryBatchCache` |     - |
|       0 | `lru`  |   1304 |       5.2 |     1.63x |    46.8% | `PCVRMemoryBatchCache` |     - |
|       0 | `opt`  |   1488 |       6.0 |     1.86x |    54.1% | `PCVRMemoryBatchCache` |   205 |
|       4 | `none` |   1642 |       6.6 |     1.00x |     0.0% | `PCVRMemoryBatchCache` |     - |
|       4 | `lru`  |   2916 |      11.7 |     1.78x |    45.9% | `PCVRSharedBatchCache` |     - |
|       4 | `opt`  |   2889 |      11.6 |     1.76x |    52.7% | `PCVRSharedBatchCache` |   205 |

观察：单 worker 下 `opt` 命中率比 `lru` 高约 7 个百分点，吞吐收益最明显（1.86x）。4 worker 下 shared cache 同步成本摊薄了 `opt` 的命中率优势，`opt` 与 `lru` 吞吐基本持平，但命中率仍高约 7 个百分点。本表为单轮 loader 迭代（trace 覆盖全部 205 步的"全量预测"场景）。

**训练多轮 sweep 场景（trace 重建收益）**：同一数据集、opt w0、容量 64、窗口 2×sweep（232 步）、模拟训练 3 轮：

| sweep |  每轮重建 | 不重建 |
| ----: | --------: | -----: |
|     0 |     33.6% |  33.6% |
|     1 |     75.9% |  75.9% |
|     2 | **82.8%** |  62.1% |

前两轮 trace 仍覆盖有效未来，两者一致；第三轮起不重建的 trace 失效，命中率掉到 62.1%，重建则继续升到 82.8%（+20.7pp）。真实训练循环（`_infinite_train_batches`）在每轮 sweep 边界自动重建，因此多轮训练的实际命中率按"每轮重建"列取值。

这页记录当前代码下 PCVR 数据管线 cache 策略的完整测评。比较对象包括 `none`、`lru`、`fifo`、`lfu`、`rr`、`opt`。所有启用的 cache 策略都使用项目内 native C++ policy index：单 worker 在 Python dict 中保存 batch payload，多 worker 在 shared-memory tensor slot 中保存 batch payload。

2026-05-07 这轮重测使用 `cache_index.cpp` 的拆锁 + slot version 实现：shared cache 只在索引查询、slot 分配和版本翻转时持全局锁，tensor payload 写入和命中 clone 在锁外完成；slot version 使用奇偶状态标记正在写入的 slot，C++ victim 选择会跳过 busy slot。训练采样口径为 step-random：默认开启 shuffle，连续消费 `1500` 个 training step，并把同样的 step horizon 传给数据层生成 OPT trace。

- 单 worker 下，所有 cache 策略都明显快于 `none`。普通策略命中率约 `36.93%` 到 `37.33%`，吞吐约为 `none` 的 `1.35x` 到 `1.43x`；`opt` 命中率 `45.40%`，吞吐 `1.66x`。
- 4 worker 下，所有非 `none` 策略都确认走 `PCVRSharedBatchCache` 和 native index。普通策略仍有约 `36%` 命中率，拆锁后吞吐提升到 `none` 的 `1.06x` 到 `1.07x`。
- 4 worker 下 `opt` 仍是最强策略：命中率 `44.22%`，吞吐 `10951 rows/s`，约为 4 worker `none` 的 `1.17x`。
- 相比上一轮全局锁 shared cache，拆锁 + slot version 对 4 worker 普通策略提升约 `259` 到 `561 rows/s`，对 `opt` 提升约 `509 rows/s`。
- step-random 抽样不再保证每个样本在一个 epoch 中只出现一次；相比旧的顺序 sweep 口径，cache 命中率明显抬升，OPT 的有限未来 trace 优势也更稳定。

## 支持范围

| 项目                | 当前状态                                                                |
| ------------------- | ----------------------------------------------------------------------- |
| CLI 入口            | `taac-benchmark-pcvr-data-pipeline`                                     |
| benchmark 源码      | `src/taac2026/application/benchmarking/pcvr_data_pipeline_benchmark.py` |
| cache 源码          | `src/taac2026/infrastructure/data/cache.py`                             |
| native policy index | `src/taac2026/infrastructure/data/native/cache_index.cpp`               |
| policy 实现         | 项目内 C++ index：`lru`、`fifo`、`lfu`、`rr`、`opt`                     |
| 单 worker cache     | `PCVRMemoryBatchCache`，Python batch 存储 + C++ policy index            |
| 多 worker cache     | `PCVRSharedBatchCache`，shared-memory batch 存储 + C++ policy index     |
| 原始输出            | `outputs/benchmarks/cache_policies_slot_version_20260507T185111Z/`      |

## 策略差异

| mode   | 实现             | 驱逐依据       | 适合场景                          | 主要风险                                             |
| ------ | ---------------- | -------------- | --------------------------------- | ---------------------------------------------------- |
| `none` | 无 cache         | 不缓存         | I/O 和 batch 构建基线             | 不复用已转换 batch                                   |
| `lru`  | native C++ index | 最近访问时间   | 局部性强、热点随时间漂移          | 随机流中可能跟不上未来重复模式                       |
| `fifo` | native C++ index | 写入顺序       | 低维护开销基线                    | 不看未来访问和访问频次                               |
| `lfu`  | native C++ index | 历史访问频次   | 随机抽样中热点稳定                | 热点漂移后旧高频 key 可能滞留                        |
| `rr`   | native C++ index | 伪随机选择     | 无结构假设对照组                  | 单次方差较大                                         |
| `opt`  | native C++ index | 下一次使用距离 | step horizon 可预生成，重复访问多 | trace horizon 太短时优势变小；shared path 有同步成本 |

PCVR cache 存的是增强前的基础 batch。cache 命中后仍会 clone batch，后续 transform 也仍会执行。因此端到端吞吐同时受 parquet 读取、batch 转换、clone、cache 维护和 transform 成本影响，不要只用命中率解释结果。

## 测评环境

| 项目    | 值                                                                 |
| ------- | ------------------------------------------------------------------ |
| 时间    | 2026-05-07 UTC                                                     |
| commit  | `9bb1cf8` + 当前未提交改动                                         |
| 主机    | `HGX-076`                                                          |
| CPU     | Intel Xeon Platinum 8468，2 sockets，192 logical CPUs              |
| GPU     | 8 x NVIDIA H800 80GB，driver `535.161.08`                          |
| Python  | 3.12.13                                                            |
| PyTorch | 2.7.1+cu126                                                        |
| CUDA    | 12.6                                                               |
| 数据集  | `outputs/perf/pcvr_synthetic_300x/demo_300000.parquet`，约 12.0 GB |
| schema  | `outputs/perf/pcvr_synthetic_300x/schema.json`                     |

**2026-08-02 复测环境**（上文"当前结论"新表）：

| 项目    | 值                                                                                                                                   |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 时间    | 2026-08-02 UTC                                                                                                                       |
| 主机    | devcontainer，Ubuntu 24.04                                                                                                           |
| Python  | 3.12.13                                                                                                                              |
| PyTorch | 2.13.0                                                                                                                               |
| 数据集  | `/tmp/pcvr_synth_30x/demo_30000.parquet`，30,000 行 / 30 row groups，约 1.2 GB                                                       |
| schema  | `/tmp/pcvr_synth_30x/schema.json`                                                                                                    |
| 口径    | `batch_size=256`、`cache_batches=64`、`max_batches=200`（一轮 116 steps）、`warmup_batches=5`、`buffer_batches=1`、`torch_threads=4` |

> 官方 300x 数据集（300,000 行，约 12 GB）本地不可用，复测使用 30 倍合成数据；绝对数字与 2026-05-07 的官方数据不可直接对比，相对收益方向与命中率可参考。

主测口径：`batch_size=256`、`cache_batches=512`、`max_batches=1500`、`warmup_batches=0`、`buffer_batches=1`、`sampling_strategy=step_random`、`torch_threads=4`、默认 shuffle。训练 split 为 270 Row Groups / 270,000 rows；1500 step 约等于 1.42 个逻辑 sweep，但访问顺序是带放回 step-random 抽样，不是 epoch 顺序扫描。

正式跑前预加载 native extension，避免 `torch.utils.cpp_extension.load` 首次编译进入任一策略的计时窗口：

```bash
export TAAC_TORCH_EXTENSIONS_DIR="$PWD/outputs/benchmarks/cache_policies_slot_version_20260507T185111Z/torch_ext"
uv run python - <<'PY'
from taac2026.infrastructure.data.native.cache_index import load_native_cache_index
load_native_cache_index()
PY
```

## 完整测评结果

### 单 worker：连续 1500 step 随机流

口径：`--cache-batches 512 --max-batches 1500 --warmup-batches 0 --num-workers 0 --buffer-batches 1 --torch-threads 4`。不传 `--no-shuffle`。

| mode   | rows/s | batches/s | elapsed sec | vs `none` | hit rate | hits / misses | cache impl             | native | trace |
| ------ | -----: | --------: | ----------: | --------: | -------: | ------------: | ---------------------- | ------ | ----: |
| `none` |   3063 |     12.25 |      122.45 |     1.00x |    0.00% |         0 / 0 | `PCVRMemoryBatchCache` | false  |     - |
| `lru`  |   4146 |     16.58 |       90.49 |     1.35x |   36.93% |     554 / 946 | `PCVRMemoryBatchCache` | true   |     - |
| `fifo` |   4219 |     16.87 |       88.91 |     1.38x |   37.20% |     558 / 942 | `PCVRMemoryBatchCache` | true   |     - |
| `lfu`  |   4301 |     17.20 |       87.22 |     1.40x |   37.27% |     559 / 941 | `PCVRMemoryBatchCache` | true   |     - |
| `rr`   |   4375 |     17.49 |       85.75 |     1.43x |   37.33% |     560 / 940 | `PCVRMemoryBatchCache` | true   |     - |
| `opt`  |   5073 |     20.29 |       73.94 |     1.66x |   45.40% |     681 / 819 | `PCVRMemoryBatchCache` | true   |  1500 |

观察：单 worker 下 cache 收益非常直接。普通策略命中率接近，吞吐集中在 `4.1k` 到 `4.4k rows/s`；`opt` 多出约 8 个百分点的命中率，并把吞吐拉到 `5.1k rows/s`。slot version 的额外 busy 检查没有造成可见退化。

### 4 worker：shared-memory cache 随机流

口径：同上，但 `--num-workers 4`。所有启用 cache 的策略都走 `PCVRSharedBatchCache`。DataLoader 预取会让 cache 访问数略高于 `measured_batches=1500`，所以 shared cache 的 hits + misses 不是严格等于 1500。

| mode   | rows/s | batches/s | elapsed sec | vs `none` | hit rate | hits / misses | cache impl             | native | trace |
| ------ | -----: | --------: | ----------: | --------: | -------: | ------------: | ---------------------- | ------ | ----: |
| `none` |   9386 |     37.53 |       39.96 |     1.00x |    0.00% |         0 / 0 | `PCVRMemoryBatchCache` | false  |     - |
| `lru`  |   9971 |     39.87 |       37.62 |     1.06x |   36.10% |     544 / 963 | `PCVRSharedBatchCache` | true   |     - |
| `fifo` |  10030 |     40.11 |       37.40 |     1.07x |   35.83% |     540 / 967 | `PCVRSharedBatchCache` | true   |     - |
| `lfu`  |  10073 |     40.28 |       37.24 |     1.07x |   36.30% |     547 / 960 | `PCVRSharedBatchCache` | true   |     - |
| `rr`   |  10061 |     40.23 |       37.28 |     1.07x |   36.03% |     543 / 964 | `PCVRSharedBatchCache` | true   |     - |
| `opt`  |  10951 |     43.79 |       34.25 |     1.17x |   44.22% |     666 / 840 | `PCVRSharedBatchCache` | true   |  1500 |

观察：4 worker 下 `none` 已经能把 parquet 读取和 batch 转换并行化到 `9386 rows/s`，因此 shared cache 的相对收益仍小于单 worker。拆锁 + slot version 后，普通策略从上一轮的 `1.02x` 到 `1.04x` 提升到 `1.06x` 到 `1.07x`；`opt` 仍有最好的命中率和吞吐提升，达到 `1.17x`。

### 4 worker：拆锁前后对比

上一轮 native C++ index 已经移除了 cachetools，但 shared cache 仍在全局锁内完成 payload 写入和命中 clone。这轮拆锁 + slot version 后，4 worker 下结果如下：

| mode   | 上一轮 rows/s | 当前 rows/s | delta rows/s | 上一轮 vs `none` | 当前 vs `none` |
| ------ | ------------: | ----------: | -----------: | ---------------: | -------------: |
| `none` |          9350 |        9386 |          +36 |            1.00x |          1.00x |
| `lru`  |          9615 |        9971 |         +356 |            1.03x |          1.06x |
| `fifo` |          9771 |       10030 |         +259 |            1.04x |          1.07x |
| `lfu`  |          9620 |       10073 |         +453 |            1.03x |          1.07x |
| `rr`   |          9500 |       10061 |         +561 |            1.02x |          1.07x |
| `opt`  |         10442 |       10951 |         +509 |            1.12x |          1.17x |

结论：多 worker 退化的主要矛盾不再只是 policy index，而是 shared-memory slot 的并发读写路径。slot version 能让长时间 payload copy 离开全局锁，收益已经可见；剩余成本主要来自 shared tensor clone、miss path 写入、DataLoader 预取调度，以及所有策略仍要串行更新少量索引元数据。

## 复现命令

先按 [快速开始](../getting-started.md) 下载 `demo_1000.parquet`，把归档 schema 复制到同一目录，再生成足够大的合成数据集。不要用 1000 行 demo 数据推断 cache 策略差异。

```bash
cp docs/archive/files/schema/sample_1000_raw.schema.json data/sample_1000_raw/schema.json
```

```bash
uv run taac-generate-pcvr-synthetic-dataset \
  --source-dir data/sample_1000_raw \
  --output-dir outputs/perf/pcvr_synthetic_300x \
  --multiplier 300 \
  --force
```

完整矩阵：

```bash
out_dir="outputs/benchmarks/cache_policies_native_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$out_dir"
export TAAC_TORCH_EXTENSIONS_DIR="$PWD/$out_dir/torch_ext"

uv run python - <<'PY'
from taac2026.infrastructure.data.native.cache_index import load_native_cache_index
load_native_cache_index()
PY

modes=(none lru fifo lfu rr opt)
for workers in 0 4; do
  for mode in "${modes[@]}"; do
    uv run taac-benchmark-pcvr-data-pipeline \
      --dataset-path outputs/perf/pcvr_synthetic_300x/demo_300000.parquet \
      --schema-path outputs/perf/pcvr_synthetic_300x/schema.json \
      --preset cache \
      --cache-mode "$mode" \
      --cache-batches 512 \
      --max-batches 1500 \
      --warmup-batches 0 \
      --num-workers "$workers" \
      --buffer-batches 1 \
      --torch-threads 4 \
      > "$out_dir/cache_${mode}_long1500_workers${workers}_random.json" \
      2> "$out_dir/cache_${mode}_long1500_workers${workers}_random.log"
  done
done
```

## JSON 字段

| 字段                                   | 含义                                                 |
| -------------------------------------- | ---------------------------------------------------- |
| `cache_mode`                           | 实际启用的 cache mode                                |
| `cache_impl`                           | `PCVRMemoryBatchCache` 或 `PCVRSharedBatchCache`     |
| `shared_cache`                         | 是否使用 shared-memory cache 路径                    |
| `rows_per_sec`                         | 端到端行吞吐                                         |
| `batches_per_sec`                      | 端到端 batch 吞吐                                    |
| `warmup_batches` / `warmup_rows`       | warmup 消耗的数据量                                  |
| `measured_passes`                      | `measured_batches / batches_per_pass` 的估算 pass 数 |
| `measured_batches`                     | 计入测速的 batch 数                                  |
| `batches_per_pass`                     | 每个逻辑 sweep 的估算 batch 数                       |
| `data_cache_stats.items`               | 当前 cache 中保留的 batch 数                         |
| `data_cache_stats.hits`                | cache 命中次数                                       |
| `data_cache_stats.misses`              | cache miss 次数                                      |
| `data_cache_stats.hit_rate`            | `hits / (hits + misses)`                             |
| `data_cache_stats.opt_active`          | OPT trace 是否启用                                   |
| `data_cache_stats.native_cache_active` | 是否启用 native C++ policy index                     |
| `data_cache_stats.native_opt_active`   | OPT 是否启用 native C++ policy index                 |
| `data_cache_stats.trace_length`        | 有限未来访问 trace 长度                              |

## 解读规则

- step-random 采样不再保证每个样本在一个 epoch 中只出现一次；cache 命中率会明显高于旧的顺序 sweep 口径。
- `opt` 需要 trace horizon 覆盖测评或训练 step horizon。发现 OPT 没有优势时，先看 `trace_length` 是否足够，以及 `shuffle` 是否保持打开。
- 单 worker 和 multi-worker 都使用 native C++ policy index；差别在于 batch 存储位置分别是 Python dict 和 shared-memory tensor。
- 多 worker 下普通 cache 策略的命中率不低；拆锁 + slot version 已降低锁成本，但 shared tensor clone、miss path 写入和 DataLoader 调度仍会压低收益。
- 当前表格是单次顺序运行。正式排名建议交错 mode 顺序并重复运行，避免 OS page cache 和后台负载影响结论。
