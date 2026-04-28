---
icon: lucide/book-open-text
---

# 官方平台用户指南

本文根据官方企业微信文档 Tencent Angel Machine Learning Platform User Guide 于 2026-04-25 的只读页面整理，重点保留训练、模型发布、评测、平台规格与环境变量约束，方便在本仓库文档站内查阅。

## 文档元信息

| 项目             | 说明                                                                                                 |
| ---------------- | ---------------------------------------------------------------------------------------------------- |
| 原文标题         | Tencent Angel Machine Learning Platform User Guide                                                   |
| 原始链接         | <https://doc.weixin.qq.com/doc/w3_ALQATQZyACgCNqA7rGAELTHOn2rc5?scode=AJEAIQdfAAo65iNyGaALQATQZyACg> |
| 本页同步时间     | 2026-04-25                                                                                           |
| 官方最近保存时间 | 2026-04-23 19:01                                                                                     |
| 获取方式         | 基于已登录只读页面人工整理；内嵌图片按原始资源直链同步                                               |

## 同步方法

这份页面的同步分成正文和图片两部分，目标是把内容固化到仓库内，而不是依赖外部登录态或截图留档。

### 正文同步

1. 在已登录的企业微信文档只读页中，优先从编辑器状态直接读取正文，而不是抓取可见 DOM 文本。
2. 这一步使用的页面内对象是 `window.pad.editor.state`，正文提取调用如下：

```javascript
window.pad.editor.state.getTextStream().readText(0)
```

3. 提取结果里会带有少量编辑器残留格式，需要人工整理成当前这份 Markdown，重点保留训练流程、模型发布、评测约束、环境变量和平台规格。

### 图片同步

1. 不使用截图。
2. 先从页面中枚举真实内嵌图片资源，优先读取 `img.melo-inline-image` 的 `src`，必要时再从资源加载记录里补全：

```javascript
Array.from(document.querySelectorAll('img.melo-inline-image')).map((img) => img.src)
```

```javascript
Array.from(
	new Set(
		performance
			.getEntriesByType('resource')
			.map((entry) => entry.name)
			.filter((name) => /picgzc\.qpic\.cn\//i.test(name))
	)
)
```

3. 确认这些 URL 指向真实图片字节后，直接下载到本仓库静态资源目录 `docs/assets/figures/guide/official-platform-guide/`。
4. 当前页面使用相对路径引用这些本地图片，因此文档站构建后不会依赖腾讯文档登录态。

### 本地落库与验证

1. 将正文整理进当前页面，并把图片以 `official-guide-01.png` 到 `official-guide-10.png` 的固定命名落库。
2. 在页面中使用相对路径引用本地图片资源。
3. 完成后运行下面的命令重建文档站：

```bash
uv run --no-project --isolated --with zensical zensical build --clean
```

4. 构建完成后，确认 `site/assets/figures/guide/official-platform-guide/` 下已经生成对应图片文件。

## 官方界面图集

下列图片来自官方文档页面中的原始内嵌图片资源，已经同步为本仓库本地静态资产，避免后续查阅依赖外部登录态。

### 赛事首页与时间线

![官方平台首页与赛事时间线](../assets/figures/guide/official-platform-guide/official-guide-01.png)

### 训练流程总览

![官方平台训练流程总览](../assets/figures/guide/official-platform-guide/official-guide-02.png)

### 创建训练任务表单

![官方平台创建训练任务表单](../assets/figures/guide/official-platform-guide/official-guide-03.png)

### 创建任务关键区域高亮

![官方平台创建训练任务关键区域高亮](../assets/figures/guide/official-platform-guide/official-guide-04.png)

### 运行入口 Run

![官方平台训练任务 Run 入口](../assets/figures/guide/official-platform-guide/official-guide-05.png)

### 运行状态示例

![官方平台 Training Job Running 状态示例](../assets/figures/guide/official-platform-guide/official-guide-06.png)

### Instances 入口

![官方平台 Instances 入口示意](../assets/figures/guide/official-platform-guide/official-guide-07.png)

### 实例页操作入口

![官方平台实例页 Output Stop Logs 操作入口](../assets/figures/guide/official-platform-guide/official-guide-08.png)

### More 菜单常见操作

![官方平台训练任务 More 菜单](../assets/figures/guide/official-platform-guide/official-guide-09.png)

### 在线脚本编辑器

![官方平台在线脚本编辑器](../assets/figures/guide/official-platform-guide/official-guide-10.png)

## 训练任务流程

### 创建与提交

1. 填写 Job Name。
2. 填写 Job Description。
3. 通过 Local Upload 上传本地脚本，或通过 New Script 在线新建脚本。
4. 确认无误后点击 Submit 保存训练任务。

### 启动与查看运行状态

1. 点击 Run 启动任务。
2. 启动后，Instance Status 会自动切换到 Training Job Running。
3. 进入 Instances 页面后，可以查看 Output 和 Logs。
4. 如果需要提前终止任务，可以点击 Stop。

### 常见任务操作

- More 菜单支持复制、删除、编辑当前训练任务。
- 模型发布入口在训练任务的实例产物页面，不在训练表单里直接出现。

## 模型发布与管理

### 发布步骤

1. 在训练任务中点击 Instances。
2. 打开 Output。
3. 选中要发布的 checkpoint，点击 Publish。
4. 填写 Model Name 和 Model Description。
5. 提交成功后，Publish Status 会切换为 Released。

### 发布后管理

- 左侧栏的 Model Management 可以查看和管理已发布模型。
- 官方文档提到可以继续做模型编辑或删除，但正文没有给出更细的字段级说明。

## 评测任务流程

### 入口

有两种进入方式：

1. 在 Model Management 页面找到目标模型，点击卡片底部的 Model Evaluation。
2. 直接进入 Model Evaluation 页面，点击 Create Evaluation。

### 提交要求

| 项目       | 官方要求                                            |
| ---------- | --------------------------------------------------- |
| 选择模型   | 从下拉框中选择已训练完成的模型                      |
| 推理代码   | 通过 Upload from Local 上传，或通过 New Script 创建 |
| 主入口文件 | 必须严格命名为 infer.py                             |
| 入口函数   | infer.py 中必须存在无参数 main()                    |
| 附加脚本   | 可上传 dataset.py、model.py 和其他自定义模块        |
| 上传位置   | 所有上传文件都会放到 EVAL_INFER_PATH                |
| 脚本总大小 | 不得超过 100 MB                                     |

### 状态流转

| 状态                             | 含义                     |
| -------------------------------- | ------------------------ |
| Pending                          | 已提交，等待排队         |
| Waiting for Inference Resources  | 等待可用推理资源         |
| Inference Running                | infer.py 正在执行        |
| Waiting for Evaluation Resources | 推理完成，等待评分资源   |
| Evaluation Running               | 平台正在对结果打分       |
| Success                          | 评测完成，可查看得分     |
| Failed                           | 评测失败，需要看日志排查 |

### 依赖安装钩子

官方文档说明，评测任务可以额外上传一个严格命名为 prepare.sh 的脚本。它会在推理开始前自动执行，并且执行时所在环境已经是平台预激活的 Conda 环境。

```bash
#!/bin/bash

# 1. 该脚本会在推理逻辑开始前自动执行
# 2. 运行位置是平台预激活环境
# 3. 在这里加入你自己的准备步骤
```

官方原文明确允许在这里执行平台侧环境准备命令；本仓库本地开发和复现实验仍建议按仓库规范使用显式 profile 的 uv 工作流。

## 平台规格

### 硬件环境

官方文档给出的单个 GPU 切片规格如下：

| 资源     | 规格     |
| -------- | -------- |
| 计算份额 | 单卡 20% |
| GPU 显存 | 19 GiB   |
| CPU 核数 | 9        |
| 内存     | 55 GiB   |

### 软件环境

| 软件   | 版本            |
| ------ | --------------- |
| Ubuntu | 22.04           |
| cuda   | 12.6            |
| cudnn  | 9.5.1           |
| cublas | 12.6.3.3        |
| nccl   | 2.26.2+cuda12.6 |
| conda  | 26.1.1          |
| python | 3.10.20         |

### 预装 Python 栈摘要

官方文档列出了完整的预装包清单。对本仓库最相关的条目包括：

| 包          | 版本         |
| ----------- | ------------ |
| torch       | 2.7.1+cu126  |
| torchaudio  | 2.7.1+cu126  |
| torchvision | 0.22.1+cu126 |
| torchrec    | 1.2.0+cu126  |
| fbgemm_gpu  | 1.2.0+cu126  |
| triton      | 3.3.1        |
| tensorflow  | 2.19.0       |
| pandas      | 2.3.3        |
| pyarrow     | 23.0.1       |
| optuna      | 4.8.0        |

原文还说明，如果平台默认环境没有你需要的依赖，可以在平台提供的环境里追加安装。

## 训练阶段环境变量

### 官方变量表

| 变量名               | 说明                                         |
| -------------------- | -------------------------------------------- |
| USER_CACHE_PATH      | 用户缓存目录，配额 20 GB；训练与评测阶段共用 |
| TRAIN_DATA_PATH      | 训练数据目录                                 |
| TRAIN_CKPT_PATH      | checkpoint 保存目录                          |
| TRAIN_TF_EVENTS_PATH | TensorBoard event 文件目录                   |

### 读取示例

Shell：

```bash
${TRAIN_DATA_PATH}
```

Python：

```python
import os

os.environ.get("TRAIN_DATA_PATH")
```

### 输出约束

1. 模型权重必须写入 TRAIN_CKPT_PATH。
2. 迭代 checkpoint 必须放在目录名前缀为 global_step 的目录下，否则平台不会识别。
3. checkpoint 目录名长度不能超过 300 个字符。
4. 目录名只允许字母、数字、下划线、连字符、等号和点号。

官方示例里给出的合法命名方式类似：

```text
global_step20.lr=0.001.layer=2.head=1.hidden=50.maxlen=200
```

### TensorBoard 指标

- 平台只支持 scalar 指标。
- 官方示例使用 TRAIN_TF_EVENTS_PATH 初始化 SummaryWriter。

```python
from torch.utils.tensorboard import SummaryWriter
import os

writer = SummaryWriter(os.environ.get("TRAIN_TF_EVENTS_PATH"))
```

## 评测阶段环境变量

### 官方变量表

| 变量名            | 说明                                         |
| ----------------- | -------------------------------------------- |
| USER_CACHE_PATH   | 用户缓存目录，配额 20 GB；训练与评测阶段共用 |
| MODEL_OUTPUT_PATH | 已发布模型产物目录                           |
| EVAL_DATA_PATH    | 推理所需测试数据目录                         |
| EVAL_RESULT_PATH  | 中间结果和 predictions.json 输出目录         |
| EVAL_INFER_PATH   | 用户上传推理脚本所在目录                     |

### 读取示例

Shell：

```bash
${EVAL_DATA_PATH}
```

Python：

```python
import os

os.environ.get("EVAL_DATA_PATH")
```

## predictions.json 规范

### 强制要求

1. 主脚本必须严格命名为 infer.py。
2. infer.py 必须包含无参数 main()。
3. main() 必须在 EVAL_RESULT_PATH 下生成 predictions.json。

### 数据格式

| 字段        | 类型               | 说明                                                              |
| ----------- | ------------------ | ----------------------------------------------------------------- |
| predictions | map<string, float> | key 是 user_id，value 是该用户交互为正样本的预测概率，范围 0 到 1 |

示例：

```json
{
	"predictions": {
		"user_001": 0.8732,
		"user_002": 0.1245,
		"user_003": 0.5621
	}
}
```

### 额外注意事项

- predictions 里的每个 key 都必须是测试集里的有效 user_id。
- 少传、漏传或出现额外 user_id 都可能影响最终得分。
- 每个团队每天最多可提交 3 个评测任务，按 AOE 时区计算；失败和主动停止的任务不计入该限制。

## 平台强制入口提醒

官方文档最后特别强调：训练模板里必须包含 run.sh，平台会在任务启动时自动执行它。

这与本仓库当前的统一入口设计是一致的：无论本地还是线上 bundle，入口都应该围绕 run.sh 展开，而不是依赖手工点击后的隐式启动逻辑。

## 与本仓库文档的对应关系

- 如果你关心平台实测容器资源和我们在线上日志中观察到的差异，见 [比赛线上服务器设备信息](competition-online-server.md)。
- 如果你要把本仓库实验包上传到线上平台，见 [线上训练打包](online-training-bundle.md)。

## 与当前仓库观察到的差异

这份官方指南与我们仓库内的线上日志观察有一处值得特别注意的差异：

- 官方指南列出的基础软件环境是 Python 3.10.20。
- 我们在 [比赛线上服务器设备信息](competition-online-server.md) 中记录到的某次任务启动参数里，平台曾显式传入 requested_python=3.13。

更稳妥的理解是：官方指南描述的是平台默认基础镜像，而实际任务仍可能叠加比赛模板、运行脚本或平台侧参数覆盖，因此线上排障时应同时参考官方指南和真实任务日志。