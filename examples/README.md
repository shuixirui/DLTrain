# Intern-S2-Preview训练示例

在仓库根目录安装并启动：

```bash
bash install.sh
bash examples/run.sh --dry-run
MODEL_PATH=/mnt/afs-a3-weight-share/Intern-S2-Preview bash examples/run.sh
```

`run.sh`负责环境加载、数据校验、分词及启动16个进程；`train_interns2.py`是实际训练入口。分词实现在`src/DLTrain-framework/tools/prepare_interns2_corpus.py`，Provider和更新验证模块位于`tools/validation/`。

`--dry-run`只打印命令，不依赖模型或NPU。真实运行要求完成[环境安装](../tools/README.md)、挂载模型权重并提供16个空闲逻辑NPU。仓库包含原始RedPajama分片，示例不采集数据。

| 环境变量 | 默认值或用途 |
| --- | --- |
| MODEL_PATH | /mnt/afs-a3-weight-share/Intern-S2-Preview |
| VENV_PATH | 仓库内.venv-dltrain，可指定其他已安装环境 |
| TRAIN_STEPS | 20个优化器更新步 |
| SCHEDULE_STEPS | 500，示例使用该调度的前20步 |
| SEQ_LENGTH / NUM_MICROBATCHES | 64 / 4，MBS1、DP8得到GBS32 |
| CORPUS_DIR | datasets/interns2-redpajama-raw |
| RUN_ID | 自动生成独立运行标识 |

```bash
VENV_PATH="$PWD/.venv-clean-test" bash examples/run.sh
TRAIN_STEPS=500 bash examples/run.sh
```

## 并行配置

### 默认拓扑与配置入口

示例使用单机16个逻辑NPU，每个NPU对应一个训练进程。并行拓扑定义在[check_interns2_provider.py](../tools/validation/check_interns2_provider.py)的`settings`字典中：

```python
"tensor_model_parallel_size": 2,
"pipeline_model_parallel_size": 1,
"expert_model_parallel_size": 8,
"expert_tensor_parallel_size": 2,
"context_parallel_size": 1,
"sequence_parallel": True,
```

| 选项 | 配置字段 | 默认值 | 含义 |
| --- | --- | ---: | --- |
| TP | tensor_model_parallel_size | 2 | 将稠密层的张量计算划分到2个进程 |
| PP | pipeline_model_parallel_size | 1 | 模型层使用1个流水线阶段 |
| EP | expert_model_parallel_size | 8 | 将256个路由专家分给8个EP rank，每个负责32个专家 |
| ETP | expert_tensor_parallel_size | 2 | 每个专家的张量计算再划分到2个进程 |
| CP | context_parallel_size | 1 | 序列使用1个上下文并行分区 |
| SP | sequence_parallel | True | 在TP组内切分部分激活的序列维度，减少激活内存 |

SP复用TP进程组，不额外增加进程数。默认拓扑的稠密数据并行大小为：

```text
DP = WORLD_SIZE / (TP × PP × CP)
   = 16 / (2 × 1 × 1)
   = 8
```

专家参数采用EP8、ETP2的分组，在此拓扑下专家数据并行大小为1。稠密参数和专家参数的优化器分片组不同，运行时可在`DISTRIBUTED_OPTIMIZER_LAYOUT`日志中查看实际分组。

### 配置全局批量

[train_interns2.py](train_interns2.py)的微批量大小MBS固定为1，`NUM_MICROBATCHES`指定每次优化器更新累计的微批次数：

```text
GBS = DP × MBS × NUM_MICROBATCHES
```

GBS按稠密DP大小计算，EP和ETP不再额外乘入该公式。默认`NUM_MICROBATCHES=4`，因此GBS为`8 × 1 × 4 = 32`。

| 目标GBS | NUM_MICROBATCHES | SEQ64时每步输入token数 |
| --- | ---: | ---: |
| 8 | 1 | 512 |
| 32 | 4 | 2,048 |
| 128 | 16 | 8,192 |

在仓库根目录选择一种配置运行：

```bash
# GBS8
NUM_MICROBATCHES=1 bash examples/run.sh

# GBS32，默认配置
NUM_MICROBATCHES=4 bash examples/run.sh

# GBS128，通过增加梯度累积次数实现
NUM_MICROBATCHES=16 bash examples/run.sh
```

这些命令沿用同一组16个NPU和默认并行拓扑，每条命令分别启动一次训练。`VENV_PATH`、`MODEL_PATH`等环境变量可以同时指定。

### 配置序列长度与检查参数

`SEQ_LENGTH`控制每条样本的token数。每次更新的输入token数为`GBS × SEQ_LENGTH`，修改序列长度不会自动改变GBS。例如预览SEQ128、GBS32的启动命令：

```bash
SEQ_LENGTH=128 NUM_MICROBATCHES=4 bash examples/run.sh --dry-run
```

确认预览中的`--seq-length 128`和`--num-microbatches 4`后，去掉`--dry-run`即可运行。序列长度需为不小于16的偶数；实际可用长度取决于设备内存和算子支持。示例默认及已完成的隔离训练测试使用SEQ64。

运行后，在`MEASUREMENT_RUN`日志及`runs/<RUN_ID>/manifest.json`中核对`world_size`、`tp`、`pp`、`ep`、`etp`、`cp`、`dp`、`micro_batch_size`、`num_microbatches`、`global_batch_size`、`seq_length`和`tokens_per_step`。其中`MEASUREMENT_RUN`只打印部分字段，完整配置以manifest为准。

### 修改并行拓扑

TP、PP、EP、ETP和CP通过Provider的`settings`配置；`run.sh`当前读取的并行相关环境变量为`NUM_MICROBATCHES`及`SEQ_LENGTH`。改变拓扑时需要同时核对以下位置：

1. 修改`tools/validation/check_interns2_provider.py`中的并行字段，保持模型结构参数与原始权重一致。
2. 核对`examples/run.sh`中的`--nproc-per-node=16`及`examples/train_interns2.py`中的`world == 16`约束。实际进程数和并行分组必须匹配。
3. 同步更新训练入口中manifest的拓扑字段；这些字段目前按默认拓扑填写。GBS使用运行时取得的DP大小计算。
4. 检查专家数与EP、各张量维度与TP/ETP的切分约束，再执行Provider检查和短步训练，核对专家权重及参数更新记录。

当前训练循环按PP1、CP1实现。启用多个流水线阶段或上下文分区，还需要适配数据分发、损失归约和专家权重抽检的层编号；仅修改Provider字段不足以完成这些改造。默认16进程拓扑下调整GBS和SEQ可直接使用上面的环境变量命令。

日志为`logs/<RUN_ID>.log`，逐步指标、配置、摘要、入口及验证代码快照为`runs/<RUN_ID>/`。成功应出现`INTERNS2_MEASURED_TRAINING_OK`，并检查命令退出码。训练不保存checkpoint；训练loss不能替代独立质量评估。
