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

日志为`logs/<RUN_ID>.log`，逐步指标、配置、摘要、入口及验证代码快照为`runs/<RUN_ID>/`。成功应出现`INTERNS2_MEASURED_TRAINING_OK`，并检查命令退出码。训练不保存checkpoint；训练loss不能替代独立质量评估。
