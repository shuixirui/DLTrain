# Intern-S2-Preview训练示例

在仓库根目录预览和启动：

```bash
bash examples/interns2/run.sh --dry-run
MODEL_PATH=/mnt/afs-a3-weight-share/Intern-S2-Preview bash examples/interns2/run.sh
```

`--dry-run`只打印准备、分词和训练命令，可在没有模型或NPU的本机检查。实际运行需准备[昇腾环境](../../tools/README.md)和16个逻辑NPU。脚本使用仓库data/en_head.json.gz的原始RedPajama分片，不执行数据采集。

| 环境变量 | 默认值或用途 |
| --- | --- |
| MODEL_PATH | /mnt/afs-a3-weight-share/Intern-S2-Preview |
| VENV_PATH | 仓库下.venv；也可使用已激活的虚拟环境 |
| TRAIN_STEPS | 20个优化器更新步 |
| SCHEDULE_STEPS | 500，示例沿用实测学习率调度的前20步 |
| SEQ_LENGTH / NUM_MICROBATCHES | 64 / 4，MBS1和DP8得到GBS32 |
| CORPUS_DIR | 仓库下datasets/interns2-redpajama-raw |
| RUN_ID | 自动生成独立运行编号 |

例如运行完整500步：

```bash
TRAIN_STEPS=500 bash examples/interns2/run.sh
```

日志在logs/<RUN_ID>.log，结构化运行记录在runs/<RUN_ID>/。报告生成工具位于[examples/benchmark](../benchmark/README.md)。针对完整500步launcher日志的解析器不能将20步短跑当成500步通过；短跑报告应使用其运行目录及plot_interns2_adamw.py。
