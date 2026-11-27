# DLTrain

面向昇腾的分布式训练代码，提供Intern-S2-Preview 35B的权重加载、训练和参数更新验证。

## 安装和运行

```bash
git clone https://github.com/shuixirui/DLTrain.git
cd DLTrain
bash install.sh
bash examples/run.sh --dry-run
MODEL_PATH=/mnt/afs-a3-weight-share/Intern-S2-Preview bash examples/run.sh
```

`install.sh`用于已有驱动和CANN的Linux ARM64开发机，默认创建`.venv-dltrain`，安装依赖及仓库源码，并检查NPU和实际导入路径。具体镜像、依赖和环境变量见[环境说明](tools/README.md)。

使用其他虚拟环境时，安装和运行指定同一路径：

```bash
VENV_PATH="$PWD/.venv-clean-test" bash install.sh --check-provider
VENV_PATH="$PWD/.venv-clean-test" bash examples/run.sh
```

示例默认运行20个优化器更新步，SEQ64、GBS32，使用16个逻辑NPU及Core原生分布式AdamW。原始RedPajama数据随仓库提供，分词后再训练；参数及运行方式见[示例说明](examples/README.md)。

## 目录

| 路径 | 用途 |
| --- | --- |
| [src/](src/) | 四个组件的完整源码 |
| [examples/](examples/) | run.sh启动脚本、train_interns2.py训练入口和参数说明 |
| [tools/environment/](tools/environment/) | 环境准备、依赖清单和组件基准版本 |
| [tools/validation/](tools/validation/) | Provider检查、分布式AdamW更新验证和相关辅助函数 |
| [data/](data/README.md) | RedPajama原始分片及来源说明 |

启动与加载输出保存在`logs/<RUN_ID>.log`，逐步指标、配置、来源快照及更新检查记录保存在`runs/<RUN_ID>/`。
