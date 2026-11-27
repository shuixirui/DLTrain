# DLTrain

面向昇腾的分布式训练代码，提供Intern-S2-Preview 35B的权重加载、训练和参数更新验证。仓库包含framework、Bridge、Adaptor和NPUEngine的全部源码，无需初始化submodule。

## 安装和运行

```bash
git clone https://github.com/shuixirui/DLTrain.git
cd DLTrain
bash install.sh
bash examples/run.sh --dry-run
MODEL_PATH=/mnt/afs-a3-weight-share/Intern-S2-Preview bash examples/run.sh
```

`install.sh`用于已有驱动和CANN的Linux ARM64开发机，默认创建`.venv-dltrain`，安装依赖及仓库源码，并检查NPU和实际导入路径。支持Git和ZIP源码目录。安装日志及实际包版本写入`logs/install-*.log`和对应的`*-packages.txt`。具体镜像、依赖和环境变量见[环境说明](tools/README.md)。

```bash
bash install.sh --dry-run
MODEL_PATH=/mnt/afs-a3-weight-share/Intern-S2-Preview bash install.sh --check-provider
```

使用其他虚拟环境时，安装和运行指定同一路径：

```bash
VENV_PATH="$PWD/.venv-clean-test" bash install.sh --check-provider
VENV_PATH="$PWD/.venv-clean-test" bash examples/run.sh
```

示例默认运行20个优化器更新步，SEQ64、GBS32，使用16个逻辑NPU及Core原生分布式AdamW。原始RedPajama数据随仓库提供，分词后再训练；参数及运行方式见[示例说明](examples/README.md)。

## 目录

| 路径 | 用途 |
| --- | --- |
| [src/](src/) | 四个组件的完整源码；分词实现在framework/tools/prepare_interns2_corpus.py |
| [examples/](examples/) | run.sh启动脚本、train_interns2.py训练入口和参数说明 |
| [tools/environment/](tools/environment/) | 环境准备、依赖清单和组件基准版本 |
| [tools/validation/](tools/validation/) | Provider检查、分布式AdamW更新验证和相关辅助函数 |
| [data/](data/README.md) | RedPajama原始分片及来源说明 |

启动与加载输出保存在`logs/<RUN_ID>.log`，逐步指标、配置、来源快照及更新检查记录保存在`runs/<RUN_ID>/`。脚本直接使用所在目录，不再将训练和验证脚本复制到仓库顶层。

## 来源与验证范围

组件基准提交记录在[component_versions.py](tools/environment/component_versions.py)，DLTrain自身变动以Git提交为准。Bridge中的ModelType兼容及专家ETP修复已合入源码。目录名称不改变原有Python模块名。

| 组件来源 | 原始许可证 |
| --- | --- |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | [LICENSE](src/DLTrain-framework/LICENSE) |
| [Ascend/MegatronAdaptor](https://gitcode.com/Ascend/MegatronAdaptor) | [LICENSE](src/DLTrain-Adaptor/LICENSE) |
| [Ascend/TransformerEngineNPU](https://gitcode.com/Ascend/TransformerEngineNPU) | [LICENSE](src/DLTrain-NPUEngine/LICENSE) |
| [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) | [LICENSE](src/DLTrain-Bridge/LICENSE) |

提交239cde3的ZIP源码已在原开发机的新目录与独立虚拟环境中完成20步训练，16个rank的专家权重抽检和首步参数更新验证通过，退出码为0。此前另有GBS8、GBS32的500步实测。当前精简和目录迁移使用本地路径及流程检查验证，尚待NPU复验；install.sh入口与全新开发机安装也待验证。完整权重/梯度数值一致性、指定起始权重身份及DLRL真实RL更新仍需专项验证。
