# DLTrain

面向昇腾的分布式训练代码，提供Intern-S2-Preview 35B模型加载、训练更新验证和日志报告。完整训练源码随仓库提供，普通clone无需初始化submodule。

## 使用

```bash
git clone https://github.com/shuixirui/DLTrain.git
cd DLTrain
bash examples/run.sh --dry-run
```

准备好昇腾环境和模型权重后，执行：

```bash
bash examples/run.sh
```

示例默认20个优化器更新步、SEQ64、GBS32和16个逻辑NPU，使用仓库中的RedPajama原始分片。脚本先准备工作目录和原生tokenizer缓存，再启动分布式AdamW训练；`--dry-run`仅预览命令。完整环境要求及参数见[训练说明](tools/README.md)和[Intern-S2示例](examples/README.md)。

## 目录

| 目录 | 内容 |
| --- | --- |
| [src/](src/) | DLTrain-framework、DLTrain-Bridge、DLTrain-Adaptor和DLTrain-NPUEngine完整源码 |
| [tools/](tools/) | 环境准备、模型加载、训练、分词和更新探针 |
| [examples/](examples/) | 可执行启动脚本和模型运行说明 |
| [tools/benchmark/](tools/benchmark/README.md) | 实测日志解析、绘图和本地检查 |
| [data/](data/README.md) | RedPajama原始分片和来源说明 |

## 源码与验证

组件基准版本保存在[环境准备代码](tools/environment/component_versions.py)，DLTrain自身修改以Git提交为准。Bridge的ModelType兼容与专家ETP修复已直接合入源码，无需额外打补丁。目录名称不改变组件的Python模块名。

| 组件来源 | 原始许可证 |
| --- | --- |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | [LICENSE](src/DLTrain-framework/LICENSE) |
| [Ascend/MegatronAdaptor](https://gitcode.com/Ascend/MegatronAdaptor) | [LICENSE](src/DLTrain-Adaptor/LICENSE) |
| [Ascend/TransformerEngineNPU](https://gitcode.com/Ascend/TransformerEngineNPU) | [LICENSE](src/DLTrain-NPUEngine/LICENSE) |
| [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) | [LICENSE](src/DLTrain-Bridge/LICENSE) |

本地检查与日志绘图：

```bash
python -m unittest discover -s tools/benchmark/tests -v
python tools/benchmark/plot_interns2_log.py /path/to/launcher.log --output-dir results/report
```

已有独立SEQ64 / GBS8和GBS32的500步实测及部分权重、首步更新抽检。目录整理后的示例启动流程已做本地预览和检查；实际NPU训练仍需相应环境。完整数值一致性、指定起始权重身份和DLRL联合RL更新及导出交接仍待专项验证。
