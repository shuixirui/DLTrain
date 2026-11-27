# DLTrain

面向昇腾的分布式训练代码，提供环境与工作目录准备、Intern-S2-Preview 35B模型加载、训练更新验证、日志记录和报告生成。训练组件的完整源码随仓库提供，普通clone无需初始化submodule。

## 安装与使用

```bash
git clone https://github.com/shuixirui/DLTrain.git
cd DLTrain
python3 tools/verify_sources.py
python3 train/environment/prepare_workspace.py --dry-run
```

当前实测环境为Linux ARM64、Ascend A3 / 910C、16个逻辑NPU，Python3.12.13、CANN9.1.0镜像、PyTorch2.10.0、torch-npu2.10.0.post7、TENPU2.17.1、Triton-Ascend3.6.0.post2和Transformers5.2.0。驱动、CANN、虚拟环境、模型权重和语料需由部署环境提供。

环境安装、工作目录准备及启动命令见[训练使用说明](train/README.md)。[Intern-S2示例](examples/interns2/README.md)列出已实测配方和操作顺序。已有editable安装在源码目录改名后，需要按新路径重新执行源码安装。

## 目录

| 目录 | 用途 |
| --- | --- |
| [src/](src/) | 四个完整训练组件的源码，目录统一使用DLTrain前缀 |
| [train/](train/) | 环境依赖、训练启动、模型注册、权重加载和更新探针 |
| [data/](data/) | 公开语料下载、预处理及原生tokenizer分词 |
| [benchmark/](benchmark/README.md) | 实测日志解析、曲线绘制、告警分析和本地检查 |
| [examples/](examples/) | 按模型整理的运行示例 |
| [tools/](tools/README.md) | 源码校验、交付审计、结果打包及WebIDE传输工具 |

`src`下的组件目录为：

```text
src/
├── DLTrain-Megatron-LM/
├── DLTrain-MegatronAdaptor/
├── DLTrain-TransformerEngineNPU-2.17/
└── DLTrain-Megatron-Bridge-interns2/
```

组件基准版本供环境准备代码使用，记录在[component_versions.py](train/environment/component_versions.py)。Bridge修复已直接保存在源码中，使用时无需再应用独立补丁，修改版本通过Git提交定位。目录改名不改变Python包名。

| 组件来源 | 原始许可证 |
| --- | --- |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | [LICENSE](src/DLTrain-Megatron-LM/LICENSE) |
| [Ascend/MegatronAdaptor](https://gitcode.com/Ascend/MegatronAdaptor) | [LICENSE](src/DLTrain-MegatronAdaptor/LICENSE) |
| [Ascend/TransformerEngineNPU](https://gitcode.com/Ascend/TransformerEngineNPU) | [LICENSE](src/DLTrain-TransformerEngineNPU-2.17/LICENSE) |
| [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) | [LICENSE](src/DLTrain-Megatron-Bridge-interns2/LICENSE) |

## 测试与报告

```bash
python -m unittest discover -s benchmark/tests -v
python benchmark/plot_interns2_log.py /path/to/launcher.log --output-dir results/report
```

报告保留原始loss、裁剪前梯度范数和最慢rank的单步耗时，另行标注移动平均及计时预热范围。PNG / SVG / PDF、逐步CSV、分析JSON和ZIP均由实际日志生成。没有本地原始日志时，依赖真实日志的检查会明确跳过。

已有独立SEQ64 / GBS8及GBS32的500步实测，并完成部分embedding、专家分片和首步优化器更新抽检。运行完成不代替完整数值验收；通信尺寸、GDN权重覆盖和stream告警仍需专项核验。指定起始权重身份、DLRL正式调用与联合RL更新、导出交接及最终镜像摘要仍需相应证据。

目录分类参考[DeepLinkEngine](https://github.com/DeepLinkNext/DeepLinkEngine)，按训练运行、数据、实测和交付工具分别管理。各组件原有版权与许可证保留于源码中。
