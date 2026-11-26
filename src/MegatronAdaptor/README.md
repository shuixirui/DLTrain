# MegatronAdaptor

<p align="left">
   <a href="./README_en.md"> English </a> | 简体中文
</p>

MegatronAdaptor 是专为 Megatron-LM 在华为昇腾（Ascend）NPU 上高效运行而生的核心组件，面向上层大模型训练框架提供能力适配，让训练框架在昇腾设备上开箱即用、高可靠地运行。依托内置的夜航自动化看护能力（megatron-adaptor-sync），项目已接入 CI 流水线持续运行：每日自动同步上游 Megatron-LM 的演进，跨多分支执行回归验证，并对上游变更引发的兼容性问题自动定位根因、自动生成修复，形成「监测—诊断—修复—回归」的闭环飞轮，让适配能力随上游持续自我进化，为昇腾大模型生态提供长期稳定、可持续演进的坚实保障。

昇腾提供基于Megatron-LM + MegatronAdaptor + TransformerEngineNPU的基础训练能力方案，使用户能够快速在昇腾设备上构建起大模型训练环境，实现开箱可用。

**·** 适配昇腾设备的TransformerEngine加速库: [TransformerEngineNPU](https://gitcode.com/Ascend/TransformerEngineNPU)

# 最新消息

**·** [Jun 22, 2026]: 🚀MegatronAdaptor + TransformerEngineNPU 支持 Mcore 0.18.0 版本

# 版本说明

当前版本推荐配套表如下：

| 组件 | 版本要求 |
| ------ | ---------- |
| **MegatronAdaptor** | core_r0.18.0 |
| **Mcore** | core_v0.18.2 |
| **TransformerEngineNPU** | 26.2.0_release_v2.17 |
| **CANN版本** | 9.2.0 |
| **PyTorch版本** | 2.10.0 |
| **TorchNPU版本** | 26.2.0 |
| **Triton-Ascend版本** | 3.6.0 |
| **Python版本** | Python3.12.x |

# 安装

拉取MegatronAdaptor及TransformerEngineNPU的源码进行安装：

```bash
git clone https://gitcode.com/Ascend/MegatronAdaptor.git
cd MegatronAdaptor
git checkout core_r0.18.0
cd ..
pip install -e MegatronAdaptor
```

请参考《[Triton-Ascend安装指南](https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide.html)》获取配套版本的Triton-Ascend安装指令。

可参考如下安装命令：

```bash
pip install triton-ascend==3.6.0 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

```bash
git clone https://gitcode.com/Ascend/TransformerEngineNPU.git
cd TransformerEngineNPU
git checkout 26.2.0_release_v2.17
pip install -e . --no-build-isolation
cd ..
```

获取并切换Megatron-LM版本，参考：

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.18.2
cd ..
```

# 快速开始

## 概述

要开始使用MegatronAdaptor + TransformerEngineNPU，只需增加一行代码，即可在昇腾设备上启用Megatron-LM。

## 操作方法

以GPT模型为例：在Megatron-LM目录下修改pretrain_gpt.py文件，在import torch下新增一行：import megatron_adaptor，即如下修改：

```python
from functools import partial
from typing import List, Optional, Tuple

import torch
import megatron_adaptor # 新增此行代码

from gpt_builders import gpt_builder
```

## 注意事项

TransformerEngineNPU 和 原生TransformerEngine 的库模块名相同，为了避免冲突，请确保同一环境内不要同时安装两者，仅保留TransformerEngineNPU即可。

## 兼容性变更说明

Megatron-LM core_r0.18.0分支默认在一次性侧流上完成 DDP/FSDP 包装初始化。部分场景下会出现训练过程中迭代内显存不断持续上涨，迭代结束时才被回收，显存回落的显存滞留现象，类似显存泄漏，部分情况可能会导致OOM。

针对该问题新增 `--fix-ddp-one-shot-stream` 修复参数（默认关闭）：启用后，eager 分支下 DDP/FSDP 初始化与主流共用同一条流，消除上述显存滞留；该修复方案与高版本 Megatron-LM 保持一致。

建议仅在出现上述问题现象时启用该修复参数。

Megatron-LM core_r0.18.0分支在开启梯度累加融合且梯度规约不与反向重叠（未开启 `--overlap-grad-reduce`，默认即未开启）时，仍会为每个权重申请一个全尺寸的占位权重梯度（dummy wgrad），形成永不释放的常驻显存占用。

针对该问题新增 `--fix-non-overlap-dummy-wgrad` 修复参数（默认关闭）：启用后，非 overlap 场景下不再申请该占位张量，`--overlap-grad-reduce` 开启时行为保持不变。

详细说明见 [FAQ](./docs/zh/FAQ.md)。

# 安全声明

[MegatronAdaptor 安全声明](./docs/zh/SECURITYNOTE.md)

# 贡献指南

欢迎贡献MegatronAdaptor！请查看[贡献指南](./CONTRIBUTING.md)了解如何参与项目贡献。

# 免责声明

## 致MegatronAdaptor使用者

1. MegatronAdaptor提供的所有内容仅供您用于非商业目的。
2. 对于MegatronAdaptor测试用例以及示例文件中所涉及的各模型和数据集，平台仅用于功能测试，华为不提供任何模型权重和数据集，如您使用这些数据进行训练，请您特别注意应遵守对应模型和数据集的License，如您因使用这些模型和数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用MegatronAdaptor过程中，发现任何问题（包括但不限于功能问题、合规问题），请在GitCode提交issue，我们将及时审视并解决。
4. MegatronAdaptor功能依赖的Megatron等第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题的修复依赖相关社区的贡献和反馈。您应理解，MegatronAdaptor仓库不保证对第三方开源软件本身的问题进行修复，也不保证会测试、纠正所有第三方开源软件的漏洞和错误。

## License声明

- MegatronAdaptor中涉及的文件，如文件目录下存在License的，以该License为准。如模型目录下不存在License的，以Apache 2.0许可证许可。
- MegatronAdaptor产品的使用许可证，具体请参见[LICENSE](./LICENSE)文件。

# 致谢

MegatronAdaptor由华为技术有限公司与开发者联合贡献。
感谢来自社区的每一个PR，欢迎贡献。
