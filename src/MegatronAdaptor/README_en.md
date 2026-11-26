# Introduction

MegatronAdaptor is a core component built specifically for Megatron-LM to run efficiently on Huawei Ascend NPUs. It provides capability adaptation for upper-layer LLM training frameworks, enabling these frameworks to run out of the box and with high reliability on Ascend devices. With the built-in nightly automated watchdog capability (megatron-adaptor-sync), the project is connected to a CI pipeline and runs continuously. Every day, it automatically syncs the evolution of upstream Megatron-LM, runs regression verification across multiple branches, and automatically locates the root cause of compatibility issues caused by upstream changes and generates fixes, forming a closed-loop flywheel of monitoring, diagnosis, fix, and regression. Therefore, the adaptation capability keeps evolving on its own along with upstream, providing solid assurance for the long-term stability and sustainable evolution of the Ascend LLM ecosystem.

Ascend provides a basic training capability solution based on Megatron-LM + MegatronAdaptor + TransformerEngineNPU, which enables you to quickly build an LLM training environment on Ascend devices that is ready to use out of the box.

**·** TransformerEngine acceleration library adapted for Ascend devices: [TransformerEngineNPU](https://gitcode.com/Ascend/TransformerEngineNPU)

# Latest Updates

**·** [Jun 22, 2026]: 🚀MegatronAdaptor + TransformerEngineNPU support Mcore 0.18.0

# Release Notes

The recommended version mapping for the current release is as follows:

| Component | Version Requirement |
| ------ | ---------- |
| **MegatronAdaptor** | 26.2.0_core_r0.18.0 |
| **Mcore** | core_v0.18.2 |
| **TransformerEngineNPU** | 26.2.0_release_v2.17 |
| **CANN version** | 9.2.0 |
| **PyTorch version** | 2.10.0 |
| **TorchNPU version** | 26.2.0 |
| **Triton-Ascend version** | 3.6.0 |
| **Python version** | Python3.12.x |

# Installation

Clone the source code of MegatronAdaptor and TransformerEngineNPU and install the packages:

```bash
git clone https://gitcode.com/Ascend/MegatronAdaptor.git
cd MegatronAdaptor
git checkout 26.2.0_core_r0.18.0
cd ..
pip install -e MegatronAdaptor
```

Refer to the [Triton-Ascend installation guide](https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide.html) for the installation command of the matching Triton-Ascend version.

The following installation command is for reference:

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

Obtain and switch to the required Megatron-LM version. For reference, run the following commands:

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.18.2
cd ..
```

# Quick Start

## Overview

To start using MegatronAdaptor + TransformerEngineNPU, you need to add only one line of code to enable Megatron-LM on Ascend devices.

## Procedure

Take the GPT model as an example. In the Megatron-LM directory, modify the `pretrain_gpt.py` file and add a new line, `import megatron_adaptor`, after `import torch`, as follows:

```python
from functools import partial
from typing import List, Optional, Tuple

import torch
import megatron_adaptor # Add this line

from gpt_builders import gpt_builder
```

## Notes

TransformerEngineNPU and the native TransformerEngine share the same library module name. To avoid conflicts, ensure that you do not install both in the same environment. Keep only TransformerEngineNPU installed.

## Compatibility Change Notes

By default, the Megatron-LM core_r0.18.0 branch completes DDP/FSDP wrapper initialization on a one-shot side stream. In some scenarios, device memory keeps growing within an iteration during training and is reclaimed only when the iteration ends. This is a memory retention phenomenon similar to a memory leak, and in some cases it may cause an out-of-memory (OOM) error.

To address this issue, a new `--fix-ddp-one-shot-stream` fix parameter is added (disabled by default). When it is enabled, DDP/FSDP initialization in the eager branch shares the same stream as the main stream, which eliminates the preceding memory retention. This fix is consistent with that in later versions of Megatron-LM.

Enable this fix parameter only when the preceding issue occurs.

On the Megatron-LM core_r0.18.0 branch, when gradient accumulation fusion is enabled and gradient reduction does not overlap with the backward pass (`--overlap-grad-reduce` is not enabled, which is the default), the branch still allocates a full-size placeholder weight gradient (dummy wgrad) for each weight, which creates permanently resident memory that is never released.

To address this issue, a new `--fix-non-overlap-dummy-wgrad` fix parameter is added (disabled by default). When it is enabled, the placeholder tensor is no longer allocated in non-overlap scenarios, and the behavior remains unchanged when `--overlap-grad-reduce` is enabled.

For details, see [FAQ](./docs/en/FAQ.md).

# Disclaimer

## To MegatronAdaptor Users

1. All content provided by MegatronAdaptor is for your non-commercial use only.
2. For the models and datasets involved in MegatronAdaptor test cases and example files, the platform uses them only for functional testing. Huawei does not provide any model weights or datasets. If you use these data for training, ensure that you comply with the license of the corresponding model and dataset. If you become involved in an infringement dispute because of your use of these models and datasets, Huawei bears no responsibility.
3. If you find any issues while using MegatronAdaptor, including but not limited to functional issues and compliance issues, submit an issue on GitCode. We will review and resolve them in a timely manner.
4. The third-party open-source software that MegatronAdaptor depends on, such as Megatron, is provided and maintained by third-party communities. Fixes for problems caused by third-party open-source software depend on contributions from and feedback to the relevant communities. You should understand that the MegatronAdaptor repository does not guarantee fixes for problems in the third-party open-source software itself, and it also does not guarantee that it tests or corrects all vulnerabilities and errors in third-party open-source software.

# Acknowledgements

MegatronAdaptor is jointly contributed by Huawei Technologies Co., Ltd. and developers.
We thank the community for every PR and welcome contributions.
