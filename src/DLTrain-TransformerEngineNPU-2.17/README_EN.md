# Introduction to TransformerEngineNPU

<p align="left">
    English  | <a href="./README.md">简体中文</a>
</p>
TransformerEngine is a library provided by NVIDIA for accelerating Transformer model training and inference, while TransformerEngineNPU is its adapted version for Ascend devices, aiming to provide efficient computing support for large model training and inference on Ascend devices.

Ascend provides a basic training capability solution based on Megatron-LM + MegatronAdaptor + TransformerEngineNPU, enabling users to quickly build a large model training environment on Ascend devices and achieve out-of-the-box usability.

**·** Adaptation library for running Megatron-LM on Ascend devices: [MegatronAdaptor](https://gitcode.com/Ascend/MegatronAdaptor)

# Latest News

**·** [July 15, 2026]: 🚀MegatronAdaptor + TransformerEngineNPU supports Mcore 0.18.0

# Release Notes

See [Release Notes](docs/en/release_notes_tenpu.md) to learn about the latest released version and historical versions of TransformerEngineNPU, including the software compatibility table for each version and the update notes for the current version.

# Installation

For detailed installation steps and environment configuration, see [TransformerEngineNPU Installation Guide](docs/en/install_guide.md).

# Quick Start

## Overview

To start using MegatronAdaptor + TransformerEngineNPU, you only need to add one line of code to enable Megatron-LM on Ascend devices.

## Procedure

Take the GPT model as an example: in the Megatron-LM directory, modify the `pretrain_gpt.py` file and add a new line below import torch: import megatron_adaptor, as shown in the following modification:

```python
from functools import partial
from typing import List, Optional, Tuple

import torch
import megatron_adaptor # Add this line of code

from gpt_builders import gpt_builder
```

## Notes

TransformerEngineNPU and the native TransformerEngine share the same library module name. To avoid conflicts, ensure that both are not installed in the same environment. Keep only TransformerEngineNPU.

# API List

See the [TransformerEngineNPU API List](docs/en/api/api_list.md) to learn about the public interface list, import paths, and environment variable descriptions of each functional module. The TENPU interfaces share the same names and signatures as the native TransformerEngine, and on this basis provide Ascend-specific extended interfaces such as HIF8 and W4A8. The differences between the two are compared in the document.

# Branch Maintenance Policy

🛠️ The maintenance phases of TransformerEngineNPU version branches are as follows:

| **Status**              | **Time**   | **Description**                                                                                                                                                     |
| --------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Planned 🕐               | 1-3 months   | Planned features                                                                                                                                                     |
| Development 🕔               | 3 months     | Feature development                                                                                                                                                     |
| Maintenance 🕚               | 6-12 months  | Merge all resolved issues and release versions. Different maintenance policies are adopted for different TransformerEngineNPU versions. The maintenance periods for regular versions and long-term support versions are 6 months and 12 months, respectively.                      |
| Unmaintained 🕛             | 0-3 months   | Merge all resolved issues, with no dedicated maintainers and no version releases.                                                                                                             |
| End of Life (EOL) 🚫 | N/A        | The branch no longer accepts any modifications.                                                                                                                                         |

🛠️ TransformerEngineNPU version maintenance policy:

| **TransformerEngineNPU Branch** | **Maintenance Policy** | **Current Status** | **Release Date** | **Subsequent Status**             | **EOL Date** |
| ---------------------------- | ------------ | ------------ | ------------ | ------------------------ | ----------- |
| 26.2.0_release_v2.17         | Regular version     | Maintenance         | 2026/09/30   | Expected to become unmaintained from 2027/03/30   |             |

# Limitations

**·** TransformerEngineNPU does not support NVFP4 low-precision quantization

# Security Statement

**⚠️** [TransformerEngineNPU Security Statement](docs/en/SECURITYNOTE.md)

# Contribution Guide

Welcome to contribute to TransformerEngineNPU! See [Contribution Guide](CONTRIBUTING_EN.md) to learn how to participate in project contributions.

# Disclaimer

## To TransformerEngineNPU Users

1. All content provided by TransformerEngineNPU is intended solely for your non-commercial use.
2. The models and datasets involved in the TransformerEngineNPU test cases and sample files are used by the platform only for functional testing. Huawei does not provide any model weights or datasets. If you use such data for training, pay special attention to complying with the licenses of the corresponding models and datasets. Huawei assumes no responsibility for any infringement disputes arising from your use of these models and datasets.
3. If you encounter any issues (including but not limited to functional issues and compliance issues) while using TransformerEngineNPU, submit an issue on GitCode, and we will review and resolve it in a timely manner.
4. Third-party open-source software such as torch, on which TransformerEngineNPU functionality depends, is provided and maintained by third-party communities. Fixes for issues caused by third-party open-source software depend on the contributions and feedback of the relevant communities. You should understand that the TransformerEngineNPU repository does not guarantee fixes for issues in the third-party open-source software itself, nor does it guarantee that all vulnerabilities and errors in third-party open-source software will be tested or fixed.

## To Data Owners

If you do not wish your models or datasets to be mentioned in TransformerEngineNPU, or if you wish to update the relevant descriptions in TransformerEngineNPU, submit an issue on GitCode, and we will delete or update the relevant descriptions as requested in your issue. We sincerely appreciate your understanding of and contributions to TransformerEngineNPU.

## License Statement

For the models involved in TransformerEngineNPU, if a License exists in the model directory, that License prevails. If no License exists in the model directory, the model is licensed under the Apache 2.0 license, and the corresponding license text can be found in the TransformerEngineNPU root directory.
For the license of the TransformerEngineNPU product, see the [LICENSE](./LICENSE) file.
The documents in the TransformerEngineNPU docs directory are licensed under the CC-BY 4.0 license. For details, see the [LICENSE](./docs/LICENSE) file.

# Acknowledgments

TransformerEngineNPU is jointly contributed by Huawei Technologies Co., Ltd. and developers. We appreciate every PR from the community and welcome contributions.
