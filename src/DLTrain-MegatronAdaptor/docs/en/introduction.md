# Introduction

## Overview

Megatron-LM is a mainstream distributed training framework for LLMs in the industry. It provides a rich set of parallelism strategies, such as tensor parallelism and pipeline parallelism, and is widely used in LLM pretraining scenarios. However, Megatron-LM is natively designed for GPUs. Therefore, running it directly on Ascend devices faces a series of challenges, including operator adaptation, parallel communication, and compute acceleration.

MegatronAdaptor is a foundational adaptation library designed specifically for running Megatron-LM on Huawei Ascend NPUs. It provides the runtime foundation for upper-layer model-side frameworks. Through a non-intrusive patch mechanism, MegatronAdaptor completes operator adaptation, compatibility fixes, and acceleration feature activation for Ascend devices without modifying the Megatron-LM source code. Therefore, you can quickly migrate your LLM training service to Ascend devices, and the service is ready to use out of the box.

## Architecture

The overall solution architecture of MegatronAdaptor consists of three layers:

- Training framework layer: native Megatron-LM, which provides model definition, distributed parallelism strategies, and the training process.
- Adaptation layer: MegatronAdaptor, which uses feature registration and the patch mechanism to adapt and enhance Megatron-LM on Ascend devices.
- Acceleration layer: [TransformerEngineNPU](https://gitcode.com/Ascend/TransformerEngineNPU), which provides a Transformer acceleration operator library for Ascend devices.

## Features

MegatronAdaptor adopts a decoupled "feature + patch" design and contains two core components:

- Feature management (`FeaturesManager`):
  Each feature inherits from the `MindSpeedFeature` base class and registers independently. Therefore, features are functionally cohesive and decoupled from one another.
  A feature can automatically determine whether it needs to be enabled based on training parameters. It also supports parameter registration and validation, which facilitates secondary development.

- Patch management (`PatchesManager`):
  It adapts Megatron-LM in a non-intrusive patch mode. Therefore, you do not need to modify the Megatron-LM source code.
  Patches are managed independently by feature dimension, which facilitates rapid iteration between versions and allows you to apply them to your own framework easily and quickly.
