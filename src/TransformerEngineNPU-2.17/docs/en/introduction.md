# Introduction to TransformerEngineNPU

## Overview

TransformerEngine is a library provided by NVIDIA for accelerating Transformer model training and inference, while TransformerEngineNPU (hereinafter referred to as TENPU) is its adapted version on Ascend devices, aiming to provide efficient computational support for large model training and inference on Ascend devices.

TENPU maintains the **same library name, module path, and interface signature** (`import transformer_engine.*`) as the native TransformerEngine. Model code already running on CUDA devices can directly switch to the Ascend backend without modifying the import statements. On top of the native TransformerEngine, TENPU also provides some Ascend-specific extensions (such as HIF8 delayed scaling, W4A8 weight quantization, and swap attention).

## TransformerEngineNPU Architecture

The overall architecture of TransformerEngineNPU is divided into four layers from top to bottom:

- Interface layer: maintains the same library name, module path, and interface signature as the native TransformerEngine, and provides basic layer modules (`Linear`, `LayerNorm`, `DotProductAttention`, etc.) and the quantization context (`autocast`) externally. Service code can switch to the Ascend backend without modifying import statements.
- Quantization framework layer: composed of quantization recipes, quantization management, and quantization tensors, providing scaling strategies, quantization state management, and quantization data structures for low-precision training such as FP8.
- Operator implementation layer: carries the operator semantics of the module layer above and encapsulates the underlying kernels below, including ops operator encapsulation (with operator fusion fuser) and Triton implementations based on Triton-Ascend.
- Underlying dependencies: the Ascend software stack composed of TorchNPU, CANN, and Triton-Ascend, providing device-side execution capabilities.

## Functional Modules

TransformerEngineNPU consists of six major functional modules: the high-performance foundational layer module, FP8/low-precision training, MoE support, optimizer and multi-Tensor operators, training parallelism and memory optimization, and numerical debugging.

For the interface list and descriptions of each module, see [API List](api/api_list.md).
