# API List

This document organizes the public interface list of TransformerEngineNPU (TENPU) according to the six functional modules in [Introduction](../introduction.md). TENPU interfaces keep the same names and signatures as the native TransformerEngine (NVTE), and provide Ascend-specific extensions on this basis.

**Interface Source Description**

- Native interface: interfaces aligned with NVTE, with consistent semantics. Refer to the [NVTE official documentation](https://docs.nvidia.com/deeplearning/transformer-engine/) for detailed behavior;
- Extension interface: Ascend extension interfaces, provided only in TENPU.

> [!NOTE]
>
> TransformerEngineNPU and the native TransformerEngine share the same library name and module path (both are transformer_engine). Do not install both in the same environment. If the NVIDIA version of TransformerEngine is already installed, run `pip uninstall transformer_engine` to uninstall it first.

**Document Structure**

Except for environment variables, each table specifies the import path of its interfaces. An example of the import method:

```python
from transformer_engine.pytorch import Linear, autocast
from transformer_engine.common import recipe
```

The following table provides a quick overview of the positioning of each functional module.

| Functional Modules | Positioning |
|------|------|
| [High-Performance Foundational Layer Module](#high-performance-foundational-layer-module) | Foundational layer modules such as `Linear` and `LayerNorm`, attention modules such as `DotProductAttention` and RoPE, and fusible operators in ops |
| [FP8/Low-Precision Training](#fp8low-precision-training) | Quantization context, capability query, Recipe strategy, quantized tensors, and Quantizer |
| [MoE Support](#moe-support) | Token permutation and MoE routing |
| [Optimizer and Multi-Tensor Operators](#optimizer-and-multi-tensor-operators) | Multi-tensor fused operators for optimizer steps |
| [Training Parallelism and Memory Optimization](#training-parallelism-and-memory-optimization) | CPU Offload, activation recomputation, and distributed utilities |
| [Numerical Debugging](#numerical-debugging) | Precision localization tool |
| [Environment Variables](#environment-variables) | Runtime behavior switches |
| [Differences from NVTE](#differences-from-nvte) | Summary of unavailable/extended/unsupported interfaces |

## High-performance Foundational Layer Module

**Table 1** Foundational layer module list

Import from **transformer_engine.pytorch**:

| API | Type | Description | Interface Source |
|-----|------|------|------|
| `Linear` | Class | Fully connected layer, supporting FP8/BF16, tensor parallelism, and delayed wgrad | Native interface |
| `GroupedLinear` | Class | Grouped fully connected layer (MoE scenario), supporting FP8 and grouped wgrad accumulation | Native interface |
| `LayerNorm` | Class | LayerNorm normalization layer, supporting zero-centered gamma and sequence parallelism | Native interface |
| `RMSNorm` | Class | RMSNorm normalization layer, supporting zero-centered gamma and sequence parallelism | Native interface |
| `LayerNormLinear` | Class | Fused normalization + linear layer module (`normalization` can be `"LayerNorm"`/`"RMSNorm"`) | Native interface |
| `LayerNormMLP` | Class | Fused normalization + two-layer MLP module, supporting GLU activation | Native interface |
| `TransformerEngineBaseModule` | Class | Common base class of the above modules | Native interface |

> [!NOTE]
>
> The following interfaces are retained in TENPU only for interface compatibility and have no actual functionality:
>
> - `Fp8Padding` / `Fp8Unpadding`: directly pass through the input on the NPU without performing padding/unpadding;
> - `initialize_ub` / `destroy_ub` / `UserBufferQuantizationMode`: the UserBuffer communication overlap mechanism of NVTE is CUDA-specific and is not implemented in TENPU.

**Table 2** Attention module list

Import from **transformer_engine.pytorch** or **transformer_engine.pytorch.attention**:

| API | Type | Description | Interface Source |
|-----|------|------|------|
| `DotProductAttention` | Class | Fused attention module: supports GQA/MQA, self/cross attention, mask types such as `no_mask`/`causal`/`padding`/`arbitrary`, sliding window attention (`window_size`), and THD variable-length sequences (`cu_seqlens_q`/`cu_seqlens_kv`); context parallelism has three built-in strategies: Ulysses, Ring, and KV-AllGather (specified by `cp_comm_type`) | Native interface |
| `DotProductAttention` (Swap strategy) | Class | Swap strategy for context parallelism (`cp_comm_type="swap"`) and the accompanying swap flash attention, an Ascend extension implementation | extension interface |
| `RotaryPositionEmbedding` | Class | RoPE positional encoding, supporting `rotary_percent`, positional interpolation, and interleaved layout | Native interface |
| `apply_rotary_pos_emb` | Function | Applies RoPE to the input tensor, supporting `sbhd`/`bshd`/`thd` formats, the `fused=True` fused implementation, and context parallelism | Native interface |
| `apply_fused_qkv_rotary_pos_emb` | Function | Performs fused RoPE on Q/K/V in one pass (commonly used in inference scenarios) | Native interface |

**Table 3** Fusible foundational operator list

Import from **transformer_engine.pytorch.ops**:

| API | Type | Description | Interface Source |
|-----|------|------|------|
| `AllGather` / `AllReduce` / `ReduceScatter` | Class | Collective communication operators that can be fused with adjacent operators for execution | Native interface |
| `LayerNorm` / `RMSNorm` / `L2Normalization` | Class | Normalization operators | Native interface |
| `Quantize` | Class | Quantization operator | Native interface |
| `Bias` / `ConstantScale` / `Identity` / `Reshape` | Class | Numeric and shape manipulation operators | Native interface |
| `AddExtraInput` / `MakeExtraOutput` | Class | Gradient accumulation auxiliary operators for multi-microbatch scenarios | Native interface |
| `GELU` / `SiLU` / `ReLU` / `QGELU` / `SReLU` / `ScaledSReLU` | Class | Activation operators | Native interface |
| `GLU` / `GEGLU` / `ReGLU` / `SReGLU` / `QGEGLU` | Class | Gated activation operators | Native interface |
| `SwiGLU` / `ScaledSwiGLU` / `ClampedSwiGLU` / `ScaledClampedQGeGLU` | Class | SwiGLU-family variant activation operators | Native interface |

**Table 4** Operator fusion and composition interface list

Import from **transformer_engine.pytorch.ops**:

| API | Type | Description | Interface Source |
|-----|------|------|------|
| `FusibleOperation` | Class | Base class of fusible operators, the common parent class of the basic operators above | Native interface |
| `BasicOperation` | Class | Base class of basic operators | Native interface |
| `FusedOperation` | Class | Fused operator that encapsulates the overall execution after fusion | Native interface |
| `Sequential` | Class | Sequential composition container that executes the operators within it in order | Native interface |
| `register_forward_fusion` / `register_backward_fusion` | Function | Register operator-level forward/backward fusion patterns | Native interface |

> [!NOTE]
>
> The fusible operator APIs are experimental interfaces and may be adjusted in later versions.

## FP8/Low-Precision Training

**Table 5** Quantization context interface list

Import from **transformer_engine.pytorch**:

| API | Type | Description | Interface Source |
|-----|------|------|------|
| `autocast` | Class (context manager) | Quantization context manager. When `enabled=True`, TENPU modules within it enable low-precision computation | Native interface |
| `fp8_autocast` | Class (context manager) | FP8 context manager (the FP8-specialized form of `autocast`) | Native interface |
| `fp8_model_init` | Function | Initializes the model in FP8 master weight format | Native interface |
| `quantized_model_init` | Function | Initializes the model in quantized format, automatically selecting the storage type according to the recipe | Native interface |
| `reset_fp8_amax_history` | Function | Resets the amax history of all Float8 quantization | Native interface |
| `reset_hif8_amax_history` | Function | Resets the amax history of HIF8 quantization | Extension interface |

**Table 6** Quantization capability query interface list

Import from **transformer_engine.pytorch.quantization** (`is_fp8_available` and `get_default_recipe` can also be imported from the top-level **transformer_engine.pytorch**):

| API | Type | Description | Interface Source |
|-----|------|------|------|
| `is_fp8_available` | Function | Queries whether the device/environment supports FP8. `return_reason=True` returns the reason | Native interface |
| `is_mxfp8_available` | Function | Queries MXFP8 support | Native interface |
| `is_mxfp4_available` | Function | Queries MXFP4 support | Extension interface |
| `is_fp8_block_scaling_available` | Function | Queries 128×128 block-level scaling support | Native interface |
| `is_w4a8_available` | Function | Queries W4A8 weight quantization support | Extension interface |
| `is_hif8_delayed_available` | Function | Queries HIF8 delayed scaling support | Extension interface |
| `is_nvfp4_available` | Function | Queries NVFP4 support (**TENPU does not currently support it and always returns False**) | Native interface |
| `get_default_recipe` | Function | Returns the default quantization strategy recommended for the current device | Native interface |

**Table 7** Quantization strategy (Recipe) list

Import from **transformer_engine.common.recipe**:

| API | Type | Description | Interface Source |
|-----|------|------|------|
| `Recipe` | Class | Strategy base class | Native interface |
| `Format` / `FormatEnum` | Class/Enum | Quantized data format definition (`kFloat8E4M3`, `kFloat8E5M2`, `kMXFP8`, etc.) | Native interface |
| `DelayedScaling` | Class | Delayed scaling: maintains amax history (`amax_history_len`) to compute scale | Native interface |
| `Float8CurrentScaling` | Class | Current scaling (per-tensor current scaling) | Native interface |
| `Float8BlockScaling` | Class | 128×128 block-level FP8 scaling | Native interface |
| `MXFP8BlockScaling` | Class | OCP MX standard 1×32 block MXFP8 scaling | Native interface |
| `MXFP832x32BlockScaling` | Class | 32×32 block MXFP8 scaling | Extension interface |
| `MXFP4BlockScaling` | Class | MXFP4 block-level scaling (the corresponding NVTE capability is NVFP4) | Extension interface |
| `W4A8BlockScaling` | Class | Weight INT4/activation INT8 block-level scaling | Extension interface |
| `HIF8DelayedScaling` | Class | HIF8 delayed scaling (Ascend FP8 format) | Extension interface |
| `CustomRecipe` | Class | Custom strategy that dynamically generates per-role strategies according to QuantizerRole | Native interface |
| `QuantizerRole` | Class | Quantization role (weight/input/output/gradient) definition, used for CustomRecipe | Native interface |

The accompanying RecipeState classes (`transformer_engine.pytorch.quantization.state`, such as `DelayedScalingRecipeState`) are strategy runtime states used internally by the framework and generally do not need to be used directly.

**Table 8** Quantized tensor type list

Import from **transformer_engine.pytorch.tensor**:

| API | Type | Description | Interface Source |
|-----|------|------|------|
| `QuantizedTensor` | Class | Quantized tensor base class | Native interface |
| `Float8Tensor` | Class | FP8 tensor with per-tensor scaling (also importable at the top level) | Native interface |
| `Float8BlockwiseQTensor` | Class | 128×128 block-level scaling FP8 tensor | Native interface |
| `MXFP8Tensor` | Class | MXFP8 scaled tensor | Native interface |
| `MXFP4Tensor` | Class | MXFP4 scaled tensor | Extension interface |
| `W4A8WeightTensor` | Class | W4A8 weight tensor | Extension interface |

**Table 9** Quantizer list

Import from **transformer_engine.pytorch.tensor**:

| API | Type | Description | Interface Source |
|-----|------|------|------|
| `Quantizer` | Class | Quantizer base class | Native interface |
| `Float8Quantizer` | Class | Delayed scaling FP8 quantizer | Native interface |
| `Float8CurrentScalingQuantizer` | Class | Current scaling FP8 quantizer | Native interface |
| `Float8BlockQuantizer` | Class | Block-level scaling FP8 quantizer | Native interface |
| `MXFP8Quantizer` | Class | MXFP8 quantizer (1×32) | Native interface |
| `MXFP832x32Quantizer` | Class | 32×32 block MXFP8 quantizer | Extension interface |
| `MXFP4Quantizer` | Class | MXFP4 quantizer | Extension interface |
| `W4A8WeightQuantizer` | Class | W4A8 weight quantizer | Extension interface |
| `HIF8Quantizer` | Class | HIF8 quantizer | Extension interface |

**Table 10** Tensor utility function list

Import from **transformer_engine.pytorch.tensor**:

| API | Type | Description | Interface Source |
|-----|------|------|------|
| `cast_master_weights_to_fp8` | Function | Convert master weights to FP8 storage | Native interface |
| `post_all_gather_processing` | Function | Quantization post-processing after AllGather | Native interface |
| `replace_raw_data` | Function | Replace the underlying data of a quantized tensor | Native interface |

## MoE Support

**Table 11** Token permutation interface list

Import from **transformer_engine.pytorch**:

| API | Type | Description | Interface Source |
|-----|------|------|------|
| `moe_permute` | Function | Rearranges tokens according to the token→expert mapping (with optional probs multiplication and padding) | Native interface |
| `moe_unpermute` | Function | The inverse operation of `moe_permute`, restoring the original token order | Native interface |
| `moe_sort_chunks_by_index` | Function | Sorts chunks by index | Native interface |
| `moe_sort_chunks_by_index_with_probs` | Function | Sorts chunks by index and probability | Native interface |
| `moe_permute_with_probs` | Function | Token permutation with expert probs | Native interface |
| `moe_permute_and_pad_with_probs` | Function | Token permutation with probs and padding | Native interface |

**Table 12** MoE routing interface list

Import from **transformer_engine.pytorch.router**:

| API | Type | Description | Interface Source |
|-----|------|------|------|
| `fused_topk_with_score_function` | Function | Fused top-k routing (with a scoring function) | Native interface |
| `fused_compute_score_for_moe_aux_loss` | Function | Fused computation of the MoE auxiliary loss score | Native interface |
| `fused_moe_aux_loss` | Function | Fused computation of the MoE auxiliary loss | Native interface |

## Optimizer and Multi-Tensor Operators

**Table 13** Multi-tensor operator list

Import from **transformer_engine.pytorch.optimizers** (`MultiTensorApply` and others are also exported at the top level):

| API | Type | Description | Interface Source |
|-----|------|------|------|
| `MultiTensorApply` | Class | Executor wrapper for multi-tensor fused operators | Native interface |
| `multi_tensor_applier` | Function | Dispatches execution by operator type | Native interface |
| `multi_tensor_scale` | Function | Multi-tensor scaling | Native interface |
| `multi_tensor_scale_tensor` | Function | Multi-tensor scaling (tensor-level interface) | Native interface |
| `multi_tensor_l2norm` | Function | Computes the L2 norm of multiple tensors | Native interface |
| `multi_tensor_unscale_l2norm` | Function | Computes the L2 norm of multiple tensors and unscales them | Native interface |
| `multi_tensor_compute_scale_and_scale_inv` | Function | Computes the FP8 scale and its inverse | Native interface |
| `multi_tensor_compute_scale_inv_e8m0` | Function | Computes the inverse scale in E8M0 format | Native interface |

## Training Parallelism and Memory Optimization

**Table 14** CPU Offload interface list

Import from **transformer_engine.pytorch**:

| API | Type | Description | Interface Source |
|-----|------|------|------|
| `get_cpu_offload_context` | Function | Creates a CPU Offload context, supporting layer-by-layer offloading of activations/weights | Native interface |
| `mark_not_offload` | Function | Marks a tensor as not participating in offload | Native interface |
| `start_offload` | Function | Immediately starts offloading the specified tensor | Native interface |
| `is_cpu_offload_enabled` | Function | Queries whether CPU Offload is enabled | Native interface |

**Table 15** Distributed and recomputation interface list

Import from **transformer_engine.pytorch.distributed** (`checkpoint` can also be imported from the top-level **transformer_engine.pytorch**):

| API | Type | Description | Interface Source |
|-----|------|------|------|
| `checkpoint` | Function | Activation recomputation wrapper (based on `torch.utils.checkpoint`), aware of the quantization state in FP8 scenarios | Native interface |
| `activation_recompute_forward` | Class (context manager) | Marks the forward pass as being in the recomputation phase | Native interface |
| `get_activation_recompute_contexts` | Function | Obtains the activation recomputation contexts | Native interface |
| `is_fp8_activation_recompute_enabled` | Function | Queries whether FP8 activation recomputation is enabled | Native interface |
| `in_fp8_activation_recompute_phase` | Function | Queries whether the current phase is the FP8 recomputation phase | Native interface |
| `get_te_classes` / `has_te_modules` | Function | Enumerates/detects TENPU modules in the network (for checkpoint compatibility handling) | Native interface |
| `set_tensor_model_parallel_attributes` | Function | Sets TP attributes for a tensor | Native interface |
| `get_distributed_world_size` / `get_distributed_rank` | Function | Obtains the world size / rank (of the specified communication group) | Native interface |
| `get_all_rng_states` / `set_all_rng_states` | Function | Obtains/restores all RNG states | Native interface |

## Numerical Debugging

**Table 16** Numerical debugging interface list

| API | Import Path | Type | Description | Interface Source |
|-----|----------|------|------|------|
| `TEDebugState` | `transformer_engine.debug.pytorch.debug_state` | Class | Debug state container | Native interface |
| `set_weight_tensor_tp_group_reduce` | `transformer_engine.debug.pytorch.debug_state` | Function | Controls the TP group reduce behavior of weight tensors | Extension interface |
| `DebugQuantizer` / `GroupedDebugQuantizer` | `transformer_engine.debug.pytorch.debug_quantization` | Class | Debug quantizer that wraps observation logic outside the quantizer, used for online numerical observation | Native interface |
| `DebugQuantizedTensor` / `GroupedDebugQuantizedTensor` | `transformer_engine.debug.pytorch.debug_quantization` | Class | Quantized tensor for debugging | Native interface |
| `GemmComparisonConfig` | `transformer_engine.pytorch` | Class | GEMM numerical comparison configuration | Extension interface |
| `configure_gemm_comparison` | `transformer_engine.pytorch` | Function | Enables numerical comparison between the quantized GEMM and the reference implementation at runtime | Extension interface |
| `get_gemm_comparison_config` | `transformer_engine.pytorch` | Function | Obtains the current GEMM comparison configuration | Extension interface |

## Environment Variables

**Table 17** Environment variable list

| Environment Variable | Description | Interface Source |
|----------|------|------|
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | When set to `0`, enables deterministic algorithms to ensure reproducible results (at the cost of some performance) | Native interface |
| `NVTE_TORCH_COMPILE` | When set to `0`, disables the `torch.compile` path for TENPU operators | Native interface |
| `NVTE_BACKWARD_OVERRIDE` | Enables backward numerical override (for precision debugging), with values `high_precision`/`dequantized`; not yet implemented for the `LayerNormMLP` scenario | Native interface |

> [!NOTE]
>
> The following NVTE environment variables are parsed and retained in TENPU only for interface compatibility, and have no actual effect after being set:
>
> - `NVTE_FWD_LAYERNORM_SM_MARGIN` / `NVTE_BWD_LAYERNORM_SM_MARGIN` / `NVTE_INF_LAYERNORM_SM_MARGIN`: SM reservation tuning for CUDA kernel scheduling, and do not participate in NPU kernel scheduling;
> - `NVTE_CPU_OFFLOAD_V1`: used to switch the CPU Offload v1 code path of NVTE, which is not implemented in TENPU.

## Differences from NVTE

### Interfaces Not Provided by TENPU

The following NVTE interfaces are **not yet provided** in TENPU. Service code that depends on these interfaces needs to be adapted:

- `MultiheadAttention`, `TransformerLayer` (high-level wrapper modules, which can be composed using `DotProductAttention` plus basic modules);
- `CudaRNGStatesTracker` (CUDA RNG management);
- `parallel_cross_entropy`, `interleave_glu_tensor` / `deinterleave_glu_tensor`;
- The `Linear`, `GroupedLinear`, `BasicLinear`, and `Dropout` operators at the ops layer, as well as the `register_forward_backward_fusion` registration interface;
- JAX/ONNX-related interfaces (`export`, `onnx_extensions`).

### Ascend Extensions of TENPU

- HIF8 delayed scaling (`HIF8DelayedScaling`, `HIF8Quantizer`, `reset_hif8_amax_history`);
- W4A8 weight quantization (`W4A8BlockScaling`, `W4A8WeightTensor`, `W4A8WeightQuantizer`);
- MXFP4 block-level scaling (`MXFP4BlockScaling`, `MXFP4Tensor`, `MXFP4Quantizer`; the corresponding NVTE capability is NVFP4);
- `MXFP832x32BlockScaling` (32×32 block MXFP8);
- Swap context-parallel attention strategy, GEMM numerical comparison tool, and so on (see the extension interfaces above).

### Explicitly Unsupported

- NVFP4 low-precision quantization (`is_nvfp4_available()` always returns `False`).
