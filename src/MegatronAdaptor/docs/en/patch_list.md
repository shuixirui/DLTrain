# Adaptor Interface List

This document summarizes, grouped by feature, all the adaptation changes (patches) that MegatronAdaptor makes to Megatron-LM, and item by item describes the patch target, the implementation location in the MA repository, and the purpose of each change. It is the checklist of non-intrusive runtime monkey-patch adaptations to upstream.

## Adaptation Mechanism

MegatronAdaptor **does not modify any source file in upstream repositories such as Megatron-LM**. All adaptations are implemented through runtime monkey-patching:

- **Registration timing**: When `import megatron_adaptor` runs, `FeaturesManager` calls `pre_register_patches` (before upstream modules are imported, to replace low-level interfaces) and `register_patches` (after upstream modules are imported, to replace service functions) in two phases by feature. Then `PatchesManager` writes the replacement functions into the namespace of the target module in a unified manner.
- **Patch modes**:
   - **Override**: When the target symbol already exists, replace it directly with the MA implementation (for example, `torch.compile` → a pass-through implementation).
   - **`create_dummy`**: Create the target symbol only when it **does not** exist (for example, provide a pure PyTorch fallback operator when FLA is not installed). When the real implementation is installed, keep it as is.
- **Feature gating**: Patches are grouped by feature, and whether a patch takes effect is determined by `--optimization-level` (default 2) together with feature switch parameters:
   - Basic features at `optimization_level=0` (requirements-basic, megatron-basic, transformer-engine-basic, profiler-default) **always take effect**.
   - Functional features (profile, npu-datadump, npu-deterministic) must satisfy both `--optimization-level ≥ 2` and the corresponding switch (`--profile`, `--npu-datadump`, `--npu-deterministic`).
- **Removable**: `FeaturesManager.remove_patches()` can restore all patches (for reuse by layered installation and removal in MindSpeed).

## Summary of Adaptations

**Table 1** Basic runtime environment adaptations

| Target Symbol | Replacement Implementation (MA Source Location) | Purpose of the Change |
|---|---|---|
| `torch.compile` | `dummy_compile` (`patches/requirements/requirements.py`) | NPU does not support CUDA graph capture-based compilation. Replace it with a pass-through to avoid errors on the upstream JIT capture path. |
| `torch.jit.script` | `dummy_compile` (`patches/requirements/requirements.py`) | Same as the preceding row. `torch.jit.script` is also passed through. |
| `torch.nn.parameter.Parameter.type` | `type_wrapper` (`patches/requirements/requirements.py`) | Replace the device name `npu` in the return value with `cuda`, keeping the upstream logic that branches on CUDA naming usable. |
| `torch.Tensor.type` | `type_wrapper` (`patches/requirements/requirements.py`) | Same as the preceding row. |
| `torch.Tensor.view` | `ensure_contiguous_wrapper` (`patches/requirements/requirements.py`) | Call `contiguous()` on a non-contiguous tensor before `view`, avoiding the restriction that `view` places on non-contiguous tensors on NPU. |
| `torch.distributed.all_reduce` | `torch_all_reduce_double_dtype_bypass_wrapper` (`patches/requirements/requirements.py`) | HCCL does not support fp64 (double) communication. A double tensor is automatically converted to fp32 for synchronous execution and then converted back. |
| `torch._C._jit_set_nvfuser_enabled` | `dummy_function` (`patches/requirements/requirements.py`) | NPU has no nvfuser. Therefore, this patch is a no-op. |
| `math.lcm` (created by default) | `lcm` (`patches/requirements/requirements.py`) | Provide a fallback implementation for older Python versions that lack `math.lcm`. |
| `importlib.metadata.version` | `version_wrapper` (`patches/requirements/requirements.py`) | Return a fixed value when upstream queries the version by the `transformer-engine` package name, avoiding a detection failure caused by TE-NPU lacking this dist metadata. |
| `transformer_engine_torch.thd_get_partitioned_indices` (created by default) | `npu_thd_get_partitioned_indices` (`patches/requirements/requirements.py`) | Provide a pure PyTorch implementation of the sharding index for THD variable-length sequence context parallelism (head/tail zigzag chunking scheme). |
| `fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32` (created by default) | `npu_wgrad_gemm_accum_fp32` (`patches/transformer_engine/transformer_engine.py`) | NPU has no CUDA fused GEMM accumulation operator. Therefore, the patch forwards to the corresponding implementation in TransformerEngineNPU. |
| `fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16` (created by default) | `npu_wgrad_gemm_accum_fp16` (`patches/transformer_engine/transformer_engine.py`) | Same as the preceding row (the fp16 version). |

**Table 2** Basic Megatron adaptations

| Target Symbol  | Replacement Implementation (MA Source Location) | Purpose of the Change |
|---|---|---|
| `megatron.core.transformer.transformer_config.TransformerConfig.__init__` | `transformer_config_init_wrapper` (`patches/megatron/transformer_config.py`) | Dynamically extend dataclass fields to accept the additional CLI parameters registered by MA features. |
| `megatron.core.transformer.transformer_config.TransformerConfig.__init_subclass__` | `transformer_config_init_subclass` (`patches/megatron/transformer_config.py`) | Same as the preceding row, applied to subclasses of TransformerConfig. |
| `megatron.core.transformer.transformer_config.TransformerConfig.__post_init__` | `transformer_config_post_init_wrapper` (`patches/megatron/transformer_config.py`) | Attach the pre-validation and post-validation pipeline of MA feature parameters before and after configuration initialization. |
| `megatron.core.transformer.transformer_config.MLATransformerConfig.__init__` | `transformer_config_init_wrapper` (`patches/megatron/transformer_config.py`) | Same as `TransformerConfig.__init__`, applied to the MLA configuration. |
| `megatron.core.tensor_parallel.random._set_cuda_rng_state` | `_set_cuda_rng_state` (`patches/megatron/megatron_basic.py`) | Route random number state saving and restoring through the NPU generator interface, which is compatible with TP random state synchronization. |
| `megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax` | `ScaledUpperTriangMaskedSoftmax` (`patches/megatron/megatron_basic.py`) | Replace the attention softmax with the NPU operator `npu_scaled_masked_softmax` (upper triangular mask). |
| `megatron.core.fusions.fused_softmax.ScaledMaskedSoftmax` | `ScaledMaskedSoftmax` (`patches/megatron/megatron_basic.py`) | Same as the preceding row (arbitrary mask). |
| `megatron.core.fusions.fused_softmax.ScaledSoftmax` | `ScaledSoftmax` (`patches/megatron/megatron_basic.py`) | Same as the preceding row (no mask. Therefore, an empty mask is used to invoke the NPU operator). |
| `megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available` | `is_kernel_available` (`patches/megatron/megatron_basic.py`) | Re-evaluate fused kernel availability according to NPU operator constraints (the sk range, divisibility of sq/sk by 16, and so on). |
| `megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax` | `forward_fused_softmax` (`patches/megatron/megatron_basic.py`) | Dispatch the forward pass to the NPU scaled masked softmax operator (distinguishing causal masks from padding masks). |
| `megatron.core.dist_checkpointing.strategies.filesystem_async.FileSystemWriterAsync.preload_tensors` | `preload_tensors` (`patches/megatron/megatron_basic.py`) | Change asynchronous checkpoint staging to a synchronous D2H copy plus a CPU tensor clone, preventing a subsequent training step from modifying the staged data in place and corrupting it. |
| `fla.modules.convolution.causal_conv1d` (created by default) | `_npu_causal_conv1d` (`patches/megatron/gdn_packed_seq.py`) | Provide a pure PyTorch causal conv1d fallback for GatedDeltaNet when FLA is not installed (supporting segmented convolution over packed sequences). |
| `fla.ops.gated_delta_rule.chunk_gated_delta_rule` (created by default) | `_npu_chunk_gated_delta_rule` (`patches/megatron/gdn_packed_seq.py`) | Same as the preceding row, forwarding to the pure PyTorch `torch_chunk_gated_delta_rule` built into Megatron-LM. |
| `fla.modules.l2norm.l2norm` (created by default) | `_npu_l2norm` (`patches/megatron/gdn_packed_seq.py`) | Same as the preceding row, implemented with `F.normalize(p=2)` (compute in fp32 and convert back to the original dtype). |
| `megatron.training.arguments.parse_args` | `parse_args_wrapper` (`patches/megatron/arguments_basic.py`) | Inject the MA feature argument parser so that MA parameters such as `--optimization-level` are recognized by the training entry point. |
| `megatron.training.arguments.validate_args` | `validate_args_wrapper` (`patches/megatron/arguments_basic.py`) | Attach pre-validation and post-validation of MA feature parameters before and after upstream argument validation. |
| `megatron.training.yaml_arguments.validate_yaml` | `validate_args_wrapper` (`patches/megatron/arguments_basic.py`) | Same as the preceding row, applied to the YAML configuration entry point. |
| `megatron.training.initialize._compile_dependencies` | `_compile_dependencies` (`patches/megatron/megatron_basic.py`) | Compilation logic for the dataset index builder (a Cython helper), executed on rank 0 only. |
| `megatron.training.dist_signal_handler.get_device` | `get_device_wrapper` (`patches/megatron/megatron_basic.py`) | Construct an NPU device by `local_rank` under the HCCL backend, so that distributed signal handling can locate the device. |
| `megatron.training.utils.get_device_arch_version` | `get_device_arch_version` (`patches/megatron/megatron_basic.py`) | Return a fixed device architecture version number, making upstream logic that branches on the architecture number usable on NPU. |
| `torch.cuda.get_device_capability` | `get_device_capability` (`patches/megatron/megatron_basic.py`) | Return a fixed capability value so that upstream can select an implementation path by GPU capability. |
| `torch.cuda.nvtx.range_push` | `_range_push_patch` (`patches/megatron/megatron_basic.py`) | Forward NVTX ranges to `torch_npu.npu.mstx.range_start`, keeping training loop performance markers visible in msprof. |
| `torch.cuda.nvtx.range_pop` | `_range_pop_patch` (`patches/megatron/megatron_basic.py`) | Forward to `torch_npu.npu.mstx.range_end` (matches the thread-local stack and reports a warning when it is unbalanced). |

**Table 3** TransformerEngine adaptations

| Target Symbol | Replacement Implementation (MA Source Location) | Purpose of the Change |
|---|---|---|
| `megatron.core.transformer.transformer_config.TransformerConfig.__post_init__` | `transformer_config_post_init_wrapper` (`patches/transformer_engine/transformer_engine.py`) | In the data dump scenario, msprobe wraps `F.gelu`/`F.silu`. This patch rebinds the stale default values left in the configuration to the currently wrapped equivalent functions, so that the upstream identity check passes. |
| `megatron.core.tensor_parallel.layers._grad_accum_fusion_available` | `get_npu_grad_accum_fusion_available` (`patches/transformer_engine/transformer_engine.py`) | Re-evaluate weight gradient accumulation fusion availability as "available on NPU", so that the fused GEMM path of TE-NPU takes effect. |
| `megatron.core.tensor_parallel.layers.ColumnParallelLinear.__init__` | `column_parallel_linear_init_wrapper` (`patches/transformer_engine/transformer_engine.py`) | When `--gradient-accumulation-fusion` is enabled but the NPU fused operator is unavailable, provide an actionable error message (check CANN or disable this option). |

**Table 4** NPU profiling

| Target Symbol  | Replacement Implementation (MA Source Location) | Purpose of the Change |
|---|---|---|
| `megatron.training.training.train` | `train_wrapper` (`patches/megatron/profile.py`) | Build a `torch_npu.profiler` context around the training loop according to the `--profile` series of parameters (collect by step range, filter by rank, and capture NPU/CPU activities and the AIC PipeUtilization metric). |
| `megatron.training.training.train_step` | `train_step_wrapper` (`patches/megatron/profile.py`) | Advance the profiler step at each training step, aligning the collection range with the training steps. |

**Table 5** NPU data dump precision comparison

| Target Symbol  | Replacement Implementation (MA Source Location) | Purpose of the Change |
|---|---|---|
| `megatron.training.training.train_step` | `dump_start_wrapper` (`patches/megatron/npu_datadump/npu_datadump.py`) | Start the msprobe `PrecisionDebugger` at the beginning of the training step (collect tensors and statistics as configured in `config.json`). |
| `megatron.training.ft_integration.on_training_step_end` | `dump_end_wrapper` (`patches/megatron/npu_datadump/npu_datadump.py`) | Stop collection and advance the step at the end of the training step, completing a per-step dump. |

>[!NOTE]
>
> NPU profiling and data dump both act on `train_step`. The two are chained in feature order (dump on the outer layer, profile on a layer further outside), and they do not affect each other.

**Table 6** NPU deterministic computation

| Target Symbol | Replacement Implementation (MA Source Location) | Purpose of the Change |
|---|---|---|
| `megatron.training.initialize._set_random_seed` | `npu_deterministic_wrapper` (`patches/megatron/npu_deterministic.py`) | Append NPU deterministic switches on top of upstream seed initialization: `HCCL_DETERMINISTIC`, `CLOSE_MATMUL_K_SHIFT`, `PYTHONHASHSEED`, `torch.use_deterministic_algorithms`, and the NPU seed setting. |

**Table 7** msprof communication group annotation

| Target Symbol | Replacement Implementation (MA Source Location) | Purpose of the Change |
|---|---|---|
| `megatron.core.parallel_state.get_nccl_options` | `get_nccl_options_add_group_info_wrapper` (`patches/megatron/parallel_state.py`) | Inject `group_name` into the HCCL communication configuration of each parallel group (dp/tp/pp/ep and so on), so that msprof trace can distinguish communication domains. |

**Table 8** Compatibility switch for DDP side-stream memory retention

| Target Symbol | Replacement Implementation (MA Source Location) | Purpose of the Change |
|---|---|---|
| `megatron.training.training.get_model` | `get_model_with_upstream_stream_selection` (`patches/megatron/ddp_one_shot_stream.py`) | Equivalent replacement of `get_model`: as in later versions of Megatron, apply three-branch stream selection in place in the DDP/FSDP wrapping section (`eager` switches to the current stream to eliminate one-shot side-stream retention, `full_iteration` switches to the shared capture stream, and the remaining cases keep the dedicated stream). |
| `megatron.training.models.dist_utils._ddp_wrap` | `ddp_wrap_with_upstream_stream_selection` (`patches/megatron/ddp_one_shot_stream.py`) | Equivalent replacement of `_ddp_wrap`: the stream selection block matches later versions of Megatron (`eager` switches to the current stream, that is, `ddp_stream == current_stream`, and no longer creates a one-shot side stream, eliminating the memory retention caused by CachingAllocator). |

**Table 9** Eliminating resident dummy wgrad memory under non-overlap gradient reduction

| Target Symbol | Replacement Implementation (MA Source Location) | Purpose of the Change |
|---|---|---|
| `megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__` | `ddp_init_dummy_wgrad_wrapper` (`patches/megatron/dummy_wgrad_non_overlap.py`) | Tag `params_with_grad` with `requires_dummy_wgrad` (whose value equals `ddp_config.overlap_grad_reduce`) and install a process-level gate. This tag is also consumed on the TransformerEngineNPU side. |
| `megatron.core.tensor_parallel.layers.get_dummy_wgrad` | `get_dummy_wgrad_skip_gate_wrapper` (`patches/megatron/dummy_wgrad_non_overlap.py`) | In the non-overlap case, it no longer requests a full-size dummy wgrad (including the resident shape cache of TransformerEngineNPU `_dummy_wgrads`), eliminating that block of resident memory. In the overlap case, it passes through as is. The switch is `--fix-non-overlap-dummy-wgrad` and is disabled by default. |

> **Note**: The paths where TransformerEngine allocates the dummy wgrad by itself (GroupedLinear/PerformanceGroupedLinear/ops basic_linear/the FSDP branch) are not the responsibility of MA. TransformerEngineNPU benefits as well only after the same gate is added on its side in `setup_dummy_wgrad` (reading `requires_dummy_wgrad`).
