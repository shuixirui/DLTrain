# 适配接口列表

本文档按特性分组汇总 MegatronAdaptor 对 Megatron-LM 的全部适配修改点（patch），逐项说明补丁目标、MA 仓内实现位置与修改目的，是对上游零侵入的运行时 monkey-patch 适配清单。

## 适配机制说明

MegatronAdaptor **不修改 Megatron-LM 等上游仓库的任何源码文件**。所有适配通过运行时 monkey-patch 实现：

- **注册时机**：`import megatron_adaptor` 时，`FeaturesManager` 按特性（feature）分两阶段调用 `pre_register_patches`（上游模块导入前，用于替换底层接口）与 `register_patches`（上游模块导入后，用于替换业务函数），再由 `PatchesManager` 统一将替换函数写入目标模块的命名空间。
- **补丁模式**：
   - **覆盖式**：目标符号已存在时直接替换为 MA 实现（例如 `torch.compile` → 直通实现）。
   - **`create_dummy`**：仅当目标符号**不存在**时才创建（例如 FLA 未安装时提供纯 PyTorch 兜底算子）；已安装真实实现时保持原样。
- **特性门控**：补丁按特性分组，是否生效由 `--optimization-level`（默认 2）与特性开关参数共同决定：
   - `optimization_level=0` 的基础特性（requirements-basic、megatron-basic、transformer-engine-basic、mla-rope、profiler-default）**始终生效**；
   - 功能特性（profile、npu-datadump、npu-deterministic）需同时满足 `--optimization-level ≥ 2` 且传入对应开关（`--profile`、`--npu-datadump`、`--npu-deterministic`）。
- **可移除**：`FeaturesManager.remove_patches()` 可还原全部补丁（供 MindSpeed 分层装卸复用）。

## 适配修改一览

**表 1** 基础运行环境适配

| 目标符号 | 替换实现（MA 源码位置） | 修改目的 |
|---|---|---|
| `torch.compile` | `dummy_compile`（patches/requirements/requirements.py） | NPU 不支持 CUDA 图捕获式编译，替换为原样透传，避免上游 JIT 捕获路径报错 |
| `torch.jit.script` | `dummy_compile`（patches/requirements/requirements.py） | 同上，`torch.jit.script` 同样透传 |
| `torch.nn.parameter.Parameter.type` | `type_wrapper`（patches/requirements/requirements.py） | 返回值中的设备名 `npu` 替换为 `cuda`，保持上游按 CUDA 命名的判断逻辑可用 |
| `torch.Tensor.type` | `type_wrapper`（patches/requirements/requirements.py） | 同上 |
| `torch.Tensor.view` | `ensure_contiguous_wrapper`（patches/requirements/requirements.py） | 非连续张量先 `contiguous()` 再执行 `view`，规避 NPU 上 view 对非连续张量的限制 |
| `torch.distributed.all_reduce` | `torch_all_reduce_double_dtype_bypass_wrapper`（patches/requirements/requirements.py） | HCCL 不支持 fp64（double）通信：double 张量自动转 fp32 同步执行后转回 |
| `torch._C._jit_set_nvfuser_enabled` | `dummy_function`（patches/requirements/requirements.py） | NPU 无 nvfuser，置为无操作 |
| `math.lcm`（缺省创建） | `lcm`（patches/requirements/requirements.py） | 旧版本 Python 无 `math.lcm` 时提供兜底实现 |
| `importlib.metadata.version` | `version_wrapper`（patches/requirements/requirements.py） | 上游按 `transformer-engine` 包名查询版本时返回固定值，规避 TE-NPU 无该 dist 元数据导致的探测失败 |
| `fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32`（缺省创建） | `npu_wgrad_gemm_accum_fp32`（patches/transformer_engine/transformer_engine.py） | NPU 无 CUDA 融合 GEMM 累加算子，转发到 TransformerEngineNPU 的对应实现 |
| `fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16`（缺省创建） | `npu_wgrad_gemm_accum_fp16`（patches/transformer_engine/transformer_engine.py） | 同上（fp16 版本） |
| `torch.cuda.device_memory_used` | `device_memory_used`（patches/megatron/device_probe_compat.py） | 该接口在 torch 2.10 走 pynvml（仅 NVIDIA），torch_npu 未覆盖，NPU 上调用报 `nvidia-ml-py does not seem to be installed`；用 `torch.npu.mem_get_info()` 的 `total - free` 提供 torch/CANN 可见内存池的精确使用量（`total` 不含驱动保留 HBM，增量/趋势与 npu-smi 一致，数值上不等同于 npu-smi 结果），开关打开时在 patch 注册阶段告警口径；每次打印实时查询；运行中失败返回 `nan` 并告警一次，不中断训练 |

**表 2** Megatron 基础适配

| 目标符号  | 替换实现（MA 源码位置） | 修改目的 |
|---|---|---|
| `megatron.core.transformer.transformer_config.TransformerConfig.__init__` | `transformer_config_init_wrapper`（patches/megatron/transformer_config.py） | 动态扩展 dataclass 字段，承接 MA 特性注册的额外命令行参数 |
| `megatron.core.transformer.transformer_config.TransformerConfig.__init_subclass__` | `transformer_config_init_subclass`（patches/megatron/transformer_config.py） | 同上，作用于 TransformerConfig 的子类 |
| `megatron.core.transformer.transformer_config.TransformerConfig.__post_init__` | `transformer_config_post_init_wrapper`（patches/megatron/transformer_config.py） | 在配置初始化前后挂接 MA 特性参数的预校验/后校验流水线 |
| `megatron.core.transformer.transformer_config.MLATransformerConfig.__init__` | `transformer_config_init_wrapper`（patches/megatron/transformer_config.py） | 同 `TransformerConfig.__init__`，作用于 MLA 配置 |
| `megatron.core.tensor_parallel.random._set_cuda_rng_state` | `_set_cuda_rng_state`（patches/megatron/megatron_basic.py） | 随机数状态保存/恢复改走 NPU 生成器接口，兼容 TP 随机状态同步 |
| `megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax` | `ScaledUpperTriangMaskedSoftmax`（patches/megatron/megatron_basic.py） | 注意力 softmax 替换为 NPU 算子 `npu_scaled_masked_softmax`（上三角掩码） |
| `megatron.core.fusions.fused_softmax.ScaledMaskedSoftmax` | `ScaledMaskedSoftmax`（patches/megatron/megatron_basic.py） | 同上（任意掩码） |
| `megatron.core.fusions.fused_softmax.ScaledSoftmax` | `ScaledSoftmax`（patches/megatron/megatron_basic.py） | 同上（无掩码，用空掩码走 NPU 算子） |
| `megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available` | `is_kernel_available`（patches/megatron/megatron_basic.py） | 按 NPU 算子约束重判融合 kernel 可用性（sk 范围、sq/sk 整除 16 等） |
| `megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax` | `forward_fused_softmax`（patches/megatron/megatron_basic.py） | 前向分发到 NPU scaled masked softmax 算子（区分 causal/padding mask） |
| `megatron.core.dist_checkpointing.strategies.filesystem_async.FileSystemWriterAsync.preload_tensors` | `preload_tensors`（patches/megatron/megatron_basic.py） | 异步 checkpoint 暂存改为同步 D2H 拷贝 + CPU 张量 clone，防止后续训练步原地修改损坏暂存数据 |
| `fla.modules.convolution.causal_conv1d`（缺省创建） | `_npu_causal_conv1d`（patches/megatron/gdn_packed_seq.py） | FLA 未安装时为 GatedDeltaNet 提供纯 PyTorch causal conv1d 兜底（支持 packed 序列按段卷积） |
| `fla.ops.gated_delta_rule.chunk_gated_delta_rule`（缺省创建） | `_npu_chunk_gated_delta_rule`（patches/megatron/gdn_packed_seq.py） | 同上，转发到 Megatron-LM 内置的纯 PyTorch `torch_chunk_gated_delta_rule` |
| `fla.modules.l2norm.l2norm`（缺省创建） | `_npu_l2norm`（patches/megatron/gdn_packed_seq.py） | 同上，`F.normalize(p=2)` 实现（fp32 计算后转回原 dtype） |
| `megatron.training.arguments.parse_args` | `parse_args_wrapper`（patches/megatron/arguments_basic.py） | 注入 MA 特性参数解析器，使 `--optimization-level` 等 MA 参数可被训练入口识别 |
| `megatron.training.arguments.validate_args` | `validate_args_wrapper`（patches/megatron/arguments_basic.py） | 在上游参数校验前后挂接 MA 特性参数的预/后校验 |
| `megatron.training.yaml_arguments.validate_yaml` | `validate_args_wrapper`（patches/megatron/arguments_basic.py） | 同上，作用于 YAML 配置入口 |
| `megatron.training.initialize._compile_dependencies` | `_compile_dependencies`（patches/megatron/megatron_basic.py） | 数据集索引构建器（Cython helper）的编译逻辑，仅 rank 0 执行 |
| `megatron.training.dist_signal_handler.get_device` | `get_device_wrapper`（patches/megatron/megatron_basic.py） | HCCL 后端下按 local_rank 构造 NPU 设备，供分布式信号处理定位设备 |
| `megatron.training.utils.get_device_arch_version` | `get_device_arch_version`（patches/megatron/megatron_basic.py） | 返回固定设备架构版本号，使上游按架构号的分支逻辑在 NPU 上可用 |
| `torch.cuda.get_device_capability` | `get_device_capability`（patches/megatron/megatron_basic.py） | 返回固定 capability 值，供上游按 GPU capability 选择实现路径 |
| `torch.cuda.nvtx.range_push` | `_range_push_patch`（patches/megatron/megatron_basic.py） | NVTX 区间转发到 `torch_npu.npu.mstx.range_start`，保持训练循环性能标记在 msprof 中可见 |
| `torch.cuda.nvtx.range_pop` | `_range_pop_patch`（patches/megatron/megatron_basic.py） | 转发到 `torch_npu.npu.mstx.range_end`（线程局部栈匹配，失衡时告警） |

**表 3** TransformerEngine适配

| 目标符号 | 替换实现（MA 源码位置） | 修改目的 |
|---|---|---|
| `megatron.core.transformer.transformer_config.TransformerConfig.__post_init__` | `transformer_config_post_init_wrapper`（patches/transformer_engine/transformer_engine.py） | 数据 dump 场景下 msprobe 会包装 `F.gelu`/`F.silu`，此处将配置中残留的旧默认值重绑定到当前包装后的等价函数，使上游恒等校验通过 |
| `megatron.core.tensor_parallel.layers._grad_accum_fusion_available` | `get_npu_grad_accum_fusion_available`（patches/transformer_engine/transformer_engine.py） | 权重梯度累加融合可用性改判为"NPU 可用"，使 TE-NPU 的融合 GEMM 路径生效 |
| `megatron.core.tensor_parallel.layers.ColumnParallelLinear.__init__` | `column_parallel_linear_init_wrapper`（patches/transformer_engine/transformer_engine.py） | 开启 `--gradient-accumulation-fusion` 但 NPU 融合算子不可用时，给出可操作的报错提示（检查 CANN 或关闭该选项） |
| `megatron.core.safe_globals.SafeUnpickler.find_class` | `safe_unpickler_find_class_patch`（patches/transformer_engine/transformer_engine.py） | 上游反序列化白名单按 NVIDIA TE 的模块布局编写，加载含 TE-NPU recipe 元数据的 FP8 extra_state checkpoint 被拒；扩展白名单放行 torch/torch_npu 的 float8 dtype 与 `transformer_engine.common.recipe` 模块，其余类仍拒绝 |

**表 4** NPU Profiling

| 目标符号  | 替换实现（MA 源码位置） | 修改目的 |
|---|---|---|
| `megatron.training.training.train` | `train_wrapper`（patches/megatron/profile.py） | 按 `--profile` 系列参数构造 `torch_npu.profiler` 上下文包裹训练循环（按 step 区间采集、rank 过滤、NPU/CPU 活动与 AIC PipeUtilization 指标） |
| `megatron.training.training.train_step` | `train_step_wrapper`（patches/megatron/profile.py） | 每个训练步推进 profiler step，使采集区间与训练 step 对齐 |

**表 5** NPU 数据 Dump 精度对照

| 目标符号  | 替换实现（MA 源码位置） | 修改目的 |
|---|---|---|
| `megatron.training.training.train_step` | `dump_start_wrapper`（patches/megatron/npu_datadump/npu_datadump.py） | 训练步开始处启动 msprobe `PrecisionDebugger`（按 config.json 配置采集张量/统计量） |
| `megatron.training.ft_integration.on_training_step_end` | `dump_end_wrapper`（patches/megatron/npu_datadump/npu_datadump.py） | 训练步结束处停止采集并推进 step，完成按步 dump |

>[!NOTE]
>
> NPU Profiling 与数据 Dump 同时作用于 `train_step`，两者按特性顺序链式叠加（dump 在外层、profile 在更外层，互不影响）。

**表 6** NPU 确定性计算

| 目标符号 | 替换实现（MA 源码位置） | 修改目的 |
|---|---|---|
| `megatron.training.initialize._set_random_seed` | `npu_deterministic_wrapper`（patches/megatron/npu_deterministic.py） | 在上游种子初始化基础上追加 NPU 确定性开关：`HCCL_DETERMINISTIC`、`CLOSE_MATMUL_K_SHIFT`、`PYTHONHASHSEED`、`torch.use_deterministic_algorithms` 及 NPU 种子设置 |

**表 7** msprof 通信组标注

| 目标符号 | 替换实现（MA 源码位置） | 修改目的 |
|---|---|---|
| `megatron.core.parallel_state.get_nccl_options` | `get_nccl_options_add_group_info_wrapper`（patches/megatron/parallel_state.py） | 为每个并行组（dp/tp/pp/ep 等）的 HCCL 通信配置注入 `group_name`，使 msprof trace 可区分各通信域 |

**表 8** DDP侧流显存滞留问题兼容性开关

| 目标符号 | 替换实现（MA 源码位置） | 修改目的 |
|---|---|---|
| `megatron.training.training.get_model` | `get_model_with_upstream_stream_selection`（patches/megatron/ddp_one_shot_stream.py） | 等价替换 `get_model`：与Megatron高版本相同，在 DDP/FSDP 包装段就地应用三分支流选择（eager 改用当前流消除一次性侧流滞留，full_iteration 改用共享捕获流，其余保持专用流）|
| `megatron.training.models.dist_utils._ddp_wrap` | `ddp_wrap_with_upstream_stream_selection`（patches/megatron/ddp_one_shot_stream.py） | 等价替换 `_ddp_wrap`：流选择块与Megatron高版本一致（eager 改用当前流 `ddp_stream == current_stream`，不再创建一次性侧流，消除 CachingAllocator 导致的显存滞留） |

**表 9** 非 overlap 梯度规约下的 dummy wgrad 常驻显存消除

| 目标符号 | 替换实现（MA 源码位置） | 修改目的 |
|---|---|---|
| `megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__` | `ddp_init_dummy_wgrad_wrapper`（patches/megatron/dummy_wgrad_non_overlap.py） | 为 `params_with_grad` 打标 `requires_dummy_wgrad`（取值等于 `ddp_config.overlap_grad_reduce`）并装备进程级闸门；该标记同时供 TransformerEngineNPU 侧消费 |
| `megatron.core.tensor_parallel.layers.get_dummy_wgrad` | `get_dummy_wgrad_skip_gate_wrapper`（patches/megatron/dummy_wgrad_non_overlap.py） | 非 overlap 时不再申请全尺寸 dummy wgrad（含 TransformerEngineNPU `_dummy_wgrads` 的常驻 shape 缓存），消除该块常驻显存；overlap 场景原样透传。开关为 `--fix-non-overlap-dummy-wgrad`，默认关闭 |

> 说明：TransformerEngine 自分配 dummy 的路径（GroupedLinear / PerformanceGroupedLinear / ops basic_linear / FSDP 分支）不属 MA 职责，需 TransformerEngineNPU 侧在 `setup_dummy_wgrad` 中加入同一道闸门（读取 `requires_dummy_wgrad`）后才同步受益。

**表 10** MLA RoPE 融合适配

| 目标符号 | 替换实现（MA 源码位置） | 修改目的 |
|---|---|---|
| `megatron.core.transformer.transformer_config.TransformerConfig.__post_init__` | `mla_rope_fusion_post_init_wrapper`（patches/megatron/mla_rope.py） | MLA + YARN RoPE 融合在 TransformerEngineNPU 后端不可用时默认关闭（上游 Triton kernel 在 NPU 上远慢于 CANN 旋转算子）并输出一次性告警；传 `--mla-rope-fusion` 可强制开启 |
| `megatron.core.fusions.fused_mla_yarn_rope_apply.fused_apply_mla_rope_for_q` | `mla_rope_npu_backend_wrapper`（patches/megatron/mla_rope.py） | 优先路由到 TransformerEngineNPU 的 CANN 实现；后端拒绝（3D/packed/interleaved 输入）时回退上游实现。仅在后端可导入时注册 |
| `megatron.core.fusions.fused_mla_yarn_rope_apply.fused_apply_mla_rope_for_kv` | `mla_rope_npu_backend_wrapper`（patches/megatron/mla_rope.py） | 同上（kv 路径），与 for_q 成对注册，保证 q/kv 旋转通道布局一致（NPU 后端为 interleaved 输出，与上游 stacked 输出相差固定通道置换） |
