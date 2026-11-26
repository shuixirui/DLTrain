# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import timedelta

import torch
from megatron.core import parallel_state as mpu
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    FullyShardedDataParallel,
)
from megatron.core.full_cuda_graph import get_shared_capture_stream
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.utils import get_model_config
from megatron.training.training import (
    Float16Module,
    ModelType,
    correct_amax_history_if_needed,
    create_all_gather_groups,
    get_args,
    get_megatron_ddp_config,
    get_pg_rank,
    get_pg_size,
    has_nvidia_modelopt,
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
    print_rank_0,
    tensor_parallel,
    to_empty_if_meta_device,
    wrap_model_chunks_with_ddp,
)

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False


def get_model_with_upstream_stream_selection(
    model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True, config=None, pg_collection=None
):
    """Drop-in replacement of upstream ``get_model`` (core_r0.18.0) mirroring #6885 exactly.

    Body copied verbatim from upstream; the DDP/FSDP wrap segment carries the #6885
    stream selection applied in place (three-way selection, see the helper below).
    The segment is extracted into ``_wrap_with_ddp_upstream_fix`` only to keep this
    function within the 200-line limit. Remove this patch once upstream cherry-picks
    #6885.
    """
    args = get_args()
    args.model_type = model_type
    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        if args.create_all_gather_group:
            timeout = timedelta(minutes=args.distributed_timeout_minutes) if args.distributed_timeout_minutes else None
            dp_cp_ag, expt_dp_ag = create_all_gather_groups(
                for_expert_parallelism=(args.expert_model_parallel_size > 1),
                timeout=timeout,
            )
            pg_collection.dp_cp_ag = dp_cp_ag
            pg_collection.expt_dp_ag = expt_dp_ag

            print_rank_0("> created all-gather process groups for AG/RS overlap")
            if expt_dp_ag is not None:
                print_rank_0(">   including expert parallelism AG group")

    if has_nvidia_modelopt:
        from megatron.post_training.checkpointing import has_modelopt_state

        # [ModelOpt]: Check if the checkpoint is a ModelOpt checkpoint and
        # set a flag to use our model provider if so.
        if args.load is not None and has_modelopt_state(args.load):
            print_rank_0('ModelOpt checkpoint detected')
            args.modelopt_enabled = True
        elif getattr(args, "export_kd_teacher_load", None):
            # For distillation ckpts without ModelOpt state
            args.modelopt_enabled = True

    # Build model.
    def build_model():
        if get_pg_size(pg_collection.pp) > 1 and args.virtual_pipeline_model_parallel_size is not None:
            model = []
            vp_size = args.virtual_pipeline_model_parallel_size
            for i in range(vp_size):
                # Set pre_process and post_process only after virtual rank is set.
                pre_process = is_pp_first_stage(pg_collection.pp) and is_vp_first_stage(vp_stage=i, vp_size=vp_size)
                post_process = is_pp_last_stage(pg_collection.pp) and is_vp_last_stage(vp_stage=i, vp_size=vp_size)
                this_model = model_provider_func(
                    pre_process=pre_process,
                    post_process=post_process,
                    vp_stage=i,
                    config=config,
                    pg_collection=pg_collection,
                )
                this_model.model_type = model_type
                this_model.vp_stage = i
                model.append(this_model)
        else:
            pre_process = is_pp_first_stage(pg_collection.pp)
            post_process = is_pp_last_stage(pg_collection.pp)
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                config=config,
                pg_collection=pg_collection,
            )
            model.model_type = model_type
        return model

    if args.init_model_with_meta_device:
        with torch.device('meta'):
            model = build_model()
    else:
        model = build_model()

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    # Verbatim from upstream get_model; keep its list-comprehension form.
    # pylint: disable=consider-using-generator
    num_parameters = sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model])
    # pylint: enable=consider-using-generator
    if get_pg_rank(pg_collection.dp) == 0 and get_pg_rank(pg_collection.cp) == 0:
        print(
            ' > number of parameters on (tensor, pipeline) model parallel rank ({}, {}): {}'.format(
                get_pg_rank(pg_collection.tp),
                get_pg_rank(pg_collection.pp),
                num_parameters,
            ),
            flush=True,
        )

    # GPU allocation.
    # For FSDP2, we don't allocate GPU memory here. We allocate GPU memory
    # in the fully_shard function of FSDP2 instead.
    if not (args.use_torch_fsdp2 and args.use_cpu_initialization) and not args.init_model_with_meta_device:
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        config = get_model_config(model[0])
        model = [Float16Module(config, model_module) for model_module in model]

    # Materialize tensors on meta device (GPU allocation) if not using FSDP2 and not using Megatron FSDP.
    if args.init_model_with_meta_device and not args.use_torch_fsdp2 and not args.use_megatron_fsdp:
        model = [to_empty_if_meta_device(model_module, device=torch.device("cuda")) for model_module in model]

    # Before TE2.x: The model_module.bfloat16()/model_module.half() above will call the inplace
    #               copy of TE's Float8Tensor, which will write an unwanted value (amax calculated
    #               from the current fp8 param) to its amax_history. The below function will correct
    #               the amax_history back.
    # After TE2.x: Below function is an empty function and does nothing.
    correct_amax_history_if_needed(model)

    if wrap_with_ddp:
        model = _wrap_with_ddp_upstream_fix(args, model, pg_collection, num_parameters)

    return model


def _wrap_with_ddp_upstream_fix(args, model, pg_collection, num_parameters):
    """The ``if wrap_with_ddp:`` segment of upstream ``get_model`` (core_r0.18.0), extracted
    verbatim to keep ``get_model`` within the 200-line limit. The stream-selection block
    below is the upstream #6885 fix applied in place.
    """
    if args.use_torch_fsdp2:
        assert HAVE_FSDP2, "Torch FSDP2 requires torch>=2.4.0"
        DP = TorchFullyShardedDataParallel
    elif args.use_megatron_fsdp:
        DP = FullyShardedDataParallel
    else:
        DP = DistributedDataParallel

    config = get_model_config(model[0])

    ddp_config = get_megatron_ddp_config(args)
    if not getattr(args, "use_torch_fsdp2", False):
        if ddp_config.num_buckets is not None:
            ddp_config.bucket_size = num_parameters // ddp_config.num_buckets

        # In the Megatron FSDP and DDP use path, we need to initialize the bucket size.
        # If bucket_size is not provided as an input, use sane default.
        # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
        # ring-reduce implementations are large enough to remain bandwidth-bound rather than
        # latency-bound.
        if ddp_config.bucket_size is None:
            ddp_config.bucket_size = max(
                40000000, 1000000 * mpu.get_data_parallel_world_size(with_context_parallel=True)
            )
        # Set bucket_size to infinity if overlap_grad_reduce is False.
        if not ddp_config.overlap_grad_reduce:
            ddp_config.bucket_size = None

    # Compute per-chunk bucket sizes / disable_bucketing flags. Bucketing is
    # disabled for non-first chunks, when overlap_param_gather_with_optimizer_step
    # is on, or for non-zero pipeline-parallel ranks.
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    per_chunk_disable_bucketing = [
        (chunk_idx > 0) or args.overlap_param_gather_with_optimizer_step for chunk_idx in range(len(model))
    ]
    per_chunk_bucket_sizes = [
        None if (disable or pp_rank > 0) else ddp_config.bucket_size for disable in per_chunk_disable_bucketing
    ]

    # Setup stream for ddp initialization (upstream #6885, applied in place).
    current_stream = torch.cuda.current_stream()
    if config.cuda_graph_impl == "full_iteration":
        # DDP initialization must use the full-iteration capture stream so its retained
        # AccumulateGrad nodes do not reference a different, non-capturing stream.
        ddp_stream = get_shared_capture_stream()
    elif config.cuda_graph_impl == "none":
        # Eager initialization is serialized with the current stream. A one-shot side stream
        # can leave cached blocks unavailable to later allocations on the current stream.
        ddp_stream = current_stream
    else:
        # Preserve a dedicated initialization stream for all other implementations.
        ddp_stream = torch.cuda.Stream()
    if ddp_stream is not current_stream:
        ddp_stream.wait_stream(current_stream)

    with torch.cuda.stream(ddp_stream):
        model = wrap_model_chunks_with_ddp(
            model,
            config,
            ddp_config,
            use_layer_wise_distributed_optimizer=getattr(args, 'use_layer_wise_distributed_optimizer', False),
            use_layer_wise_param_layout=getattr(args, 'use_layer_wise_param_layout', True),
            DP=DP,
            pg_collection=pg_collection if args.use_megatron_fsdp else None,
            bucket_sizes=per_chunk_bucket_sizes,
            disable_bucketing_per_chunk=per_chunk_disable_bucketing,
        )
    # Ensure the initialization work completes before touching params on the current stream.
    if ddp_stream is not current_stream:
        current_stream.wait_stream(ddp_stream)

    # Broadcast params from data parallel src rank to other data parallel ranks.
    if args.data_parallel_random_init:
        for model_module in model:
            model_module.broadcast_params()

    return model


def ddp_wrap_with_upstream_stream_selection(
    model: list[MegatronModule],
    data_parallel_random_init: bool,
    ddp_config: DistributedDataParallelConfig,
    overlap_param_gather_with_optimizer_step: bool,
    use_megatron_fsdp: bool = False,
    use_torch_fsdp2: bool = False,
    *,
    pg_collection: ProcessGroupCollection,
) -> list[MegatronModule]:
    """Drop-in replacement of upstream ``_ddp_wrap`` mirroring the #6885 fix exactly.

    Body copied from upstream core_r0.18.0; the stream-selection block below is the
    upstream #6885 change applied verbatim (three-way selection with same-stream
    guards). Remove this patch once upstream cherry-picks #6885.
    """
    if use_megatron_fsdp:
        DP = FullyShardedDataParallel
        if use_torch_fsdp2:
            raise ValueError("Using use_megatron_fsdp and use_torch_fsdp2 at the same time is not supported.")
    elif use_torch_fsdp2:
        assert HAVE_FSDP2, "Torch FSDP2 requires torch>=2.4.0"
        DP = TorchFullyShardedDataParallel
    else:
        DP = DistributedDataParallel

    if not use_torch_fsdp2:
        # Default bucket size when unset; larger buckets for large DP sizes keep NCCL
        # ring-reduce bandwidth-bound. Bucketing is disabled when overlap_grad_reduce is off.
        if ddp_config.num_buckets is not None:
            # Verbatim from upstream _ddp_wrap; keep its list-comprehension form.
            # pylint: disable=consider-using-generator
            num_parameters = sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model])
            # pylint: enable=consider-using-generator
            ddp_config.bucket_size = num_parameters // ddp_config.num_buckets

        if ddp_config.bucket_size is None:
            ddp_config.bucket_size = max(40000000, 1000000 * pg_collection.dp_cp.size())
        if not ddp_config.overlap_grad_reduce:
            ddp_config.bucket_size = None

    # Upstream #6885 stream selection, verbatim.
    current_stream = torch.cuda.current_stream()
    cuda_graph_impl = get_model_config(model[0]).cuda_graph_impl
    if cuda_graph_impl == "full_iteration":
        # DDP initialization must use the full-iteration capture stream so its retained
        # AccumulateGrad nodes do not reference a different, non-capturing stream.
        ddp_stream = get_shared_capture_stream()
    elif cuda_graph_impl == "none":
        # Eager initialization is serialized with the current stream. A one-shot side stream
        # can leave cached blocks unavailable to later allocations on the current stream.
        ddp_stream = current_stream
    else:
        # Preserve a dedicated initialization stream for all other implementations.
        ddp_stream = torch.cuda.Stream()
    if ddp_stream is not current_stream:
        ddp_stream.wait_stream(current_stream)

    with torch.cuda.stream(ddp_stream):
        dp_init_kwargs = {}
        if not use_torch_fsdp2:
            dp_init_kwargs["pg_collection"] = pg_collection

        wrapped_model = []
        for model_chunk_idx, model_chunk in enumerate(model):
            chunk_kwargs = dict(dp_init_kwargs)
            disable_bucketing = (model_chunk_idx > 0) or overlap_param_gather_with_optimizer_step

            # Pre-compute parameter layouts for the distributed optimizer; only DDP
            # accepts full_param_layout.
            if ddp_config.use_distributed_optimizer and DP is DistributedDataParallel:
                all_params = [p for p in model_chunk.parameters() if p.requires_grad]
                pp_rank = pg_collection.pp.rank()
                effective_bucket_size = None if disable_bucketing or pp_rank > 0 else ddp_config.bucket_size
                chunk_kwargs["full_param_layout"] = DistributedOptimizer.compute_full_param_layout(
                    all_params,
                    effective_bucket_size,
                    pg_collection.dp_cp.size(),
                    ddp_config,
                    expert_data_parallel_world_size=pg_collection.expt_dp.size(),
                )

            wrapped_chunk = DP(
                config=get_model_config(model_chunk),
                ddp_config=ddp_config,
                module=model_chunk,
                disable_bucketing=disable_bucketing,
                **chunk_kwargs,
            )
            wrapped_model.append(wrapped_chunk)
        model = wrapped_model

    # Ensure side-stream initialization completes before touching params on the current stream.
    if ddp_stream is not current_stream:
        current_stream.wait_stream(ddp_stream)

    if data_parallel_random_init:
        for model_module in model:
            model_module.broadcast_params()

    return model
