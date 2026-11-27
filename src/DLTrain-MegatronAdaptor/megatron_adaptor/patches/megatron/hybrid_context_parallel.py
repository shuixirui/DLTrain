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
"""Dense hybrid context parallelism for Megatron 0.18 on NPU."""

import inspect
import logging
from contextlib import nullcontext
from contextvars import ContextVar
from copy import copy
from functools import lru_cache, wraps
from math import lcm

import torch


TOKEN_KEYS = ("tokens", "labels", "loss_mask", "position_ids")


@lru_cache(maxsize=1)
def _log_conversion():
    logging.getLogger(__name__).info(
        "MA DCP automatic EOD conversion active: Attention is isolated at EOD; positions restart per sequence."
    )


def prepare_raw_batch(batch, pool_size, allow_eod=True, sequence_parallel_size=1):
    """Prepare GPT EOD or explicit packed samples in native routing field order."""
    # Align each document before assignment: every eligible CP width must split
    # it into mirrored chunks while keeping the local sequence TP/SP-divisible.
    alignment = lcm(2, sequence_parallel_size) * pool_size
    prepared = []
    eod = None
    for sample in batch:
        if sample.get("attention_mask") is not None:
            raise ValueError("MA DCP requires no dense attention_mask in raw data")
        tokens = sample["tokens"]
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 1 or not tokens.numel():
            raise ValueError("MA DCP raw tokens must be a nonempty 1-D tensor")
        length = tokens.numel()
        for key in TOKEN_KEYS:
            value = sample[key]
            dtype = torch.float32 if key == "loss_mask" else torch.int64
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 1
                or value.numel() != length
                or value.dtype != dtype
                or value.device != tokens.device
            ):
                raise ValueError(f"MA DCP raw {key} must be matching 1-D {dtype} on the tokens device")
        # Native reroute transports fields in dictionary order on every rank.
        source = {key: sample[key] for key in TOKEN_KEYS}
        if "cu_seqlens" in sample or not allow_eod:
            prepared.append(dict(source, cu_seqlens=sample["cu_seqlens"]))
            continue
        if eod is None:
            from megatron.training import get_tokenizer

            eod = get_tokenizer().eod
        ends = (torch.nonzero(tokens == eod, as_tuple=False).flatten() + 1).tolist()
        if not ends or ends[-1] != length:
            ends.append(length)
        starts = [0, *ends[:-1]]
        # ends indexes the original tokens; cu below indexes the padded buffer.
        # Padding each document separately keeps CP chunks within document boundaries.
        sizes = [((end - start + alignment - 1) // alignment) * alignment for start, end in zip(starts, ends)]
        total = sum(sizes)
        output = {key: source[key].new_zeros(total) for key in TOKEN_KEYS}
        cu = [0]
        for start, end, size in zip(starts, ends, sizes):
            offset = cu[-1]
            for key in ("tokens", "labels", "loss_mask"):
                output[key][offset : offset + end - start] = source[key][start:end]
            # Labels were shifted by GPTDataset before padding; never shift again.
            output["position_ids"][offset : offset + size] = torch.arange(size, device=tokens.device)
            cu.append(offset + size)
        output["cu_seqlens"] = torch.tensor(cu, dtype=torch.int32, device=tokens.device)
        prepared.append(output)
        _log_conversion()
    return prepared


class PackedBatchIterator:
    """Prepare provider data and propagate conversion failures before routing collectives."""

    def __init__(
        self,
        source,
        group,
        allow_eod=True,
        sequence_parallel_size=1,
    ):
        self.source = source
        self.group = group
        self.allow_eod = allow_eod
        self.sequence_parallel_size = sequence_parallel_size

    def __next__(self):
        # Each provider rank must either return a prepared batch or fail with its peers.
        # Raising here on just one rank could strand the others in native sample routing.
        batch, error = None, None
        try:
            raw = next(self.source)
            batch = prepare_raw_batch(
                raw,
                self.group.size(),
                self.allow_eod,
                self.sequence_parallel_size,
            )
        except Exception as exc:  # Keep provider/tokenizer failures inside the collective error boundary.
            error = str(exc) or 'raw provider was exhausted before a complete global batch'
        if self.group.size() == 1:
            if error is not None:
                raise ValueError(f'Hybrid CP raw-provider validation failed: {error}')
            return batch
        # Report provider failures before other source ranks enter routing collectives.
        failed = torch.tensor([error is not None], dtype=torch.int32, device=torch.cuda.current_device())
        torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX, group=self.group)
        if failed.item():
            raise ValueError(f'Hybrid CP raw-provider validation failed: {error or "invalid data on a peer rank"}')
        return batch


def hybrid_loader_init_wrapper(original):
    """Validate and prepare provider data before native scheduling and routing."""

    @wraps(original)
    def wrapper(self, data_iterator, config, *args, **kwargs):
        from megatron.training import get_args

        original(self, data_iterator, config, *args, **kwargs)
        # Native HybridCPDataLoaderWrapper leaves nonzero TP ranks without a
        # provider. They receive every routed sub-sample through TP broadcast.
        if self.data_iterator is None:
            return
        # Native TransformerConfig does not own these provider-level fields.
        # Read training arguments only here, after global training initialization.
        run_args = get_args()
        # The native loader keeps its scheduler and groups. Wrap only its input so the
        # scheduler sees physical document lengths even when GPT supplies just EOD tokens.
        self.data_iterator = PackedBatchIterator(
            self.data_iterator,
            self.dp_cp_group,
            allow_eod=not getattr(run_args, 'sft', False),
            sequence_parallel_size=(
                getattr(run_args, 'tensor_model_parallel_size', 1)
                if getattr(run_args, 'sequence_parallel', False)
                else 1
            ),
        )

    return wrapper


# One job per model call, including activation recomputation in its backward.
_JOB_WIDTH = ContextVar("ma_018_dcp_width", default=None)


def hybrid_loader_wrapper(original):
    # Return a native loader subclass with the same (assigned samples, waves) interface.
    # At DP1 the samples already exist locally; at DP>1 native routing owns the exchange.
    class HybridCPDataLoader(original):
        __init__ = hybrid_loader_init_wrapper(original.__init__)

        def __next__(self):
            if self.data_iterator is None or self.dp_group.size() != 1:
                return super().__next__()
            # DP1 CP replicas read identical packs. Global IDs are unpack order.
            batch = next(self.data_iterator)
            lengths = [length for sample in batch for length in sample["cu_seqlens"].diff().tolist()]
            _, waves = self.cp_balancing_scheduler.get_groups_and_subsamples(list(enumerate(lengths)), self.config)
            samples = self.unpack_batch(batch)
            rank = self.dp_cp_group.rank()
            return {gid: samples[gid] for wave in waves for gid in wave[rank]}, waves

    return HybridCPDataLoader


def _share_plan(widths, tp, src):
    # A zero width separates waves. Native TP batch broadcast carries each physical length.
    if tp.size() == 1:
        return widths
    device = torch.cuda.current_device()
    # TP peers start with no widths. Send the count first so their receive buffers
    # match the source even when the number of jobs changes between steps.
    count = torch.tensor([len(widths)], device=device, dtype=torch.int64)
    torch.distributed.broadcast(count, src, group=tp)
    payload = (
        torch.tensor(widths, dtype=torch.int64, device=device)
        if widths
        else torch.empty((int(count.item()),), dtype=torch.int64, device=device)
    )
    torch.distributed.broadcast(payload, src, group=tp)
    return widths or payload.tolist()


def hybrid_schedule(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    output_tensor_grad,
    forward_data_store,
    config,
    collect_non_loss_data,
    first_val_step,
    forward_only,
    no_sync_func,
    total_num_tokens,
    check_first_val_step,
    model_type,
):
    # Execute the native assignments and return the forward data store and token total
    # expected by the outer schedule. Each TP peer must run the same local job sequence.
    from megatron.core import parallel_state as ps
    from megatron.core.pipeline_parallel.schedules import backward_step, forward_step
    from megatron.core.rerun_state_machine import RerunDataIterator

    pool = ps.get_data_parallel_group(with_context_parallel=True)
    tp = ps.get_tensor_model_parallel_group()
    source = tp.rank() == 0
    widths, ids = [], []
    if source:
        samples, waves = next(data_iterator)
        for wave_index, wave in enumerate(waves):
            # Peers receive only local jobs, but must enter the same pool barriers
            # between native waves even when their numbers of jobs differ.
            if wave_index:
                widths.append(0)
            for gid in wave[pool.rank()]:
                ids.append(gid)
                # Keep one model call per document, even when two documents share a group.
                width = sum(gid in members for members in wave)
                widths.append(width)
    torch.distributed.barrier()
    plan = _share_plan(widths, tp, ps.get_tensor_model_parallel_src_rank())
    job_index = 0
    for index, width in enumerate(plan):
        if width == 0:
            torch.distributed.barrier(group=pool)
            continue
        iterator = None
        if source:
            sample = samples[ids[job_index]]
            length = sample["tokens"].numel()
            batch = {key: sample[key].reshape(1, length) for key in TOKEN_KEYS}
            cu = torch.tensor([[0, length]], dtype=torch.int32)
            batch.update(
                cu_seqlens=cu,
                cu_seqlens_padded=cu,
                max_seqlen=torch.tensor([length], dtype=torch.int32),
                local_cp_size=torch.tensor([width], dtype=torch.int32),
            )
            iterator = RerunDataIterator(iter([batch]))
        token = _JOB_WIDTH.set(width)
        try:
            # Keep the descriptor available to batch fetches during forward and recompute.
            # Only the final local job runs outside the gradient no-sync context.
            with nullcontext() if index == len(plan) - 1 else no_sync_func():
                # The 0.18 forward_step needs the job width for loss scaling;
                # backward_step takes config directly, without the old model_type.
                output, num_tokens = forward_step(
                    forward_step_func,
                    iterator,
                    model,
                    num_microbatches,
                    input_tensor,
                    forward_data_store,
                    config,
                    width,
                    collect_non_loss_data,
                    is_first_microbatch=check_first_val_step(first_val_step, forward_only, job_index == 0),
                    current_microbatch=job_index,
                )
                total_num_tokens += num_tokens
                if not forward_only:
                    backward_step(input_tensor, output, output_tensor_grad, config)
                del output
        finally:
            _JOB_WIDTH.reset(token)
        job_index += 1
    return forward_data_store, total_num_tokens


def hybrid_tp_batch_wrapper(original):
    @wraps(original)
    def wrapper(batch, *args, **kwargs):
        width = _JOB_WIDTH.get()
        if width is None:
            return original(batch, *args, **kwargs)
        # Configured MBS belongs to the sampler. Each scheduled document has one row.
        kwargs.update(micro_batch_size=1, cp_size=width)
        result = original(batch, *args, **kwargs)
        if width == 1:
            from megatron.core import parallel_state

            # Native CP1 skips padded endpoints and does not install its singleton.
            result["cu_seqlens_padded"] = result["cu_seqlens"]
            result["hybrid_cp_group"] = parallel_state.get_hybrid_data_context_parallel_groups(group_size=1)
        return result

    return wrapper


def hybrid_groups_wrapper(original):
    """Add the singleton missing from 0.18's native hybrid group registry."""

    @wraps(original)
    def wrapper(rank, ranks, pg_options):
        from megatron.core import parallel_state

        groups = original(rank, ranks, pg_options)
        # All participants create singletons in the same order. Each rank retains
        # its own handle under width 1 so a local job cannot fall back to the static PG.
        for member in ranks:
            group = parallel_state.create_group([member], pg_options=pg_options, group_desc='HYBRID_DP_CP_GROUP_1')
            if rank == member:
                groups[1] = group
        return groups

    return wrapper


def hybrid_attention_forward_wrapper(original):
    signature = inspect.signature(original)

    @wraps(original)
    def wrapper(self, *args, **kwargs):
        if not getattr(self.config, 'hybrid_context_parallel', False):
            return original(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        group = getattr(bound.arguments.get('packed_seq_params'), 'cp_group', None)
        previous = self.pg_collection
        if group is None or group is previous.cp:
            return original(self, *args, **kwargs)
        # Native attention passes pg_collection.cp to RoPE. Leave the shared
        # collection untouched and restore the layer's reference on every exit.
        current = copy(previous)
        current.cp = group
        self.pg_collection = current
        try:
            return original(self, *args, **kwargs)
        finally:
            self.pg_collection = previous

    return wrapper


def hybrid_te_init_wrapper(original):
    signature = inspect.signature(original)

    @wraps(original)
    def wrapper(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        # Static CP1 construction may omit TE's communication settings. Keep the
        # requested per-layer strategy for subsequent jobs with a wider CP group.
        comm_type = bound.arguments.get('cp_comm_type') or 'p2p'
        original(self, *args, **kwargs)
        if not getattr(self.config, 'hybrid_context_parallel', False):
            return
        self.cp_comm_type = comm_type
        from megatron.core.extensions.transformer_engine import TEDotProductAttention

        if TEDotProductAttention.cp_stream is None:
            TEDotProductAttention.cp_stream = torch.cuda.Stream()
        self.cp_stream = TEDotProductAttention.cp_stream
        # Enable once per layer. The actual job PG arrives later, at forward time.
        self.enable_dynamic_context_parallel()

    return wrapper
