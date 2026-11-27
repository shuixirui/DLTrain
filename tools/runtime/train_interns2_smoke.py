"""Candidate real-weight, text-only Megatron SGD step on 16 logical Ascend devices."""

import hashlib
import json
import math
import os
import runpy
from collections import Counter
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from safetensors import safe_open
from optimizer_update_probe import capture_update_points, verify_update_points

workdir = Path(os.environ["WORKDIR"])
model_dir = Path(os.environ.get("MODEL_PATH", "/mnt/afs-a3-weight-share/Intern-S2-Preview"))
local_rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(local_rank)
dist.init_process_group("hccl", timeout=timedelta(minutes=15))
rank, world = dist.get_rank(), dist.get_world_size()
assert world == 16
state = runpy.run_path(str(workdir / "check_interns2_provider.py"))
provider = state["provider"]
for flag in ("bias_activation_fusion", "bias_dropout_fusion", "masked_softmax_fusion", "cross_entropy_loss_fusion", "apply_rope_fusion"):
    setattr(provider, flag, False)
provider.deallocate_pipeline_outputs = False
provider.initialize_model_parallel(seed=42)

from megatron.core import parallel_state as mpu
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.utils import get_model_config

ddp_config = DistributedDataParallelConfig(
    grad_reduce_in_fp32=True, overlap_grad_reduce=False,
    use_distributed_optimizer=False, check_for_nan_in_grad=True,
)
if rank == 0:
    print("START_DDP_WEIGHT_LOAD", flush=True)
models = provider.provide_distributed_model(
    ddp_config=ddp_config, wrap_with_ddp=True, data_parallel_random_init=False,
)
assert len(models) == 1
model = models[0]
model.train()
named_params = dict(model.named_parameters())
dtype_counts = Counter()
for param in named_params.values():
    dtype_counts[str(param.dtype)] += param.numel()
fp32_largest = sorted(
    ((name, param.numel()) for name, param in named_params.items() if param.dtype == torch.float32),
    key=lambda pair: pair[1], reverse=True,
)[:8]
if rank == 0:
    print("LOCAL_PARAMETER_DTYPES:", dict(dtype_counts), flush=True)
    print("LARGEST_LOCAL_FP32:", fp32_largest, flush=True)

# Check the first local expert of layer 0 on every EP/ETP rank, not just embeddings.
module_map = dict(model.named_modules())
fc1 = next(module for name, module in module_map.items() if name.endswith("language_model.decoder.layers.0.mlp.experts.linear_fc1"))
fc2 = next(module for name, module in module_map.items() if name.endswith("language_model.decoder.layers.0.mlp.experts.linear_fc2"))
expert_number = mpu.get_expert_model_parallel_rank() * (provider.num_moe_experts // provider.expert_model_parallel_size)
etp_rank = mpu.get_expert_tensor_parallel_rank()
etp_size = mpu.get_expert_tensor_parallel_world_size()
index = json.loads((model_dir / "model.safetensors.index.json").read_text())
prefix = "model.language_model.layers.0.mlp.experts."

def read_expert(suffix):
    key = prefix + suffix
    with safe_open(str(model_dir / index["weight_map"][key]), framework="pt", device="cpu") as source:
        return source.get_slice(key)[expert_number]

gate_up = read_expert("gate_up_proj")
hidden, ffn = provider.hidden_size, provider.moe_ffn_hidden_size
if tuple(gate_up.shape) == (hidden, 2 * ffn):
    gate_up = gate_up.t().contiguous()
assert tuple(gate_up.shape) == (2 * ffn, hidden)
gate, up = gate_up.chunk(2, dim=0)
expected = torch.cat((gate.chunk(etp_size, dim=0)[etp_rank], up.chunk(etp_size, dim=0)[etp_rank]), dim=0)
torch.testing.assert_close(fc1.weight0.detach().cpu(), expected.to(fc1.weight0.dtype), rtol=0, atol=0)
down = read_expert("down_proj")
if tuple(down.shape) == (ffn, hidden):
    down = down.t().contiguous()
assert tuple(down.shape) == (hidden, ffn)
expected = down.chunk(etp_size, dim=1)[etp_rank].contiguous()
torch.testing.assert_close(fc2.weight0.detach().cpu(), expected.to(fc2.weight0.dtype), rtol=0, atol=0)
print(f"rank={rank}: EXPERT_EXACT_MATCH global_expert={expert_number} etp_rank={etp_rank}", flush=True)
del gate_up, gate, up, down, expected

# Combine small tensors with gradient-selected points in large matrices below.
candidates = [(name, param) for name, param in named_params.items() if param.requires_grad and "language_model" in name and 0 < param.numel() <= 8192]
candidates.sort(key=lambda pair: ("decoder.final_layernorm" not in pair[0], pair[1].dtype != torch.float32, "norm" not in pair[0], pair[0]))
probes = candidates[:128]
assert probes

optimizer_config = OptimizerConfig(
    optimizer="sgd", lr=1e-3, sgd_momentum=0.0, weight_decay=0.0,
    bf16=True, fp16=False, params_dtype=torch.bfloat16,
    use_distributed_optimizer=False, clip_grad=1.0,
)
optimizer = get_megatron_optimizer(optimizer_config, models, use_gloo_process_groups=False)
config = get_model_config(model)
config.grad_scale_func = optimizer.scale_loss
config.finalize_model_grads_func = finalize_model_grads
config.no_sync_func = model.no_sync
config.deallocate_pipeline_outputs = False
model.zero_grad_buffer()
optimizer.zero_grad()

seq_length = 64
generator = torch.Generator(device="cpu").manual_seed(20261125 + mpu.get_data_parallel_rank())
sequence = torch.randint(100, 20000, (1, seq_length + 1), generator=generator, device="cpu", dtype=torch.long).to(f"npu:{local_rank}")
tokens, labels = sequence[:, :-1].contiguous(), sequence[:, 1:].contiguous()
mask = torch.ones_like(labels, dtype=torch.float32)

def forward_step(data_iterator, wrapped_model):
    token_batch, label_batch, loss_mask = next(data_iterator)
    output = wrapped_model(input_ids=token_batch, labels=label_batch, attention_mask=None, loss_mask=loss_mask, extra_block_kwargs={})
    assert tuple(output.shape) == tuple(label_batch.shape), (output.shape, label_batch.shape)

    def loss_func(losses):
        total = (losses.float() * loss_mask).sum()
        count = loss_mask.sum().to(torch.int32)
        assert torch.isfinite(total).item(), "Non-finite forward loss"
        return total, count, {"lm_loss": torch.stack((total.detach(), count.float()))}

    return output, loss_func

if rank == 0:
    print("START_FORWARD_BACKWARD", flush=True)
reports = get_forward_backward_func()(
    forward_step_func=forward_step,
    data_iterator=iter([(tokens, labels, mask)]),
    model=models, num_microbatches=1, seq_length=seq_length,
    micro_batch_size=1, forward_only=False,
)
torch.npu.synchronize()
metric = reports[0]["lm_loss"].detach().clone()
dist.all_reduce(metric, group=mpu.get_data_parallel_group())
mean_loss = (metric[0] / metric[1]).item()
assert math.isfinite(mean_loss)
if rank == 0:
    print("FORWARD_BACKWARD_GRAD_SYNC_OK loss:", mean_loss, flush=True)

update_points, effective_lrs = capture_update_points(named_params, optimizer, probes)
if rank == 0:
    print("EFFECTIVE_LEARNING_RATES:", effective_lrs, flush=True)
success, grad_norm, _ = optimizer.step()
assert success, "Optimizer rejected the step"
assert grad_norm is not None and math.isfinite(float(grad_norm)) and float(grad_norm) > 0
torch.npu.synchronize()
update_summary = verify_update_points(update_points, grad_norm, optimizer_config.clip_grad, rank, effective_lrs)
max_delta = update_summary["model_delta_max"]
changed_name = update_summary["changed_name"]
ok = torch.tensor([float(update_summary["master_delta_max"] > 0), float(max_delta > 0)], device=f"npu:{local_rank}")
dist.all_reduce(ok)
assert ok[0].item() == world, f"Only {ok[0].item()}/{world} ranks observed a master-weight update"
assert ok[1].item() == world, f"Only {ok[1].item()}/{world} ranks observed a model-weight update; see OPTIMIZER_UPDATE_DIAGNOSTICS"
print(f"rank={rank}: PARAMETER_UPDATED delta={max_delta} name={changed_name}", flush=True)

if rank == 0:
    result = {
        "run_id": os.environ.get("RUN_ID", "manual"),
        "model_path": str(model_dir), "world_size": world,
        "tp": 2, "pp": 1, "ep": 8, "etp": 2, "cp": 1,
        "input_seq_length": seq_length, "micro_batch_size": 1,
        "num_microbatches": 1, "optimizer": "Megatron SGD", "lr": 1e-3,
        "loss": mean_loss, "grad_norm": float(grad_norm),
        "probe_name": changed_name, "probe_max_abs_delta": max_delta,
        "optimizer_update_diagnostics": update_summary,
        "local_parameter_dtype_elements": dict(dtype_counts),
        "largest_local_fp32": fp32_largest,
        "peak_npu_memory_gib": torch.npu.max_memory_allocated(local_rank) / 2**30,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "forward_backward_grad_sync_optimizer_verified": True,
        "scope": "Text-only synthetic tokens, MTP disabled, vision frozen; no checkpoint written",
    }
    (workdir / "logs" / f"{result['run_id']}-step-result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print("INTERNS2_MEGATRON_TRAIN_STEP_OK", flush=True)
dist.destroy_process_group()
