bash <<'PJLAB_UPLOAD_SUBSHELL'
set -eu
mkdir -p /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/logs
cat > '/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/train_interns2_smoke.py' <<'PJLAB_SCRIPT_0'
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
model_dir = Path("/mnt/afs-a3-weight-share/Intern-S2-Preview")
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
PJLAB_SCRIPT_0
printf '%s\n' '6433cf21d2a93b14b0ad146689519bbe8ca9535e27922765a1d548caa1dc73d5  /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/train_interns2_smoke.py' | sha256sum -c -
cat > '/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/optimizer_update_probe.py' <<'PJLAB_SCRIPT_1'
"""Observe actual master/model SGD updates without changing optimizer behavior."""

import json
import math

import torch


def scalar(value):
    return float(value.detach().item()) if torch.is_tensor(value) else float(value)


@torch.no_grad()
def active_indices(gradient, master, chunk_size=4 * 1024 * 1024):
    """Pick points using pre-step gradients; bound temporary allocations."""
    grad = gradient.detach().reshape(-1)
    weight = master.detach().reshape(-1)
    ratios, ratio_indices, magnitudes, magnitude_indices = [], [], [], []
    for start in range(0, grad.numel(), chunk_size):
        g = grad[start:start + chunk_size].float().abs()
        denominator = weight[start:start + chunk_size].float().abs().clamp_min(1e-12)
        ratio, ratio_index = (g / denominator).max(dim=0)
        magnitude, magnitude_index = g.max(dim=0)
        ratios.append(ratio)
        ratio_indices.append(ratio_index + start)
        magnitudes.append(magnitude)
        magnitude_indices.append(magnitude_index + start)
    best_ratio = torch.stack(ratios).argmax()
    best_magnitude = torch.stack(magnitudes).argmax()
    return torch.stack((torch.stack(ratio_indices)[best_ratio], torch.stack(magnitude_indices)[best_magnitude]))


@torch.no_grad()
def capture_update_points(named_params, optimizer, small_probes):
    groups = [group for group in optimizer.param_groups if group["params"]]
    rates = [scalar(group["lr"]) for group in groups]
    assert rates and all(math.isfinite(rate) and rate > 0 for rate in rates), ("Invalid effective learning rates", rates)
    owners = {id(param): group for group in groups for param in group["params"]}
    chosen = list(small_probes)
    suffixes = (
        "language_model.decoder.layers.0.self_attention.in_proj.weight",
        "language_model.decoder.layers.0.self_attention.out_proj.weight",
        "language_model.decoder.layers.0.mlp.experts.linear_fc1.weight0",
        "language_model.decoder.layers.0.mlp.experts.linear_fc2.weight0",
        "language_model.decoder.layers.3.self_attention.linear_qkv.weight",
        "language_model.decoder.layers.3.self_attention.linear_proj.weight",
        "language_model.output_layer.weight",
    )
    chosen.extend((name, param) for name, param in named_params.items() if param.requires_grad and param.ndim == 2 and name.endswith(suffixes))
    seen, snapshots = set(), []
    for name, param in chosen:
        if id(param) in seen:
            continue
        seen.add(id(param))
        grad = getattr(param, "main_grad", None)
        if grad is None:
            grad = param.grad
        if grad is None:
            continue
        master = getattr(param, "main_param", param)
        assert id(master) in owners, ("Probe parameter is not owned by optimizer", name)
        group = owners[id(master)]
        assert scalar(group.get("momentum", 0)) == 0
        assert scalar(group.get("weight_decay", 0)) == 0
        indices = torch.arange(param.numel(), device=param.device) if param.numel() <= 8192 else active_indices(grad, master)
        snapshots.append({
            "name": name, "param": param, "master": master, "indices": indices,
            "lr": scalar(group["lr"]),
            "model_before": param.detach().reshape(-1).index_select(0, indices).clone(),
            "master_before": master.detach().reshape(-1).index_select(0, indices).float().clone(),
            "grad_before": grad.detach().reshape(-1).index_select(0, indices).float().clone(),
        })
    assert snapshots, "No gradient-bearing update probes found"
    return snapshots, sorted(set(rates))


@torch.no_grad()
def verify_update_points(snapshots, grad_norm, clip_grad, rank, rates):
    norm = scalar(grad_norm)
    factor = min(1.0, clip_grad / (norm + 1e-6)) if clip_grad > 0 else 1.0
    rows = []
    for probe in snapshots:
        param, master, indices = probe["param"], probe["master"], probe["indices"]
        actual_master = master.detach().reshape(-1).index_select(0, indices).float()
        actual_model = param.detach().reshape(-1).index_select(0, indices)
        expected_master = probe["master_before"] - probe["lr"] * (probe["grad_before"] * factor)
        rows.append({
            "name": probe["name"], "dtype": str(param.dtype),
            "master_delta": (actual_master - probe["master_before"]).abs().max().item(),
            "model_delta": (actual_model.float() - probe["model_before"].float()).abs().max().item(),
            "gradient_max": probe["grad_before"].abs().max().item(),
            "master_formula_ok": torch.isclose(actual_master, expected_master, rtol=1e-6, atol=1e-8).all().item(),
            "master_formula_max_error": (actual_master - expected_master).abs().max().item(),
            "model_copy_ok": torch.equal(actual_model, actual_master.to(param.dtype)),
        })
    best = max(rows, key=lambda row: row["model_delta"])
    summary = {
        "rank": rank, "effective_lrs": rates, "grad_norm": norm, "clip_factor": factor,
        "master_delta_max": max(row["master_delta"] for row in rows),
        "model_delta_max": best["model_delta"], "changed_name": best["name"],
        "master_formula_and_model_copy_checked": all(row["master_formula_ok"] and row["model_copy_ok"] for row in rows),
        "failed_checks": [row for row in rows if not (row["master_formula_ok"] and row["model_copy_ok"])],
        "probes": len(rows), "largest_model_changes": sorted(rows, key=lambda row: row["model_delta"], reverse=True)[:5],
    }
    print("OPTIMIZER_UPDATE_DIAGNOSTICS:", json.dumps(summary), flush=True)
    assert summary["master_formula_and_model_copy_checked"], "SGD update or master-to-model copy mismatch; see OPTIMIZER_UPDATE_DIAGNOSTICS"
    return summary
PJLAB_SCRIPT_1
printf '%s\n' '62a2be06db172d1964b8bcb262bb9277c1ed1c9c321f91574c599a7672ae5b08  /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/optimizer_update_probe.py' | sha256sum -c -
cat > '/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/run_interns2_smoke.sh' <<'PJLAB_SCRIPT_2'
#!/usr/bin/env bash
set -eo pipefail

export WORKDIR=/mnt/afs-ptdata-shared/sxr/dltrain-mcore018
export LD_LIBRARY_PATH="/usr/local/python3.12.13/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export QWEN35_DEBUG=0
export OMP_NUM_THREADS=1
export RUN_ID="interns2-step-$(date -u +%Y%m%dT%H%M%SZ)"

source "$WORKDIR/.venv/bin/activate"
source /usr/local/Ascend/cann/set_env.sh
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

python -m py_compile "$WORKDIR/train_interns2_smoke.py" "$WORKDIR/optimizer_update_probe.py"
python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=16 \
    "$WORKDIR/train_interns2_smoke.py" \
    2>&1 | tee "$WORKDIR/logs/${RUN_ID}.log"
PJLAB_SCRIPT_2
printf '%s\n' '23e7f8eb1a26eb7179fec71a66da6609e2aefd441ef336139bb4bb55e2e64247  /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/run_interns2_smoke.sh' | sha256sum -c -
printf '%s\n' 'PJLAB_SCRIPTS_READY'
PJLAB_UPLOAD_SUBSHELL
