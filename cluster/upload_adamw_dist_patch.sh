bash <<'PJLAB_ADAMW_DIST_UPLOAD' || printf '%s\n' 'ADAMW_DIST_UPLOAD_FAILED'
set -eu
mkdir -p /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/logs
cat > '/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/train_interns2_adamw_dist.py' <<'PJLAB_ADAMW_DIST_FILE_0'
"""Measured text-only training iterations; every plotted point comes from JSONL metrics."""

import argparse
import hashlib
import importlib.metadata
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import json
import math
import os
import runpy
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from safetensors import safe_open
from adamw_dist_update_probe import capture_update_points, verify_update_points
from adamw_math import scheduled_lr
from adamw_shard_layout import owned_parameter_shards

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--seq-length", type=int, default=64)
parser.add_argument("--num-microbatches", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--min-lr", type=float, default=1e-6)
parser.add_argument("--warmup-steps", type=int, default=20)
parser.add_argument("--schedule-steps", type=int, default=500)
parser.add_argument("--run-role", choices=("validation", "measurement"), default="measurement")
parser.add_argument("--data-mode", choices=("text", "synthetic"), default="text")
parser.add_argument("--corpus-dir", type=Path)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--seed", type=int, default=20261125)
args = parser.parse_args()
assert 1 <= args.steps <= args.schedule_steps
scheduled_lr(1, args.lr, args.min_lr, args.warmup_steps, args.schedule_steps)
assert args.steps > 0 and args.seq_length >= 16 and args.seq_length % 2 == 0
assert args.num_microbatches > 0 and math.isfinite(args.lr) and args.lr > 0
if args.data_mode == "text" and args.corpus_dir is None:
    parser.error("text mode requires --corpus-dir prepared by prepare_interns2_corpus.py")

workdir = Path(os.environ["WORKDIR"])
model_dir = Path("/mnt/afs-a3-weight-share/Intern-S2-Preview")
local_rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(local_rank)
dist.init_process_group("hccl", timeout=timedelta(minutes=15))
rank, world = dist.get_rank(), dist.get_world_size()
assert world == 16
state = runpy.run_path(str(workdir / "check_interns2_provider.py"))
provider = state["provider"]
provider.seq_length = args.seq_length
provider.language_max_sequence_length = args.seq_length
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
    overlap_param_gather=False,
    use_distributed_optimizer=True, check_for_nan_in_grad=True,
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

# All TP ranks in one DP replica consume identical tokens; EP partitions experts,
# not the global batch. With TP2/PP1/CP1 on 16 ranks, dense DP is 8.
dp_rank, dp_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
global_batch = dp_size * args.num_microbatches  # micro batch is fixed to 1
vocab_size = int(getattr(provider, "vocab_size", 251392))
corpus, corpus_meta = None, None
if args.data_mode == "text":
    corpus_meta = json.loads((args.corpus_dir / "tokens.meta.json").read_text(encoding="utf-8"))
    binary = args.corpus_dir / "tokens.bin"
    assert binary.stat().st_size == 4 * corpus_meta["num_tokens"]
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    assert digest == corpus_meta["tokens_sha256"], "Token corpus checksum mismatch"
    corpus = np.memmap(binary, mode="r", dtype="<u4")
    assert len(corpus) >= global_batch * args.seq_length + 1, "Corpus too short for one distinct global batch"
    assert int(corpus.max()) < vocab_size, "Corpus token exceeds model vocabulary"

optimizer_config = OptimizerConfig(
    optimizer="adam", lr=args.lr, weight_decay=0.01, decoupled_weight_decay=True,
    adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8, use_precision_aware_optimizer=False,
    bf16=True, fp16=False, params_dtype=torch.bfloat16,
    use_distributed_optimizer=True, clip_grad=1.0,
)
optimizer = get_megatron_optimizer(optimizer_config, models, use_gloo_process_groups=False)
local_shards = owned_parameter_shards(named_params, optimizer)
optimizer_layout = {
    "rank": rank, "use_distributed_optimizer": True,
    "dense_dp_size": mpu.get_data_parallel_world_size(),
    "expert_dp_size": mpu.get_expert_data_parallel_world_size(),
    "local_master_elements": sum(shard["master"].numel() for shard in local_shards),
    "local_dense_master_elements": sum(shard["master"].numel() for shard in local_shards if getattr(shard["param"], "allreduce", True)),
    "local_expert_master_elements": sum(shard["master"].numel() for shard in local_shards if not getattr(shard["param"], "allreduce", True)),
    "optimizer_types": sorted({type(shard["leaf"]).__name__ for shard in local_shards}),
}
print("DISTRIBUTED_OPTIMIZER_LAYOUT:", json.dumps(optimizer_layout), flush=True)
del local_shards
config = get_model_config(model)
config.grad_scale_func = optimizer.scale_loss
config.finalize_model_grads_func = finalize_model_grads
config.no_sync_func = model.no_sync
config.deallocate_pipeline_outputs = False
schedule = get_forward_backward_func()
small_probes = sorted(
    ((name, param) for name, param in named_params.items() if param.requires_grad and "language_model" in name and 0 < param.numel() <= 8192),
    key=lambda pair: ("decoder.final_layernorm" not in pair[0], pair[0]),
)[:128]
generator = torch.Generator(device="cpu").manual_seed(args.seed + dp_rank)


def batches_for_step(step):
    for micro in range(args.num_microbatches):
        if corpus is None:
            values = torch.randint(100, min(20000, vocab_size), (1, args.seq_length + 1), generator=generator, device="cpu", dtype=torch.long)
        else:
            sample_number = (step - 1) * global_batch + micro * dp_size + dp_rank
            indices = (sample_number * args.seq_length + np.arange(args.seq_length + 1, dtype=np.int64)) % len(corpus)
            values = torch.from_numpy(np.asarray(corpus[indices], dtype=np.int64).copy()).unsqueeze(0)
        values = values.to(f"npu:{local_rank}")
        yield values[:, :-1].contiguous(), values[:, 1:].contiguous()


def forward_step(data_iterator, wrapped_model):
    tokens, labels = next(data_iterator)
    loss_mask = torch.ones_like(labels, dtype=torch.float32)
    losses = wrapped_model(input_ids=tokens, labels=labels, attention_mask=None, loss_mask=loss_mask, extra_block_kwargs={})
    assert tuple(losses.shape) == tuple(labels.shape)

    def loss_func(output):
        total = (output.float() * loss_mask).sum()
        count = loss_mask.sum().to(torch.int32)
        return total, count, {"lm_loss": torch.stack((total.detach(), count.float()))}

    return losses, loss_func


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def package_versions():
    versions = {}
    for package in ("torch", "torch-npu", "transformers", "numpy", "megatron-core", "megatron-adaptor", "transformer-engine", "triton-ascend", "nvidia-modelopt"):
        try:
            versions[package] = importlib.metadata.distribution(package).version
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed-as-distribution"
    return versions


def source_versions():
    versions = {}
    for name in ("Megatron-LM", "MegatronAdaptor", "TransformerEngineNPU-2.17", "Megatron-Bridge-interns2"):
        path = workdir / "src" / name
        if (path / ".git").exists():
            command = ["/usr/bin/git", "-c", f"safe.directory={path}", "-C", str(path)]
            commit = subprocess.check_output(command + ["rev-parse", "HEAD"], text=True).strip()
            changes = subprocess.check_output(command + ["diff", "HEAD", "--binary"])
            (args.run_dir / f"{name}-changes.patch").write_bytes(changes)
            versions[name] = {"commit": commit, "changes_sha256": hashlib.sha256(changes).hexdigest()}
        elif (path / "source-commit.txt").exists():
            versions[name] = {"source_commit": (path / "source-commit.txt").read_text().strip()}
    return versions


if rank == 0:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    assert not (args.run_dir / "metrics.jsonl").exists(), "Refusing to append a new training run to existing metrics"
    script_hashes = {}
    for name in (Path(__file__).name, "adamw_dist_update_probe.py", "adamw_shard_layout.py", "adamw_math.py", "optimizer_update_probe.py", "check_interns2_provider.py", "run_interns2_adamw_dist.sh", "plot_interns2_adamw.py"):
        path = workdir / name
        if path.is_file():
            script_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            (args.run_dir / name).write_bytes(path.read_bytes())
    bridge_root = workdir / "src/Megatron-Bridge-interns2/src/megatron/bridge"
    bridge_hashes = {}
    for path in sorted(bridge_root.rglob("*.py")):
        bridge_hashes[str(path.relative_to(bridge_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1, "run_id": args.run_dir.name, "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_steps": args.steps, "model_name": "Intern-S2-Preview", "model_path": str(model_dir),
        "num_layers": provider.num_layers, "num_experts": provider.num_moe_experts,
        "world_size": world, "tp": 2, "pp": 1, "ep": 8, "etp": 2, "cp": 1, "dp": dp_size,
        "seq_length": args.seq_length, "micro_batch_size": 1, "num_microbatches": args.num_microbatches,
        "global_batch_size": global_batch, "tokens_per_step": global_batch * args.seq_length,
        "optimizer": "Megatron distributed AdamW", "use_distributed_optimizer": True,
        "optimizer_layout_rank0": optimizer_layout, "overlap_param_gather": False, "lr": args.lr, "weight_decay": 0.01, "clip_grad": 1.0,
        "adam_betas": [0.9, 0.999], "adam_eps": 1e-8, "run_role": args.run_role,
        "lr_schedule": {"type": "linear_warmup_cosine", "peak_lr": args.lr, "min_lr": args.min_lr, "warmup_steps": args.warmup_steps, "schedule_steps": args.schedule_steps, "convention": "step 1 uses peak_lr/warmup_steps; final schedule step uses min_lr"},
        "dtype": "bfloat16", "seed": args.seed, "data_mode": args.data_mode,
        "corpus": corpus_meta, "held_out_evaluation": False,
        "checkpoint_saved": False, "mtp_num_layers": 0, "vision_frozen": True,
        "step_time_definition": "maximum rank duration of synchronized input/zero-grad/forward/backward/gradient-sync/AdamW; excludes update audit and metric aggregation; excludes initialization/loading",
        "timing_warmup_steps_omitted": 4, "gradient_norm_definition": "MCore optimizer global L2 norm before clip_grad=1; not an auxiliary loss",
        "python": sys.version, "platform": platform.platform(), "packages": package_versions(),
        "source_versions": source_versions(), "script_sha256": script_hashes,
        "bridge_python_sha256": bridge_hashes, "local_parameter_dtype_elements_rank0": dict(dtype_counts),
        "model_config_sha256": hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest(),
        "model_index_sha256": hashlib.sha256((model_dir / "model.safetensors.index.json").read_bytes()).hexdigest(),
        "formal_checkpoint_revision_verified": False,
        "warnings_suppressed": False,
    }
    write_json(args.run_dir / "manifest.json", manifest)
    print("MEASUREMENT_RUN:", json.dumps({k: manifest[k] for k in ("run_id", "requested_steps", "data_mode", "global_batch_size", "seq_length")}), flush=True)
    metrics_file = (args.run_dir / "metrics.jsonl").open("x", encoding="utf-8", buffering=1)

torch.npu.synchronize()
dist.barrier()
torch.npu.reset_peak_memory_stats(local_rank)
times, losses_recorded, norms = [], [], []
first_update_audit = None
run_started = time.perf_counter()
try:
    for step in range(1, args.steps + 1):
        torch.npu.synchronize()
        dist.barrier()
        started = time.perf_counter()
        step_lr = scheduled_lr(step, args.lr, args.min_lr, args.warmup_steps, args.schedule_steps)
        for group in optimizer.param_groups:
            group["lr"] = step_lr * group.get("lr_mult", 1.0)
            group["weight_decay"] = optimizer_config.weight_decay * group.get("wd_mult", 1.0)
        model.zero_grad_buffer()
        optimizer.zero_grad()
        reports = schedule(
            forward_step_func=forward_step, data_iterator=iter(batches_for_step(step)), model=models,
            num_microbatches=args.num_microbatches, seq_length=args.seq_length,
            micro_batch_size=1, forward_only=False,
        )
        # The first-step audit preserves the AdamW state/formula/copy checks.
        # Its device synchronizations and scans are excluded from step timing.
        audit_seconds = 0.0
        if step == 1:
            torch.npu.synchronize()
            audit_started = time.perf_counter()
            update_points, effective_lrs = capture_update_points(named_params, optimizer, small_probes)
            torch.npu.synchronize()
            audit_seconds = time.perf_counter() - audit_started
        success, grad_norm, _ = optimizer.step()
        torch.npu.synchronize()
        elapsed = time.perf_counter() - started - audit_seconds
        assert success, f"Optimizer rejected step {step}"
        assert grad_norm is not None and math.isfinite(float(grad_norm)) and float(grad_norm) > 0, (step, grad_norm)

        sums = torch.stack([report["lm_loss"] for report in reports]).sum(dim=0).detach()
        dist.all_reduce(sums, group=mpu.get_data_parallel_group())
        mean_loss, token_count = (sums[0] / sums[1]).item(), int(sums[1].item())
        assert math.isfinite(mean_loss) and mean_loss >= 0, (step, mean_loss)
        assert token_count == global_batch * args.seq_length, (token_count, global_batch, args.seq_length)
        step_stats = torch.tensor([elapsed, torch.npu.max_memory_allocated(local_rank) / 2**30, torch.npu.max_memory_reserved(local_rank) / 2**30], dtype=torch.float32, device=f"npu:{local_rank}")
        dist.all_reduce(step_stats, op=dist.ReduceOp.MAX)
        step_seconds, peak_allocated, peak_reserved = step_stats.tolist()
        if step == 1:
            first_update_audit = verify_update_points(update_points, grad_norm, optimizer_config.clip_grad, rank, effective_lrs)
            ok = torch.tensor([float(first_update_audit["master_delta_max"] > 0), float(first_update_audit["model_delta_max"] > 0)], device=f"npu:{local_rank}")
            dist.all_reduce(ok)
            assert ok.tolist() == [float(world), float(world)], "Not all ranks observed the first parameter update"
            if rank == 0:
                write_json(args.run_dir / "first-step-update-audit-rank0.json", first_update_audit)
            del update_points

        times.append(step_seconds)
        losses_recorded.append(mean_loss)
        norms.append(float(grad_norm))
        row = {
            "step": step, "lm_loss": mean_loss, "grad_norm_before_clip": float(grad_norm),
            "step_seconds_max_rank": step_seconds, "lr": step_lr,
            "optimizer_update_success": bool(success), "tokens": token_count,
            "tokens_per_second": token_count / step_seconds,
            "peak_allocated_gib_max_rank": peak_allocated, "peak_reserved_gib_max_rank": peak_reserved,
            "elapsed_wall_seconds": time.perf_counter() - run_started,
            "data_tokens_consumed": step * token_count,
            "corpus_passes_equivalent": step * token_count / len(corpus) if corpus is not None else None,
        }
        if rank == 0:
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            if step % 10 == 0 or step == args.steps:
                os.fsync(metrics_file.fileno())
            print(f"ITER {step}/{args.steps} loss={mean_loss:.6f} grad_norm={float(grad_norm):.6f} lr={step_lr:.8g} step_s={step_seconds:.3f} tokens_s={row['tokens_per_second']:.1f} peak_GiB={peak_allocated:.2f}", flush=True)
        del reports, sums, step_stats

    local_total_gib = torch.npu.get_device_properties(local_rank).total_memory / 2**30
    local_peak_gib = torch.npu.max_memory_allocated(local_rank) / 2**30
    local_headroom_gib = local_total_gib - local_peak_gib
    headroom = torch.tensor([local_headroom_gib], device=f"npu:{local_rank}")
    dist.all_reduce(headroom, op=dist.ReduceOp.MIN)
    if args.run_role == "validation":
        assert headroom.item() >= 2.0, f"AdamW validation leaves only {headroom.item():.2f} GiB of peak allocated memory headroom"
    finished = torch.tensor([args.steps], dtype=torch.int32, device=f"npu:{local_rank}")
    dist.all_reduce(finished, op=dist.ReduceOp.MIN)
    assert finished.item() == args.steps
    if rank == 0:
        metrics_file.close()
        measured_times = times[4:]
        result = {
            "schema_version": 1, "run_id": args.run_dir.name, "status": "passed",
            "completed_steps": args.steps, "requested_steps": args.steps,
            "initial_lm_loss": losses_recorded[0], "final_lm_loss": losses_recorded[-1],
            "minimum_lm_loss": min(losses_recorded), "maximum_grad_norm_before_clip": max(norms),
            "median_step_seconds_after_warmup": statistics.median(measured_times) if measured_times else None,
            "timing_samples_after_warmup": len(measured_times),
            "first_step_updates_verified_on_all_ranks": True,
            "run_role": args.run_role, "min_peak_allocated_headroom_gib_all_ranks": headroom.item(),
            "peak_allocated_gib_max_rank": peak_allocated, "peak_reserved_gib_max_rank": peak_reserved,
            "wall_seconds_excluding_initialization": time.perf_counter() - run_started,
            "scope": "training loss only; no held-out quality claim; no checkpoint saved",
        }
        write_json(args.run_dir / "summary.json", result)
        manifest["status"] = "passed"
        write_json(args.run_dir / "manifest.json", manifest)
        print("INTERNS2_MEASURED_TRAINING_OK:", json.dumps(result), flush=True)
except BaseException as error:
    if rank == 0:
        metrics_file.close()
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        write_json(args.run_dir / "manifest.json", manifest)
    raise
finally:
    dist.destroy_process_group()

PJLAB_ADAMW_DIST_FILE_0
cat > '/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/adamw_dist_update_probe.py' <<'PJLAB_ADAMW_DIST_FILE_1'
"""Check AdamW state, FP32 shard updates, and their BF16 model ranges after all-gather."""

import json
import math

import torch

from adamw_math import first_step_expectation
from adamw_shard_layout import choose_probe_shards, owned_parameter_shards
from optimizer_update_probe import active_indices, scalar


@torch.no_grad()
def capture_update_points(named_params, optimizer, small_probes):
    shards = owned_parameter_shards(named_params, optimizer)
    rates = sorted({scalar(shard["group"]["lr"]) for shard in shards})
    assert rates and all(math.isfinite(rate) and rate > 0 for rate in rates)
    snapshots = []
    for shard in choose_probe_shards(shards, small_probes):
        param, master = shard["param"], shard["master"]
        start, end = shard["param_start"], shard["param_end"]
        assert param.main_grad is not None, shard["name"]
        # Reduce-scatter has finished. Only this owned gradient range is valid.
        grad = param.main_grad.detach().view(-1)[start:end]
        model_view = param.detach().view(-1)[start:end]
        group, base = shard["group"], shard["base"]
        state = base.state.get(master, {})
        assert scalar(state.get("step", group.get("step", 0))) == 0, "Requires a fresh optimizer"
        indices = torch.arange(master.numel(), device=param.device) if master.numel() <= 8192 else active_indices(grad, master)
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                assert state[key].reshape(-1).index_select(0, indices).count_nonzero().item() == 0
        snapshots.append({
            **shard, "indices": indices, "lr": scalar(group["lr"]),
            "betas": group["betas"], "eps": scalar(group["eps"]), "weight_decay": scalar(group.get("weight_decay", 0)),
            "model_before": model_view.index_select(0, indices).clone(),
            "master_before": master.detach().reshape(-1).index_select(0, indices).float().clone(),
            "grad_before": grad.index_select(0, indices).float().clone(),
        })
    return snapshots, rates


@torch.no_grad()
def verify_update_points(snapshots, grad_norm, clip_grad, rank, rates):
    norm = scalar(grad_norm)
    factor = min(1.0, clip_grad / (norm + 1e-6)) if clip_grad > 0 else 1.0
    rows = []
    for probe in snapshots:
        param, master, indices = probe["param"], probe["master"], probe["indices"]
        actual_master = master.detach().reshape(-1).index_select(0, indices).float()
        model_view = param.detach().view(-1)[probe["param_start"]:probe["param_end"]]
        actual_model = model_view.index_select(0, indices)
        state = probe["base"].state[master]
        actual_m = state["exp_avg"].detach().reshape(-1).index_select(0, indices).float()
        actual_v = state["exp_avg_sq"].detach().reshape(-1).index_select(0, indices).float()
        expected_master, expected_m, expected_v = first_step_expectation(
            probe["master_before"], probe["grad_before"], probe["lr"],
            *probe["betas"], probe["eps"], probe["weight_decay"], factor,
        )
        rows.append({
            "name": probe["name"], "dtype": str(param.dtype),
            "owned_model_range": [probe["param_start"], probe["param_end"]],
            "master_delta": (actual_master - probe["master_before"]).abs().max().item(),
            "model_delta": (actual_model.float() - probe["model_before"].float()).abs().max().item(),
            "gradient_max": probe["grad_before"].abs().max().item(),
            "master_formula_ok": torch.isclose(actual_master, expected_master, rtol=1e-6, atol=2e-9).all().item(),
            "moment1_ok": torch.isclose(actual_m, expected_m, rtol=2e-4, atol=1e-12).all().item(),
            "moment2_ok": torch.isclose(actual_v, expected_v, rtol=3e-4, atol=1e-16).all().item(),
            "model_copy_ok": torch.equal(actual_model, actual_master.to(param.dtype)),
            "optimizer_step": scalar(state.get("step", probe["group"].get("step", -1))),
        })
    checks = ("master_formula_ok", "moment1_ok", "moment2_ok", "model_copy_ok")
    failed = [row for row in rows if not all(row[key] for key in checks) or row["optimizer_step"] != 1]
    best = max(rows, key=lambda row: row["model_delta"])
    summary = {
        "rank": rank, "optimizer": "MCore DistributedOptimizer AdamW", "effective_lrs": rates,
        "grad_norm": norm, "clip_factor": factor,
        "master_delta_max": max(row["master_delta"] for row in rows), "model_delta_max": best["model_delta"],
        "changed_name": best["name"], "master_formula_and_model_copy_checked": not failed,
        "failed_checks": failed, "probes": len(rows), "probe_scope": "owned DP ranges after synchronous parameter all-gather",
        "largest_model_changes": sorted(rows, key=lambda row: row["model_delta"], reverse=True)[:5],
    }
    print("ADAMW_DISTRIBUTED_UPDATE_DIAGNOSTICS:", json.dumps(summary), flush=True)
    assert not failed, "AdamW state/update/copy check failed; see ADAMW_DISTRIBUTED_UPDATE_DIAGNOSTICS"
    return summary
PJLAB_ADAMW_DIST_FILE_1
cat > '/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/adamw_shard_layout.py' <<'PJLAB_ADAMW_DIST_FILE_2'
"""Describe MCore-owned parameter ranges without assuming a full FP32 master copy."""


def owned_parameter_shards(named_params, optimizer):
    names = {id(param): name for name, param in named_params.items()}
    shards, seen = [], set()
    for leaf in getattr(optimizer, "chained_optimizers", [optimizer]):
        assert hasattr(leaf, "_get_model_param_range_map"), "Expected MCore DistributedOptimizer"
        assert not leaf.config.use_precision_aware_optimizer, "Probe requires explicit FP32 masters"
        base = leaf.optimizer
        if hasattr(base, "adam_w_mode"):
            assert base.adam_w_mode, "Expected decoupled AdamW weight decay"
        owners = {id(master): group for group in base.param_groups for master in group["params"]}
        model_groups = leaf.model_float16_groups
        master_groups = leaf.shard_fp32_from_float16_groups
        assert len(model_groups) == len(master_groups)
        assert not any(leaf.model_fp32_groups), "Expected all model parameters to be BF16"
        for model_group, master_group in zip(model_groups, master_groups):
            assert len(model_group) == len(master_group)
            for param, master in zip(model_group, master_group):
                assert master is not None and id(master) in owners
                assert id(param) in names and id(param) not in seen
                seen.add(id(param))
                owned = leaf._get_model_param_range_map(param)["param"]
                start, end = int(owned.start), int(owned.end)
                assert 0 <= start < end <= param.numel(), (names[id(param)], start, end)
                assert end - start == master.numel(), "Master size must match the owned model range"
                assert getattr(param, "main_param", None) is master
                shards.append({
                    "name": names[id(param)], "param": param, "master": master,
                    "base": base, "group": owners[id(master)], "leaf": leaf,
                    "param_start": start, "param_end": end,
                })
    assert shards, "No local optimizer shards"
    return shards


def choose_probe_shards(shards, small_probes):
    requested = {id(param) for _, param in small_probes}
    suffixes = (
        "language_model.decoder.layers.0.mlp.experts.linear_fc1.weight0",
        "language_model.decoder.layers.0.mlp.experts.linear_fc2.weight0",
        "language_model.decoder.layers.3.self_attention.linear_qkv.weight",
        "language_model.decoder.layers.3.self_attention.linear_proj.weight",
        "language_model.output_layer.weight",
    )
    chosen, seen = [], set()
    # Some dense DP ranks own none of the originally requested parameters.
    # Audit two owned matrices per leaf as well, so each optimizer is checked.
    fallback_counts = {}
    for shard in shards:
        param = shard["param"]
        key = id(shard["leaf"])
        fallback = param.ndim == 2 and "language_model" in shard["name"] and fallback_counts.get(key, 0) < 2
        if fallback:
            fallback_counts[key] = fallback_counts.get(key, 0) + 1
        if id(param) in requested or shard["name"].endswith(suffixes) or fallback:
            if id(param) not in seen:
                seen.add(id(param))
                chosen.append(shard)
    assert chosen, "No owned probe parameters"
    assert {id(shard["leaf"]) for shard in chosen} == {id(shard["leaf"]) for shard in shards}, "Every local optimizer must have a probe"
    return chosen
PJLAB_ADAMW_DIST_FILE_2
cat > '/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/adamw_math.py' <<'PJLAB_ADAMW_DIST_FILE_3'
"""Explicit AdamW schedule and first-step expectations; works with scalar/array tensors."""

import math


def scheduled_lr(step, peak_lr, min_lr, warmup_steps=20, schedule_steps=500):
    if not (1 <= step <= schedule_steps and 0 <= warmup_steps < schedule_steps):
        raise ValueError("Invalid optimizer step or schedule length")
    if not (math.isfinite(peak_lr) and math.isfinite(min_lr) and 0 < min_lr <= peak_lr):
        raise ValueError("Invalid peak/minimum learning rate")
    if warmup_steps and step <= warmup_steps:
        return peak_lr * step / warmup_steps
    progress = (step - warmup_steps) / (schedule_steps - warmup_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


def first_step_expectation(weight, raw_grad, lr, beta1, beta2, eps, weight_decay, clip_factor):
    """Fresh optimizer, bias correction enabled, epsilon outside the square root."""
    grad = raw_grad * clip_factor
    moment1 = (1 - beta1) * grad
    moment2 = (1 - beta2) * grad * grad
    corrected1 = moment1 / (1 - beta1)
    corrected2 = moment2 / (1 - beta2)
    updated = weight * (1 - lr * weight_decay) - lr * corrected1 / (corrected2 ** 0.5 + eps)
    return updated, moment1, moment2
PJLAB_ADAMW_DIST_FILE_3
cat > '/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/optimizer_update_probe.py' <<'PJLAB_ADAMW_DIST_FILE_4'
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
PJLAB_ADAMW_DIST_FILE_4
cat > '/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/plot_interns2_adamw.py' <<'PJLAB_ADAMW_DIST_FILE_5'
"""Plot measured training records; never interpolate missing steps or infer success from loss."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import warnings
from pathlib import Path
from adamw_math import scheduled_lr


def load_run(run):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    requested = int(manifest["requested_steps"])
    if requested <= 0:
        raise ValueError("requested_steps must be positive")
    lines = (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    rows = []
    fields = ("lm_loss", "grad_norm_before_clip", "step_seconds_max_rank", "lr", "tokens")
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                warnings.warn("Ignoring a truncated final JSONL record; run cannot pass with missing records")
                break
            raise ValueError(f"Invalid metrics JSON at line {index + 1}") from None
        if row.get("step") != len(rows) + 1:
            raise ValueError("Steps must be contiguous, ordered, and unique starting at 1")
        if row["step"] > requested:
            raise ValueError("More measured steps than the requested run length")
        for field in fields:
            if not isinstance(row.get(field), (int, float)) or not math.isfinite(row[field]):
                raise ValueError(f"Non-finite/missing {field} at step {row['step']}")
        if row["lm_loss"] < 0 or min(row[field] for field in fields[1:]) <= 0:
            raise ValueError(f"Invalid loss, gradient, duration, learning rate or tokens at step {row['step']}")
        if row.get("optimizer_update_success") is not True:
            raise ValueError(f"Unsuccessful optimizer update at step {row['step']}")
        schedule = manifest["lr_schedule"]
        expected_lr = scheduled_lr(row["step"], schedule["peak_lr"], schedule["min_lr"], schedule["warmup_steps"], schedule["schedule_steps"])
        if row["tokens"] != manifest["tokens_per_step"] or not math.isclose(row["lr"], expected_lr, rel_tol=1e-12, abs_tol=1e-16):
            raise ValueError("Per-step batch tokens or scheduled learning rate differs from the recorded manifest")
        rows.append(row)
    if not rows:
        raise ValueError("No complete measured steps found; refusing to fabricate a curve")
    summary_path, exit_path = run / "summary.json", run / "launcher-exit.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    launcher = json.loads(exit_path.read_text(encoding="utf-8")) if exit_path.exists() else {}
    passed = (
        len(rows) == requested and manifest.get("status") == "passed"
        and summary.get("status") == "passed" and summary.get("completed_steps") == requested
        and summary.get("first_step_updates_verified_on_all_ranks") is True
        and launcher.get("exit_code") == 0
    )
    status = "passed" if passed else "incomplete / not passed"
    omitted = int(manifest.get("timing_warmup_steps_omitted", 4))
    if omitted < 0:
        raise ValueError("Negative warmup count")
    timings = [row["step_seconds_max_rank"] for row in rows if row["step"] > omitted]
    report = {
        "run_id": manifest["run_id"], "status": status,
        "observed_steps": len(rows), "requested_steps": requested,
        "timing_warmup_steps_omitted": omitted, "timing_sample_count": len(timings),
        "median_step_seconds": statistics.median(timings) if timings else None,
        "initial_lm_loss": rows[0]["lm_loss"], "final_lm_loss": rows[-1]["lm_loss"],
        "max_grad_norm_before_clip": max(row["grad_norm_before_clip"] for row in rows),
        "metrics_sha256": hashlib.sha256((run / "metrics.jsonl").read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest(),
        "loss_smoothing": "none", "missing_steps_filled": False,
    }
    return manifest, rows, report


def plot_run(run, output_dir):
    manifest, rows, report = load_run(run)
    # Imported only when plotting. Training on the cluster does not need matplotlib.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    steps = [row["step"] for row in rows]
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(13, 11))
    axes[0].plot(steps, [row["lm_loss"] for row in rows], color="#2768b2", linewidth=1.25)
    axes[0].set_ylabel("Training LM loss")
    axes[1].plot(steps, [row["grad_norm_before_clip"] for row in rows], color="#a85ace", linewidth=1.0)
    axes[1].set_ylabel("Gradient norm (before clipping)")
    omitted = report["timing_warmup_steps_omitted"]
    timed = [row for row in rows if row["step"] > omitted]
    if timed:
        axes[2].plot([row["step"] for row in timed], [row["step_seconds_max_rank"] for row in timed], color="#267c70", linewidth=1.0, label=f"Max rank step time (first {omitted} omitted)")
        median = report["median_step_seconds"]
        axes[2].axhline(median, color="#666666", linestyle="--", linewidth=1, label=f"Median: {median:.3f} s")
        axes[2].legend(loc="upper left", fontsize=10)
    else:
        axes[2].text(0.5, 0.5, "No timing samples after warmup", transform=axes[2].transAxes, ha="center")
    axes[2].set_ylabel("Seconds / optimizer step")
    axes[2].set_xlabel("Optimizer iteration")
    for axis in axes:
        axis.grid(True, alpha=0.2)
        axis.tick_params(direction="out")
    if len(steps) == 1:
        axes[2].set_xlim(0, max(2, steps[-1] + 1))
    title = f"{manifest['model_name']} L{manifest['num_layers']} / E{manifest['num_experts']} on Ascend A3 - {len(rows)}/{report['requested_steps']} observed steps"
    layout = f"TP{manifest['tp']} | PP{manifest['pp']} | EP{manifest['ep']} | ETP{manifest['etp']} | CP{manifest['cp']} | DP{manifest['dp']} | SEQ{manifest['seq_length']} | GBS{manifest['global_batch_size']}"
    fig.suptitle(title + "\n" + layout + f"\n{manifest['optimizer']} | peak lr={manifest['lr']:g}, warmup+cosine | BF16 | {manifest['world_size']} logical NPUs", y=0.985, fontsize=13)
    data_label = "real text, native model tokenizer" if manifest["data_mode"] == "text" else "SYNTHETIC RANDOM TOKENS - stability test"
    footer = manifest.get("run_role", "measurement") + "; " + data_label + "; training loss only, no held-out quality evaluation. Status: " + report["status"]
    if manifest["data_mode"] == "text":
        footer += f"\nCorpus tokens: {manifest['corpus']['num_tokens']:,}; EOS-joined documents; repeated cyclically when exhausted."
    fig.text(0.5, 0.022, footer, ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.095, right=0.98, hspace=0.07)
    fig.savefig(output_dir / "report.png", dpi=180)
    fig.savefig(output_dir / "report.svg")
    plt.close(fig)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "plot-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("MEASURED_REPORT:", output_dir / "report.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    plot_run(args.run_dir, args.output_dir or args.run_dir)
PJLAB_ADAMW_DIST_FILE_5
cat > '/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/run_interns2_adamw_dist.sh' <<'PJLAB_ADAMW_DIST_FILE_6'
#!/usr/bin/env bash
set -eo pipefail
export WORKDIR=/mnt/afs-ptdata-shared/sxr/dltrain-mcore018
export LD_LIBRARY_PATH="/usr/local/python3.12.13/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export QWEN35_DEBUG=0
export OMP_NUM_THREADS=1
export ADAMW_JOB_ID="${ADAMW_JOB_ID:-interns2-redpajama-adamw-dist-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
export CORPUS_DIR="$WORKDIR/datasets/interns2-redpajama-500"
if [[ ! "$ADAMW_JOB_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
    printf '%s\n' 'Invalid ADAMW_JOB_ID' >&2
    exit 2
fi
mkdir -p "$WORKDIR/runs" "$WORKDIR/logs"
source "$WORKDIR/.venv/bin/activate"
source /usr/local/Ascend/cann/set_env.sh
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

write_bundle() {
    export ADAMW_RUN_EXIT_CODE="$1"
    python - <<'PY'
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
run = Path(os.environ["CURRENT_RUN_DIR"])
(run / "launcher-exit.json").write_text(json.dumps({"exit_code": int(os.environ["ADAMW_RUN_EXIT_CODE"])}) + "\n", encoding="utf-8")
with ZipFile(run.with_suffix(".zip"), "w", ZIP_DEFLATED) as bundle:
    for path in sorted(run.rglob("*")):
        if path.is_file():
            bundle.write(path, arcname=str(path.relative_to(run.parent)))
print("ADAMW_RESULT_BUNDLE:", run.with_suffix(".zip"), flush=True)
PY
}

finish_job() {
    pjlab_status=$?
    trap - EXIT
    if [ -n "${CURRENT_RUN_DIR:-}" ]; then
        npu-smi info > "$CURRENT_RUN_DIR/npu-smi-end.txt" 2>&1 || true
        write_bundle "$pjlab_status" || true
    fi
    printf 'ADAMW_PIPELINE_EXIT_CODE=%s\n' "$pjlab_status"
    exit "$pjlab_status"
}
trap finish_job EXIT

python -m py_compile "$WORKDIR/train_interns2_adamw_dist.py" \
    "$WORKDIR/adamw_dist_update_probe.py" "$WORKDIR/adamw_shard_layout.py" "$WORKDIR/adamw_math.py" \
    "$WORKDIR/plot_interns2_adamw.py"

python - <<'PY'
import json
import os
import time
from pathlib import Path
root = Path(os.environ["CORPUS_DIR"])
assert (root / "tokens.bin").is_file() and (root / "tokens.meta.json").is_file(), "Prepare RedPajama first"
meta = json.loads((root / "tokens.meta.json").read_text())
assert meta["num_tokens"] >= 256001
names = {"train_interns2_500.py", "train_interns2_smoke.py", "load_interns2_weights.py", "train_interns2_adamw.py", "train_interns2_adamw_dist.py"}
while True:
    busy = []
    for path in Path("/proc").glob("[0-9]*"):
        try:
            args = (path / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if any(Path(arg).name in names for arg in args if arg):
            busy.append(path.name)
    if not busy:
        break
    print("WAITING_FOR_EXISTING_TRAINING_PIDS:", ",".join(busy), flush=True)
    time.sleep(30)
PY

run_one() {
    pjlab_role="$1"
    pjlab_steps="$2"
    export RUN_ID="$ADAMW_JOB_ID-$3"
    export CURRENT_RUN_DIR="$WORKDIR/runs/$RUN_ID"
    mkdir "$CURRENT_RUN_DIR"
    printf '%s\n' "$CURRENT_RUN_DIR" > "$WORKDIR/logs/latest-adamw-run.txt"
    npu-smi info > "$CURRENT_RUN_DIR/npu-smi-start.txt" 2>&1 || true
    date -u +%Y-%m-%dT%H:%M:%SZ > "$CURRENT_RUN_DIR/start-utc.txt"
    printf 'ADAMW_RUN_START role=%s steps=%s directory=%s\n' "$pjlab_role" "$pjlab_steps" "$CURRENT_RUN_DIR"
    python -m torch.distributed.run \
        --standalone --nnodes=1 --nproc-per-node=16 \
        "$WORKDIR/train_interns2_adamw_dist.py" \
        --run-dir "$CURRENT_RUN_DIR" --run-role "$pjlab_role" \
        --steps "$pjlab_steps" --schedule-steps 500 --warmup-steps 20 \
        --lr 0.00001 --min-lr 0.000001 \
        --seq-length 64 --num-microbatches 1 \
        --data-mode text --corpus-dir "$CORPUS_DIR" \
        2>&1 | tee "$CURRENT_RUN_DIR/train.log"
    npu-smi info > "$CURRENT_RUN_DIR/npu-smi-end.txt" 2>&1 || true
    write_bundle 0
    unset CURRENT_RUN_DIR
}

run_one validation 3 check3
export ADAMW_CHECK_DIR="$WORKDIR/runs/$ADAMW_JOB_ID-check3"
python - <<'PY'
import json
import os
from pathlib import Path
run = Path(os.environ["ADAMW_CHECK_DIR"])
summary = json.loads((run / "summary.json").read_text())
assert summary["status"] == "passed" and summary["completed_steps"] == 3
assert summary["first_step_updates_verified_on_all_ranks"]
assert summary["min_peak_allocated_headroom_gib_all_ranks"] >= 2.0
assert json.loads((run / "launcher-exit.json").read_text())["exit_code"] == 0
print("ADAMW_SHORT_CHECK_PASSED:", json.dumps(summary), flush=True)
PY

# Start a new measurement from the original HF weights, not from pilot updates.
run_one measurement 500 steps500
printf '%s\n' 'ADAMW_500_PIPELINE_OK'
PJLAB_ADAMW_DIST_FILE_6
sha256sum -c - <<'PJLAB_ADAMW_DIST_HASHES'
2b19bf76d7fbd91c5dc88c2ea2b80473335c31c606f89b7500d31d4de379079f  /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/train_interns2_adamw_dist.py
d0fd3004efb6e82a7a629fa49fdecf3f5bae6d928416f4fce66b98dc9a93860a  /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/adamw_dist_update_probe.py
8d9a58357b0d9faf2a762a524914c9a483bf0d34f17f93ab99f91f960b9c058e  /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/adamw_shard_layout.py
d5d5e061349b02f3592fedb92d8d9dc43c83ef3dcc7c89d0f269a42f00d8d3d9  /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/adamw_math.py
62a2be06db172d1964b8bcb262bb9277c1ed1c9c321f91574c599a7672ae5b08  /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/optimizer_update_probe.py
d81da7ab8f203bf57f590651ab0310d6228c48d31c0fc5011b60e8528dba8149  /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/plot_interns2_adamw.py
4b3c8bfed8c57cad402b977e8d80a4a0df56ba5c0a57aea5b515b1ae29e44f84  /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/run_interns2_adamw_dist.sh
PJLAB_ADAMW_DIST_HASHES
printf '%s\n' 'PJLAB_ADAMW_DIST_SCRIPTS_READY'
PJLAB_ADAMW_DIST_UPLOAD
