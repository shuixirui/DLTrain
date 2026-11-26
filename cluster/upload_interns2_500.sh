bash <<'PJLAB_500_UPLOAD' || printf '%s\n' 'UPLOAD_FAILED: inspect errors above; training was not started.'
set -eu
mkdir -p /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/logs
cat > '/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/train_interns2_500.py' <<'PJLAB_500_FILE_0'
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
from optimizer_update_probe import capture_update_points, verify_update_points

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--seq-length", type=int, default=64)
parser.add_argument("--num-microbatches", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--data-mode", choices=("text", "synthetic"), default="text")
parser.add_argument("--corpus-dir", type=Path)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--seed", type=int, default=20261125)
args = parser.parse_args()
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
    optimizer="sgd", lr=args.lr, sgd_momentum=0.0, weight_decay=0.0,
    bf16=True, fp16=False, params_dtype=torch.bfloat16,
    use_distributed_optimizer=False, clip_grad=1.0,
)
optimizer = get_megatron_optimizer(optimizer_config, models, use_gloo_process_groups=False)
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
    for name in (Path(__file__).name, "optimizer_update_probe.py", "check_interns2_provider.py", "run_interns2_500.sh", "plot_interns2_500.py"):
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
        "optimizer": "Megatron SGD", "lr": args.lr, "momentum": 0, "weight_decay": 0, "clip_grad": 1.0,
        "dtype": "bfloat16", "seed": args.seed, "data_mode": args.data_mode,
        "corpus": corpus_meta, "held_out_evaluation": False,
        "checkpoint_saved": False, "mtp_num_layers": 0, "vision_frozen": True,
        "step_time_definition": "maximum rank duration of synchronized input/zero-grad/forward/backward/gradient-sync/SGD; excludes update audit and metric aggregation; excludes initialization/loading",
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
        model.zero_grad_buffer()
        optimizer.zero_grad()
        reports = schedule(
            forward_step_func=forward_step, data_iterator=iter(batches_for_step(step)), model=models,
            num_microbatches=args.num_microbatches, seq_length=args.seq_length,
            micro_batch_size=1, forward_only=False,
        )
        # The first-step audit preserves the proven SGD formula/copy checks.
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
            "step_seconds_max_rank": step_seconds, "lr": args.lr,
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
            print(f"ITER {step}/{args.steps} loss={mean_loss:.6f} grad_norm={float(grad_norm):.6f} step_s={step_seconds:.3f} tokens_s={row['tokens_per_second']:.1f} peak_GiB={peak_allocated:.2f}", flush=True)
        del reports, sums, step_stats

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
PJLAB_500_FILE_0
printf '%s\n' '348b855c7547d7fdf1e263730972f51298b30a3e1d09e8051f4100cc60b8340d  /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/train_interns2_500.py' | sha256sum -c -
cat > '/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/optimizer_update_probe.py' <<'PJLAB_500_FILE_1'
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
PJLAB_500_FILE_1
printf '%s\n' '62a2be06db172d1964b8bcb262bb9277c1ed1c9c321f91574c599a7672ae5b08  /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/optimizer_update_probe.py' | sha256sum -c -
cat > '/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/prepare_interns2_corpus.py' <<'PJLAB_500_FILE_2'
"""Tokenize a local UTF-8 text/JSONL corpus for an explicitly recorded training run."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def documents(path, field):
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            if path.suffix.lower() == ".jsonl":
                record = json.loads(line)
                text = record.get(field) if isinstance(record, dict) else None
                if not isinstance(text, str):
                    raise ValueError(f"Line {line_number}: expected a string field {field!r}")
            else:
                text = line
            if text.strip():
                yield text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", dest="model", type=Path, default=Path("/mnt/afs-a3-weight-share/Intern-S2-Preview"))
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-tokens", type=int, default=2_000_000)
    args = parser.parse_args()
    if args.input.suffix.lower() not in (".txt", ".jsonl"):
        parser.error("--input must be a UTF-8 .txt or .jsonl file")
    if args.max_tokens < 1024:
        parser.error("--max-tokens must be at least 1024")

    import numpy as np
    import transformers
    from transformers import AutoTokenizer

    # Hash the source once. No corpus or model weights are modified.
    source_hash = sha256_file(args.input)
    tokenizer_files = {}
    for pattern in ("config.json", "tokenizer*", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "tokenization*.py"):
        for path in args.model.glob(pattern):
            if path.is_file():
                tokenizer_files[path.name] = sha256_file(path)
    signature = {
        "input_sha256": source_hash, "text_field": args.text_field,
        "max_tokens": args.max_tokens, "tokenizer_files_sha256": tokenizer_files,
        "transformers_version": transformers.__version__,
        "document_policy": "each non-empty txt line / JSONL record is a document; append EOS",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    binary, manifest = args.output_dir / "tokens.bin", args.output_dir / "tokens.meta.json"
    if binary.is_file() and manifest.is_file():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old.get("signature") == signature and sha256_file(binary) == old["tokens_sha256"]:
            print("CORPUS_CACHE_VERIFIED:", manifest, flush=True)
            return

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    eos = tokenizer.eos_token_id
    if not isinstance(eos, int):
        raise ValueError("Tokenizer must expose a single EOS token ID")
    excluded_ids = set()
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    for key, value in config.items():
        if key.endswith("token_id") and any(tag in key for tag in ("image", "video", "vision", "ts_", "time_series")):
            if isinstance(value, int):
                excluded_ids.add(value)
    token_count, doc_count, skipped = 0, 0, 0
    temporary = binary.with_name(f"tokens.bin.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as output:
            for text in documents(args.input, args.text_field):
                ids = tokenizer.encode(text, add_special_tokens=False)
                if excluded_ids.intersection(ids):
                    raise ValueError("Corpus contains multimodal placeholder tokens; this run supports text only")
                if not ids:
                    skipped += 1
                    continue
                ids.append(eos)
                ids = ids[:args.max_tokens - token_count]
                np.asarray(ids, dtype="<u4").tofile(output)
                token_count += len(ids)
                doc_count += 1
                if token_count >= args.max_tokens:
                    break
            output.flush()
            os.fsync(output.fileno())
        if token_count < 1024:
            raise ValueError(f"Only {token_count} tokens found; use at least 1024 tokens of real text")
        temporary.replace(binary)
    finally:
        if temporary.exists():
            temporary.unlink()
    meta = {
        "schema_version": 1, "signature": signature,
        "input_path": str(args.input.resolve()), "model_path": str(args.model.resolve()),
        "dtype": "<u4", "num_tokens": token_count, "num_documents": doc_count,
        "empty_token_documents_skipped": skipped, "eos_token_id": eos,
        "tokens_sha256": sha256_file(binary),
        "scope": "training text only; no held-out evaluation; documents joined with EOS",
    }
    meta_tmp = manifest.with_name(f"tokens.meta.json.tmp-{os.getpid()}")
    meta_tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    meta_tmp.replace(manifest)
    print("CORPUS_READY:", json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
PJLAB_500_FILE_2
printf '%s\n' '70cf2c48aa15bdda480c50afa53caf016d08562cb0649b806e852538a64ddc39  /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/prepare_interns2_corpus.py' | sha256sum -c -
cat > '/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/plot_interns2_500.py' <<'PJLAB_500_FILE_3'
"""Plot measured training records; never interpolate missing steps or infer success from loss."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import warnings
from pathlib import Path


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
        if row["tokens"] != manifest["tokens_per_step"] or row["lr"] != manifest["lr"]:
            raise ValueError("Per-step batch tokens or learning rate differs from the recorded manifest")
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
    fig.suptitle(title + "\n" + layout + f"\n{manifest['optimizer']} | lr={manifest['lr']:g} | BF16 | {manifest['world_size']} logical NPUs", y=0.985, fontsize=13)
    data_label = "real text, native model tokenizer" if manifest["data_mode"] == "text" else "SYNTHETIC RANDOM TOKENS - stability test"
    footer = data_label + "; training loss only, no held-out quality evaluation. Status: " + report["status"]
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
PJLAB_500_FILE_3
printf '%s\n' 'd075174f5aa45409305c808aca57055d8cb0f79e12dab044bc5f94e29ea9c13c  /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/plot_interns2_500.py' | sha256sum -c -
cat > '/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/run_interns2_500.sh' <<'PJLAB_500_FILE_4'
#!/usr/bin/env bash
set -eo pipefail

export WORKDIR=/mnt/afs-ptdata-shared/sxr/dltrain-mcore018
export LD_LIBRARY_PATH="/usr/local/python3.12.13/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export QWEN35_DEBUG=0
export OMP_NUM_THREADS=1
export RUN_ID="${RUN_ID:-interns2-500-$(date -u +%Y%m%dT%H%M%SZ)-$$}"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || [[ "$RUN_ID" == "." || "$RUN_ID" == ".." ]]; then
    printf '%s\n' 'Invalid RUN_ID' >&2
    exit 2
fi
export RUN_DIR="$WORKDIR/runs/$RUN_ID"
mkdir -p "$WORKDIR/runs" "$WORKDIR/logs"
mkdir "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$WORKDIR/logs/latest-500-run.txt"

source "$WORKDIR/.venv/bin/activate"
source /usr/local/Ascend/cann/set_env.sh
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

finish_run() {
    pjlab_exit_code=$?
    trap - EXIT
    export PJLAB_TRAIN_EXIT_CODE="$pjlab_exit_code"
    python - <<'PY'
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

run = Path(os.environ["RUN_DIR"])
(run / "launcher-exit.json").write_text(json.dumps({"exit_code": int(os.environ["PJLAB_TRAIN_EXIT_CODE"])}) + "\n", encoding="utf-8")
archive = run.with_suffix(".zip")
with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
    for path in sorted(run.rglob("*")):
        if path.is_file():
            bundle.write(path, arcname=str(path.relative_to(run.parent)))
print("MEASUREMENT_BUNDLE:", archive, flush=True)
PY
    printf 'TRAIN_EXIT_CODE=%s\n' "$pjlab_exit_code"
    exit "$pjlab_exit_code"
}
trap finish_run EXIT

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/start-utc.txt"
uname -a > "$RUN_DIR/uname.txt"
npu-smi info > "$RUN_DIR/npu-smi-start.txt" 2>&1 || true

python -m py_compile \
    "$WORKDIR/train_interns2_500.py" \
    "$WORKDIR/optimizer_update_probe.py" \
    "$WORKDIR/prepare_interns2_corpus.py" \
    "$WORKDIR/plot_interns2_500.py"

pjlab_data_mode="${DATA_MODE:-text}"
pjlab_corpus_dir="${CORPUS_DIR:-$WORKDIR/datasets/interns2-500}"
if [ "$pjlab_data_mode" = text ]; then
    if [ -z "${TRAIN_TEXT_PATH:-}" ]; then
        printf '%s\n' 'Set TRAIN_TEXT_PATH to a real UTF-8 .txt or .jsonl file. No random text is used implicitly.' >&2
        exit 2
    fi
    python "$WORKDIR/prepare_interns2_corpus.py" \
        --input "$TRAIN_TEXT_PATH" \
        --model-path /mnt/afs-a3-weight-share/Intern-S2-Preview \
        --output-dir "$pjlab_corpus_dir" \
        --text-field "${TEXT_FIELD:-text}" \
        --max-tokens "${MAX_CORPUS_TOKENS:-2000000}" \
        2>&1 | tee "$RUN_DIR/prepare-corpus.log"
elif [ "$pjlab_data_mode" != synthetic ]; then
    printf '%s\n' 'DATA_MODE must be text or synthetic' >&2
    exit 2
fi

python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=16 \
    "$WORKDIR/train_interns2_500.py" \
    --run-dir "$RUN_DIR" \
    --steps "${TRAIN_STEPS:-500}" \
    --seq-length "${SEQ_LENGTH:-64}" \
    --num-microbatches "${NUM_MICROBATCHES:-1}" \
    --lr "${LEARNING_RATE:-0.001}" \
    --data-mode "$pjlab_data_mode" \
    --corpus-dir "$pjlab_corpus_dir" \
    2>&1 | tee "$RUN_DIR/train.log"
PJLAB_500_FILE_4
printf '%s\n' 'e6bf743f6d5e81fa5a37eee8161758e47e4ce838ddde752b27a8482a86e67341  /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/run_interns2_500.sh' | sha256sum -c -
cat > '/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/start_interns2_500.sh' <<'PJLAB_500_FILE_5'
#!/usr/bin/env bash
# Run this script with bash. It launches a detached job and returns to the terminal.
set -eo pipefail
export WORKDIR=/mnt/afs-ptdata-shared/sxr/dltrain-mcore018
export DATA_MODE="${DATA_MODE:-text}"
export RUN_ID="${RUN_ID:-interns2-500-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || [[ "$RUN_ID" == "." || "$RUN_ID" == ".." ]]; then
    printf '%s\n' 'Invalid RUN_ID' >&2
    exit 2
fi
if [ "$DATA_MODE" = text ] && [ ! -f "${TRAIN_TEXT_PATH:-}" ]; then
    printf '%s\n' 'Set TRAIN_TEXT_PATH to the complete path of a readable .txt or .jsonl corpus.' >&2
    exit 2
fi
if [ -d "$WORKDIR/runs/$RUN_ID" ]; then
    printf '%s\n' 'RUN_ID already exists; choose a fresh run ID.' >&2
    exit 2
fi
mkdir -p "$WORKDIR/logs"
nohup bash "$WORKDIR/run_interns2_500.sh" \
    > "$WORKDIR/logs/$RUN_ID-launcher.log" 2>&1 < /dev/null &
pjlab_pid=$!
printf '%s\n' "$pjlab_pid" > "$WORKDIR/logs/$RUN_ID.pid"
printf 'STARTED_PID=%s\nRUN_ID=%s\n' "$pjlab_pid" "$RUN_ID"
printf 'Watch: tail -F %s/logs/%s-launcher.log\n' "$WORKDIR" "$RUN_ID"
printf 'Result directory: %s/runs/%s\n' "$WORKDIR" "$RUN_ID"
printf '%s\n' 'Started in background. Check INTERNS2_MEASURED_TRAINING_OK and TRAIN_EXIT_CODE=0 at completion.'
PJLAB_500_FILE_5
printf '%s\n' 'fe92cfad23c3c77978f87c2ffdb8e0d6db2c747aaf4aabd67158e7986e82f1ad  /mnt/afs-ptdata-shared/sxr/dltrain-mcore018/start_interns2_500.sh' | sha256sum -c -
printf '%s\n' 'PJLAB_500_SCRIPTS_READY'
PJLAB_500_UPLOAD
