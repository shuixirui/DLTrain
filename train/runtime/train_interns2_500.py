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
model_dir = Path(os.environ.get("MODEL_PATH", "/mnt/afs-a3-weight-share/Intern-S2-Preview"))
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
    for name in ("DLTrain-framework", "DLTrain-Adaptor", "DLTrain-NPUEngine", "DLTrain-Bridge"):
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
    bridge_root = workdir / "src/DLTrain-Bridge/src/megatron/bridge"
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
