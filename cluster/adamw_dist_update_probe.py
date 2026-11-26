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
