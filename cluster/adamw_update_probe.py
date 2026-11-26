"""Audit fresh AdamW moments, FP32 master updates, and BF16 copy at selected points."""

import json
import math

import torch

from adamw_math import first_step_expectation
from optimizer_update_probe import active_indices, scalar


@torch.no_grad()
def capture_update_points(named_params, optimizer, small_probes):
    owners = {}
    for leaf in getattr(optimizer, "chained_optimizers", [optimizer]):
        base = leaf.optimizer
        if hasattr(base, "adam_w_mode"):
            assert base.adam_w_mode, "Expected decoupled AdamW weight decay"
        for group in base.param_groups:
            for master in group["params"]:
                owners[id(master)] = (base, group)
    rates = sorted({scalar(group["lr"]) for _, group in owners.values()})
    assert rates and all(math.isfinite(rate) and rate > 0 for rate in rates)
    suffixes = (
        "language_model.decoder.layers.0.mlp.experts.linear_fc1.weight0",
        "language_model.decoder.layers.0.mlp.experts.linear_fc2.weight0",
        "language_model.decoder.layers.3.self_attention.linear_qkv.weight",
        "language_model.decoder.layers.3.self_attention.linear_proj.weight",
        "language_model.output_layer.weight",
    )
    chosen = list(small_probes) + [(name, param) for name, param in named_params.items() if param.requires_grad and param.ndim == 2 and name.endswith(suffixes)]
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
        base, group = owners[id(master)]
        state = base.state.get(master, {})
        assert scalar(state.get("step", group.get("step", 0))) == 0, "First-step probe requires a fresh optimizer"
        indices = torch.arange(param.numel(), device=param.device) if param.numel() <= 8192 else active_indices(grad, master)
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                assert state[key].reshape(-1).index_select(0, indices).count_nonzero().item() == 0
        snapshots.append({
            "name": name, "param": param, "master": master, "base": base, "group": group,
            "indices": indices, "lr": scalar(group["lr"]),
            "betas": group["betas"], "eps": scalar(group["eps"]), "weight_decay": scalar(group.get("weight_decay", 0)),
            "model_before": param.detach().reshape(-1).index_select(0, indices).clone(),
            "master_before": master.detach().reshape(-1).index_select(0, indices).float().clone(),
            "grad_before": grad.detach().reshape(-1).index_select(0, indices).float().clone(),
        })
    assert snapshots, "No update probes"
    return snapshots, rates


@torch.no_grad()
def verify_update_points(snapshots, grad_norm, clip_grad, rank, rates):
    norm = scalar(grad_norm)
    factor = min(1.0, clip_grad / (norm + 1e-6)) if clip_grad > 0 else 1.0
    rows = []
    for probe in snapshots:
        param, master, indices = probe["param"], probe["master"], probe["indices"]
        actual_master = master.detach().reshape(-1).index_select(0, indices).float()
        actual_model = param.detach().reshape(-1).index_select(0, indices)
        state = probe["base"].state[master]
        actual_m = state["exp_avg"].detach().reshape(-1).index_select(0, indices).float()
        actual_v = state["exp_avg_sq"].detach().reshape(-1).index_select(0, indices).float()
        expected_master, expected_m, expected_v = first_step_expectation(
            probe["master_before"], probe["grad_before"], probe["lr"],
            *probe["betas"], probe["eps"], probe["weight_decay"], factor,
        )
        rows.append({
            "name": probe["name"], "dtype": str(param.dtype),
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
        "rank": rank, "optimizer": "AdamW", "effective_lrs": rates, "grad_norm": norm, "clip_factor": factor,
        "master_delta_max": max(row["master_delta"] for row in rows), "model_delta_max": best["model_delta"],
        "changed_name": best["name"], "master_formula_and_model_copy_checked": not failed,
        "failed_checks": failed, "probes": len(rows),
        "largest_model_changes": sorted(rows, key=lambda row: row["model_delta"], reverse=True)[:5],
    }
    print("ADAMW_UPDATE_DIAGNOSTICS:", json.dumps(summary), flush=True)
    assert not failed, "AdamW state/update/copy check failed; see ADAMW_UPDATE_DIAGNOSTICS"
    return summary
