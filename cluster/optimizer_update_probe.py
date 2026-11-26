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
