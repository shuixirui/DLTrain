"""Shared scalar and gradient-index helpers for parameter update checks."""

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
