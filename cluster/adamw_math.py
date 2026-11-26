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
