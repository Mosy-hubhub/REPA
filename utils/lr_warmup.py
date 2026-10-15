"""Learning-rate scheduling helpers for REPA training scripts."""

from __future__ import annotations

import math

from torch.optim import Optimizer


def get_scheduled_lr(
    step: int,
    base_lr: float,
    schedule_type: str = "constant",
    warmup_steps: int = 0,
    decay_steps: int = 0,
    min_lr: float = 0.0,
) -> float:
    """Return the learning rate for the requested training step.

    The caller is expected to pass a 1-based optimization step so that
    warmup schedules start from a small positive learning rate.
    """
    step = max(int(step), 1)

    if schedule_type == "constant":
        return float(base_lr)

    if schedule_type == "warmup_constant":
        if warmup_steps <= 0:
            return float(base_lr)
        warmup_progress = min(step / warmup_steps, 1.0)
        return float(base_lr) * warmup_progress

    if schedule_type == "decay":
        effective_decay_steps = max(int(decay_steps), 1)
        decay_progress = min(step / effective_decay_steps, 1.0)
        return float(base_lr) + (float(min_lr) - float(base_lr)) * decay_progress

    if schedule_type == "cosinedecay":
        effective_decay_steps = max(int(decay_steps), 1)
        decay_progress = min(step / effective_decay_steps, 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return float(min_lr) + (float(base_lr) - float(min_lr)) * cosine_factor

    raise ValueError(f"Unsupported lr schedule: {schedule_type}")


def apply_scheduled_lr(
    optimizer: Optimizer,
    step: int,
    base_lr: float,
    schedule_type: str = "constant",
    warmup_steps: int = 0,
    decay_steps: int = 0,
    min_lr: float = 0.0,
) -> float:
    """Update every optimizer param group with the scheduled learning rate."""
    lr = get_scheduled_lr(
        step=step,
        base_lr=base_lr,
        schedule_type=schedule_type,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        min_lr=min_lr,
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr
