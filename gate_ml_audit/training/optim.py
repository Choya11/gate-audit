"""build_optimizer / build_scheduler: factories driven entirely by config.

Per architecture §3's explicit design decision, hyperparameters are never
set programmatically inside mechanisms/ or training/ code -- only via
config -- so these factories read every tunable (lr, weight_decay,
warmup_steps, optimizer name) from OptimizerConfig and nowhere else.

RUNTIME VERIFICATION NOTE: imports torch; syntax-checked only in this
sandbox, not executed (see training/model.py's docstring for why).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config.schema import OptimizerConfig


_SUPPORTED_OPTIMIZERS = ("adamw",)


def build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    if cfg.name not in _SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"unsupported optimizer {cfg.name!r}; supported: {_SUPPORTED_OPTIMIZERS}"
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: OptimizerConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup over cfg.warmup_steps, constant (multiplier 1.0)
    afterward.

    No decay schedule is specified anywhere in the project docs beyond
    "warmup_steps" existing as a config field, so this deliberately
    implements only what is actually specified rather than inventing a decay
    shape (cosine, linear-to-zero, etc.) that was never asked for. If a decay
    schedule is wanted later, it is a change to this one function, not a
    change to training/loop.py, which only ever calls scheduler.step().
    """
    warmup_steps = max(1, cfg.warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
