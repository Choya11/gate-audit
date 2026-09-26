"""Per-checkpoint, per-language perplexity: exp(mean cross-entropy loss)
over a held-out split, token-weighted (not sequence-averaged, which would
weight short and long sequences equally rather than by how many tokens they
actually contribute to the loss).

RUNTIME VERIFICATION NOTE: imports torch. Syntax-checked only in this
sandbox.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_perplexity(
    model: torch.nn.Module, batches: Iterable, device: torch.device
) -> float:
    """batches: an iterable of objects with a `.token_ids` LongTensor
    [batch, seq_len + 1] attribute (i.e. a data.Batch, or anything sharing
    that shape convention) drawn from a held-out split.

    Returns exp(total_loss / total_target_tokens) -- the standard
    token-level perplexity definition. Raises ValueError if the eval split
    is empty rather than returning a nonsensical perplexity of exp(0) = 1.0,
    which would look like a real (suspiciously perfect) result rather than
    "there was nothing to evaluate."
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in batches:
        token_ids = batch.token_ids.to(device)
        input_ids = token_ids[:, :-1]
        target_ids = token_ids[:, 1:]
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += target_ids.numel()

    if total_tokens == 0:
        raise ValueError(
            "compute_perplexity received no target tokens -- empty eval split"
        )
    mean_loss = total_loss / total_tokens
    return math.exp(mean_loss)
