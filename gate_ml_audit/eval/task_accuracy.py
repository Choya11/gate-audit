"""Exact-match task accuracy over a held-out example set (used for
ChakmaBridge / IndicXTREME-style tasks, per architecture §1's eval/
responsibility). Deliberately generic: this module knows nothing about
ChakmaBridge or IndicXTREME's specific file formats -- it scores
greedy-decoded model output against an expected answer for each example,
which is the one operation every exact-match task benchmark reduces to.
Dataset-specific loading (turning a raw ChakmaBridge file into
(prompt_token_ids, expected_token_ids) pairs) belongs in data/, not here,
and is out of scope for this pass -- no real task datasets are reachable in
this sandbox to build or test that loader against (see
data/real_loader.py's runtime-verification note).

RUNTIME VERIFICATION NOTE: imports torch (greedy decoding). Syntax-checked
only.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


@torch.no_grad()
def _greedy_decode(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> List[int]:
    """Greedy (argmax) decoding, one token at a time, each new token
    appended to the running sequence before the next forward pass. Greedy
    rather than sampling: sampling would make accuracy nondeterministic
    across runs, which is exactly wrong for an audit whose whole point is
    reproducible cross-mechanism, cross-condition comparison."""
    model.eval()
    generated = prompt_ids.to(device).unsqueeze(0)  # [1, prompt_len]
    new_tokens: List[int] = []
    for _ in range(max_new_tokens):
        if generated.size(1) > model.max_seq_len:
            break
        logits = model(generated)  # [1, seq_len, vocab_size]
        next_token = int(torch.argmax(logits[0, -1, :]).item())
        new_tokens.append(next_token)
        next_token_tensor = torch.tensor([[next_token]], device=device, dtype=generated.dtype)
        generated = torch.cat([generated, next_token_tensor], dim=1)
    return new_tokens


def compute_task_accuracy(
    model: torch.nn.Module,
    examples: Sequence[Tuple[List[int], List[int]]],
    device: torch.device,
) -> float:
    """examples: a sequence of (prompt_token_ids, expected_answer_token_ids)
    pairs. Returns the fraction of examples where greedy decoding for
    len(expected_answer_token_ids) new tokens exactly matches the expected
    answer token-for-token.

    Raises ValueError on an empty example set rather than silently
    returning 0.0, which would look like a real (zero) accuracy score
    rather than "there was nothing to score."
    """
    if not examples:
        raise ValueError("compute_task_accuracy received an empty example set")

    correct = 0
    for prompt_ids, expected_ids in examples:
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)
        predicted = _greedy_decode(model, prompt_tensor, len(expected_ids), device)
        if predicted == list(expected_ids):
            correct += 1
    return correct / len(examples)
