"""eval/harness.py: the single evaluate() implementation shared by both call
sites -- periodic in-loop eval during training, and post-hoc eval on a saved
checkpoint via orchestration/report.py (architecture §7's non-negotiable
rule: "periodic" and "final headline" numbers must be computed identically,
which matters when a reviewer asks whether early stopping or checkpoint
selection quietly changed the metric definition -- the guarantee only holds
if both call sites are backed by this one function, never two).

RUNTIME VERIFICATION NOTE: imports torch (model construction, device
placement, delegated calls into eval/perplexity.py and
eval/task_accuracy.py). Syntax-checked only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from checkpointing.state import CheckpointState
from eval.perplexity import compute_perplexity
from eval.resource_gap import resource_gap
from eval.task_accuracy import compute_task_accuracy
from training.model import TinyDecoderLM


@dataclass
class EvalResult:
    checkpoint_step: int
    perplexity_by_lang: Dict[str, float]
    task_accuracy: Dict[str, float] = field(default_factory=dict)
    resource_gap_by_lang: Dict[str, float] = field(default_factory=dict)


def evaluate(
    model: torch.nn.Module,
    checkpoint_step: int,
    held_out_batches_by_lang: Dict[str, Sequence],
    device: torch.device,
    high_resource_language: str = "english",
    task_examples_by_lang: Optional[Dict[str, Sequence[Tuple[List[int], List[int]]]]] = None,
) -> EvalResult:
    """Compute perplexity for every language in held_out_batches_by_lang,
    task accuracy for every language in task_examples_by_lang (if given),
    and the resource gap relative to high_resource_language.

    Both call sites -- the training loop's periodic eval and
    orchestration/report.py's post-hoc eval on a loaded checkpoint -- call
    this exact function with a model already placed on `device`. Neither
    site re-implements perplexity or accuracy scoring, which is what
    guarantees the two numbers can never silently diverge in how they're
    computed.
    """
    if not held_out_batches_by_lang:
        raise ValueError(
            "held_out_batches_by_lang must be non-empty -- evaluate() with "
            "no languages to score is a caller bug, not a valid empty result"
        )

    perplexity_by_lang = {
        lang: compute_perplexity(model, batches, device)
        for lang, batches in held_out_batches_by_lang.items()
    }

    task_accuracy_by_lang: Dict[str, float] = {}
    if task_examples_by_lang:
        task_accuracy_by_lang = {
            lang: compute_task_accuracy(model, examples, device)
            for lang, examples in task_examples_by_lang.items()
        }

    gap = (
        resource_gap(perplexity_by_lang, high_resource_language)
        if high_resource_language in perplexity_by_lang
        else {}
    )

    return EvalResult(
        checkpoint_step=checkpoint_step,
        perplexity_by_lang=perplexity_by_lang,
        task_accuracy=task_accuracy_by_lang,
        resource_gap_by_lang=gap,
    )


def evaluate_from_checkpoint(
    checkpoint_path: str,
    model_ctor_kwargs: Dict,
    held_out_batches_by_lang: Dict[str, Sequence],
    device: torch.device,
    high_resource_language: str = "english",
    task_examples_by_lang: Optional[Dict[str, Sequence[Tuple[List[int], List[int]]]]] = None,
) -> EvalResult:
    """Post-hoc call site: load a saved checkpoint's model weights into a
    freshly constructed TinyDecoderLM, then delegate to evaluate() -- the
    exact same function the training loop's periodic eval calls, per this
    module's own docstring guarantee.

    model_ctor_kwargs must match the architecture the checkpoint was
    actually trained with (vocab_size, d_model, n_layers, n_heads,
    max_seq_len) -- this is not re-derived from the checkpoint itself,
    since CheckpointState does not store model hyperparameters (only
    model_state, the state_dict), so a mismatch here would fail inside
    load_state_dict with a shape-mismatch error rather than silently
    producing a wrong result.
    """
    raw = torch.load(checkpoint_path, map_location="cpu")
    ckpt = CheckpointState.from_dict(raw)
    model = TinyDecoderLM(**model_ctor_kwargs).to(device)
    model.load_state_dict(ckpt.model_state)
    return evaluate(
        model=model,
        checkpoint_step=ckpt.step,
        held_out_batches_by_lang=held_out_batches_by_lang,
        device=device,
        high_resource_language=high_resource_language,
        task_examples_by_lang=task_examples_by_lang,
    )
