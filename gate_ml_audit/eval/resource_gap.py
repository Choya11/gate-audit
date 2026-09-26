"""Resource-gap metric and steps-to-generalization, per architecture §1's
eval/ responsibility ("resource-gap metric, steps-to-generalization").

Neither term's precise definition is specified anywhere in the available
project docs beyond the name itself -- this module implements the most
standard reading of each from the multilingual-NLP-audit literature this
project sits in, flagged explicitly rather than silently assumed, exactly
like GATE-ML's trigger direction in mechanisms/gate_ml/mechanism.py's
docstring. Both readings should be confirmed (or overridden) before either
metric appears in a headline table.

Pure Python (operates on already-computed per-language/per-step metric
values, never touches a model or tensor directly) -- fully runtime-tested
in this sandbox.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple


def resource_gap(
    perplexity_by_language: Dict[str, float], high_resource_language: str
) -> Dict[str, float]:
    """Per-language perplexity gap relative to the designated high-resource
    language (English, in this project): gap[lang] = perplexity[lang] -
    perplexity[high_resource_language] for every other language. A larger
    positive gap means that language does worse relative to the
    high-resource anchor at the same checkpoint.

    OPEN QUESTION, FLAGGED RATHER THAN SILENTLY ASSUMED: whether this
    should be an absolute gap (implemented here) or a relative/normalized
    gap (e.g. divided by the high-resource perplexity) is not specified in
    the available docs. An absolute gap is scale-dependent -- it shrinks as
    overall perplexity falls during training even if the *relative*
    disparity between languages stays constant -- which matters directly
    for how this number should be read across different checkpoints or
    different mechanisms with different overall convergence speeds.
    """
    if high_resource_language not in perplexity_by_language:
        raise KeyError(
            f"high_resource_language {high_resource_language!r} not present "
            f"in perplexity_by_language keys: {sorted(perplexity_by_language)}"
        )
    anchor = perplexity_by_language[high_resource_language]
    return {
        lang: value - anchor
        for lang, value in perplexity_by_language.items()
        if lang != high_resource_language
    }


def steps_to_generalization(
    metric_by_step: Sequence[Tuple[int, float]],
    threshold: float,
    lower_is_better: bool,
) -> Optional[int]:
    """The first step at which a per-step metric crosses `threshold` and
    stays on the "good" side of it for the remainder of the observed
    sequence -- not merely touches it once, which could be noise. This
    "crosses and stays" reading is this module's chosen definition of
    "generalization," flagged for the same reason as resource_gap above:
    the term is not defined further anywhere in the project docs.

    metric_by_step: a sequence of (step, value) pairs in increasing step
    order (not checked/re-sorted here -- the caller is expected to have
    already produced them in order, since re-sorting silently would hide a
    caller bug that fed them out of order). Returns the step of the
    earliest such crossing, or None if the metric never satisfies the
    threshold for the remainder of the sequence.
    """
    if not metric_by_step:
        raise ValueError("metric_by_step must be non-empty")

    for i, (step, _value) in enumerate(metric_by_step):
        remainder_ok = all(
            (value <= threshold if lower_is_better else value >= threshold)
            for _, value in metric_by_step[i:]
        )
        if remainder_ok:
            return step
    return None
