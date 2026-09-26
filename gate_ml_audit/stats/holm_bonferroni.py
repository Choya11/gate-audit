"""Holm-Bonferroni step-down correction for multiple comparisons
(experiment design §9: "Holm-Bonferroni correction applied across the full
family of battery-outcome tests (4 mechanisms x 3 conditions x multiple
metrics), not per-mechanism in isolation").

Pure Python, stdlib only -- runtime-tested in this sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class HolmBonferroniResult:
    original_index: int
    p_value: float
    adjusted_alpha: float
    reject_null: bool


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> List[HolmBonferroniResult]:
    """Standard Holm step-down procedure: sort p-values ascending, compare
    the k-th smallest (1-indexed) against alpha / (m - k + 1), and once one
    comparison fails to reject, every later (larger-p-value, larger-index)
    comparison in the sorted order is also marked as not rejecting,
    regardless of its own p-value against its own threshold.

    This "stop at the first failure" behavior is not a shortcut -- it is
    what makes Holm's procedure valid (family-wise error rate controlled at
    alpha): once the (k)-th smallest p-value exceeds its threshold, every
    later comparison's threshold is >= that one's, so continuing to test
    them individually rather than marking them all as failed would silently
    inflate the false-positive rate the whole procedure exists to control.
    """
    if not p_values:
        raise ValueError("p_values must be non-empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    for p in p_values:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p-values must be in [0, 1], got {p}")

    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])

    results: List[Optional[HolmBonferroniResult]] = [None] * m
    still_rejecting = True
    for rank, (original_index, p) in enumerate(indexed):
        k = rank + 1  # 1-indexed rank, per Holm's own convention
        adjusted_alpha = alpha / (m - k + 1)
        if still_rejecting and p <= adjusted_alpha:
            reject = True
        else:
            reject = False
            still_rejecting = False
        results[original_index] = HolmBonferroniResult(
            original_index=original_index,
            p_value=p,
            adjusted_alpha=adjusted_alpha,
            reject_null=reject,
        )

    assert all(r is not None for r in results)  # every index was filled exactly once
    return results  # type: ignore[return-value]
