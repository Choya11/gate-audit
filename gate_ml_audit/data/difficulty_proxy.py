"""Length + rarity composite difficulty proxy, consumed only by
mechanisms/curriculum (still a registry stub in this pass -- see that
module's docstring). Defined here, in data/, rather than inside
mechanisms/curriculum itself, because corpus-level scoring is corpus
engineering, not mechanism logic, and per architecture §1, three of the
four mechanisms already share this module's output directly.

Pure Python, no torch dependency -- fully runtime-tested in this sandbox.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


def build_frequency_table(all_token_ids: Sequence[Sequence[int]]) -> Counter:
    """Corpus-wide token frequency table, built once and passed into every
    score() call rather than recomputed per example."""
    counter: Counter = Counter()
    for seq in all_token_ids:
        counter.update(seq)
    return counter


def _neg_log_or_ceiling(p: float) -> float:
    """-log(p), or a fixed ceiling for p<=0 (an unseen token) rather than
    raising or returning inf -- a single out-of-vocabulary token in one
    example should not crash difficulty scoring for the whole example, and
    inf would make every example containing an unseen token score identically
    (infinitely) difficult regardless of how many unseen tokens it has."""
    if p <= 0:
        return 20.0
    return -math.log(p)


def score(token_ids: Sequence[int], token_frequencies: Counter) -> float:
    """Return a single difficulty score for one tokenized example.

    Additive combination of a length component and a rarity component
    (mean per-token surprisal, -log(frequency)), rather than multiplicative:
    additive degrades gracefully if one component is uninformative (e.g. a
    corpus with near-uniform token frequencies makes the rarity component
    roughly constant, but the length component still varies meaningfully),
    where a multiplicative combination would let a near-zero component
    silently zero out the whole score.
    """
    if not token_ids:
        raise ValueError("token_ids must be non-empty")
    total_count = sum(token_frequencies.values())
    if total_count == 0:
        raise ValueError("token_frequencies must have at least one observed token")

    length_component = float(len(token_ids))
    rarity_component = sum(
        _neg_log_or_ceiling(token_frequencies.get(t, 0) / total_count)
        for t in token_ids
    ) / len(token_ids)
    return length_component + rarity_component
