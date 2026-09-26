"""Percentile bootstrap confidence interval (architecture §1's stats/
responsibility; experiment design §9: "bootstrap confidence intervals
(percentile method, >=1,000 resamples) on every battery pass/fail outcome
and MI-with-step score").

Pure Python + numpy, no torch dependency -- runtime-tested in this sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    point_estimate: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int


def bootstrap_ci(
    data: Sequence[float],
    statistic: Callable[[np.ndarray], float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI for `statistic` computed over `data`.

    Resamples `data` with replacement `n_resamples` times, computes
    `statistic` on each resample, and returns the
    [(1-confidence)/2, 1-(1-confidence)/2] percentiles of that resampled
    distribution as the CI bounds -- the standard percentile-bootstrap
    method. The more complex bias-corrected-and-accelerated (BCa) variant is
    not implemented: nothing in the project docs calls for it, and adding it
    without that requirement would be exactly the kind of unrequested
    complexity this project's own config-system reasoning (architecture §3)
    argues against elsewhere.
    """
    if len(data) < 2:
        raise ValueError(
            f"bootstrap_ci requires at least 2 data points, got {len(data)}"
        )
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    rng = np.random.default_rng(seed)
    data_array = np.asarray(data, dtype=float)
    n = len(data_array)

    point_estimate = float(statistic(data_array))
    resampled_stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        resample = rng.choice(data_array, size=n, replace=True)
        resampled_stats[i] = statistic(resample)

    alpha = 1.0 - confidence
    lower = float(np.percentile(resampled_stats, 100 * (alpha / 2)))
    upper = float(np.percentile(resampled_stats, 100 * (1 - alpha / 2)))
    return BootstrapCI(
        point_estimate=point_estimate,
        lower=lower,
        upper=upper,
        confidence=confidence,
        n_resamples=n_resamples,
    )
