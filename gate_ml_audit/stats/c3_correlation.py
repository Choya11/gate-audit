"""C3 correlation: battery-outcome vs. reported-effect-size correlation
across the three *audited* mechanisms only (experiment design §4: GATE-ML,
RigL, GrokFast). Curriculum-ordering, the positive control, is deliberately
excluded from this specific analysis -- its role is calibration against a
known published result (Wu et al.), not being a genuine unknown under
audit; it still appears in the main pass/fail table, just not here
(experiment design §4's explicit rationale).

Uses Spearman's rho over two rankings (originally-reported effect size,
normalized upstream to a relative-improvement-over-baseline scale, and
battery outcome/MI score) -- not Pearson correlation over raw values, which
nothing in the project docs calls for.

Architecture §12 requires descriptive-only to be the *only* mode available:
no p_value field on the result type at all. scipy.stats.spearmanr returns a
p-value alongside rho; this module discards it immediately on the same line
it is computed and never stores or returns it anywhere.

Experiment design §15 flags an explicit degeneracy risk distinct from the
usual "underpowered at N=3" caveat: if all three mechanisms land on the same
side of pass/fail (or tie on effect size), one of the two ranked series is
constant and Spearman's rho is mathematically undefined -- not just
statistically weak. This is handled as a first-class outcome
(C3Result.degenerate=True), not computed as a nonsensical number and not
raised as an error.

Pure Python + numpy/scipy, no torch dependency -- runtime-tested in this
sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

_AUDITED_MECHANISMS: Tuple[str, ...] = ("gate_ml", "rigl", "grokfast")


@dataclass(frozen=True)
class C3Result:
    """Deliberately has NO p_value field and no method that could compute
    one -- see the module docstring. Do not add one here; a future need for
    inferential statistics on this specific comparison is a new, separately
    named result type and function, not an addition to this one.
    """
    correlation: Optional[float]
    n: int
    mechanism_names: Tuple[str, ...]
    degenerate: bool
    # (mechanism, effect_size, battery_outcome) per mechanism, always
    # populated regardless of `degenerate` -- experiment design §15's
    # required fallback ("has to be rewritten ... as a qualitative
    # case-by-case discussion") needs this table whether or not a
    # correlation number exists.
    case_by_case: Tuple[Tuple[str, float, float], ...]


def compute_c3(
    effect_size_by_mechanism: Dict[str, float],
    battery_outcome_by_mechanism: Dict[str, float],
) -> C3Result:
    """effect_size_by_mechanism / battery_outcome_by_mechanism: dicts keyed
    by mechanism name, each covering at least the three audited mechanisms.
    A caller passing the full four-mechanism dict it already has lying
    around from the main pass/fail table (curriculum included) is fine --
    curriculum's entry is simply ignored, not an error, since experiment
    design §4 excludes it from this analysis specifically, not from having
    a battery result computed for it at all.
    """
    missing_effect = [m for m in _AUDITED_MECHANISMS if m not in effect_size_by_mechanism]
    if missing_effect:
        raise KeyError(
            f"effect_size_by_mechanism missing required mechanisms: {missing_effect}"
        )
    missing_outcome = [m for m in _AUDITED_MECHANISMS if m not in battery_outcome_by_mechanism]
    if missing_outcome:
        raise KeyError(
            f"battery_outcome_by_mechanism missing required mechanisms: {missing_outcome}"
        )

    effect_sizes = [effect_size_by_mechanism[m] for m in _AUDITED_MECHANISMS]
    battery_outcomes = [battery_outcome_by_mechanism[m] for m in _AUDITED_MECHANISMS]
    case_by_case = tuple(
        (m, effect_size_by_mechanism[m], battery_outcome_by_mechanism[m])
        for m in _AUDITED_MECHANISMS
    )

    if len(set(effect_sizes)) == 1 or len(set(battery_outcomes)) == 1:
        return C3Result(
            correlation=None,
            n=3,
            mechanism_names=_AUDITED_MECHANISMS,
            degenerate=True,
            case_by_case=case_by_case,
        )

    rho, _p_value_deliberately_discarded = scipy_stats.spearmanr(
        np.asarray(effect_sizes, dtype=float),
        np.asarray(battery_outcomes, dtype=float),
    )
    return C3Result(
        correlation=float(rho),
        n=3,
        mechanism_names=_AUDITED_MECHANISMS,
        degenerate=False,
        case_by_case=case_by_case,
    )
