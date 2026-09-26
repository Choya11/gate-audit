"""4-bucket failure classification (experiment design §10), applied to
every audit run -- all four mechanisms -- before any battery conclusion is
drawn.

Bucket definitions, kept close to verbatim from experiment design §10
rather than paraphrased loosely, since getting this exactly right matters
for what's allowed to happen downstream (bucket "a" is the only one that
goes in the headline table):
  (a) Clean reproduction, clean battery result -- trustworthy, headline table.
  (b) Reproduction diverges from published numbers, battery still run --
      report both, flagged; a battery verdict on a bad reproduction is not
      evidence about the original mechanism.
  (c) Reproduction fails to converge -- excluded from the headline table,
      documented as its own small negative reproducibility finding.
  (d) Battery result is ambiguous -- Real and Step-matched statistically
      indistinguishable from each other AND from Scrambled -- reported as
      inconclusive, not forced into a binary verdict.

Pure Python, no torch dependency -- runtime-tested in this sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from stats.bootstrap_ci import BootstrapCI


@dataclass(frozen=True)
class ReproductionFidelity:
    converged: bool
    # None is only valid when converged is False -- "does it match published
    # numbers" is not a meaningful question for a run that never converged
    # at all, so this field is deliberately Optional rather than defaulting
    # to False in that case (False would wrongly imply "converged, but to
    # the wrong numbers," which is bucket b, not bucket c).
    matches_published: Optional[bool]

    def __post_init__(self) -> None:
        if not self.converged and self.matches_published is not None:
            raise ValueError(
                "matches_published must be None when converged is False -- "
                "a run that never converged cannot meaningfully be said to "
                "match or not match published numbers"
            )
        if self.converged and self.matches_published is None:
            raise ValueError(
                "matches_published must be True or False when converged is "
                "True"
            )


def cis_overlap(a: BootstrapCI, b: BootstrapCI) -> bool:
    """Operationalizes "statistically indistinguishable" as confidence
    interval overlap -- a standard, conservative reading (CI overlap is a
    weaker claim than a dedicated two-sample test failing to reject a
    difference) that needs no machinery beyond the BootstrapCI objects this
    project already computes elsewhere, consistent with this project's own
    general discipline against adding machinery beyond what's actually
    needed (architecture §3). Public (not prefixed with an underscore)
    because orchestration/report.py needs this exact same check when
    building the pass/fail heatmap, and duplicating it there would risk the
    two call sites' notions of "indistinguishable" silently drifting apart.
    """
    return a.lower <= b.upper and b.lower <= a.upper


def classify(
    reproduction: ReproductionFidelity,
    real_ci: Optional[BootstrapCI] = None,
    step_matched_ci: Optional[BootstrapCI] = None,
    scrambled_ci: Optional[BootstrapCI] = None,
) -> str:
    """Return exactly one of "a", "b", "c", "d".

    real_ci/step_matched_ci/scrambled_ci are each a BootstrapCI over the
    native-metric performance-gap statistic for that condition (experiment
    design §9's "bootstrap-CI'd performance gap between Real and each
    control"). All three are required once reproduction has converged and
    matches published numbers (the only case where bucket a or d applies);
    they are simply ignored (may be left as None) when the run never
    converged, since bucket c is returned immediately in that case without
    a battery result to classify against at all.
    """
    if not reproduction.converged:
        return "c"
    if reproduction.matches_published is False:
        return "b"

    if real_ci is None or step_matched_ci is None or scrambled_ci is None:
        raise ValueError(
            "real_ci, step_matched_ci, and scrambled_ci are all required "
            "once reproduction has converged and matches published numbers "
            "-- both bucket a and bucket d need a battery result to "
            "classify against"
        )

    real_vs_step_matched_indistinguishable = cis_overlap(real_ci, step_matched_ci)
    real_vs_scrambled_indistinguishable = cis_overlap(real_ci, scrambled_ci)

    if real_vs_step_matched_indistinguishable and real_vs_scrambled_indistinguishable:
        return "d"
    return "a"
