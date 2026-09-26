"""Registry entry for curriculum-ordering (difficulty-based data ordering).

Not yet implemented -- see mechanisms/rigl/mechanism.py's docstring for the
rationale shared by all three stub mechanisms in this pass. Calling apply()
raises NotImplementedError rather than silently doing nothing.

When this is implemented for real: this mechanism's "signal" and
"intervention" are unusual relative to the other three -- the intervention
is data mixing/ordering (consuming data/difficulty_proxy.py's length+rarity
composite), not an architecture or optimizer change -- which is exactly why
it is good category diversity for the audit (per the Implementation Plan),
and also why its Mechanism-protocol mapping needs its own care when
implemented: apply() would reorder/reweight the upcoming data batch rather
than mutate the model or optimizer directly.
"""
from __future__ import annotations

from mechanisms.base import StubMechanism


class CurriculumMechanism(StubMechanism):
    def __init__(self, **params: object) -> None:
        super().__init__(name="curriculum", **params)
