"""Registry entry for RigL (Rigging the Lottery).

Not yet implemented -- per the current scope decision, only GATE-ML is
fully implemented in this pass; RigL, GrokFast, and curriculum-ordering are
registry-valid stubs so config validation, the battery wrapper, and
checkpointing can all reference these mechanism names as real, importable
entries (needed so the four-mechanism registry shape is correct end-to-end)
without pretending a real gradient-magnitude-triggered regrowth
implementation exists. Calling apply() raises NotImplementedError (see
mechanisms.base.StubMechanism) rather than silently doing nothing.

When this is implemented for real: RigL's signal is gradient magnitude
(used to decide which pruned connections to regrow), and should_intervene
fires on a periodic regrowth schedule per the original RigL paper -- this
docstring is the placeholder for that design, not the implementation.
"""
from __future__ import annotations

from mechanisms.base import StubMechanism


class RiglMechanism(StubMechanism):
    def __init__(self, **params: object) -> None:
        super().__init__(name="rigl", **params)
