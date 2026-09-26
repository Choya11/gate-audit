"""Registry entry for GrokFast.

Not yet implemented -- see mechanisms/rigl/mechanism.py's docstring for the
rationale shared by all three stub mechanisms in this pass. Calling apply()
raises NotImplementedError rather than silently doing nothing.

When this is implemented for real: GrokFast's signal is the low-frequency
(EMA-filtered) component of the gradient, and its intervention amplifies
that component to accelerate generalization -- this docstring is the
placeholder for that design, not the implementation.
"""
from __future__ import annotations

from mechanisms.base import StubMechanism


class GrokFastMechanism(StubMechanism):
    def __init__(self, **params: object) -> None:
        super().__init__(name="grokfast", **params)
