"""The three battery-condition wrappers (architecture §2/§14): each one
conforms to the Mechanism protocol itself, wrapping a real Mechanism
instance. The training loop only ever sees "a Mechanism" -- it cannot tell
which condition it is running, because all three conditions are the exact
same three methods (signal/should_intervene/apply), just with one of them
substituted.

Typed loosely (Any) for model/optimizer rather than importing torch.nn.Module
/ torch.optim.Optimizer directly -- these wrappers never touch a tensor
themselves, they only pass model/optimizer straight through to the wrapped
mechanism's own apply(). Avoiding the torch import here is a deliberate
"stdlib/already-available before a new dependency" choice (ponytail rung 5):
it is also what makes this entire file torch-free and fully runtime-testable
in a sandbox where torch cannot be installed, which is exactly what happened
below -- this file, unlike mechanisms/gate_ml/mechanism.py, has been executed
against a mock Mechanism, not just syntax-checked.
"""
from __future__ import annotations

import random
from typing import Any, Callable, List


class RealWrapper:
    """Passthrough: the unmodified mechanism, wrapped only so it shares a
    common type shape with StepMatchedWrapper/ScrambledWrapper for whatever
    code assembles a battery (battery-assembly code can treat all three
    conditions uniformly rather than special-casing Real as "the bare
    mechanism, no wrapper")."""

    def __init__(self, mechanism: Any) -> None:
        self._mechanism = mechanism

    def signal(self, state: Any) -> float:
        return self._mechanism.signal(state)

    def should_intervene(self, signal_history: List[float], state: Any) -> bool:
        return self._mechanism.should_intervene(signal_history, state)

    def apply(self, model: Any, optimizer: Any, state: Any) -> None:
        self._mechanism.apply(model, optimizer, state)


class StepMatchedWrapper:
    """should_intervene() is replaced by a fixed schedule derived from the
    Real condition's own trigger log (battery/trigger_log.py) -- isolates
    timing from the signal itself: if performance matches Real, timing
    alone explains the result, independent of whatever the true signal was
    doing at those steps.

    signal() still delegates to the wrapped mechanism (rather than
    returning a constant or omitting it) so the per-mechanism
    signal-trajectory diagnostic plots (architecture §10) have something
    real to plot for Step-matched runs too, even though should_intervene()
    below ignores whatever signal() returns entirely.
    """

    def __init__(self, mechanism: Any, is_trigger_step: Callable[[int], bool]) -> None:
        self._mechanism = mechanism
        self._is_trigger_step = is_trigger_step

    def signal(self, state: Any) -> float:
        return self._mechanism.signal(state)

    def should_intervene(self, signal_history: List[float], state: Any) -> bool:
        return self._is_trigger_step(state.step)

    def apply(self, model: Any, optimizer: Any, state: Any) -> None:
        self._mechanism.apply(model, optimizer, state)


class ScrambledWrapper:
    """signal() is replaced by a randomized, statistically-similar-but-
    information-free surrogate (per experiment design §3: "randomized,
    information-free version of the divergence signal"): each call still
    computes the wrapped mechanism's real signal (so the wrapped
    mechanism's own internal state -- e.g. GateMLState's EMA buffer --
    keeps updating on genuine observations, exactly as it does in the Real
    condition), but what gets *returned* (and therefore logged, and
    therefore fed into should_intervene()'s history) is a uniformly random
    past observation from this same run, not the value that actually
    corresponds to the current step.

    This keeps the surrogate's marginal distribution equal to the real
    signal's own distribution (it is a resampling of that exact run's own
    observed values, not a draw from some external reference distribution)
    while decorrelating it from what the model is actually doing at each
    step -- which is precisely "statistically similar but information-free."
    should_intervene() then delegates its threshold/window logic to the
    wrapped mechanism unchanged, fed this scrambled history: if performance
    still matches Real, the signal wasn't causally load-bearing.
    """

    def __init__(self, mechanism: Any, rng_seed: int) -> None:
        self._mechanism = mechanism
        self._rng = random.Random(rng_seed)
        self._observed: List[float] = []

    def signal(self, state: Any) -> float:
        raw = self._mechanism.signal(state)
        self._observed.append(raw)
        # self._observed always has at least one element here (raw was just
        # appended), so random.choice never raises on an empty sequence.
        return self._rng.choice(self._observed)

    def should_intervene(self, signal_history: List[float], state: Any) -> bool:
        return self._mechanism.should_intervene(signal_history, state)

    def apply(self, model: Any, optimizer: Any, state: Any) -> None:
        self._mechanism.apply(model, optimizer, state)


def make_real_and_scrambled(mechanism: Any, rng_seed: int) -> dict:
    """Build the two conditions that can be constructed immediately, with no
    sequencing dependency on any other run. Step-matched is deliberately not
    included here -- see make_step_matched below and architecture §4/§13.2:
    it requires the Real condition's own run to have already completed and
    its trigger log extracted, so it cannot be built at the same time as
    these two without either lying about that dependency or silently
    ignoring it.
    """
    return {
        "real": RealWrapper(mechanism),
        "scrambled": ScrambledWrapper(mechanism, rng_seed=rng_seed),
    }


def make_step_matched(mechanism: Any, is_trigger_step: Callable[[int], bool]) -> Any:
    """Build the Step-matched condition once its one prerequisite (a fixed
    schedule extracted from a completed Real run, via
    battery.trigger_log.to_fixed_schedule) is available. Kept as a separate
    function from make_real_and_scrambled specifically so that dependency is
    visible in the call signature, not just in a docstring."""
    return StepMatchedWrapper(mechanism, is_trigger_step)
