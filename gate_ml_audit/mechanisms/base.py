"""The Mechanism protocol -- the single most important interface in this
architecture (Publication_System_Architecture.md §2).

The training loop calls exactly these three methods on whatever Mechanism it
was handed and never anything else. It cannot tell which concrete mechanism,
or which of the three battery-condition wrappers (battery/conditions.py), it
is running -- that is the architectural property that makes the
cross-mechanism, cross-condition comparison fair by construction rather than
by promise.

NOTE ON RUNTIME VERIFICATION: this module imports torch. In the sandbox this
was written in, torch cannot be installed (confirmed: the CPU-only wheel
index is network-blocked, and the default CUDA-bundled install exhausts the
sandbox's disk budget). This file has been syntax-checked with
`python3 -m py_compile` and manually reviewed against the documented
torch.nn / torch.optim / typing.Protocol APIs, but has not been executed.
Runtime verification (`import mechanisms.base`, constructing a Mechanism,
calling all three methods against a real model) still needs to happen on an
environment where torch actually installs.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from training.state import TrainingState


@runtime_checkable
class Mechanism(Protocol):
    def signal(self, state: TrainingState) -> float:
        """Return the current value of this mechanism's monitored signal."""
        ...

    def should_intervene(self, signal_history: list, state: TrainingState) -> bool:
        """Decide whether to fire the intervention at this step."""
        ...

    def apply(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        state: TrainingState,
    ) -> None:
        """Apply the intervention (tie/untie, prune/regrow, gradient
        filter, reorder) in-place."""
        ...


class StubMechanism:
    """Registry-valid placeholder for a mechanism not yet implemented.

    signal() and should_intervene() return inert defaults so a battery can be
    *assembled* around a stub without erroring -- this lets config
    validation, the mechanism registry, and checkpoint-schema code exercise
    every registered mechanism name uniformly, stub or not, through the same
    code path. apply() raises unconditionally: a stub must fail loudly the
    instant the battery actually tries to run its intervention, rather than
    silently doing nothing and producing a BatteryResult that looks like a
    real, uneventful run.
    """

    def __init__(self, name: str, **_ignored_params: object) -> None:
        self.name = name

    def signal(self, state: TrainingState) -> float:
        return 0.0

    def should_intervene(self, signal_history: list, state: TrainingState) -> bool:
        return False

    def apply(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        state: TrainingState,
    ) -> None:
        raise NotImplementedError(
            f"mechanism {self.name!r} is a registry-only stub -- apply() has "
            f"no real implementation yet. See config/mechanisms/{self.name}.yaml "
            f"(status: stub) and the Implementation Plan's mechanism-selection "
            f"phase."
        )
