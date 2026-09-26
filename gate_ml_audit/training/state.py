"""TrainingState: the per-step state object passed to every Mechanism method.

Carries exactly the fields needed for (a) mechanisms to read config/context
without importing the training loop, (b) logging (architecture §9's JSONL
schema logs device/mechanism/condition/seed/data_source unconditionally on
every line), and (c) checkpoint tagging (CheckpointState needs the same
identifying fields). It intentionally does not decide *what to do* with any
of these fields -- that is training/loop.py's job; this class only carries
data.

`extra` is a deliberately loose escape hatch for exactly one documented use:
the training loop places a reference to the live model under
extra["model"] before calling mechanism.signal(), because some mechanisms
(GATE-ML) need to read live gradient tensors off the model that a plain
float/int state object has no other way to expose. Nothing else should be
routed through `extra` -- if a second mechanism needs a second kind of
extra data, that is a signal this dataclass's fixed fields should grow, not
that `extra` should become a general-purpose bag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingState:
    step: int
    mechanism: str
    condition: str
    seed: int
    device: str
    data_source: str
    loss: Optional[float] = None
    extra: dict = field(default_factory=dict)

    def with_step(self, step: int) -> "TrainingState":
        """Return a shallow copy advanced to a new step.

        Mutating step in place would let a stale TrainingState reference
        held elsewhere (e.g. inside a Mechanism's own signal_history) silently
        observe a step number that no longer matches when it was recorded.
        Returning a fresh copy makes that class of bug structurally
        impossible rather than relying on callers to remember not to mutate.
        `extra` is shallow-copied (new dict, same value references) since its
        one sanctioned use (`extra["model"]`) is meant to always point at the
        one live model instance for the whole run, not a per-step snapshot.
        """
        return TrainingState(
            step=step,
            mechanism=self.mechanism,
            condition=self.condition,
            seed=self.seed,
            device=self.device,
            data_source=self.data_source,
            loss=self.loss,
            extra=dict(self.extra),
        )
