"""GateMLState: the mechanism-specific internal state for GATE-ML.

Holds the EMA (exponential moving average) buffer over per-step cosine
distances between the input- and output-embedding gradients, plus the signal
history needed by should_intervene()'s persistence-window check, and the
`tied` flag recording whether the one-shot intervention has already fired.

This is exactly the object that gets serialized into
CheckpointState.mechanism_state on save and restored from it on resume
(architecture §8/§2's "mechanism_state: dict" convention) -- keeping it as
its own small dataclass, rather than scattering these fields across
GateMLMechanism's instance attributes, is what makes that serialize/restore
round-trip a single, obvious to_dict()/from_dict() pair instead of a
checkpoint writer that has to know GateMLMechanism's internals directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GateMLState:
    beta: float
    ema_cosine_distance: float = 0.0
    initialized: bool = False
    tied: bool = False
    signal_history: list = field(default_factory=list)

    def update(self, raw_cosine_distance: float) -> float:
        """Update the EMA with a new raw observation, return the new EMA value.

        Seeds the EMA directly from the first observation rather than from
        0.0 -- seeding from 0.0 would bias every early step toward "already
        similar" regardless of the true signal, which is exactly the kind of
        silent inaccuracy a project whose whole premise is auditing
        signal-driven claims cannot afford to reproduce in its own tooling.
        """
        if not self.initialized:
            self.ema_cosine_distance = raw_cosine_distance
            self.initialized = True
        else:
            self.ema_cosine_distance = (
                self.beta * self.ema_cosine_distance
                + (1.0 - self.beta) * raw_cosine_distance
            )
        self.signal_history.append(self.ema_cosine_distance)
        return self.ema_cosine_distance

    def to_dict(self) -> dict:
        return {
            "beta": self.beta,
            "ema_cosine_distance": self.ema_cosine_distance,
            "initialized": self.initialized,
            "tied": self.tied,
            "signal_history": list(self.signal_history),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GateMLState":
        required = ("beta", "ema_cosine_distance", "initialized", "tied", "signal_history")
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"GateMLState dict is missing required keys: {missing}")
        obj = cls(beta=data["beta"])
        obj.ema_cosine_distance = data["ema_cosine_distance"]
        obj.initialized = data["initialized"]
        obj.tied = data["tied"]
        obj.signal_history = list(data["signal_history"])
        return obj
