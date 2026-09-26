"""CheckpointState: the single schema for both periodic and trigger-event
checkpoints, across all four mechanisms (architecture §8, extended from the
original GATE-ML CheckpointState convention to be mechanism-generic).

Mechanism-specific fields live in `mechanism_state: dict` rather than as
mechanism-specific dataclass subclasses, so checkpoint-loading code
(checkpointing/reader.py) never needs to know which mechanism produced a
given checkpoint file ahead of time -- it loads the common fields, then hands
mechanism_state off to whichever Mechanism's own state class knows how to
interpret it (e.g. mechanisms.gate_ml.state.GateMLState.from_dict).

This module intentionally has no torch dependency: model_state and
optimizer_state are plain dicts here (the caller is responsible for having
already produced them via model.state_dict()/optimizer.state_dict() and for
converting back via model.load_state_dict()/optimizer.load_state_dict() --
that conversion is checkpointing/writer.py and checkpointing/reader.py's job,
not this file's). Keeping this file torch-free means its schema and
round-trip logic can be fully unit-tested (including in a sandbox where
torch itself cannot be installed) independent of whether the actual tensor
serialization works.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CheckpointState:
    step: int
    mechanism: str
    condition: str
    seed: int
    device: str
    data_source: str
    model_state: dict
    optimizer_state: dict
    mechanism_state: dict = field(default_factory=dict)
    is_trigger_event: bool = False

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "mechanism": self.mechanism,
            "condition": self.condition,
            "seed": self.seed,
            "device": self.device,
            "data_source": self.data_source,
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "mechanism_state": self.mechanism_state,
            "is_trigger_event": self.is_trigger_event,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointState":
        required = (
            "step", "mechanism", "condition", "seed", "device", "data_source",
            "model_state", "optimizer_state",
        )
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"checkpoint dict is missing required keys: {missing}")
        return cls(
            step=data["step"],
            mechanism=data["mechanism"],
            condition=data["condition"],
            seed=data["seed"],
            device=data["device"],
            data_source=data["data_source"],
            model_state=data["model_state"],
            optimizer_state=data["optimizer_state"],
            mechanism_state=data.get("mechanism_state", {}),
            is_trigger_event=data.get("is_trigger_event", False),
        )

    def naming_key(self) -> str:
        """The deterministic filename stem for this checkpoint.

        Matches architecture §12's flat-file naming scheme
        ({mechanism}_{condition}_{seed}_{step}) exactly, so checkpoints
        written by different people on different uncoordinated machines
        never collide and are trivially syncable via git or ordinary cloud
        storage without a shared database server.
        """
        return f"{self.mechanism}_{self.condition}_{self.seed}_{self.step}"
