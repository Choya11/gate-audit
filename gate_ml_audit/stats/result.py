"""BatteryResult: one instance per (mechanism, condition, seed) -- native
metric value, MI score + p-value, bootstrap CI bounds, and the data_source
tag orchestration/report.py's DataProvenanceError gates on (architecture
§2's secondary interfaces; §0.5's "every BatteryResult carries the
data_source it was produced under").

Placed in stats/ rather than battery/ because assembling one requires the
falsification-audit stats layer's output (MI score, CI bounds) on top of
what battery/ itself produces -- it is the unit stats/ and viz/ both
consume, per architecture §2.

Pure Python, no torch dependency -- runtime-tested in this sandbox.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BatteryResult:
    mechanism: str
    condition: str
    seed: int
    data_source: str  # "synthetic" | "real" -- tagged unconditionally, per §0.5
    native_metric: float
    mi_score: float
    mi_p_value: float
    ci_lower: float
    ci_upper: float

    def __post_init__(self) -> None:
        if self.data_source not in ("synthetic", "real"):
            raise ValueError(
                f"data_source must be 'synthetic' or 'real', got "
                f"{self.data_source!r}"
            )
        if self.condition not in ("real", "step_matched", "scrambled"):
            raise ValueError(
                f"condition must be one of real/step_matched/scrambled, "
                f"got {self.condition!r}"
            )
        if not (0.0 <= self.mi_p_value <= 1.0):
            raise ValueError(
                f"mi_p_value must be in [0, 1], got {self.mi_p_value!r}"
            )
        if self.ci_lower > self.ci_upper:
            raise ValueError(
                f"ci_lower ({self.ci_lower}) must not exceed ci_upper "
                f"({self.ci_upper})"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BatteryResult":
        required = (
            "mechanism", "condition", "seed", "data_source", "native_metric",
            "mi_score", "mi_p_value", "ci_lower", "ci_upper",
        )
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"BatteryResult dict is missing required keys: {missing}")
        return cls(**{k: data[k] for k in required})
