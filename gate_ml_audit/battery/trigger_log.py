"""Extraction of the Real condition's trigger-event steps from its JSONL
log, and their conversion into the fixed schedule the Step-matched condition
replays (architecture §4's "Step-matched's fixed schedule is derived from
the Real condition's own trigger log").

Pure Python, no torch dependency -- fully runtime-tested in this sandbox.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List


def extract_trigger_steps(real_run_log_path: str) -> List[int]:
    """Read a completed Real run's JSONL log and return every step at which
    its intervention actually fired.

    Raises FileNotFoundError if the log doesn't exist yet. This is the
    concrete, code-level form of architecture §4's sequencing dependency:
    Step-matched cannot be constructed until this function can actually run
    against a completed Real run's log -- there is no silent fallback to an
    empty schedule here, because an empty schedule would make Step-matched
    look like "never intervenes," which is a different (and wrong) run, not
    a safe default.
    """
    path = Path(real_run_log_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Real run log not found at {real_run_log_path!r} -- Step-matched's "
            f"fixed schedule cannot be derived until the Real condition has "
            f"run to completion and produced this log (architecture §4)"
        )
    trigger_steps: List[int] = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "step" not in record:
                raise ValueError(
                    f"{real_run_log_path}:{line_no}: log line missing required "
                    f"'step' field"
                )
            if record.get("intervention_fired", False):
                trigger_steps.append(record["step"])
    return trigger_steps


def to_fixed_schedule(steps: List[int]) -> Callable[[int], bool]:
    """Return is_trigger_step(step) -> bool implementing the fixed schedule.

    A plain closure over a set, not a class -- there is exactly one
    operation here (membership lookup), so a class would be an unrequested
    abstraction for a single method.
    """
    step_set = set(steps)

    def is_trigger_step(step: int) -> bool:
        return step in step_set

    return is_trigger_step
