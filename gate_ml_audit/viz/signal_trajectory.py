"""Per-mechanism signal-trajectory plot (architecture §10, artifact 1):
diagnostic, supports the reproduction-fidelity check (architecture §6) --
shows the raw per-step signal value over training with intervention-fired
steps marked, so a person can visually sanity-check that a mechanism's
signal behaves the way its own paper describes before trusting any battery
result built on top of it.

Matplotlib, static PNG, no interactive-dashboard dependency -- consistent
with the existing auditbench_calibration.png pattern and for the same
reason: no GPU, no server, works identically on a laptop or a notebook.

Pure Python + matplotlib, no torch dependency -- runtime-tested in this
sandbox: a real PNG is generated and its existence/size checked, not just
"the function ran without raising."
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_signal_trajectory(
    steps: Sequence[int],
    signal_values: Sequence[float],
    intervention_fired: Sequence[bool],
    mechanism_name: str,
    condition_name: str,
    output_path: str,
) -> str:
    """Plot signal value over step, with a dashed vertical line at every
    step where the intervention actually fired. Returns output_path (so
    callers can chain straight into presenting the file without needing to
    remember the path they passed in)."""
    if not (len(steps) == len(signal_values) == len(intervention_fired)):
        raise ValueError(
            "steps, signal_values, and intervention_fired must all be the "
            "same length"
        )
    if not steps:
        raise ValueError("steps must be non-empty")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, signal_values, color="tab:blue", linewidth=1.2, label="signal")

    fired_steps = [s for s, fired in zip(steps, intervention_fired) if fired]
    for i, s in enumerate(fired_steps):
        ax.axvline(
            s, color="tab:red", linestyle="--", alpha=0.6, linewidth=0.8,
            label="intervention fired" if i == 0 else None,
        )

    ax.set_xlabel("training step")
    ax.set_ylabel("signal value")
    ax.set_title(f"{mechanism_name} / {condition_name} -- signal trajectory")
    ax.legend(loc="best")
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
