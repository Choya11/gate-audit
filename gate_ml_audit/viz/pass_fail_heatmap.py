"""Pass/fail heatmap across mechanisms x conditions -- the headline
cross-mechanism table (architecture §10, artifact 2, tagged [C2]).

This module only draws whatever it is handed; the data_source=real gate
itself lives in orchestration/report.py (architecture §11's
DataProvenanceError), not here -- the caller is responsible for having
already filtered to data_source=real results before calling this.

Pure Python + matplotlib + numpy, no torch dependency -- runtime-tested.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_LABELS = {0: "n/a", 1: "fail", 2: "pass"}


def plot_pass_fail_heatmap(
    mechanisms: Sequence[str],
    conditions: Sequence[str],
    pass_fail_grid: Dict[str, Dict[str, bool]],
    output_path: str,
) -> str:
    """pass_fail_grid[mechanism][condition] -> bool (True = pass).

    A missing (mechanism, condition) pair is rendered as a distinct "n/a"
    cell rather than silently defaulting to pass or fail -- "no result
    exists yet" is a different thing from either verdict and should look
    different on the plot, not just be numerically ambiguous with one of
    the real outcomes.
    """
    if not mechanisms or not conditions:
        raise ValueError("mechanisms and conditions must both be non-empty")

    grid = np.zeros((len(mechanisms), len(conditions)), dtype=int)
    for i, mech in enumerate(mechanisms):
        for j, cond in enumerate(conditions):
            if mech not in pass_fail_grid or cond not in pass_fail_grid[mech]:
                grid[i, j] = 0
            else:
                grid[i, j] = 2 if pass_fail_grid[mech][cond] else 1

    fig, ax = plt.subplots(
        figsize=(1.5 + 1.5 * len(conditions), 1.0 + 0.8 * len(mechanisms))
    )
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_yticks(range(len(mechanisms)))
    ax.set_yticklabels(mechanisms)

    for i in range(len(mechanisms)):
        for j in range(len(conditions)):
            ax.text(
                j, i, _LABELS[int(grid[i, j])],
                ha="center", va="center", color="black", fontsize=9,
            )

    ax.set_title("Falsification battery: pass/fail (data_source=real only)")
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
