"""C3 correlation scatter -- with the N=3 caveat rendered directly on the
plot as a text annotation, not left to the caption (architecture §10,
artifact 3), so the honesty already built into experiment design §4/§15
survives even if the plot gets reused or excerpted somewhere the
surrounding text doesn't travel with it.

Pure Python + matplotlib, no torch dependency -- runtime-tested.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stats.c3_correlation import C3Result


def plot_c3_scatter(result: C3Result, output_path: str) -> str:
    """Plot each of the 3 audited mechanisms' (effect_size, battery_outcome)
    point from result.case_by_case, with the correlation value -- or, if
    degenerate, an explicit degeneracy notice -- annotated directly on the
    plot itself."""
    fig, ax = plt.subplots(figsize=(5, 5))

    for mech, effect_size, battery_outcome in result.case_by_case:
        ax.scatter(effect_size, battery_outcome, s=80)
        ax.annotate(
            mech, (effect_size, battery_outcome),
            textcoords="offset points", xytext=(6, 6),
        )

    ax.set_xlabel("reported effect size (normalized)")
    ax.set_ylabel("battery outcome (MI score)")
    ax.set_title("C3: battery outcome vs. reported effect size")

    if result.degenerate:
        caption = (
            f"N={result.n}: correlation UNDEFINED -- one axis is constant "
            f"across all mechanisms (degenerate case, not merely "
            f"underpowered). See experiment design §15."
        )
    else:
        caption = (
            f"N={result.n}: Spearman's rho = {result.correlation:.3f} -- "
            f"DESCRIPTIVE ONLY, not a hypothesis test (no p-value at N=3; "
            f"see experiment design §4/§9)."
        )
    fig.text(0.02, 0.02, caption, fontsize=8, wrap=True, ha="left", va="bottom")
    fig.subplots_adjust(bottom=0.22)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
