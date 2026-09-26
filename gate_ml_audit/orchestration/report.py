"""orchestration/report.py: CLI entrypoint. Loads BatteryResult[] (post-hoc,
from completed runs' JSON files), enforces the data-provenance gate (§0.5,
architecture §11's new DataProvenanceError), aggregates per-(mechanism,
condition) bootstrap CIs across seeds, classifies each mechanism into one of
the four failure buckets (experiment design §10), computes C3 (experiment
design §4), and produces the two required plots.

This is the module that makes §0.5's fix structural rather than a
documented convention: it raises DataProvenanceError -- loudly, not a
silently-empty report -- if asked to produce a headline table/plot from a
result set that contains zero data_source=real entries.

REPRODUCTION-FIDELITY INPUT, FLAGGED: whether a mechanism's toy-scale
reproduction "matches published numbers" (experiment design §10's bucket
a/b distinction) is inherently a qualitative judgment against a paper's own
reported numbers -- it is not something this module can compute from
BatteryResult's purely numeric fields. This module therefore requires it as
a small, separately-supplied JSON file (one {"converged": bool,
"matches_published": bool|null} entry per mechanism) rather than inventing
an automated proxy for a judgment nothing in the project docs claims can be
automated.

CI-AGGREGATION DESIGN DECISION, FLAGGED: each BatteryResult already carries
its own ci_lower/ci_upper (a per-seed quantity, e.g. bootstrapped over
per-step values within that one run). The bucket classification in
stats/failure_classifier.py needs a per-(mechanism, condition) CI
aggregated ACROSS seeds instead, to compare Real against each control at
the level the experiment design's §9/§10 actually operate at. This module
computes that aggregate CI fresh, via stats/bootstrap_ci.py over each
condition's collection of native_metric point estimates (one per seed) --
it does not attempt to combine or reuse each seed's own already-stored
ci_lower/ci_upper, which is a different statistical object. Worth
confirming this reading matches the intended methodology before this
report's bucket assignments are trusted; it is this module's own
best-effort synthesis of §9 and §10 together, not something spelled out
verbatim in one place in the docs.

RUNTIME VERIFICATION NOTE: this module has NO torch dependency anywhere --
it only ever consumes already-computed scalars (BatteryResult's fields
loaded from JSON) and calls stats/ + viz/, none of which touch a model or
tensor. It is therefore fully runtime-tested end to end in this sandbox,
not merely syntax-checked, including the plots it produces.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from stats.bootstrap_ci import BootstrapCI, bootstrap_ci
from stats.c3_correlation import C3Result, compute_c3
from stats.failure_classifier import ReproductionFidelity, cis_overlap, classify
from stats.result import BatteryResult
from viz.c3_scatter import plot_c3_scatter
from viz.pass_fail_heatmap import plot_pass_fail_heatmap

_CONTROL_CONDITIONS = ("step_matched", "scrambled")


class DataProvenanceError(RuntimeError):
    """Raised when asked to produce a headline table/plot from a result set
    containing zero data_source='real' entries -- every available result
    was a dev-mode artifact, which is a loud failure, not a quietly-empty
    report (architecture §11, new per the §0.5 amendment)."""


def enforce_data_provenance_gate(results: List[BatteryResult]) -> List[BatteryResult]:
    """Filter to data_source=='real' results, raising DataProvenanceError if
    that leaves nothing. Filtering out data_source=='synthetic' entries
    silently (rather than raising on their mere presence) is deliberate:
    synthetic entries legitimately coexist with real ones during ordinary
    development (wiring tests run before the real audit even starts), so
    their presence alone is not an error -- only an *all-synthetic* result
    set is.
    """
    real_results = [r for r in results if r.data_source == "real"]
    if not real_results:
        raise DataProvenanceError(
            f"refusing to produce a headline report: {len(results)} "
            f"result(s) were provided but ZERO have data_source='real' -- "
            f"every available result is a dev-mode/wiring-test artifact "
            f"(architecture §0.5/§11)"
        )
    return real_results


def load_battery_results(results_dir: str) -> List[BatteryResult]:
    """Load every *.json BatteryResult file in results_dir -- the
    deterministic flat-file convention (architecture §12), one file per
    (mechanism, condition, seed)."""
    directory = Path(results_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"results directory not found: {results_dir}")
    results = []
    for path in sorted(directory.glob("*.json")):
        with open(path, "r") as f:
            raw = json.load(f)
        results.append(BatteryResult.from_dict(raw))
    return results


def load_reproduction_fidelity(path: str) -> Dict[str, ReproductionFidelity]:
    """Load the hand-supplied reproduction-fidelity judgment per mechanism
    -- see module docstring's REPRODUCTION-FIDELITY INPUT note."""
    with open(path, "r") as f:
        raw = json.load(f)
    return {
        mechanism: ReproductionFidelity(
            converged=entry["converged"],
            matches_published=entry.get("matches_published"),
        )
        for mechanism, entry in raw.items()
    }


def _group_by_mechanism_condition(
    results: List[BatteryResult],
) -> Dict[str, Dict[str, List[BatteryResult]]]:
    grouped: Dict[str, Dict[str, List[BatteryResult]]] = {}
    for r in results:
        grouped.setdefault(r.mechanism, {}).setdefault(r.condition, []).append(r)
    return grouped


def _condition_ci(results: List[BatteryResult], seed: int) -> BootstrapCI:
    """Bootstrap CI over this (mechanism, condition)'s native_metric point
    estimates, one per seed -- see module docstring's CI-AGGREGATION DESIGN
    DECISION note for why this is computed fresh rather than reusing each
    seed's own stored ci_lower/ci_upper."""
    native_metrics = [r.native_metric for r in results]
    return bootstrap_ci(native_metrics, statistic=np.mean, seed=seed)


def build_pass_fail_grid_and_buckets(
    real_results: List[BatteryResult],
    reproduction_fidelity: Dict[str, ReproductionFidelity],
    bootstrap_seed: int = 0,
) -> "tuple[Dict[str, Dict[str, bool]], Dict[str, str]]":
    """Returns (pass_fail_grid, bucket_by_mechanism).

    pass_fail_grid[mechanism][control_condition] = True iff Real is
    distinguishable from that control (CIs do not overlap) -- "Real" itself
    is not a column, since it is the reference every other condition is
    compared against, not something compared against itself.

    bucket_by_mechanism[mechanism] is exactly one of "a"/"b"/"c"/"d", per
    stats/failure_classifier.py's classify(), for every mechanism present in
    reproduction_fidelity (a mechanism absent from real_results entirely --
    e.g. bucket "c", never converged, no battery ever run -- still gets a
    bucket, using only its ReproductionFidelity and no CIs).
    """
    grouped = _group_by_mechanism_condition(real_results)
    pass_fail_grid: Dict[str, Dict[str, bool]] = {}
    bucket_by_mechanism: Dict[str, str] = {}

    for mechanism, fidelity in reproduction_fidelity.items():
        if not fidelity.converged or fidelity.matches_published is False:
            bucket_by_mechanism[mechanism] = classify(fidelity)
            continue  # no battery CIs needed or available to build a grid row from

        conditions_for_mech = grouped.get(mechanism, {})
        missing = [c for c in ("real",) + _CONTROL_CONDITIONS if c not in conditions_for_mech]
        if missing:
            raise ValueError(
                f"mechanism {mechanism!r} has converged, published-matching "
                f"reproduction fidelity but is missing data_source=real "
                f"battery results for condition(s) {missing} -- cannot "
                f"classify or build its heatmap row without all three"
            )

        real_ci = _condition_ci(conditions_for_mech["real"], seed=bootstrap_seed)
        control_cis = {
            c: _condition_ci(conditions_for_mech[c], seed=bootstrap_seed)
            for c in _CONTROL_CONDITIONS
        }

        pass_fail_grid[mechanism] = {
            c: not cis_overlap(real_ci, control_cis[c]) for c in _CONTROL_CONDITIONS
        }
        bucket_by_mechanism[mechanism] = classify(
            fidelity, real_ci, control_cis["step_matched"], control_cis["scrambled"]
        )

    return pass_fail_grid, bucket_by_mechanism


def build_c3(
    real_results: List[BatteryResult], reported_effect_size_by_mechanism: Dict[str, float]
) -> C3Result:
    """MI score, averaged across seeds per (mechanism, condition='real'), is
    this module's chosen "battery outcome" input to C3 -- experiment design
    §4 says C3 correlates effect size against "battery outcome (MI score +
    pass/fail)" without pinning down exactly how MI score alone should be
    reduced to one number per mechanism when multiple seeds exist; mean
    across seeds is the natural, unflagged-elsewhere choice, so it is
    flagged here instead.
    """
    grouped = _group_by_mechanism_condition(real_results)
    battery_outcome_by_mechanism = {}
    for mechanism in ("gate_ml", "rigl", "grokfast"):
        if mechanism not in grouped or "real" not in grouped[mechanism]:
            raise ValueError(
                f"C3 requires data_source=real 'real'-condition results for "
                f"{mechanism!r}, none found"
            )
        mi_scores = [r.mi_score for r in grouped[mechanism]["real"]]
        battery_outcome_by_mechanism[mechanism] = float(np.mean(mi_scores))

    return compute_c3(reported_effect_size_by_mechanism, battery_outcome_by_mechanism)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble the headline pass/fail table, C3 correlation, and "
            "plots from completed BatteryResults."
        )
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--reproduction-fidelity-file", required=True)
    parser.add_argument("--effect-sizes-file", required=True,
                         help="JSON: {mechanism_name: normalized_effect_size}")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    try:
        all_results = load_battery_results(args.results_dir)
        real_results = enforce_data_provenance_gate(all_results)
        reproduction_fidelity = load_reproduction_fidelity(args.reproduction_fidelity_file)
        with open(args.effect_sizes_file, "r") as f:
            effect_sizes = json.load(f)
    except (FileNotFoundError, DataProvenanceError, KeyError, ValueError) as e:
        print(f"[report] REFUSING to run: {e}", file=sys.stderr)
        return 1

    pass_fail_grid, bucket_by_mechanism = build_pass_fail_grid_and_buckets(
        real_results, reproduction_fidelity
    )
    c3_result = build_c3(real_results, effect_sizes)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mechanisms = sorted(reproduction_fidelity.keys())
    if pass_fail_grid:
        plot_pass_fail_heatmap(
            mechanisms, list(_CONTROL_CONDITIONS), pass_fail_grid,
            str(output_dir / "pass_fail_heatmap.png"),
        )
    plot_c3_scatter(c3_result, str(output_dir / "c3_scatter.png"))

    with open(output_dir / "failure_buckets.json", "w") as f:
        json.dump(bucket_by_mechanism, f, indent=2)

    print(f"[report] wrote report artifacts to {output_dir}")
    print(f"[report] failure buckets: {bucket_by_mechanism}")
    print(
        f"[report] C3: degenerate={c3_result.degenerate}, "
        f"correlation={c3_result.correlation}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
