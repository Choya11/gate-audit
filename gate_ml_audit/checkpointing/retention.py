"""Retention policy (architecture §8): keep every trigger-event checkpoint
and the final checkpoint; keep periodic checkpoints only as a rolling
window of the most recent N (default 2, from config's
checkpointing.periodic_retain) -- resumability, not archival. Enforced by
code, not by someone remembering to clean up manually on a deadline
(architecture §13.4's storage-exhaustion bottleneck).

Pure Python, no torch dependency (this file only ever manipulates
filenames, never checkpoint contents) -- fully runtime-tested in this
sandbox.

MATCHING NOTE: mechanism names ("gate_ml") and condition names
("step_matched") both legitimately contain underscores, so the naming
scheme "{mechanism}_{condition}_{seed}_{step}_periodic.pt" cannot be parsed
back apart with a generic underscore-splitting regex -- an earlier version
of this file tried exactly that and it silently matched nothing (caught by
this file's own test suite, see the runtime verification that follows).
Since every caller already knows which mechanism/condition/seed it is
filtering for, this file never needs to *parse* a filename into components
at all -- it only needs to check whether a filename matches the *known*
prefix "{mechanism}_{condition}_{seed}_" and suffix "_periodic.pt" for the
specific values the caller passed in, then extract the step number from
between them. This assumes the four registered mechanism names and three
condition names never collide as string prefixes of one another (true for
the current fixed vocab: gate_ml/rigl/grokfast/curriculum and
real/step_matched/scrambled) -- worth re-checking by hand if a 5th
mechanism or 4th condition is ever added with a name that could be a prefix
of an existing one.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


def _matching_periodic_files(
    checkpoint_dir: str, mechanism: str, condition: str, seed: int
) -> List[Tuple[int, Path]]:
    directory = Path(checkpoint_dir)
    if not directory.is_dir():
        return []

    prefix = f"{mechanism}_{condition}_{seed}_"
    suffix = "_periodic.pt"
    matches: List[Tuple[int, Path]] = []
    for entry in directory.iterdir():
        name = entry.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        step_str = name[len(prefix): -len(suffix)]
        if not step_str.isdigit():
            continue
        matches.append((int(step_str), entry))

    matches.sort(key=lambda pair: pair[0], reverse=True)
    return matches


def prune_periodic(
    checkpoint_dir: str, mechanism: str, condition: str, seed: int, keep: int
) -> List[str]:
    """Delete all but the `keep` most recent periodic checkpoints for
    (mechanism, condition, seed). Returns the paths deleted, for logging.

    Trigger-event checkpoints are never touched by this function -- they
    are named with a "_trigger.pt" suffix by checkpointing/writer.py, not
    "_periodic.pt", so _PERIODIC_PATTERN never matches them; there is no
    risk of this function accidentally deleting an audit-trail checkpoint.
    """
    if keep < 0:
        raise ValueError(f"keep must be >= 0, got {keep}")
    candidates = _matching_periodic_files(checkpoint_dir, mechanism, condition, seed)
    to_delete = candidates[keep:]
    deleted_paths = []
    for _, path in to_delete:
        path.unlink()
        deleted_paths.append(str(path))
    return deleted_paths


def on_run_complete(
    checkpoint_dir: str, mechanism: str, condition: str, seed: int
) -> List[str]:
    """Called once a run finishes cleanly: deletes ALL periodic checkpoints
    for this run (keep=0). A completed run's periodic checkpoints existed
    only for mid-run resumability; once the run is done, only its final
    checkpoint (saved separately, as a permanent checkpoint by whatever code
    calls this) and its trigger-event checkpoints need to survive."""
    return prune_periodic(checkpoint_dir, mechanism, condition, seed, keep=0)
