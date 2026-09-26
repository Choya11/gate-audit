"""Checkpoint reader: discovers and loads the latest valid periodic
checkpoint for a given run, so orchestration/train.py's resume-by-default
path (architecture §11: "resume is a first-class code path, not a manual
recovery procedure") has something concrete to call at startup.

MATCHING NOTE: see checkpointing/retention.py's module docstring -- mechanism
and condition names can themselves contain underscores ("gate_ml",
"step_matched"), so filenames are matched against a known prefix/suffix for
the specific (mechanism, condition, seed) the caller already knows it wants,
never parsed apart with a generic underscore-splitting regex. This file and
retention.py intentionally duplicate this small amount of matching logic
rather than sharing a helper across a torch-dependent and a torch-free
module -- keeping retention.py torch-free is what makes it runtime-testable
in a sandbox where torch cannot be installed (see its own tests), and that
was worth more than deduplicating roughly ten lines.

RUNTIME VERIFICATION NOTE: torch.load requires torch, unavailable in this
sandbox. Syntax-checked only.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch

from checkpointing.state import CheckpointState


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


def find_latest_valid(
    checkpoint_dir: str, mechanism: str, condition: str, seed: int
) -> Optional[CheckpointState]:
    """Return the highest-step periodic checkpoint for (mechanism,
    condition, seed) that loads successfully, or None if there is no valid
    periodic checkpoint yet (a fresh run, not an error).

    "Valid" means: matches the naming scheme AND torch.load succeeds AND
    CheckpointState.from_dict succeeds against its contents. A checkpoint
    that fails either of the latter two -- which per checkpointing/writer.py's
    atomic-rename guarantee should never happen from an interrupted write,
    but a corrupted-on-disk file from some other cause still could -- is
    skipped in favor of the next-highest step, rather than raised: resume
    must not become an all-or-nothing gate on the single newest file when an
    older, still-good checkpoint is right there.
    """
    candidates = _matching_periodic_files(checkpoint_dir, mechanism, condition, seed)
    for _, path in candidates:
        try:
            raw = torch.load(str(path), map_location="cpu")
            return CheckpointState.from_dict(raw)
        except Exception:
            continue
    return None
