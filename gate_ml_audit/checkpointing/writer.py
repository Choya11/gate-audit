"""Checkpoint writer: serializes a CheckpointState to disk with an atomic
temp-path + rename, so a killed process (Colab/Kaggle session limits,
architecture §12/§13.2) never leaves a half-written checkpoint that looks
valid but isn't (architecture §11's explicit error-handling requirement).

RUNTIME VERIFICATION NOTE: torch.save requires torch, unavailable in this
sandbox. Syntax-checked only; the atomic-write mechanics (tempfile in the
same directory + os.replace, cleanup on any exception) are standard,
documented stdlib usage, not novel logic, which is exactly why this file
is kept this thin rather than adding a hand-rolled serialization format on
top of torch.save.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from checkpointing.state import CheckpointState


def save(state: CheckpointState, path: str) -> None:
    """Serialize `state` to `path`, atomically.

    Writes to a temp file in the same directory as `path` (same filesystem,
    so the final os.replace is a true atomic rename, not a cross-filesystem
    copy) and only renames it into place after the write and an explicit
    fsync have both succeeded. On any failure during the write, the temp
    file is removed so no stray .tmp files accumulate -- `target` itself is
    never touched until the rename, so it is never left half-written either
    way.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state.to_dict(), f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(target))
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def save_trigger_event(state: CheckpointState, checkpoint_dir: str) -> str:
    """Save a trigger-event checkpoint -- always kept, never pruned by
    checkpointing/retention.py's rolling-window policy (architecture §8).
    Returns the path written to."""
    state.is_trigger_event = True
    path = str(Path(checkpoint_dir) / f"{state.naming_key()}_trigger.pt")
    save(state, path)
    return path


def save_periodic(state: CheckpointState, checkpoint_dir: str) -> str:
    """Save a periodic checkpoint -- subject to retention.py's rolling
    window of the most recent N (architecture §8). Returns the path written
    to."""
    state.is_trigger_event = False
    path = str(Path(checkpoint_dir) / f"{state.naming_key()}_periodic.pt")
    save(state, path)
    return path
