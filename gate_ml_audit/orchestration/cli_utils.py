"""CLI override parsing, split into its own file specifically so it has no
transitive torch dependency and can be imported/tested in isolation --
orchestration/train.py itself imports training.loop, which imports torch,
so anything living in train.py's own module cannot be runtime-tested in a
sandbox where torch is unavailable, no matter how torch-free that
particular function's own body is. Splitting this one function out is a
real design decision earned by that constraint, not just file organization
for its own sake.

Pure Python, no dependencies beyond stdlib -- runtime-tested in this
sandbox.
"""
from __future__ import annotations

from typing import Dict, List


def parse_cli_overrides(raw_overrides: List[str]) -> Dict[str, object]:
    """Parse repeated --override key=value strings into a dict, with
    best-effort type coercion (int, then float, then bool, else str) -- this
    covers every value type actually used in base.yaml without requiring the
    caller to know which type each dotted config path expects ahead of time.
    """
    overrides: Dict[str, object] = {}
    for item in raw_overrides:
        if "=" not in item:
            raise ValueError(f"--override must be of the form key=value, got {item!r}")
        key, _, raw_value = item.partition("=")
        if not key:
            raise ValueError(f"--override key must be non-empty, got {item!r}")

        value: object
        try:
            value = int(raw_value)
        except ValueError:
            try:
                value = float(raw_value)
            except ValueError:
                if raw_value.lower() in ("true", "false"):
                    value = raw_value.lower() == "true"
                else:
                    value = raw_value
        overrides[key] = value
    return overrides
