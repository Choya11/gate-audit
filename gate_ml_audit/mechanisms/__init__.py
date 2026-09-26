"""Mechanism registry: maps a config-declared mechanism name to the class
that implements it, plus whether that class is a real implementation or a
registry-valid stub.

config/validate.py imports this to reject unregistered mechanism names and
to cross-check a mechanism's declared config status ("implemented" vs
"stub") against what is actually registered here -- the two are required to
agree, so a config typo can't silently run a stub as if it were real, or
vice versa. This is also the single place a 5th mechanism gets added later
(architecture §12's "additive via the Mechanism registry" scalability
claim) -- a new submodule plus one new line here, no change to training/,
battery/, or stats/.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from mechanisms.base import Mechanism
from mechanisms.gate_ml.mechanism import GateMLMechanism
from mechanisms.rigl.mechanism import RiglMechanism
from mechanisms.grokfast.mechanism import GrokFastMechanism
from mechanisms.curriculum.mechanism import CurriculumMechanism


@dataclass(frozen=True)
class RegistryEntry:
    cls: Type
    is_stub: bool


MECHANISM_REGISTRY: dict = {
    "gate_ml": RegistryEntry(cls=GateMLMechanism, is_stub=False),
    "rigl": RegistryEntry(cls=RiglMechanism, is_stub=True),
    "grokfast": RegistryEntry(cls=GrokFastMechanism, is_stub=True),
    "curriculum": RegistryEntry(cls=CurriculumMechanism, is_stub=True),
}


def build_mechanism(name: str, params: dict):
    """Instantiate the registered Mechanism for `name` from its config params.

    Raises KeyError if `name` is not registered. Callers (config/validate.py,
    orchestration/*.py) are expected to have already validated the name
    against MECHANISM_REGISTRY before calling this -- this function
    constructs, it does not re-validate.
    """
    if name not in MECHANISM_REGISTRY:
        raise KeyError(
            f"unregistered mechanism name {name!r}; known mechanisms: "
            f"{sorted(MECHANISM_REGISTRY)}"
        )
    entry = MECHANISM_REGISTRY[name]
    return entry.cls(**params)
