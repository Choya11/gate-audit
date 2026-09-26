"""DataSource protocol: the small interface both the real loader
(data/real_loader.py) and the synthetic generator (data/synthetic_loader.py)
implement, so training/loop.py never branches on which one it received
(architecture §2's secondary interfaces; the §0.5 amendment's device/
data_source config knob is what selects which implementation gets
constructed, never the training loop itself).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable

import torch


@dataclass
class Batch:
    # LongTensor [batch, seq_len + 1] -- the training loop slices this into
    # input_ids = token_ids[:, :-1] and target_ids = token_ids[:, 1:] for
    # standard next-token-prediction causal LM training. The +1 keeps this
    # shift entirely inside the training loop, so DataSource implementations
    # never need to know about it.
    token_ids: torch.Tensor


@runtime_checkable
class DataSource(Protocol):
    def __iter__(self) -> Iterator[Batch]:
        ...
