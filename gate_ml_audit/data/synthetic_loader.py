"""SyntheticDataSource: an in-code, filesystem- and network-free DataSource
for data_source=synthetic -- wiring tests in constrained sandboxes only,
per the §0.5 amendment. Structurally identical `__iter__` signature to
RealDataSource so training/loop.py cannot tell them apart.

Per §0.5 item 3: any BatteryResult produced against this data source is
tagged data_source="synthetic" downstream and is hard-gated out of
orchestration/report.py's headline output. This class exists to validate
wiring (config composition, the Mechanism interface, checkpoint format,
logging schema), never to produce a trusted mechanism-audit result -- there
is nothing for a reproduction-fidelity check to compare a synthetic run
against.

RUNTIME VERIFICATION NOTE: imports torch (torch.randint, torch.Generator).
Syntax-checked only in this sandbox; the batch-shape logic has been traced
by hand (see inline comment) but not executed against a real torch install.
"""
from __future__ import annotations

from typing import Iterator

import torch

from data import Batch


class SyntheticDataSource:
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        batch_size: int,
        num_batches: int,
        seed: int,
    ) -> None:
        if vocab_size <= 0 or seq_len <= 0 or batch_size <= 0 or num_batches <= 0:
            raise ValueError(
                "vocab_size, seq_len, batch_size, and num_batches must all be "
                "positive integers"
            )
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_batches = num_batches
        self._generator = torch.Generator().manual_seed(seed)

    def __iter__(self) -> Iterator[Batch]:
        for _ in range(self.num_batches):
            # +1 so the training loop can slice input/target via a
            # one-position shift; shape [batch_size, seq_len + 1].
            token_ids = torch.randint(
                low=0,
                high=self.vocab_size,
                size=(self.batch_size, self.seq_len + 1),
                generator=self._generator,
            )
            yield Batch(token_ids=token_ids)
