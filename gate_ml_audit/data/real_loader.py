"""RealDataSource: per-language corpus loading + tokenization + toy-scale
subsampling for data_source=real.

Streams tokens from disk one line at a time rather than loading a full
corpus into memory -- architecture §13.3's 16GB-system-RAM bottleneck
applies independently of the 8GB VRAM ceiling, so the data pipeline has to
respect it regardless of which GPU is available.

RUNTIME VERIFICATION NOTE: imports torch (torch.tensor at batch-yield time
only). Not executable in this sandbox (no real corpus files, no network
route to FineWeb-Edu/IndicCorp/ChakmaBridge, and torch itself unavailable
here regardless). Syntax-checked and manually traced only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

import torch

from data import Batch
from data.tokenizer import Tokenizer


class RealDataSource:
    def __init__(
        self,
        shard_paths: List[str],
        tokenizer: Tokenizer,
        seq_len: int,
        batch_size: int,
        token_budget: int,
    ) -> None:
        if not shard_paths:
            raise ValueError("shard_paths must be non-empty for data_source=real")
        missing = [p for p in shard_paths if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"corpus shard(s) not found: {missing}")
        if seq_len <= 0 or batch_size <= 0 or token_budget <= 0:
            raise ValueError(
                "seq_len, batch_size, and token_budget must all be positive"
            )

        self.shard_paths = shard_paths
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.token_budget = token_budget

    def _stream_tokens(self) -> Iterator[int]:
        """Yield token ids one shard, one line at a time. Never materializes
        more than one line's worth of text or token ids in memory at once,
        per the streaming/chunked requirement in architecture §13.3."""
        tokens_yielded = 0
        for shard_path in self.shard_paths:
            with open(shard_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    for token_id in self.tokenizer.encode(line):
                        if tokens_yielded >= self.token_budget:
                            return
                        yield token_id
                        tokens_yielded += 1

    def __iter__(self) -> Iterator[Batch]:
        buffer: List[int] = []
        batch_rows: List[List[int]] = []
        for token_id in self._stream_tokens():
            buffer.append(token_id)
            if len(buffer) == self.seq_len + 1:
                batch_rows.append(buffer)
                buffer = []
                if len(batch_rows) == self.batch_size:
                    yield Batch(token_ids=torch.tensor(batch_rows, dtype=torch.long))
                    batch_rows = []
        # ponytail: a final partial batch (fewer than batch_size full rows)
        # and any leftover `buffer` tokens (fewer than seq_len + 1) are
        # dropped rather than padded. Padding a causal-LM batch correctly
        # needs an attention mask threaded through TinyDecoderLM.forward(),
        # which is out of scope for this toy-scale pass; the data lost this
        # way is bounded by (batch_size * seq_len) + seq_len tokens per shard
        # sweep. Add padding + an attention mask if the toy-scale token
        # budget ever gets small enough for that loss to matter.
