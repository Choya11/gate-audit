"""Shared BPE tokenizer, trained once across all languages in a run (per
the original GATE-ML tokenizer design referenced in architecture §0/§4:
"shared BPE vocab across languages"). Built on Hugging Face `tokenizers`
(added to requirements.txt) rather than a hand-rolled BPE trainer -- BPE
merge-order edge cases are exactly the kind of subtly-wrong-at-the-margins
logic worth depending on a documented, widely-used library for, rather than
a from-scratch implementation whose bugs would surface as silently-corrupted
real-corpus token sequences months into the audit.

RUNTIME VERIFICATION NOTE: `tokenizers` was not attempted for install in
this sandbox -- unlike torch, this module is on the data_source=real path
only, which this sandbox cannot exercise regardless (no corpus files, no
network route to the real corpora). Syntax-checked only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

_SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>"]


class Tokenizer:
    """Thin wrapper exposing only encode/decode/vocab_size/save/load, so
    nothing outside this file depends on the `tokenizers` library's own API
    surface directly -- swapping the underlying BPE implementation later
    only touches this one file."""

    def __init__(self, hf_tokenizer: HFTokenizer) -> None:
        self._hf = hf_tokenizer

    def encode(self, text: str) -> List[int]:
        return self._hf.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self._hf.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._hf.get_vocab_size()

    def save(self, path: str) -> None:
        self._hf.save(path)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        if not Path(path).is_file():
            raise FileNotFoundError(f"tokenizer file not found: {path}")
        return cls(HFTokenizer.from_file(path))


def train_bpe(corpus_paths: Iterable[str], vocab_size: int) -> Tokenizer:
    """Train a single shared BPE tokenizer across every corpus file in
    corpus_paths (one file per language). Raises FileNotFoundError up front
    for any missing path, rather than letting the underlying trainer fail
    with a less specific error partway through training."""
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    paths = [str(p) for p in corpus_paths]
    if not paths:
        raise ValueError("corpus_paths must be non-empty")
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"corpus file(s) not found: {missing}")

    hf_tokenizer = HFTokenizer(BPE(unk_token="<unk>"))
    hf_tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=_SPECIAL_TOKENS)
    hf_tokenizer.train(paths, trainer)
    return Tokenizer(hf_tokenizer)
