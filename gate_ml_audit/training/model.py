"""TinyDecoderLM: the toy-scale, mechanism-agnostic decoder-only LM.

Built from documented torch.nn building blocks only (nn.Embedding,
nn.TransformerEncoderLayer/nn.TransformerEncoder used as a causal decoder
stack via an explicit causal mask, nn.Parameter) -- no custom attention
implementation, so there is no hand-rolled tensor-shape logic to get subtly
wrong in the attention math itself.

TIED/UNTIED OUTPUT PROJECTION -- see mechanisms/gate_ml/mechanism.py's
docstring for the full rationale. Concretely: this model always owns two
d_model x vocab_size-shaped parameters --
`input_embedding.weight` (shape [vocab_size, d_model], via nn.Embedding) and
`output_head_weight` (shape [vocab_size, d_model], a bare nn.Parameter) --
for its entire lifetime. Both are handed to the optimizer at construction
time and neither is ever removed or replaced. The `tied` boolean only
changes which of the two the forward pass reads from when projecting hidden
states to vocabulary logits. This sidesteps the real, documented
torch.optim pitfall where reassigning which Parameter *object* backs an
attribute after Optimizer construction leaves the optimizer holding a
reference to the old (now orphaned) Parameter.

RUNTIME VERIFICATION NOTE: this file imports torch and has not been
executed in the sandbox it was written in (torch cannot be installed here
-- see requirements.txt / the earlier confirmed disk+network failures).
It has been syntax-checked with `python3 -m py_compile` and manually
traced for tensor-shape correctness at every step (see inline comments);
it has not been run against a real torch installation yet.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyDecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        if vocab_size <= 0 or d_model <= 0 or n_layers <= 0 or n_heads <= 0 or max_seq_len <= 0:
            raise ValueError(
                "vocab_size, d_model, n_layers, n_heads, and max_seq_len must "
                "all be positive integers"
            )

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.tied = False

        # [vocab_size, d_model] -- input_embedding(input_ids) yields
        # [batch, seq_len, d_model].
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        # [max_seq_len, d_model] -- learned absolute position embeddings,
        # looked up by position index, added to the token embedding.
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        # Used as a causal decoder stack: an explicit causal mask (built
        # fresh per forward() call, see below) is passed in, so no token
        # attends to a future position -- the standard documented pattern
        # for building a GPT-style decoder-only model out of
        # nn.TransformerEncoder blocks.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Bare nn.Parameter, same shape as input_embedding.weight
        # ([vocab_size, d_model]), always present and always registered with
        # the optimizer -- see module docstring for why this must never be
        # reassigned to a *different* Parameter object after construction.
        self.output_head_weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self.output_head_weight, mean=0.0, std=d_model ** -0.5)

    def set_tied(self, tied: bool) -> None:
        """Switch which parameter the output projection reads from.

        On a False -> True transition, copies the current input-embedding
        values into output_head_weight first, so the function this model
        computes has no discontinuity at the exact step of the switch --
        the projection weight's *values* carry over even though which
        Parameter object is live for future gradient updates changes.
        """
        if tied and not self.tied:
            with torch.no_grad():
                self.output_head_weight.copy_(self.input_embedding.weight)
        self.tied = tied

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: LongTensor [batch, seq_len] -> logits [batch, seq_len, vocab_size]."""
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2D [batch, seq_len], got shape {tuple(input_ids.shape)}"
            )
        batch, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        # [batch, seq_len, d_model] + [batch, seq_len, d_model] (broadcast on
        # batch via expand above) -> [batch, seq_len, d_model].
        hidden = self.input_embedding(input_ids) + self.position_embedding(positions)

        # [seq_len, seq_len] additive mask: 0 on/below the diagonal, -inf
        # above it, so position i cannot attend to position j > i.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_ids.device)
        hidden = self.encoder(hidden, mask=causal_mask)  # -> [batch, seq_len, d_model]

        weight = self.input_embedding.weight if self.tied else self.output_head_weight
        # F.linear(x, W) computes x @ W.T: [batch, seq_len, d_model] @
        # [d_model, vocab_size] -> [batch, seq_len, vocab_size].
        logits = F.linear(hidden, weight)
        return logits


def count_params(model: nn.Module) -> int:
    """Total trainable parameter count, logged once per run so the actual
    instantiated model size can be cross-checked against the scale declared
    in config (architecture §12's model-scale scalability claim depends on
    this being an honest count, not an assumed one)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
