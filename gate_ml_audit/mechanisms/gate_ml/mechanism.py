"""GateMLMechanism: cosine-divergence-triggered embedding tying.

Signal: an EMA-smoothed cosine distance between the input-embedding and
output-projection gradients (GateMLState.update). Intervention: once that
EMA has stayed at or below tau_l for `window` consecutive observations, tie
the two embedding matrices together (model.set_tied(True)) and never untie
again for the rest of the run -- this is a one-shot, permanent intervention,
not a repeated one.

DESIGN NOTE ON TIE/UNTIE MECHANICS (read before touching training/model.py):
naively reassigning model.output_head_weight to literally be the same
nn.Parameter object as model.input_embedding.weight, after a
torch.optim.Optimizer has already been constructed over the model's original
parameter list, does not work: Optimizer.param_groups holds direct
references to the original Parameter objects, so a later reassignment on the
model is invisible to an already-built optimizer and the "shared" parameter
would silently stop receiving updates through that optimizer. To avoid this
bug entirely, TinyDecoderLM (training/model.py) is built with a fixed set of
parameters for the model's whole lifetime -- both input_embedding.weight and
output_head_weight always exist and are both registered with the optimizer
at construction time -- and a boolean `tied` flag on the model that only
changes *which tensor the forward pass reads from* for the output
projection. model.set_tied(True) additionally copies the current
input-embedding values into output_head_weight at the moment of the switch
(so the function computed has no discontinuity right at the transition),
then forward() ignores output_head_weight from then on; it receives no
further gradient, which torch.optim.Adam handles as a well-defined no-op
(grad is None -> that parameter's step is skipped), not an error.

OPEN QUESTION, FLAGGED RATHER THAN SILENTLY ASSUMED: the exact trigger
direction implemented here (tie once the two embeddings' gradients have
become similar enough that their cosine distance falls to or below tau_l)
is this implementation's best-effort reading of "cosine-divergence-triggered
tying" from the available project docs, which do not specify the trigger
direction or the exact role of the persistence window explicitly. This must
be checked against GATE-ML's original paper/reference implementation as part
of the reproduction-fidelity check (architecture §6) before any
BatteryResult produced with this mechanism is trusted. If the direction
turns out to be wrong, that is a reproduction-fidelity failure (experiment
design §10, buckets c/d), not a falsification-battery finding about the
mechanism itself -- these are two different kinds of "the result was bad"
and must not be conflated in the write-up.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from mechanisms.gate_ml.state import GateMLState
from training.state import TrainingState


class GateMLMechanism:
    def __init__(self, tau_l: float, window: int, beta: float) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"beta must be in [0.0, 1.0), got {beta}")
        self.tau_l = float(tau_l)
        self.window = int(window)
        self.state = GateMLState(beta=float(beta))

    def signal(self, state: TrainingState) -> float:
        """Compute (and EMA-smooth) the cosine distance between the input-
        and output-embedding gradients.

        Requires state.extra["model"] to be set by the training loop to the
        live model instance, and requires this to be called after
        loss.backward() but before optimizer.zero_grad() this step, so that
        .grad is populated on both embedding parameters. If either gradient
        is missing (first step of the run before any backward call, or the
        embeddings are already tied so output_head_weight receives no
        independent gradient) the EMA is held steady rather than fed a
        fabricated 0.0 -- a fabricated 0.0 would look like "already
        identical," which could spuriously help satisfy the persistence
        window with no real signal behind it.
        """
        model = state.extra.get("model")
        if model is None:
            raise KeyError(
                "GateMLMechanism.signal() requires state.extra['model'] to be "
                "set by the training loop -- this mechanism needs live "
                "gradient tensors on the model's embedding parameters"
            )

        input_grad = model.input_embedding.weight.grad
        output_grad = model.output_head_weight.grad

        if self.state.tied or input_grad is None or output_grad is None:
            return self.state.ema_cosine_distance if self.state.initialized else 0.0

        raw_similarity = F.cosine_similarity(
            input_grad.flatten().unsqueeze(0),
            output_grad.flatten().unsqueeze(0),
        ).item()
        raw_distance = 1.0 - raw_similarity
        return self.state.update(raw_distance)

    def should_intervene(self, signal_history: list, state: TrainingState) -> bool:
        """Fire once the EMA has stayed at or below tau_l for `window`
        consecutive observations, and never again once already tied.

        Uses the training loop's own `signal_history` (the same values
        self.state.signal_history holds, since both are appended to every
        time signal() is called) rather than reaching into self.state
        directly -- this keeps should_intervene() testable in isolation with
        a synthetic history list, independent of whether signal() has ever
        actually been called on this instance.
        """
        if self.state.tied:
            return False
        if len(signal_history) < self.window:
            return False
        recent = signal_history[-self.window:]
        return all(v <= self.tau_l for v in recent)

    def apply(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        state: TrainingState,
    ) -> None:
        """Tie the embeddings, once, permanently.

        Idempotent by construction (checks self.state.tied first) so that a
        battery wrapper calling apply() more than once in edge-case timing
        does not double-fire or raise.
        """
        if self.state.tied:
            return
        model.set_tied(True)
        self.state.tied = True
