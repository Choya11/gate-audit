# Supported Mechanisms

[← back to README](../README.md)

Adding a 5th mechanism later is additive — a new `mechanisms/<name>/` submodule plus one line in `MECHANISM_REGISTRY` — with no change to `training/`, `battery/`, or `stats/`. This is the whole point of the `Mechanism` protocol.

## GATE-ML — implemented

- **File:** `mechanisms/gate_ml/mechanism.py` + `state.py`
- **Signal:** EMA-smoothed cosine distance between the input-embedding and output-projection gradients.
- **Intervention:** tie the two embedding matrices together — once, permanently, for the rest of the run.
- **Trigger condition (implemented):** fires once the EMA has stayed at-or-below `tau_l` for `window` consecutive observations.

  > ⚠️ **Flagged, not confirmed:** neither the trigger *direction* nor the exact role of the persistence window is stated anywhere in the available project docs. This is a best-effort reading, and needs checking against GATE-ML's own paper/reference implementation as part of reproduction-fidelity — a wrong direction here is a reproduction-fidelity failure, not a finding about the mechanism.

- **A real PyTorch design decision worth knowing:** tying the embeddings is implemented as a boolean flag that changes which of two *always-registered* parameters the forward pass reads from — never by reassigning which `nn.Parameter` object is live. Reassigning the object after `torch.optim.Optimizer` construction is a documented footgun: the optimizer keeps a reference to the original object and the "shared" parameter silently stops receiving updates. See `TinyDecoderLM.set_tied()` in `training/model.py`.

## RigL — registry stub

- **File:** `mechanisms/rigl/mechanism.py`
- **Signal (when implemented):** gradient magnitude, deciding which pruned connections to regrow.
- **Current behavior:** `signal()` returns `0.0`, `should_intervene()` always returns `False`, `apply()` raises `NotImplementedError` if ever called (it won't be, under normal operation, since it's never triggered).

## GrokFast — registry stub

- **File:** `mechanisms/grokfast/mechanism.py`
- **Signal (when implemented):** EMA-filtered low-frequency component of the gradient.
- **Current behavior:** same stub pattern as RigL.

## Curriculum-ordering — registry stub

- **File:** `mechanisms/curriculum/mechanism.py`
- **Signal (when implemented):** the length + rarity difficulty proxy already built in `data/difficulty_proxy.py`.
- **Note:** this mechanism's intervention is data reordering/reweighting, not an architecture or optimizer change — its eventual `apply()` will need its own care mapping onto the `Mechanism` protocol (it would reorder the *next* batch rather than mutate the model or optimizer directly).

## Why stubs raise instead of silently no-op-ing

`StubMechanism.apply()` raises `NotImplementedError` on the (currently unreachable) path where it would be called. A silent no-op would let a battery run against an unimplemented mechanism complete "successfully" and produce a `BatteryResult` that looks like a real, uneventful run — a stub has to fail loudly the instant anything actually depends on it, not produce a quiet false negative.
