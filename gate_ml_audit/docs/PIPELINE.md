# Pipeline Components

[← back to README](../README.md)

Every stage in execution order, what it's built from, and what it hands to the next stage.

## 1. Configuration — `config/`

- `schema.py` — dataclasses (`BaseConfig`, `MechanismConfig`, `ConditionConfig`, `RunConfig`) + a hand-written ~40-line merge utility. No config framework (Hydra, OmegaConf) — the project runs across too many heterogeneous environments for a shared dependency version to be a safe assumption.
- `validate.py` — fails fast on an unregistered mechanism name, a mismatched stub/implemented status, a malformed condition, or an invalid numeric field, before any compute time is spent.
- Composition order: `base.yaml` → `mechanisms/{name}.yaml` → `conditions/{name}.yaml` → CLI overrides (always logged, never silent).

## 2. Data pipeline — `data/`

One `DataSource` protocol (`__iter__` → token batches), two interchangeable implementations:

- `synthetic_loader.py` — `torch.randint` batches, no filesystem or network I/O. Wiring-only; never a trusted audit result.
- `real_loader.py` — streams tokens from disk one line at a time (bounded memory), tokenized via `tokenizer.py` (Hugging Face `tokenizers`, BPE).
- `difficulty_proxy.py` — length + rarity composite, consumed only by the curriculum-ordering mechanism (currently a stub).

## 3. Mechanism + Battery — `mechanisms/`, `battery/`

- `mechanisms/base.py` — the `Mechanism` protocol (`signal`, `should_intervene`, `apply`) and `StubMechanism` (registry-valid placeholder whose `apply()` raises `NotImplementedError` rather than silently no-op-ing).
- `mechanisms/gate_ml/` — the one fully-implemented mechanism.
- `mechanisms/{rigl,grokfast,curriculum}/` — registry stubs.
- `battery/conditions.py` — `RealWrapper` (passthrough), `StepMatchedWrapper` (fixed schedule, ignores the live signal), `ScrambledWrapper` (resamples the mechanism's own observed signal values out of order — same marginal distribution, decorrelated from the actual per-step state).

## 4. Training engine — `training/`

- `model.py` — `TinyDecoderLM`, a small decoder-only transformer built from documented `torch.nn` blocks only.
- `loop.py` — the single loop for every mechanism × condition × device × data_source. `.to(device)` is the only device-specific line.
- `optim.py` — AdamW + linear-warmup scheduler, both driven entirely by config.

## 5. Checkpointing & Logging — `checkpointing/`, `logs/*.jsonl`

- Two checkpoint types: **trigger-event** (kept forever — the audit trail) and **periodic** (rolling window, default 2, for resumability only).
- Atomic writes (temp file + `os.replace`) so a killed process never leaves a half-written checkpoint.
- Resume is checked at every startup by default, not treated as an edge case.
- JSONL log: one line per step, local-first, `device`/`data_source` on every single line.

## 6. Eval pipeline — `eval/`

One `evaluate()` in `harness.py`, called identically by the in-loop periodic path and the post-hoc path in `orchestration/report.py` — so the two numbers can never be computed by two different implementations that quietly drift apart.

## 7. Validation pipeline — `falsification_audit/` (pre-existing, untouched)

`mi_estimator.py`'s permutation test + `auditbench.py`'s synthetic calibration sweep. Sits beside the main line, not inside it — it validates the *statistics*, never a real mechanism's result.

## 8. Audit entrypoint — `orchestration/audit.py`

Checksum-gates on the two files above before allowing anything downstream to trust their output.

## 9. Statistics — `stats/`

Bootstrap CI (percentile method) → Holm–Bonferroni correction → 4-bucket failure classification (a/b/c/d) → C3 correlation (the 3 audited mechanisms only, Spearman, descriptive-only — no `p_value` field exists on the result type).

## 10. Report & Visualization — `orchestration/report.py`, `viz/`

Filters to `data_source=real`, assembles the pass/fail grid and C3 result, and produces three static PNGs (signal trajectory, pass/fail heatmap, C3 scatter with the N=3 caveat rendered on the plot itself, not just the caption).

---

See [`docs/DATA_FLOW.md`](DATA_FLOW.md) for how state moves between these stages, and [`docs/MODULES.md`](MODULES.md) for file-by-file detail and verification status.
