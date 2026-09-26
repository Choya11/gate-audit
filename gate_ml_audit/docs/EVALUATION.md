# Evaluation Metrics

[← back to README](../README.md)

All of these are produced by `eval/harness.py`'s single `evaluate()` function — called identically whether it's the periodic in-loop check during training or the post-hoc check on a saved checkpoint.

## Perplexity — `eval/perplexity.py`

Standard token-level perplexity: `exp(total_loss / total_target_tokens)`, computed with the model in eval mode and no gradient tracking. Token-weighted, not sequence-averaged — a sequence-average would weight a 5-token and a 500-token example equally, which is the wrong thing to do when comparing across languages with very different average sequence lengths.

## Task accuracy — `eval/task_accuracy.py`

Exact-match accuracy via greedy (argmax) decoding, one new token at a time. Greedy, not sampled — sampling would make accuracy nondeterministic across runs, which is exactly wrong for an audit whose whole point is reproducible comparison.

> This module is deliberately generic — it knows nothing about ChakmaBridge or IndicXTREME's specific file formats. Turning a raw task dataset into `(prompt_token_ids, expected_answer_token_ids)` pairs is a `data/`-layer job, not built in this pass (no real task datasets were reachable to build or test that loader against).

## Resource gap — `eval/resource_gap.py`

`gap[lang] = perplexity[lang] − perplexity[high_resource_language]` (English, by default), computed at one checkpoint.

> ⚠️ **Flagged, not confirmed:** this is an *absolute* gap, not a relative/normalized one. An absolute gap shrinks as overall perplexity falls during training even if the real disparity between languages is constant — worth a deliberate choice before this number appears in a headline table, not left as this module's silent default.

## Steps-to-generalization — `eval/resource_gap.py`

The first training step at which a per-step metric crosses a threshold **and stays on the good side of it for the rest of the run** — not merely touches it once, which could be noise.

> ⚠️ **Flagged, not confirmed:** the "crosses and stays" reading is this codebase's own definition. Nothing in the project docs pins down "generalization" more precisely than the name itself.

## Falsification-battery statistics — `stats/`

These are a different layer from the per-checkpoint metrics above — they operate on a mechanism's *battery outcome*, not on one checkpoint:

- **`bootstrap_ci.py`** — percentile bootstrap CI (not BCa; nothing in the docs calls for the more complex variant).
- **`holm_bonferroni.py`** — step-down multiple-comparison correction across the full family of tests (not per-mechanism in isolation).
- **`failure_classifier.py`** — the 4-bucket classification every audit run gets, before any conclusion is drawn:

  | Bucket | Meaning |
  |---|---|
  | **a** | Clean reproduction, clean battery result — trustworthy, goes in the headline table |
  | **b** | Reproduction diverges from published numbers, battery still run — report both, flagged |
  | **c** | Reproduction fails to converge — excluded from the headline table |
  | **d** | Real is statistically indistinguishable from *both* Step-matched and Scrambled — inconclusive, not forced into a binary verdict |

  "Statistically indistinguishable" is operationalized as bootstrap-CI overlap — a conservative reading, and reused identically by `orchestration/report.py` so the two call sites' notion of "indistinguishable" can't drift apart.

- **`c3_correlation.py`** — Spearman's ρ between each of the **3 audited mechanisms'** (GATE-ML, RigL, GrokFast — curriculum-ordering is a calibration case, not a genuine unknown, and is excluded here specifically) originally-reported effect size and its battery outcome (mean MI score). Two things enforced structurally, not by convention:
  - **No `p_value` field exists on `C3Result` at all.** At N=3, a p-value would misleadingly imply more statistical power than three data points can support — this is reported as a descriptive pattern, never a hypothesis test, and the type system makes the mistake of reporting one impossible rather than merely discouraged.
  - **Degeneracy is a first-class outcome, not a crash.** If all three mechanisms land on the same side of pass/fail (or tie on effect size), the correlation is mathematically undefined — `C3Result.degenerate=True`, `correlation=None`, and the case-by-case table is still populated for a qualitative write-up.
