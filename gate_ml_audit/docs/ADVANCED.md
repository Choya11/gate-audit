# Advanced Features & Known Gaps

[← back to README](../README.md)

## The one real gap: nothing assembles a `BatteryResult` yet

`orchestration/report.py` is fully built and fully tested — but it *starts* from a folder of `BatteryResult` JSON files that are assumed to already exist. The step in between — take a completed run's JSONL log, feed its signal column into `falsification_audit/mi_estimator.py`'s `permutation_test_mi`, combine that with the corresponding `EvalResult`, and write out one `BatteryResult` JSON per (mechanism, condition, seed) — was never written.

The schema (`stats/result.py`'s `BatteryResult`), the consumer (`report.py`), and the key ingredient (`permutation_test_mi`, pre-existing and verified) all exist and are tested independently. This is the single biggest remaining task — probably a new `orchestration/assemble.py`, or a new mode of `audit.py`.

## Environment constraint that shaped several design decisions

An earlier session tried to build this pipeline in a disk- and network-restricted dev sandbox. `pip install torch` failed two ways: the CPU-only wheel index (`download.pytorch.org`) was network-blocked, and the default CUDA-bundled install exhausted the sandbox's disk budget (the `nvidia-*` dependency packages alone run several GB). The response at the time was a NumPy-only reimplementation using synthetic data — which was on track to quietly become the *actual* pipeline, not a throwaway test.

That was reversed. Two reasons, both structural now:

1. **NumPy has no GPU/CUDA path.** If it were the real pipeline, the whole toy-scale hardware plan (why 10–25M params, why bf16/gradient-checkpointing) would assume a GPU that was never actually in the loop.
2. **Synthetic data breaks the actual research claim**, not just precision — there's nothing to compare fabricated sequences against for reproduction-fidelity.

The fix that's now load-bearing throughout the codebase:

- **PyTorch is the sole training framework.** No NumPy-based training path exists anywhere in `training/`, `mechanisms/`, or `battery/`.
- **`device` and `data_source` are two orthogonal, required config knobs** — never inferred, never defaulted. A restricted sandbox runs `device=cpu, data_source=synthetic` for wiring tests only; real hardware runs `device=cuda, data_source=real`. Both paths execute through the *identical* training/mechanism/battery code.
- **`orchestration/report.py` structurally refuses** to build a headline report from an all-synthetic result set (`DataProvenanceError`) — this is the concrete fix for the exact risk described above, not a documented convention someone has to remember.

If a future session proposes reimplementing `training/`, `mechanisms/`, or `battery/` in a different framework because of a sandbox constraint: that proposal has to clear the same bar this one failed to clear. Treat it as a request, not a reasonable default.

## A real bug found and fixed during the original build

`checkpointing/retention.py` and `reader.py` originally used a regex assuming `_`-delimited filename fields never contain `_` themselves. But `gate_ml` and `step_matched` — real names in this project — both contain underscores. The regex silently matched *nothing*, so periodic-checkpoint pruning would have quietly never run at all in the one condition (`step_matched`) most likely to be used. Caught by the module's own test suite before it shipped; fixed by matching against a known prefix/suffix for the caller-supplied `(mechanism, condition, seed)` rather than trying to parse an unknown filename apart.

## Design decisions made without a spec to check against

Confirm these before trusting a downstream result:

| Decision | File | What to check |
|---|---|---|
| GATE-ML's trigger direction (tie when EMA cosine-distance ≤ `tau_l`, held for `window` steps) | `mechanisms/gate_ml/mechanism.py` | the original GATE-ML paper/reference code, as part of reproduction-fidelity |
| Resource-gap is absolute, not relative/normalized | `eval/resource_gap.py` | whether a scale-independent version is actually wanted |
| Steps-to-generalization is "crosses and stays," not "first touch" | `eval/resource_gap.py` | same |
| Report's per-condition CI is a fresh bootstrap over per-seed point estimates, not a reuse of each seed's own stored CI | `orchestration/report.py` | whether this matches the intended methodology in experiment design §9/§10 |
| Reproduction-fidelity is manual, supplied as a small JSON file | `orchestration/report.py` | intentional — this is a judgment call against a paper's numbers, not something code can compute |

## Not yet built, beyond the one gap above

- A `pytest` suite. Everything described as "tested" in [`docs/MODULES.md`](MODULES.md) was verified with one-off scripts during the build session, not a checked-in, repeatable test suite. Worth converting before this grows further.
- Real corpora and tokenizer training have never been run — `data/real_loader.py` and `data/tokenizer.py` are syntax-checked only.
- RigL, GrokFast, and curriculum-ordering are registry stubs (see [`docs/MECHANISMS.md`](MECHANISMS.md)).
