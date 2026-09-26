# GATE-ML Falsification Audit

A falsification-audit benchmark for adaptive training-signal mechanisms in language models, plus AuditBench — a synthetic-ground-truth validation suite for the audit's own statistics.

**Status:** implemented and self-tested without a GPU (27/33 modules genuinely executed and verified in a CPU-only, no-torch dev sandbox); the remaining 6 need a real run on actual hardware to confirm. One real gap remains in the pipeline — see [`docs/ADVANCED.md`](docs/ADVANCED.md). Nothing here has been run on real training data yet.

## Table of Contents

1. [Overview](#overview) — this file
2. [System Architecture](docs/ARCHITECTURE.md)
3. [Pipeline Components](docs/PIPELINE.md)
4. [Data Flow Diagrams](docs/DATA_FLOW.md)
5. [Module Documentation](docs/MODULES.md)
6. [Configuration System](docs/CONFIGURATION.md)
7. [Supported Mechanisms](docs/MECHANISMS.md)
8. [Usage Guide](docs/USAGE.md)
9. [Evaluation Metrics](docs/EVALUATION.md)
10. [Advanced Features & Known Gaps](docs/ADVANCED.md)

## Overview

Four training-time mechanisms — each one gates an intervention on a monitored signal — are audited with the same standardized battery:

| Mechanism | Signal | Intervention | Status |
|---|---|---|---|
| **GATE-ML** | EMA cosine-distance between input/output embedding gradients | tie the two embedding matrices, once, permanently | implemented |
| RigL | gradient magnitude | prune/regrow connections | registry stub |
| GrokFast | EMA-filtered low-frequency gradient | amplify that component | registry stub |
| Curriculum-ordering | length + rarity difficulty proxy | reorder training data | registry stub |

The **falsification battery** runs three conditions per mechanism, per seed:

- **Real** — the mechanism's actual signal-driven intervention.
- **Step-matched** — the same intervention, fired on a fixed schedule copied from Real's own trigger timing (isolates *timing* from the *signal*).
- **Scrambled** — the intervention fired on a randomized, statistically-similar-but-information-free surrogate of the signal (isolates whether the signal is *causally load-bearing* at all).

If Real doesn't clearly beat both controls, the mechanism's own signal wasn't doing the work its paper claims.

A separate, pre-existing, numpy-only validation suite (`falsification_audit/`) checks the statistics themselves — the mutual-information estimator and its permutation test — against synthetic ground truth before any real-mechanism result is trusted. This is a **hard pre-flight gate**, not a one-time check: `orchestration/audit.py` refuses to run if either `mi_estimator.py` or `auditbench.py` has changed since the last recorded pass.

### Quick facts

- Toy-scale models only (tens of millions of parameters), meant to run on a single consumer GPU (RTX 5050), Colab, or Kaggle — not a datacenter cluster.
- Single framework: PyTorch. There is no NumPy-based training path; an earlier attempt at one was deliberately reversed (see `docs/ADVANCED.md`).
- `device` and `data_source` are required, no-default config fields everywhere in the codebase, specifically so a dev-mode (`data_source=synthetic`) result can never be silently mistaken for a real one downstream.

### Repository layout

```
gate_ml_audit/
├── config/              YAML configs + dataclass schema + validation
├── data/                real & synthetic data loaders (one shared interface)
├── mechanisms/          GATE-ML (implemented) + 3 registry stubs
├── battery/             the Real / Step-matched / Scrambled wrappers
├── training/            the one training loop, model, optimizer
├── checkpointing/       trigger-event + periodic checkpoints, retention policy
├── eval/                perplexity, task accuracy, resource-gap metrics
├── falsification_audit/ pre-existing MI estimator + AuditBench (untouched)
├── stats/               bootstrap CI, Holm–Bonferroni, failure buckets, C3
├── viz/                 matplotlib plots (trajectory, heatmap, C3 scatter)
├── orchestration/       train.py, audit.py, report.py CLI entrypoints
├── docs/                this documentation set
└── requirements.txt
```

See [`docs/USAGE.md`](docs/USAGE.md) for the exact commands, in order, including the one step that isn't built yet.

## License

Not yet chosen — add one before making this repository public.
