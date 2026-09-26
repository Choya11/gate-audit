# Usage Guide

[← back to README](../README.md)

## 0. Prerequisites

You need a machine where PyTorch actually installs — the RTX 5050 laptop, Colab, or Kaggle all work. A generic cloud sandbox or codespace often does **not**: this exact project hit a wall where `pip install torch` either got network-blocked (the CPU-only wheel index) or ran out of disk (the default CUDA-bundled install pulls several GB of `nvidia-*` packages). If `python3 -c "import torch"` fails, fix that first — don't work around it by skipping torch.

```bash
pip install torch                # or: pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 1. Run the pre-flight validation gate once

```bash
python3 -m orchestration.audit \
  --refresh-validation \
  --toy-scale-log-length 500
```

This checks that the statistics code (`mi_estimator.py`/`auditbench.py`) is trustworthy before anything is allowed to use it — a real ~100-seed calibration sweep, takes about a minute. Re-run it (drop `--refresh-validation` to just check freshness) any time either of those two files changes; `audit.py` refuses to run the real battery otherwise.

## 2. Train one mechanism's `real` condition first

```bash
python3 -m orchestration.train \
  --base-config config/base.yaml \
  --mechanism-config config/mechanisms/gate_ml.yaml \
  --condition-config config/conditions/real.yaml \
  --override data_source=synthetic
```

Start with `data_source=synthetic` — no real corpus needed — just to prove the wiring works end to end on your machine. Switch to `data_source=real` once `config/base.yaml`'s `data:` section points at a real corpus + tokenizer. Only `real` results ever count toward the final report.

## 3. Then `step_matched`, then `scrambled`

```bash
python3 -m orchestration.train --base-config config/base.yaml \
  --mechanism-config config/mechanisms/gate_ml.yaml \
  --condition-config config/conditions/step_matched.yaml --override data_source=synthetic

python3 -m orchestration.train --base-config config/base.yaml \
  --mechanism-config config/mechanisms/gate_ml.yaml \
  --condition-config config/conditions/scrambled.yaml --override data_source=synthetic
```

`step_matched` **must** come after `real` for the same mechanism — it reads real's log to build its own fixed schedule. Order between mechanisms doesn't matter. Repeat all three conditions for `rigl`/`grokfast`/`curriculum` once those are implemented (they're stubs today — see [`docs/MECHANISMS.md`](MECHANISMS.md)).

## 4. ⚠️ Assemble each run's log into a `BatteryResult` — not built yet

This step doesn't exist as a script. See [`docs/ADVANCED.md`](ADVANCED.md) for exactly what's missing. Until it's written, step 5 has nothing real to read.

## 5. Build the report

```bash
python3 -m orchestration.report \
  --results-dir ./results \
  --reproduction-fidelity-file ./fidelity.json \
  --effect-sizes-file ./effect_sizes.json \
  --output-dir ./report_out
```

`--reproduction-fidelity-file` is a small hand-written JSON file, one entry per mechanism:

```json
{
  "gate_ml": {"converged": true, "matches_published": true},
  "rigl": {"converged": true, "matches_published": true},
  "grokfast": {"converged": true, "matches_published": true},
  "curriculum": {"converged": false, "matches_published": null}
}
```

`--effect-sizes-file` is the normalized, originally-reported effect size per mechanism (used only by the C3 correlation, over the 3 audited mechanisms — curriculum is excluded there by design):

```json
{"gate_ml": 0.9, "rigl": 0.3, "grokfast": 0.5}
```

Output: `pass_fail_heatmap.png`, `c3_scatter.png`, `failure_buckets.json` in `--output-dir`. Exits with code 1 and a clear message if every result found is `data_source=synthetic`.

## Running the test suite yourself

Most of `stats/`, `battery/`, `checkpointing/`, `viz/`, and all of `orchestration/audit.py`/`report.py` have no torch dependency and can be exercised on any machine, GPU or not — useful for a quick sanity check before committing to a real training run. There's no `pytest` suite checked in yet; see [`docs/ADVANCED.md`](ADVANCED.md).
