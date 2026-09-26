# Configuration System

[← back to README](../README.md)

Plain YAML + dataclasses, composed by a small hand-written merge utility — not Hydra or OmegaConf. The project runs across too many heterogeneous, uncoordinated environments (a laptop, Colab, Kaggle, occasional restricted dev sandboxes) for a shared config-framework dependency version to be a safe assumption.

## Composition order

```
config/base.yaml
    → config/mechanisms/{name}.yaml
        → config/conditions/{name}.yaml
            → CLI --override key=value (always logged, never silent)
```

## `base.yaml` — every field is required, none silently default

```yaml
device: cpu              # cpu | cuda — REQUIRED, no default
data_source: synthetic   # synthetic | real — REQUIRED, no default
seed_list: [0, 1, 2]
batch_size_ceiling: 32
total_steps: 2000
logging:
  jsonl_dir: ./logs
  mirror_to_wandb: false   # optional, additive, never load-bearing
checkpointing:
  periodic_interval_steps: 500
  periodic_retain: 2
model:
  d_model: 256
  n_layers: 4
  n_heads: 4
  vocab_size: 8000
  max_seq_len: 128
eval:
  cadence_steps: 200
optimizer:
  name: adamw
  lr: 0.0003
  weight_decay: 0.01
  warmup_steps: 100
data:                      # only required when data_source: real
  real_shard_paths: [/path/to/shard0.txt]
  tokenizer_path: /path/to/tokenizer.json
  token_budget: 5000000
```

**Why `device` and `data_source` have no default anywhere in the load path:** an earlier session drifted toward a NumPy-only training path when PyTorch couldn't install in a constrained sandbox. A silent default would let that class of mistake happen again, unnoticed. `config/schema.py`'s `load_base()` raises `ConfigError` if either key is absent from the YAML file at all.

## `config/mechanisms/*.yaml`

```yaml
# gate_ml.yaml
name: gate_ml
status: implemented
tau_l: 0.15
window: 50
beta: 0.9
```

```yaml
# rigl.yaml / grokfast.yaml / curriculum.yaml
name: rigl
status: stub
```

`status` must match what's actually registered in `mechanisms/__init__.py`'s `MECHANISM_REGISTRY` — a config claiming `status: implemented` for a registry stub (or vice versa) is rejected at validation time, before any compute is spent.

## `config/conditions/*.yaml`

```yaml
# real.yaml
wrapper: real
```

```yaml
# step_matched.yaml
wrapper: step_matched
requires_trigger_log_from: real   # structural reminder of the sequencing dependency
```

```yaml
# scrambled.yaml
wrapper: scrambled
rng_seed_offset: 1000
```

## CLI overrides

```bash
python3 -m orchestration.train \
  --base-config config/base.yaml \
  --mechanism-config config/mechanisms/gate_ml.yaml \
  --condition-config config/conditions/real.yaml \
  --override device=cuda \
  --override total_steps=5000
```

Dotted paths into the **base** config only (never into mechanism or condition configs — an override is for a deliberate one-off deviation, not a way to silently swap which mechanism is running). Every applied override is recorded on the resulting `RunConfig.overrides_applied` so it shows up in logs, never silently.

## Validation

`config/validate.py` runs before any GPU time is spent and checks, in order: device/data_source validity, mechanism registration, stub/implemented status agreement, condition wrapper validity, the `step_matched` → `requires_trigger_log_from` structural requirement, positive numeric fields, `d_model % n_heads == 0`, and — only when `data_source: real` — that `data.real_shard_paths`, `data.tokenizer_path`, and `data.token_budget` are all present and well-formed.
