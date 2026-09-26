"""Fail-fast validation for a fully merged RunConfig (architecture §11):
missing/invalid hyperparameters, an unregistered mechanism name, a condition
config that doesn't resolve to a valid Mechanism wrapper, and an unset or
invalid device/data_source pair must all be caught here, at config-load
time, before any GPU time is spent -- not discovered mid-run.

RUNTIME VERIFICATION NOTE: imports mechanisms (which imports torch via
mechanisms.base). Syntax-checked only in this sandbox; the mechanism-name
and stub/implemented cross-checks below have been traced by hand against
mechanisms/__init__.py's actual MECHANISM_REGISTRY contents and the four
config/mechanisms/*.yaml files, not executed end-to-end.
"""
from __future__ import annotations

from config.schema import ConfigError, RunConfig
from mechanisms import MECHANISM_REGISTRY

_VALID_WRAPPERS = ("real", "step_matched", "scrambled")


def validate_run_config(cfg: RunConfig) -> None:
    """Raise ConfigError on any structurally invalid RunConfig.

    Checks, in order:
      1. device / data_source are set and valid.
      2. the mechanism name is registered.
      3. the mechanism's declared config status ("implemented"/"stub")
         agrees with what is actually registered -- a config typo cannot
         silently run a stub as if it were real, or vice versa.
      4. the condition wrapper name is one of the three defined battery
         conditions.
      5. step_matched's declared dependency on a Real trigger log is at
         least structurally present in condition.params (this is a
         config-shape check; whether the log file actually exists yet is
         checked later, at battery-assembly time in battery/trigger_log.py).
      6. basic numeric sanity on batch_size_ceiling, total_steps, and every
         model dimension, including the d_model/n_heads divisibility
         TinyDecoderLM itself also enforces -- checked here too so a bad
         config fails at load time, not at model-construction time deep
         inside a training run.
    """
    cfg.require_device_and_data_source()

    if cfg.mechanism.name not in MECHANISM_REGISTRY:
        raise ConfigError(
            f"unregistered mechanism name {cfg.mechanism.name!r}; known "
            f"mechanisms: {sorted(MECHANISM_REGISTRY)}"
        )

    registry_entry = MECHANISM_REGISTRY[cfg.mechanism.name]
    if registry_entry.is_stub and cfg.mechanism.status != "stub":
        raise ConfigError(
            f"mechanism {cfg.mechanism.name!r} is a registry stub but its "
            f"config declares status={cfg.mechanism.status!r}; the config "
            f"must say status: stub explicitly"
        )
    if not registry_entry.is_stub and cfg.mechanism.status != "implemented":
        raise ConfigError(
            f"mechanism {cfg.mechanism.name!r} is fully implemented but its "
            f"config declares status={cfg.mechanism.status!r}; the config "
            f"must say status: implemented explicitly"
        )

    if cfg.condition.wrapper not in _VALID_WRAPPERS:
        raise ConfigError(
            f"condition wrapper must be one of {_VALID_WRAPPERS}, got "
            f"{cfg.condition.wrapper!r}"
        )

    if cfg.condition.wrapper == "step_matched":
        if "requires_trigger_log_from" not in cfg.condition.params:
            raise ConfigError(
                "condition wrapper 'step_matched' requires a "
                "'requires_trigger_log_from' key naming the Real run its "
                "fixed schedule is derived from (architecture §4)"
            )

    if cfg.base.batch_size_ceiling <= 0:
        raise ConfigError("batch_size_ceiling must be a positive integer")
    if cfg.base.total_steps <= 0:
        raise ConfigError("total_steps must be a positive integer")

    model = cfg.base.model
    for field_name, value in (
        ("d_model", model.d_model),
        ("n_layers", model.n_layers),
        ("n_heads", model.n_heads),
        ("vocab_size", model.vocab_size),
        ("max_seq_len", model.max_seq_len),
    ):
        if value <= 0:
            raise ConfigError(f"model.{field_name} must be a positive integer, got {value}")

    if model.d_model % model.n_heads != 0:
        raise ConfigError(
            f"model.d_model ({model.d_model}) must be divisible by "
            f"model.n_heads ({model.n_heads})"
        )

    if cfg.base.checkpointing.periodic_interval_steps <= 0:
        raise ConfigError("checkpointing.periodic_interval_steps must be positive")
    if cfg.base.checkpointing.periodic_retain <= 0:
        raise ConfigError("checkpointing.periodic_retain must be positive")
    if cfg.base.eval.cadence_steps <= 0:
        raise ConfigError("eval.cadence_steps must be positive")

    if cfg.base.data_source == "real":
        data = cfg.base.data
        for key in ("real_shard_paths", "tokenizer_path", "token_budget"):
            if key not in data:
                raise ConfigError(
                    f"data_source=real requires a 'data.{key}' config value, "
                    f"but it is missing"
                )
        if not isinstance(data["real_shard_paths"], list) or not data["real_shard_paths"]:
            raise ConfigError(
                "data.real_shard_paths must be a non-empty list when "
                "data_source=real"
            )
        if not isinstance(data["tokenizer_path"], str) or not data["tokenizer_path"]:
            raise ConfigError(
                "data.tokenizer_path must be a non-empty string when "
                "data_source=real"
            )
        if not isinstance(data["token_budget"], int) or data["token_budget"] <= 0:
            raise ConfigError(
                "data.token_budget must be a positive integer when "
                "data_source=real"
            )
