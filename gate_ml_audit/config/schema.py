"""Config dataclasses and the hand-written merge utility (architecture §3).

Deliberately not a config framework (no Hydra, no OmegaConf) -- this project
runs across several heterogeneous, uncoordinated environments (RTX 5050
laptop, Colab, Kaggle, and occasional restricted dev sandboxes), and every
added dependency is one more thing that can silently be a different version
across those environments. This module is the ~40-line merge utility the
architecture doc calls for, not a general-purpose config system.

The one rule enforced everywhere in this file: `device` and `data_source`
have no implicit default anywhere in the load path. If a YAML file omits
either key, loading raises ConfigError rather than silently assuming
device=cpu or data_source=synthetic. This is the direct code-level
enforcement of the §0.5 amendment's "no code path is allowed to silently
default" requirement.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml


class ConfigError(Exception):
    """Raised for any malformed, incomplete, or internally inconsistent config."""


# ---------------------------------------------------------------------------
# Base config sub-sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoggingConfig:
    jsonl_dir: str
    mirror_to_wandb: bool


@dataclass(frozen=True)
class CheckpointingConfig:
    periodic_interval_steps: int
    periodic_retain: int


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    n_layers: int
    n_heads: int
    vocab_size: int
    max_seq_len: int


@dataclass(frozen=True)
class EvalConfig:
    cadence_steps: int


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float
    warmup_steps: int


@dataclass(frozen=True)
class BaseConfig:
    device: str          # "cpu" | "cuda" -- required, no default (§0.5)
    data_source: str      # "synthetic" | "real" -- required, no default (§0.5)
    seed_list: list
    batch_size_ceiling: int
    total_steps: int
    logging: LoggingConfig
    checkpointing: CheckpointingConfig
    model: ModelConfig
    eval: EvalConfig
    optimizer: OptimizerConfig
    # Only meaningful when data_source == "real"; deliberately NOT in
    # _REQUIRED_BASE_KEYS below and defaults to {} -- unlike device/
    # data_source (which change execution mode and must never silently
    # default), an empty `data` dict on a data_source=synthetic run means
    # exactly what it says: no real-data settings are needed. Required keys
    # inside it (real_shard_paths, tokenizer_path, token_budget) are
    # enforced by config/validate.py, but only when data_source == "real".
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MechanismConfig:
    name: str
    status: str  # "implemented" | "stub"
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionConfig:
    wrapper: str  # "real" | "step_matched" | "scrambled"
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    base: BaseConfig
    mechanism: MechanismConfig
    condition: ConditionConfig
    overrides_applied: dict = field(default_factory=dict)

    def require_device_and_data_source(self) -> None:
        if self.base.device not in ("cpu", "cuda"):
            raise ConfigError(
                f"device must be 'cpu' or 'cuda', got {self.base.device!r}"
            )
        if self.base.data_source not in ("synthetic", "real"):
            raise ConfigError(
                f"data_source must be 'synthetic' or 'real', got "
                f"{self.base.data_source!r}"
            )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_REQUIRED_BASE_KEYS = (
    "device", "data_source", "seed_list", "batch_size_ceiling", "total_steps",
    "logging", "checkpointing", "model", "eval", "optimizer",
)


def _require(d: dict, key: str, context: str) -> Any:
    if key not in d:
        raise ConfigError(f"missing required key {key!r} in {context}")
    return d[key]


def _read_yaml(path: Union[str, Path]) -> dict:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ConfigError(f"{path} is empty")
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return raw


def load_base(path: Union[str, Path]) -> BaseConfig:
    raw = _read_yaml(path)
    for key in _REQUIRED_BASE_KEYS:
        _require(raw, key, str(path))

    logging_raw = raw["logging"]
    checkpointing_raw = raw["checkpointing"]
    model_raw = raw["model"]
    eval_raw = raw["eval"]
    optimizer_raw = raw["optimizer"]

    return BaseConfig(
        device=raw["device"],
        data_source=raw["data_source"],
        seed_list=list(raw["seed_list"]),
        batch_size_ceiling=int(raw["batch_size_ceiling"]),
        total_steps=int(raw["total_steps"]),
        logging=LoggingConfig(
            jsonl_dir=_require(logging_raw, "jsonl_dir", "logging"),
            mirror_to_wandb=bool(_require(logging_raw, "mirror_to_wandb", "logging")),
        ),
        checkpointing=CheckpointingConfig(
            periodic_interval_steps=int(
                _require(checkpointing_raw, "periodic_interval_steps", "checkpointing")
            ),
            periodic_retain=int(
                _require(checkpointing_raw, "periodic_retain", "checkpointing")
            ),
        ),
        model=ModelConfig(
            d_model=int(_require(model_raw, "d_model", "model")),
            n_layers=int(_require(model_raw, "n_layers", "model")),
            n_heads=int(_require(model_raw, "n_heads", "model")),
            vocab_size=int(_require(model_raw, "vocab_size", "model")),
            max_seq_len=int(_require(model_raw, "max_seq_len", "model")),
        ),
        eval=EvalConfig(
            cadence_steps=int(_require(eval_raw, "cadence_steps", "eval")),
        ),
        optimizer=OptimizerConfig(
            name=_require(optimizer_raw, "name", "optimizer"),
            lr=float(_require(optimizer_raw, "lr", "optimizer")),
            weight_decay=float(_require(optimizer_raw, "weight_decay", "optimizer")),
            warmup_steps=int(_require(optimizer_raw, "warmup_steps", "optimizer")),
        ),
        data=dict(raw.get("data", {})),
    )


def load_mechanism(path: Union[str, Path]) -> MechanismConfig:
    raw = _read_yaml(path)
    name = _require(raw, "name", str(path))
    status = _require(raw, "status", str(path))
    if status not in ("implemented", "stub"):
        raise ConfigError(
            f"mechanism status must be 'implemented' or 'stub', got "
            f"{status!r} in {path}"
        )
    params = {k: v for k, v in raw.items() if k not in ("name", "status")}
    return MechanismConfig(name=name, status=status, params=params)


def load_condition(path: Union[str, Path]) -> ConditionConfig:
    raw = _read_yaml(path)
    wrapper = _require(raw, "wrapper", str(path))
    if wrapper not in ("real", "step_matched", "scrambled"):
        raise ConfigError(
            f"condition wrapper must be one of real/step_matched/scrambled, "
            f"got {wrapper!r} in {path}"
        )
    params = {k: v for k, v in raw.items() if k != "wrapper"}
    return ConditionConfig(wrapper=wrapper, params=params)


def _rebuild_base_from_dict(d: dict) -> BaseConfig:
    return BaseConfig(
        device=d["device"],
        data_source=d["data_source"],
        seed_list=list(d["seed_list"]),
        batch_size_ceiling=d["batch_size_ceiling"],
        total_steps=d["total_steps"],
        logging=LoggingConfig(**d["logging"]),
        checkpointing=CheckpointingConfig(**d["checkpointing"]),
        model=ModelConfig(**d["model"]),
        eval=EvalConfig(**d["eval"]),
        optimizer=OptimizerConfig(**d["optimizer"]),
        data=dict(d.get("data", {})),
    )


def merge(
    base: BaseConfig,
    mechanism: MechanismConfig,
    condition: ConditionConfig,
    cli_overrides: Optional[dict] = None,
) -> RunConfig:
    """Compose base + mechanism + condition, then apply CLI overrides last.

    CLI overrides use dot-separated paths into the base config only (e.g.
    "device", "checkpointing.periodic_retain") -- per architecture §3, an
    override is meant for "the rare case of an actual deliberate deviation,"
    which in this design is always a base-config value (device/data_source
    switches, a one-off step-count change), never a silent way to smuggle in
    a different mechanism or condition than what was explicitly loaded.
    Every applied override is recorded in RunConfig.overrides_applied so it
    shows up in logs rather than being a silent deviation.
    """
    cli_overrides = cli_overrides or {}
    working_base = base
    applied: dict = {}

    if cli_overrides:
        base_dict = dataclasses.asdict(base)
        for dotted_key, value in cli_overrides.items():
            parts = dotted_key.split(".")
            node = base_dict
            for part in parts[:-1]:
                if not isinstance(node, dict) or part not in node:
                    raise ConfigError(
                        f"CLI override path {dotted_key!r} does not exist in "
                        f"base config"
                    )
                node = node[part]
            if not isinstance(node, dict) or parts[-1] not in node:
                raise ConfigError(
                    f"CLI override path {dotted_key!r} does not exist in "
                    f"base config"
                )
            node[parts[-1]] = value
            applied[dotted_key] = value
        working_base = _rebuild_base_from_dict(base_dict)

    run_cfg = RunConfig(
        base=working_base,
        mechanism=mechanism,
        condition=condition,
        overrides_applied=applied,
    )
    run_cfg.require_device_and_data_source()
    return run_cfg
