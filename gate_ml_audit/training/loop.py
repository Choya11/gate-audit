"""training/loop.py: the single, mechanism-/device-/data_source-agnostic
training loop (architecture §5).

Deliberately knows nothing about which mechanism, which battery condition,
which device, or which data source it is running -- it only calls the
Mechanism interface and reads device/data_source off the config
(architecture §2's load-bearing "training engine can't tell them apart"
property, extended one level up by the §0.5 amendment to dev-vs-real mode).
`.to(device)` is the only device-specific line in the whole loop.

RUNTIME VERIFICATION NOTE: imports torch throughout (model construction,
optimizer, forward/backward, .to(device)). Not executable in this sandbox
-- syntax-checked and manually traced against every other already-verified
piece it calls (TrainingState, GateMLMechanism, battery/conditions.py,
checkpointing/*, data/*, all of which either were runtime-tested directly
or are honestly flagged as syntax-checked-only in their own files).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from battery.conditions import RealWrapper, ScrambledWrapper, make_step_matched
from battery.trigger_log import extract_trigger_steps, to_fixed_schedule
from checkpointing.reader import find_latest_valid
from checkpointing.retention import prune_periodic
from checkpointing.state import CheckpointState
from checkpointing.writer import save_periodic, save_trigger_event
from config.schema import RunConfig
from data.real_loader import RealDataSource
from data.synthetic_loader import SyntheticDataSource
from data.tokenizer import Tokenizer
from mechanisms import build_mechanism
from training.model import TinyDecoderLM
from training.optim import build_optimizer, build_scheduler
from training.state import TrainingState


class NaNLossError(RuntimeError):
    """Raised when the training loop detects a non-finite loss -- halts the
    run immediately rather than continuing to log garbage (architecture
    §11's inner-loop sanity-check discipline)."""


def _log_path(mechanism_name: str, condition_wrapper: str, seed: int, jsonl_dir: str) -> str:
    """Deterministic JSONL log path, matching checkpointing/state.py's
    {mechanism}_{condition}_{seed} naming convention (architecture §12's
    flat-file naming scheme, extended to logs)."""
    return str(Path(jsonl_dir) / f"{mechanism_name}_{condition_wrapper}_{seed}.jsonl")


def _checkpoint_dir(cfg: RunConfig) -> str:
    return str(Path(cfg.base.logging.jsonl_dir).parent / "checkpoints")


def _build_data_source(cfg: RunConfig, seed: int):
    if cfg.base.data_source == "synthetic":
        return SyntheticDataSource(
            vocab_size=cfg.base.model.vocab_size,
            seq_len=cfg.base.model.max_seq_len,
            batch_size=cfg.base.batch_size_ceiling,
            num_batches=cfg.base.total_steps,
            seed=seed,
        )
    elif cfg.base.data_source == "real":
        tokenizer = Tokenizer.load(cfg.base.data["tokenizer_path"])
        return RealDataSource(
            shard_paths=list(cfg.base.data["real_shard_paths"]),
            tokenizer=tokenizer,
            seq_len=cfg.base.model.max_seq_len,
            batch_size=cfg.base.batch_size_ceiling,
            token_budget=cfg.base.data["token_budget"],
        )
    # config/validate.py (via RunConfig.require_device_and_data_source())
    # should already have rejected any other data_source value before this
    # function is ever called -- this branch exists so a caller that skips
    # validation fails loudly here too, rather than this function silently
    # returning None.
    raise ValueError(f"unsupported data_source {cfg.base.data_source!r}")


def _build_condition_mechanism(cfg: RunConfig, raw_mechanism, seed: int):
    wrapper = cfg.condition.wrapper
    if wrapper == "real":
        return RealWrapper(raw_mechanism)
    if wrapper == "scrambled":
        return ScrambledWrapper(raw_mechanism, rng_seed=seed)
    if wrapper == "step_matched":
        # Sequencing dependency, architecture §4/§13.2: derive the fixed
        # schedule from the Real condition's own completed log, at the same
        # (mechanism, seed) -- constructed here as a ConditionConfig(wrapper=
        # "real") purely to reuse _log_path's naming convention, not as an
        # actual Mechanism instantiation.
        real_log_path = _log_path(
            cfg.mechanism.name, "real", seed, cfg.base.logging.jsonl_dir
        )
        trigger_steps = extract_trigger_steps(real_log_path)
        return make_step_matched(raw_mechanism, to_fixed_schedule(trigger_steps))
    # config/validate.py should already have rejected any other wrapper
    # value -- see _build_data_source's identical reasoning above.
    raise ValueError(f"unsupported condition wrapper {wrapper!r}")


def run(cfg: RunConfig, seed: int) -> None:
    """Run one (mechanism, condition, seed) run to completion, or resume it
    from its latest valid periodic checkpoint (architecture §11's
    resume-by-default policy).

    One seed per call, deliberately: architecture §12 treats seed as "a
    config value... a loop variable, not a hardcoded assumption," and the
    natural place for the outer seed loop is orchestration/train.py (which
    can run seeds sequentially or in parallel across machines), not buried
    inside this function.
    """
    torch.manual_seed(seed)

    device = torch.device(cfg.base.device)
    model = TinyDecoderLM(
        vocab_size=cfg.base.model.vocab_size,
        d_model=cfg.base.model.d_model,
        n_layers=cfg.base.model.n_layers,
        n_heads=cfg.base.model.n_heads,
        max_seq_len=cfg.base.model.max_seq_len,
    ).to(device)

    optimizer = build_optimizer(model, cfg.base.optimizer)
    scheduler = build_scheduler(optimizer, cfg.base.optimizer)

    raw_mechanism = build_mechanism(cfg.mechanism.name, cfg.mechanism.params)
    mechanism = _build_condition_mechanism(cfg, raw_mechanism, seed)

    data_source = _build_data_source(cfg, seed)

    checkpoint_dir = _checkpoint_dir(cfg)
    log_path = _log_path(
        cfg.mechanism.name, cfg.condition.wrapper, seed, cfg.base.logging.jsonl_dir
    )
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Resume-by-default (architecture §11): always check for the latest
    # valid periodic checkpoint before starting fresh.
    start_step = 0
    resumed = find_latest_valid(
        checkpoint_dir, cfg.mechanism.name, cfg.condition.wrapper, seed
    )
    if resumed is not None:
        model.load_state_dict(resumed.model_state)
        optimizer.load_state_dict(resumed.optimizer_state)
        # Generic restore: any real (non-stub) mechanism whose .state object
        # follows the to_dict()/from_dict() convention (GateMLState does;
        # future mechanisms should too) gets its internal state restored
        # without this loop needing to special-case the mechanism by name.
        if hasattr(raw_mechanism, "state") and hasattr(type(raw_mechanism.state), "from_dict"):
            raw_mechanism.state = type(raw_mechanism.state).from_dict(resumed.mechanism_state)
        start_step = resumed.step + 1

    data_iter = iter(data_source)
    if start_step > 0:
        # Skip exactly the batches already consumed before the checkpoint
        # being resumed from, so a resumed run sees the same batch sequence
        # a from-scratch run would have seen at these step numbers (both
        # SyntheticDataSource and RealDataSource iterate deterministically
        # for a given seed/shard order) -- without this, resuming would
        # silently re-feed already-trained-on batches as if they were new.
        for _ in range(start_step):
            try:
                next(data_iter)
            except StopIteration:
                break

    signal_history: list = []
    log_file = open(log_path, "a")
    try:
        step = start_step
        for batch in data_iter:
            if step >= cfg.base.total_steps:
                break

            token_ids = batch.token_ids.to(device)
            input_ids = token_ids[:, :-1]
            target_ids = token_ids[:, 1:]

            model.train()
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

            if not torch.isfinite(loss):
                raise NaNLossError(
                    f"non-finite loss ({loss.item()!r}) at step {step} -- "
                    f"halting immediately per architecture §11"
                )

            loss.backward()

            state = TrainingState(
                step=step,
                mechanism=cfg.mechanism.name,
                condition=cfg.condition.wrapper,
                seed=seed,
                device=cfg.base.device,
                data_source=cfg.base.data_source,
                loss=loss.item(),
                extra={"model": model},
            )

            signal_value = mechanism.signal(state)
            signal_history.append(signal_value)
            intervention_fired = mechanism.should_intervene(signal_history, state)
            if intervention_fired:
                mechanism.apply(model, optimizer, state)

            optimizer.step()
            scheduler.step()

            log_file.write(
                json.dumps(
                    {
                        "step": step,
                        "wall_clock": time.time(),
                        "mechanism": cfg.mechanism.name,
                        "condition": cfg.condition.wrapper,
                        "seed": seed,
                        "device": cfg.base.device,
                        "data_source": cfg.base.data_source,
                        "signal": signal_value,
                        "intervention_fired": intervention_fired,
                        "loss": loss.item(),
                    }
                )
                + "\n"
            )
            log_file.flush()

            is_periodic_step = (
                (step + 1) % cfg.base.checkpointing.periodic_interval_steps == 0
            )
            if intervention_fired or is_periodic_step:
                mechanism_state = (
                    raw_mechanism.state.to_dict()
                    if hasattr(raw_mechanism, "state") and hasattr(raw_mechanism.state, "to_dict")
                    else {}
                )
                ckpt = CheckpointState(
                    step=step,
                    mechanism=cfg.mechanism.name,
                    condition=cfg.condition.wrapper,
                    seed=seed,
                    device=cfg.base.device,
                    data_source=cfg.base.data_source,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    mechanism_state=mechanism_state,
                )
                if intervention_fired:
                    save_trigger_event(ckpt, checkpoint_dir)
                if is_periodic_step:
                    save_periodic(ckpt, checkpoint_dir)
                    prune_periodic(
                        checkpoint_dir,
                        cfg.mechanism.name,
                        cfg.condition.wrapper,
                        seed,
                        keep=cfg.base.checkpointing.periodic_retain,
                    )

            step += 1
    finally:
        log_file.close()
