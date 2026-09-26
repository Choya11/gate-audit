"""orchestration/train.py: CLI entrypoint. Loads and validates config, then
runs training/loop.py's run() once per seed in cfg.base.seed_list.

The seed loop lives HERE, not inside training/loop.py's run() -- architecture
§12 treats seed as "a config value... a loop variable, not a hardcoded
assumption," and putting the loop at the orchestration layer (rather than
inside run() itself) is what leaves room for seeds to later run in parallel
across machines (architecture §12's "heterogeneous, uncoordinated compute")
without training/loop.py needing to change at all.

RUNTIME VERIFICATION NOTE: imports torch (via training.loop), so this file
itself cannot be imported in this sandbox at all. The CLI-override parsing
logic is factored into orchestration/cli_utils.py specifically so it has no
transitive torch dependency and IS runtime-tested there, in isolation --
see that file.
"""
from __future__ import annotations

import argparse
import sys

from config.schema import ConfigError, load_base, load_condition, load_mechanism, merge
from config.validate import validate_run_config
from orchestration.cli_utils import parse_cli_overrides
from training.loop import run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one mechanism/condition training run, once per configured seed."
    )
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--mechanism-config", required=True)
    parser.add_argument("--condition-config", required=True)
    parser.add_argument(
        "--override", action="append", default=[],
        help="key=value, dotted path into base config, repeatable",
    )
    args = parser.parse_args(argv)

    try:
        base = load_base(args.base_config)
        mechanism = load_mechanism(args.mechanism_config)
        condition = load_condition(args.condition_config)
        overrides = parse_cli_overrides(args.override)
        cfg = merge(base, mechanism, condition, cli_overrides=overrides)
        validate_run_config(cfg)
    except (ConfigError, ValueError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 1

    for seed in cfg.base.seed_list:
        print(f"[train] starting {cfg.mechanism.name}/{cfg.condition.wrapper}/seed={seed}")
        run(cfg, seed)
        print(f"[train] finished {cfg.mechanism.name}/{cfg.condition.wrapper}/seed={seed}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
