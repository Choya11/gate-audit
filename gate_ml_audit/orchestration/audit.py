"""orchestration/audit.py: CLI entrypoint for the falsification-audit
pipeline. Enforces the hard pre-flight validation gate (architecture §6):
refuses to run the real-mechanism battery analysis unless the estimator's
synthetic validation and the full AuditBench sweep have both been re-run
and passed since the last change to mi_estimator.py or auditbench.py.

Wires to falsification_audit/mi_estimator.py's run_synthetic_validation and
falsification_audit/auditbench.py's run_auditbench, both already built and
already passing (architecture §7.2) -- this module does not modify or
reimplement either, only gates on and calls them.

IMPORT-STYLE NOTE: both existing files use bare top-level imports
(auditbench.py contains `from mi_estimator import (...)`, not a relative
`from .mi_estimator import`), confirmed by inspecting the actual files --
they are written to be run with their own directory on sys.path, not as a
properly dotted package. Rather than restructure that (architecture §6:
"this design doesn't touch it, just confirms it's the right boundary"),
_import_falsification_audit_modules below does the same thing running them
as scripts would: add their directory to sys.path, then import normally.

RUNTIME VERIFICATION NOTE: falsification_audit/ itself is numpy-only (no
torch import anywhere in either file -- confirmed directly). Every function
in this file that does NOT call into training/eval is therefore genuinely
runtime-testable in this sandbox, and has been -- including an actual call
into the real run_synthetic_validation from this project's real
mi_estimator.py, not a mock. Only main()'s final battery-analysis line
(which would need torch-based training results to operate on) is out of
scope for this pass' runtime testing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


class ValidationStaleError(RuntimeError):
    """Raised when the estimator/AuditBench validation gate has not been
    re-run since the last change to mi_estimator.py or auditbench.py --
    architecture §6's "failing loudly rather than silently trusting stale
    validation," made concrete."""


def _file_checksum(path: str) -> str:
    """SHA-256 of a file's contents -- used to detect ANY change to
    mi_estimator.py or auditbench.py since the last recorded validation
    pass, including changes that only touch comments or formatting. This is
    conservative on purpose: re-running a two-minute, CPU-only validation
    sweep on a false-positive "changed" signal is far cheaper than trusting
    stale validation on a false-negative "unchanged" signal.
    """
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def record_validation_pass(estimator_path: str, auditbench_path: str, record_path: str) -> None:
    """Write the current checksums of the estimator and auditbench modules
    to record_path. Only ever called by run_preflight_validation, after both
    checks have actually passed -- never called speculatively."""
    record = {
        "estimator_checksum": _file_checksum(estimator_path),
        "auditbench_checksum": _file_checksum(auditbench_path),
    }
    Path(record_path).parent.mkdir(parents=True, exist_ok=True)
    with open(record_path, "w") as f:
        json.dump(record, f)


def check_validation_freshness(estimator_path: str, auditbench_path: str, record_path: str) -> None:
    """Raise ValidationStaleError unless record_path exists and its
    recorded checksums match the CURRENT contents of estimator_path and
    auditbench_path exactly. No recorded pass, or any mismatch, is stale."""
    if not Path(record_path).is_file():
        raise ValidationStaleError(
            f"no recorded validation pass found at {record_path!r} -- run "
            f"the pre-flight validation (run_synthetic_validation + "
            f"run_auditbench) before running the real-mechanism battery "
            f"analysis (architecture §6)"
        )
    with open(record_path, "r") as f:
        record = json.load(f)

    if record.get("estimator_checksum") != _file_checksum(estimator_path):
        raise ValidationStaleError(
            f"{estimator_path} has changed since the last recorded "
            f"validation pass -- re-run the pre-flight validation before "
            f"trusting any real-mechanism MI result (architecture §6)"
        )
    if record.get("auditbench_checksum") != _file_checksum(auditbench_path):
        raise ValidationStaleError(
            f"{auditbench_path} has changed since the last recorded "
            f"validation pass -- re-run the pre-flight validation before "
            f"trusting any real-mechanism MI result (architecture §6)"
        )


def _import_falsification_audit_modules(estimator_path: str, auditbench_path: str):
    """See module docstring's IMPORT-STYLE NOTE. Adds each file's directory
    to sys.path (idempotently) and imports both modules the same way running
    them as scripts would, so auditbench.py's own `from mi_estimator import
    (...)` resolves correctly."""
    estimator_dir = str(Path(estimator_path).resolve().parent)
    auditbench_dir = str(Path(auditbench_path).resolve().parent)
    if estimator_dir not in sys.path:
        sys.path.insert(0, estimator_dir)
    if auditbench_dir not in sys.path:
        sys.path.insert(0, auditbench_dir)

    import mi_estimator
    import auditbench

    return mi_estimator, auditbench


def run_preflight_validation(
    estimator_path: str,
    auditbench_path: str,
    record_path: str,
    toy_scale_log_length: int,
) -> None:
    """Run both pre-flight checks and, only if both succeed, record a fresh
    validation pass.

    toy_scale_log_length is passed through to run_synthetic_validation's
    n_steps -- this project's toy-scale runs will produce shorter logs than
    the ~2,000-step default the estimator was originally validated at
    (architecture §13.5's explicitly flagged bottleneck), so validating at
    the default length and trusting it at a different real length would be
    exactly the kind of stale-by-a-different-axis mistake this whole gate
    exists to prevent.

    Neither run_synthetic_validation nor run_auditbench is caught here --
    both raise AssertionError on failure per their own docstrings, and a
    failed pre-flight validation must stop everything downstream, not be
    silently logged and proceeded past.
    """
    mi_estimator, auditbench = _import_falsification_audit_modules(
        estimator_path, auditbench_path
    )
    mi_estimator.run_synthetic_validation(n_steps=toy_scale_log_length)
    auditbench.run_auditbench()
    record_validation_pass(estimator_path, auditbench_path, record_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Falsification-audit pre-flight gate + battery analysis entrypoint."
    )
    parser.add_argument("--estimator-path", default="falsification_audit/mi_estimator.py")
    parser.add_argument("--auditbench-path", default="falsification_audit/auditbench.py")
    parser.add_argument("--record-path", default="./logs/.validation_pass.json")
    parser.add_argument(
        "--toy-scale-log-length", type=int, required=True,
        help="actual number of logged signal points per run (architecture §13.5)",
    )
    parser.add_argument(
        "--refresh-validation", action="store_true",
        help="run the pre-flight validation now, even if a recorded pass already looks fresh",
    )
    args = parser.parse_args(argv)

    if args.refresh_validation:
        print(
            "[audit] running pre-flight validation (estimator synthetic "
            "check + full AuditBench sweep)..."
        )
        run_preflight_validation(
            args.estimator_path, args.auditbench_path, args.record_path,
            args.toy_scale_log_length,
        )
        print("[audit] pre-flight validation passed and recorded.")

    try:
        check_validation_freshness(
            args.estimator_path, args.auditbench_path, args.record_path
        )
    except ValidationStaleError as e:
        print(f"[audit] REFUSING to run: {e}", file=sys.stderr)
        return 1

    print("[audit] validation is fresh -- real-mechanism battery analysis would run here.")
    # The actual battery-analysis wiring (loading each mechanism's logged
    # signal via checkpointing/reader.py + battery/trigger_log.py, feeding
    # it to mi_estimator.permutation_test_mi, assembling BatteryResult) is
    # orchestration/report.py's job (post-hoc, on completed runs, per
    # architecture §7), not audit.py's -- audit.py's scope per architecture
    # §6 is specifically the pre-flight gate itself.
    return 0


if __name__ == "__main__":
    sys.exit(main())
