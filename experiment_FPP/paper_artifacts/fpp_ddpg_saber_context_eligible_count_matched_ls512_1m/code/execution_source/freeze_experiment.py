#!/usr/bin/env python3
"""Freeze production semantics only after the preregistered technical gates pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from source_fingerprint import (
    collect_source_hashes,
    make_fingerprint,
    validate_fingerprint_payload,
)


HERE = Path(__file__).resolve().parent
FROZEN_FILES = (
    "config.json",
    "run.py",
    "validate_config_parity.py",
    "validate_results.py",
    "freeze_experiment.py",
    "source_fingerprint.py",
    "test_source_fingerprint.py",
    "run_smoke_5k_seed1.sh",
    "run_technical_gate_50k_seed1.sh",
    "run_main_1m_seeds1_5_parallel.sh",
    "run_tests.sh",
    "test_validation_optional_nan.py",
    "analysis/strict_analysis.py",
    "analysis/analyze_results.py",
    "analysis/build_artifact.py",
    "analysis/verify_artifact.py",
    "analysis/test_analysis.py",
    "analysis/test_artifact.py",
    "code/src/train_fetch_pick_and_place_lite_v0.py",
    "code/src/metacog_lite_v0/config.py",
    "code/src/metacog_lite_v0/weights.py",
    "code/src/metacog_lite_v0/assignment_diagnostics.py",
    "code/src/metacog_lite_v0/trainer.py",
    "code/src/metacog_lite_v0/numba_support_scan_fpp.py",
    "code/src/test_context_eligible_count_matched.py",
    "code/src/test_ddpg_strict_loss_scope.py",
    "code/src/test_trainer_resume_contract.py",
    "code/src/test_resume_output_rollback.py",
    "code/src/test_support_query_modes.py",
    "code/src/test_ddpg_query_loss_isolation.py",
    "code/src/test_current_policy_legacy_parity.py",
    "code/src/test_numba_runtime_opt_fpp.py",
    "code/src/test_numba_support_scan_fpp.py",
    "code/src/test_b3_runtime_opt_lookup.py",
    "code/src/weight_parity_fixture.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_passed(path: Path, expected_kind: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("kind") != expected_kind or value.get("passed") is not True:
        raise ValueError(f"{path}: {expected_kind} validation did not pass")
    if value.get("performance_used_for_gate_decision") is not False:
        raise ValueError(f"{path}: performance was used as a gate")
    expected_steps = {"smoke": 5_000, "gate": 50_000}[expected_kind]
    if value.get("steps") != expected_steps or value.get("seeds") != [1]:
        raise ValueError(
            f"{path}: expected steps={expected_steps}, seeds=[1]"
        )
    runs = value.get("runs")
    if not isinstance(runs, list) or len(runs) != 1:
        raise ValueError(f"{path}: expected exactly one seed-1 validation record")
    record = runs[0]
    if (
        record.get("seed") != 1
        or record.get("steps") != expected_steps
        or record.get("stage") != expected_kind
    ):
        raise ValueError(f"{path}: seed/stage/step record mismatch")
    diagnostics = record.get("assignment_diagnostics", {})
    for field in (
        "exact_k_mismatch_count", "raw_histogram_mismatch_count",
        "normalized_histogram_mismatch_count", "invariant_failure_count",
    ):
        if int(diagnostics.get(field, -1)) != 0:
            raise ValueError(f"{path}: {field} is not zero")
    checkpoint = record.get("checkpoint", {})
    if checkpoint.get("assignment_rng_state_present") is not True or checkpoint.get("diagnostic_state_present") is not True:
        raise ValueError(f"{path}: checkpoint state evidence missing")
    if record.get("log", {}).get("resume_exercised") is not True:
        raise ValueError(f"{path}: checkpoint resume was not exercised")
    fingerprint = value.get("source_fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise ValueError(f"{path}: source fingerprint evidence missing")
    validate_fingerprint_payload(fingerprint, expected_stage=expected_kind)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-validation", type=Path, required=True)
    parser.add_argument("--gate-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "freeze_manifest.json")
    args = parser.parse_args()
    smoke = load_passed(args.smoke_validation.resolve(), "smoke")
    gate = load_passed(args.gate_validation.resolve(), "gate")
    smoke_hashes = validate_fingerprint_payload(
        smoke["source_fingerprint"], expected_stage="smoke"
    )
    gate_hashes = validate_fingerprint_payload(
        gate["source_fingerprint"], expected_stage="gate"
    )
    current_fingerprint = make_fingerprint("freeze", HERE)
    hashes = collect_source_hashes(HERE)
    if smoke_hashes != gate_hashes or smoke_hashes != hashes:
        raise ValueError(
            "smoke, gate, and current full-package source fingerprints differ"
        )
    for relative in FROZEN_FILES:
        if relative not in hashes:
            raise FileNotFoundError(f"frozen source is missing: {HERE / relative}")
    repo = HERE.parents[1]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    payload: Mapping[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head_at_freeze": head,
        "production_config_frozen": True,
        "performance_used_for_gate_decision": False,
        "smoke_validation": smoke,
        "gate_validation": gate,
        "source_fingerprint": current_fingerprint,
        "frozen_files_sha256": hashes,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
