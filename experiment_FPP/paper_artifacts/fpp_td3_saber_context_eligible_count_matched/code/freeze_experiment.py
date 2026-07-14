#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


FROZEN_FILES = (
    "config.json",
    "run.py",
    "analyze_results.py",
    "test_analyze_results.py",
    "test_run_config_parity.py",
    "validate_technical_gate.py",
    "freeze_experiment.py",
    "code/src/train_fetch_pick_and_place_lite_v0.py",
    "code/src/metacog_lite_v0/config.py",
    "code/src/metacog_lite_v0/weights.py",
    "code/src/metacog_lite_v0/assignment_diagnostics.py",
    "code/src/metacog_lite_v0/trainer.py",
    "code/src/test_context_eligible_count_matched.py",
    "code/src/test_support_query_modes.py",
    "code/src/test_current_policy_legacy_parity.py",
    "code/src/test_numba_runtime_opt_fpp.py",
    "code/src/test_numba_support_scan_fpp.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_passed_validation(path: Path, kind: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != kind or payload.get("passed") is not True:
        raise ValueError(f"{path}: {kind} validation did not pass")
    if payload.get("performance_used_for_gate_decision") is not False:
        raise ValueError(f"{path}: performance was used for gate decision")
    for field in (
        "invariant_failure_count",
        "exact_k_mismatch_count",
        "raw_histogram_mismatch_count",
        "normalized_histogram_mismatch_count",
    ):
        if int(payload.get(field, -1)) != 0:
            raise ValueError(f"{path}: {field} is not zero")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the count-matched production config and analyzer."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--smoke-validation", type=Path, required=True)
    parser.add_argument("--gate-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    folder = (
        repo_root
        / "experiment_FPP"
        / "td3_saber_context_eligible_count_matched_ls512"
    )
    smoke = load_passed_validation(
        args.smoke_validation.resolve(),
        "smoke",
    )
    gate = load_passed_validation(
        args.gate_validation.resolve(),
        "gate",
    )

    files: dict[str, str] = {}
    for relative in FROZEN_FILES:
        path = folder / relative
        if not path.is_file():
            raise FileNotFoundError(f"frozen source missing: {path}")
        files[relative] = sha256(path)

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit_at_freeze": head,
        "production_config_frozen": True,
        "analyzer_frozen": True,
        "performance_used_for_gate_decision": False,
        "smoke_validation": smoke,
        "gate_validation": gate,
        "frozen_files_sha256": files,
        "required_tests": [
            "code/src/test_context_eligible_count_matched.py",
            "code/src/test_support_query_modes.py",
            "code/src/test_current_policy_legacy_parity.py",
            "code/src/test_numba_runtime_opt_fpp.py",
            "code/src/test_numba_support_scan_fpp.py",
            "test_run_config_parity.py",
            "test_analyze_results.py",
        ],
    }
    output = (
        args.output.resolve()
        if args.output is not None
        else folder / "freeze_manifest.json"
    )
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
