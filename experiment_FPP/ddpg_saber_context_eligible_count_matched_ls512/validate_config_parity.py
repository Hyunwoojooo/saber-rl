#!/usr/bin/env python3
"""Fail-closed config/source parity audit for the DDPG strict assignment control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
CANONICAL_DIR = EXPERIMENT_DIR.parent / "b3_critic_only_runtime_opt_v4_ls512"
ADDED_CONFIG_KEYS = {
    "support_query_mode",
    "support_assignment_mode",
    "assignment_seed_offset",
    "assignment_diagnostic_window_updates",
}
CHANGED_CONFIG_KEYS = {"folder", "output_root", "purpose", "reference", "run_slug"}
MODIFIED_PRODUCTION_FILES = {
    Path("metacog_lite_v0/config.py"),
    Path("metacog_lite_v0/numba_support_scan_fpp.py"),
    Path("metacog_lite_v0/trainer.py"),
    Path("metacog_lite_v0/weights.py"),
    Path("train_fetch_pick_and_place_lite_v0.py"),
}
ADDED_PRODUCTION_FILES = {Path("metacog_lite_v0/assignment_diagnostics.py")}
REQUIRED_TESTS = {
    "test_context_eligible_count_matched.py",
    "test_ddpg_strict_loss_scope.py",
    "test_trainer_resume_contract.py",
    "test_resume_output_rollback.py",
    "test_support_query_modes.py",
    "test_ddpg_query_loss_isolation.py",
    "test_current_policy_legacy_parity.py",
    "test_b3_runtime_opt_lookup.py",
    "test_numba_runtime_opt_fpp.py",
    "test_numba_support_scan_fpp.py",
    "weight_parity_fixture.py",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON object {path}: {exc}") from exc
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def production_files(root: Path) -> dict[Path, Path]:
    require(root.is_dir(), f"missing source directory: {root}")
    output: dict[Path, Path] = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.name.startswith("test_"):
            continue
        if path.name == "weight_parity_fixture.py":
            continue
        output[relative] = path
    return output


def validate_config() -> dict[str, Any]:
    current = read_object(EXPERIMENT_DIR / "config.json")
    canonical = read_object(CANONICAL_DIR / "config.json")
    added = set(current) - set(canonical)
    missing = set(canonical) - set(current)
    changed = {key for key in canonical if current[key] != canonical[key]}
    require(added == ADDED_CONFIG_KEYS, f"config added-key mismatch: {sorted(added)}")
    require(not missing, f"config missing canonical keys: {sorted(missing)}")
    require(
        changed == CHANGED_CONFIG_KEYS,
        f"config changed-key mismatch: expected={sorted(CHANGED_CONFIG_KEYS)}, actual={sorted(changed)}",
    )
    expected = {
        "folder": EXPERIMENT_DIR.name,
        "run_slug": "ddpg_saber_context_eligible_count_matched_ls512",
        "output_root": "experiment_FPP/runs/ddpg_saber_context_eligible_count_matched_ls512",
        "reference": "experiment_FPP/b3_critic_only_runtime_opt_v4_ls512",
        "trainer_method": "b3_support_runtime_opt",
        "support_query_mode": "current_policy",
        "support_assignment_mode": "context_eligible_count_matched",
        "assignment_seed_offset": 30_000,
        "assignment_diagnostic_window_updates": 1_000,
        "weight_scope": "critic_only",
        "weighting_mode": "batch_norm",
        "seeds": [1, 2, 3, 4, 5],
        "total_steps": 1_000_000,
    }
    for key, value in expected.items():
        require(current.get(key) == value, f"config {key!r} expected {value!r}")
    require(canonical.get("support_query_mode", "current_policy") == "current_policy", "canonical query default drift")
    require(canonical.get("support_assignment_mode", "support") == "support", "canonical assignment default drift")
    return {
        "added_keys": sorted(added),
        "changed_keys": sorted(changed),
        "scientific_change": {
            "support_assignment_mode": ["support", "context_eligible_count_matched"]
        },
        "explicit_default": {"support_query_mode": "current_policy"},
        "control_seed_rule": "assignment_seed=30000+experiment_seed",
    }


def validate_source() -> dict[str, Any]:
    current_root = EXPERIMENT_DIR / "code/src"
    canonical_root = CANONICAL_DIR / "code/src"
    current = production_files(current_root)
    canonical = production_files(canonical_root)
    added = set(current) - set(canonical)
    missing = set(canonical) - set(current)
    modified = {
        relative
        for relative in set(current) & set(canonical)
        if current[relative].read_bytes() != canonical[relative].read_bytes()
    }
    require(added == ADDED_PRODUCTION_FILES, f"production source added-file drift: {sorted(map(str, added))}")
    require(not missing, f"production source missing files: {sorted(map(str, missing))}")
    require(
        modified == MODIFIED_PRODUCTION_FILES,
        "production delta mismatch: "
        f"expected={sorted(map(str, MODIFIED_PRODUCTION_FILES))}, actual={sorted(map(str, modified))}",
    )
    for relative in (
        Path("metacog_lite_v0/agent.py"),
        Path("metacog_lite_v0/replay.py"),
        Path("metacog_lite_v0/networks.py"),
    ):
        require(current[relative].read_bytes() == canonical[relative].read_bytes(), f"forbidden DDPG core drift: {relative}")
    present_tests = {path.name for path in current_root.iterdir() if path.is_file()}
    require(REQUIRED_TESTS <= present_tests, f"missing required tests: {sorted(REQUIRED_TESTS - present_tests)}")
    changed_paths = sorted(added | modified, key=lambda value: value.as_posix())
    changed_hashes = {path.as_posix(): sha256(current[path]) for path in changed_paths}
    unchanged_tree = hashlib.sha256()
    unchanged_count = 0
    for relative in sorted(set(canonical) - modified, key=lambda value: value.as_posix()):
        digest = sha256(current[relative])
        unchanged_tree.update(relative.as_posix().encode("utf-8") + b"\0" + bytes.fromhex(digest))
        unchanged_count += 1
    return {
        "added_files": sorted(path.as_posix() for path in added),
        "modified_files": sorted(path.as_posix() for path in modified),
        "changed_sha256": changed_hashes,
        "canonical_identical_file_count": unchanged_count,
        "canonical_identical_tree_sha256": unchanged_tree.hexdigest(),
        "required_tests": sorted(REQUIRED_TESTS),
    }


def main() -> int:
    print(json.dumps({
        "ok": True,
        "canonical": "experiment_FPP/b3_critic_only_runtime_opt_v4_ls512",
        "config": validate_config(),
        "production_source": validate_source(),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
