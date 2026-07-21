#!/usr/bin/env python3
"""Freeze the one-factor config and generic DDPG query-mode source delta."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
CANONICAL_DIR = EXPERIMENT_DIR.parent / "b3_critic_only_runtime_opt_v4_ls512"
ADDED_CONFIG_KEYS = {"support_query_mode"}
CHANGED_CONFIG_KEYS = {
    "folder",
    "output_root",
    "purpose",
    "reference",
    "run_slug",
}
QUERY_PRODUCTION_FILES = {
    Path("metacog_lite_v0/config.py"),
    Path("metacog_lite_v0/numba_support_scan_fpp.py"),
    Path("metacog_lite_v0/trainer.py"),
    Path("metacog_lite_v0/weights.py"),
    Path("train_fetch_pick_and_place_lite_v0.py"),
}
# These hashes freeze the reviewed generic query-mode port.  Any production
# edit, even inside an allowlisted file, requires an explicit scientific audit.
EXPECTED_QUERY_SOURCE_SHA256 = {
    "metacog_lite_v0/config.py": "c81f187f16b91a82b123d3d959c9aa2bbbaac104ee1d90c77f3ca5736b8b9114",
    "metacog_lite_v0/numba_support_scan_fpp.py": "4be4be463a54deddd0a2bc9a94ead3d821c768e1e8fa1dd1a72d652081d6d28a",
    "metacog_lite_v0/trainer.py": "fa5b768fda0e1d634030b403edd8138cede4710b57c08f1d818305cb98fe516c",
    "metacog_lite_v0/weights.py": "2b7c68c8c1eebf65159eb057a0241086e58cdd4efcf7ee985a4a5c9e4f4f2875",
    "train_fetch_pick_and_place_lite_v0.py": "a52c35a8e1583149cfd5050d9a57dabf9ebf12276996dcac9483034bab73ab3b",
}
REQUIRED_QUERY_TESTS = {
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


def production_files(source_root: Path) -> dict[Path, Path]:
    require(source_root.is_dir(), f"missing source directory: {source_root}")
    files: dict[Path, Path] = {}
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root)
        if "__pycache__" in relative.parts:
            continue
        if path.name.startswith("test_") or path.name == "weight_parity_fixture.py":
            continue
        files[relative] = path
    require(files, f"no production Python under {source_root}")
    return files


def validate_config() -> dict[str, Any]:
    current = read_object(EXPERIMENT_DIR / "config.json")
    canonical = read_object(CANONICAL_DIR / "config.json")
    added = set(current) - set(canonical)
    missing = set(canonical) - set(current)
    require(added == ADDED_CONFIG_KEYS, f"config added keys mismatch: {sorted(added)}")
    require(not missing, f"config missing canonical keys: {sorted(missing)}")
    changed = {key for key in canonical if current[key] != canonical[key]}
    require(
        changed == CHANGED_CONFIG_KEYS,
        "config changed-key mismatch: "
        f"expected={sorted(CHANGED_CONFIG_KEYS)}, actual={sorted(changed)}",
    )
    require(current["support_query_mode"] == "replay_action", "query mode must be replay_action")
    require(canonical.get("support_query_mode", "current_policy") == "current_policy", "canonical query default drift")
    require(current["trainer_method"] == canonical["trainer_method"] == "b3_support_runtime_opt", "trainer drift")
    require(current["weight_scope"] == canonical["weight_scope"] == "critic_only", "weight scope drift")
    require(current["weighting_mode"] == canonical["weighting_mode"] == "batch_norm", "weight mode drift")
    require(current["folder"] == EXPERIMENT_DIR.name, "folder/package mismatch")
    require(current["run_slug"] == "ddpg_saber_query_replay_ls512", "run_slug mismatch")
    require(
        current["output_root"] == "experiment_FPP/runs/ddpg_saber_query_replay_ls512",
        "main output root mismatch",
    )
    require(current["reference"] == "experiment_FPP/b3_critic_only_runtime_opt_v4_ls512", "reference mismatch")
    require(current["seeds"] == [1, 2, 3, 4, 5], "seed registration drift")
    require(current["total_steps"] == 1_000_000, "main budget drift")
    return {
        "added_keys": sorted(added),
        "changed_keys": sorted(changed),
        "support_query_mode": current["support_query_mode"],
        "weight_scope": current["weight_scope"],
    }


def validate_source() -> dict[str, Any]:
    current_root = EXPERIMENT_DIR / "code" / "src"
    canonical_root = CANONICAL_DIR / "code" / "src"
    current = production_files(current_root)
    canonical = production_files(canonical_root)
    require(
        set(current) == set(canonical),
        "production source file-set drift: "
        f"added={sorted(map(str, set(current) - set(canonical)))}, "
        f"missing={sorted(map(str, set(canonical) - set(current)))}",
    )

    modified = {
        relative
        for relative in current
        if current[relative].read_bytes() != canonical[relative].read_bytes()
    }
    require(
        modified == QUERY_PRODUCTION_FILES,
        "generic query production delta mismatch: "
        f"expected={sorted(map(str, QUERY_PRODUCTION_FILES))}, "
        f"actual={sorted(map(str, modified))}",
    )
    require(
        set(EXPECTED_QUERY_SOURCE_SHA256) == {path.as_posix() for path in QUERY_PRODUCTION_FILES},
        "frozen query hash registry does not match source allowlist",
    )

    changed_hashes = {relative.as_posix(): sha256(current[relative]) for relative in sorted(modified)}
    require(
        changed_hashes == EXPECTED_QUERY_SOURCE_SHA256,
        "allowlisted query source hash mismatch: "
        f"expected={EXPECTED_QUERY_SOURCE_SHA256}, actual={changed_hashes}",
    )

    unchanged_tree = hashlib.sha256()
    unchanged_count = 0
    for relative in sorted(set(current) - modified, key=lambda value: value.as_posix()):
        digest = sha256(current[relative])
        unchanged_tree.update(relative.as_posix().encode("utf-8"))
        unchanged_tree.update(b"\0")
        unchanged_tree.update(bytes.fromhex(digest))
        unchanged_count += 1

    present_tests = {path.name for path in current_root.iterdir() if path.is_file()}
    require(
        REQUIRED_QUERY_TESTS <= present_tests,
        f"missing required query/parity tests: {sorted(REQUIRED_QUERY_TESTS - present_tests)}",
    )
    # The actor/critic optimizer implementation and replay buffer are outside
    # the generic query delta and must therefore be canonical byte-for-byte.
    for relative in (Path("metacog_lite_v0/agent.py"), Path("metacog_lite_v0/replay.py")):
        require(current[relative].read_bytes() == canonical[relative].read_bytes(), f"forbidden DDPG core drift: {relative}")

    return {
        "modified_files": sorted(changed_hashes),
        "modified_sha256": changed_hashes,
        "canonical_identical_file_count": unchanged_count,
        "canonical_identical_tree_sha256": unchanged_tree.hexdigest(),
        "required_tests": sorted(REQUIRED_QUERY_TESTS),
    }


def main() -> int:
    payload = {
        "ok": True,
        "canonical": "experiment_FPP/b3_critic_only_runtime_opt_v4_ls512",
        "config": validate_config(),
        "production_source": validate_source(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
