#!/usr/bin/env python3
"""Validate that the actor-critic arm changes only its registered scope metadata.

The completed DDPG SABER v4/LS512 critic-only experiment is the scientific
reference for this arm.  Production Python must remain byte-for-byte identical
to that snapshot; the only intended learning change is ``weight_scope``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
CANONICAL_DIR = EXPERIMENT_DIR.parent / "b3_critic_only_runtime_opt_v4_ls512"
ALLOWED_CONFIG_CHANGES = {
    "folder",
    "output_root",
    "purpose",
    "reference",
    "run_slug",
    "weight_scope",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON object {path}: {exc}") from exc
    require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def production_files(source_root: Path) -> dict[Path, Path]:
    """Return production .py files, excluding package-local validation tests."""
    require(source_root.is_dir(), f"missing source directory: {source_root}")
    files: dict[Path, Path] = {}
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root)
        if "__pycache__" in relative.parts or path.name.startswith("test_"):
            continue
        files[relative] = path
    require(files, f"no production Python files found under {source_root}")
    return files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config() -> dict[str, Any]:
    current = read_object(EXPERIMENT_DIR / "config.json")
    canonical = read_object(CANONICAL_DIR / "config.json")

    require(
        set(current) == set(canonical),
        "actor-critic and canonical config keys differ: "
        f"added={sorted(set(current) - set(canonical))}, "
        f"missing={sorted(set(canonical) - set(current))}",
    )
    changed = {
        key for key in canonical if current.get(key) != canonical.get(key)
    }
    unexpected = changed - ALLOWED_CONFIG_CHANGES
    require(
        not unexpected,
        f"non-allowlisted config changes: {sorted(unexpected)}",
    )
    require(
        changed == ALLOWED_CONFIG_CHANGES,
        "registered metadata/scope changes must all be explicit: "
        f"expected={sorted(ALLOWED_CONFIG_CHANGES)}, actual={sorted(changed)}",
    )

    require(
        canonical.get("trainer_method") == "b3_support_runtime_opt",
        "canonical trainer_method is not b3_support_runtime_opt",
    )
    require(
        current.get("trainer_method") == "b3_support_runtime_opt",
        "actor-critic trainer_method must be b3_support_runtime_opt",
    )
    require(
        canonical.get("weight_scope") == "critic_only",
        "canonical weight_scope is not critic_only",
    )
    require(
        current.get("weight_scope") == "actor_critic",
        "actor-critic weight_scope must be actor_critic",
    )
    require(
        current.get("folder") == EXPERIMENT_DIR.name,
        f"folder must equal package name {EXPERIMENT_DIR.name!r}",
    )
    require(
        current.get("run_slug") == "ddpg_saber_actor_critic_v4_ls512",
        "run_slug must be ddpg_saber_actor_critic_v4_ls512",
    )
    require(
        current.get("output_root")
        == "experiment_FPP/runs/ddpg_saber_actor_critic_v4_ls512",
        "output_root must isolate actor-critic main runs under experiment_FPP/runs",
    )
    purpose = current.get("purpose")
    require(isinstance(purpose, str) and purpose.strip(), "purpose must be nonempty")
    require(
        "both critic and actor" in purpose.lower(),
        "purpose must identify weighting of both critic and actor losses",
    )
    require(
        current.get("reference")
        == "experiment_FPP/b3_critic_only_runtime_opt_v4_ls512",
        "reference must name the exact canonical critic-only package",
    )

    return {
        "changed_keys": sorted(changed),
        "trainer_method": current["trainer_method"],
        "weight_scope": current["weight_scope"],
        "run_slug": current["run_slug"],
        "output_root": current["output_root"],
    }


def validate_source() -> dict[str, Any]:
    current_root = EXPERIMENT_DIR / "code" / "src"
    canonical_root = CANONICAL_DIR / "code" / "src"
    current = production_files(current_root)
    canonical = production_files(canonical_root)
    require(
        set(current) == set(canonical),
        "production source file sets differ: "
        f"added={sorted(map(str, set(current) - set(canonical)))}, "
        f"missing={sorted(map(str, set(canonical) - set(current)))}",
    )

    mismatches: list[str] = []
    file_hashes: dict[str, str] = {}
    tree_digest = hashlib.sha256()
    for relative in sorted(current, key=lambda value: value.as_posix()):
        current_bytes = current[relative].read_bytes()
        canonical_bytes = canonical[relative].read_bytes()
        if current_bytes != canonical_bytes:
            mismatches.append(relative.as_posix())
            continue
        digest = sha256(current[relative])
        file_hashes[relative.as_posix()] = digest
        tree_digest.update(relative.as_posix().encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(bytes.fromhex(digest))

    require(not mismatches, f"production source byte mismatches: {mismatches}")
    return {
        "file_count": len(current),
        "tree_sha256": tree_digest.hexdigest(),
        "files": file_hashes,
    }


def main() -> int:
    payload = {
        "ok": True,
        "canonical": str(CANONICAL_DIR.relative_to(EXPERIMENT_DIR.parents[1])),
        "config": validate_config(),
        "production_source": validate_source(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
