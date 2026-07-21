#!/usr/bin/env python3
"""Standalone fail-closed verifier for the compact replay-query artifact."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from query_ablation import (
    ARTIFACT_ID,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    GRID,
    MAIN_KEY,
    PAIR_CONFIG_EXEMPT_KEYS,
    REPLAY_KEY,
    SCHEMA_VERSION,
    SEEDS,
    TOTAL_STEPS,
    TOTAL_UPDATES,
    AnalysisError,
    analyze_query_ablation,
    config_differences,
    read_json,
    verify_rendered_outputs,
)


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REPLAY_SOURCE_PATH = "experiment_FPP/ddpg_saber_query_replay_ls512"
MAIN_SOURCE_PATH = "experiment_FPP/b3_critic_only_runtime_opt_v4_ls512"
EXCLUDED_SOURCE_DIR_NAMES = frozenset({"analysis", "__pycache__", ".pytest_cache"})
EXCLUDED_SOURCE_SUFFIXES = frozenset({".pyc", ".pt", ".pth", ".ckpt"})
ROOT_FILES = frozenset(
    {
        "README.md",
        "input_manifest.json",
        "artifact_manifest.json",
        "SHA256SUMS",
        "source_commit.txt",
    }
)
ROOT_DIRS = frozenset({"code", "data", "results", "tests"})
ANALYSIS_CODE_FILES = frozenset(
    {"query_ablation.py", "analyze_results.py", "build_artifact.py", "verify_artifact.py"}
)
TEST_FILES = frozenset({"test_analysis.py", "test_artifact.py", "test_evidence.json"})
SEED_FILES = frozenset(
    {"config.json", "env_snapshot.json", "summary.json", "metrics_selected_10k.csv"}
)
RESULT_FILES = frozenset(
    {
        "per_seed_results.csv",
        "aggregate_results.csv",
        "paired_differences.csv",
        "paired_inference.csv",
        "leave_one_seed_out.csv",
        "mean_learning_curve.csv",
        "query_diagnostics_per_seed.csv",
        "query_diagnostics_aggregate.csv",
        "analysis_results.json",
        "report.md",
        "validation.json",
    }
)
FORBIDDEN_NAMES = frozenset(
    {"metrics.csv", "metrics.jsonl", "run.log", "checkpoint.pt", "best_checkpoint.pt"}
)


class ArtifactVerificationError(AnalysisError):
    """Raised when packaged contents do not exactly satisfy the artifact contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_object_oid(kind: str, payload: bytes) -> str:
    header = f"{kind} {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _decode_base64(value: Any, *, label: str) -> bytes:
    if not isinstance(value, str):
        raise ArtifactVerificationError(f"{label} must be base64 text")
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ArtifactVerificationError(f"Invalid base64 payload for {label}") from exc


def _parse_git_tree(payload: bytes) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    cursor = 0
    while cursor < len(payload):
        space = payload.find(b" ", cursor)
        nul = payload.find(b"\0", space + 1)
        if space < 0 or nul < 0 or nul + 21 > len(payload):
            raise ArtifactVerificationError("Malformed packaged Git tree object")
        try:
            mode = payload[cursor:space].decode("ascii")
            name = payload[space + 1 : nul].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ArtifactVerificationError("Non-UTF-8 packaged Git tree entry") from exc
        oid = payload[nul + 1 : nul + 21].hex()
        if not name or "/" in name or name in entries or not HEX40.fullmatch(oid):
            raise ArtifactVerificationError(f"Invalid packaged Git tree entry: {name!r}")
        entries[name] = (mode, oid)
        cursor = nul + 21
    return entries


def _resolve_git_path(
    root_tree_oid: str,
    tree_objects: Mapping[str, bytes],
    parts: Sequence[str],
) -> tuple[str, str]:
    if not parts:
        return "40000", root_tree_oid
    current = root_tree_oid
    for index, part in enumerate(parts):
        payload = tree_objects.get(current)
        if payload is None:
            raise ArtifactVerificationError(
                f"Git provenance is missing tree object {current} while resolving {'/'.join(parts)}"
            )
        entry = _parse_git_tree(payload).get(part)
        if entry is None:
            raise ArtifactVerificationError(
                f"Git provenance path is absent: {'/'.join(parts[: index + 1])}"
            )
        mode, oid = entry
        if index < len(parts) - 1 and mode not in {"40000", "040000"}:
            raise ArtifactVerificationError(
                f"Non-tree Git provenance path component: {'/'.join(parts[: index + 1])}"
            )
        current = oid
    return mode, current


def _safe_relative_parts(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise ArtifactVerificationError(f"{label} must be a relative path string")
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactVerificationError(f"Unsafe {label}: {value!r}")
    return path.parts


def _enumerate_eligible_source_blobs(
    tree_oid: str,
    tree_objects: Mapping[str, bytes],
    prefix: tuple[str, ...] = (),
) -> dict[str, tuple[str, str]]:
    payload = tree_objects.get(tree_oid)
    if payload is None:
        raise ArtifactVerificationError(
            f"Git provenance is missing authenticated source subtree {tree_oid}"
        )
    result: dict[str, tuple[str, str]] = {}
    for name, (mode, oid) in _parse_git_tree(payload).items():
        parts = prefix + (name,)
        if mode in {"40000", "040000"}:
            if name in EXCLUDED_SOURCE_DIR_NAMES:
                continue
            result.update(_enumerate_eligible_source_blobs(oid, tree_objects, parts))
            continue
        relative = Path(*parts)
        if relative.suffix in EXCLUDED_SOURCE_SUFFIXES:
            continue
        if mode not in {"100644", "100755"}:
            raise ArtifactVerificationError(
                f"Unsupported eligible committed source mode {mode}: {relative.as_posix()}"
            )
        result[relative.as_posix()] = (mode, oid)
    return result


def _verify_one_source_provenance(
    root: Path,
    payload: Mapping[str, Any],
    *,
    expected_name: str,
    expected_commit: str,
    expected_source_path: str,
    expected_packaged_root: str,
    expected_execution_captured: bool,
) -> dict[str, Any]:
    expected_scalars = {
        "schema_version": 1,
        "name": expected_name,
        "git_object_format": "sha1",
        "commit": expected_commit,
        "source_path_at_commit": expected_source_path,
        "packaged_root": expected_packaged_root,
        "execution_commit_captured": expected_execution_captured,
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise ArtifactVerificationError(
                f"{expected_name} provenance {key} expected {expected!r}, got {payload.get(key)!r}"
            )
    commit_payload = _decode_base64(
        payload.get("commit_object_base64"), label=f"{expected_name}.commit_object_base64"
    )
    if _git_object_oid("commit", commit_payload) != expected_commit:
        raise ArtifactVerificationError(f"{expected_name} packaged commit object hash mismatch")
    commit_tree_lines = [
        line[5:].decode("ascii", errors="strict")
        for line in commit_payload.split(b"\n\n", 1)[0].splitlines()
        if line.startswith(b"tree ")
    ]
    if len(commit_tree_lines) != 1 or not HEX40.fullmatch(commit_tree_lines[0]):
        raise ArtifactVerificationError(f"{expected_name} commit object has invalid root tree")
    root_tree_oid = commit_tree_lines[0]
    if payload.get("commit_root_tree_oid") != root_tree_oid:
        raise ArtifactVerificationError(f"{expected_name} recorded commit root tree mismatch")

    encoded_trees = payload.get("tree_objects_base64")
    if not isinstance(encoded_trees, Mapping) or not encoded_trees:
        raise ArtifactVerificationError(f"{expected_name} tree object map is empty")
    trees: dict[str, bytes] = {}
    for oid, encoded in encoded_trees.items():
        if not isinstance(oid, str) or not HEX40.fullmatch(oid):
            raise ArtifactVerificationError(f"{expected_name} has invalid tree OID key: {oid!r}")
        tree_payload = _decode_base64(encoded, label=f"{expected_name}.tree_objects[{oid}]")
        if _git_object_oid("tree", tree_payload) != oid:
            raise ArtifactVerificationError(f"{expected_name} packaged tree object hash mismatch: {oid}")
        trees[oid] = tree_payload

    source_parts = _safe_relative_parts(
        payload.get("source_path_at_commit"), label=f"{expected_name}.source_path_at_commit"
    )
    source_mode, source_oid = _resolve_git_path(root_tree_oid, trees, source_parts)
    if source_mode not in {"40000", "040000"} or source_oid != payload.get(
        "source_root_tree_oid"
    ):
        raise ArtifactVerificationError(f"{expected_name} source-root tree chain mismatch")
    code_mode, code_oid = _resolve_git_path(source_oid, trees, ("code", "src"))
    if code_mode not in {"40000", "040000"} or code_oid != payload.get("code_src_tree_oid"):
        raise ArtifactVerificationError(f"{expected_name} code/src tree chain mismatch")
    config_mode, config_oid = _resolve_git_path(source_oid, trees, ("config.json",))
    if config_mode not in {"100644", "100755"} or config_oid != payload.get("config_blob_oid"):
        raise ArtifactVerificationError(f"{expected_name} config blob chain mismatch")

    eligible_committed = _enumerate_eligible_source_blobs(source_oid, trees)

    records = payload.get("selected_files")
    if not isinstance(records, list) or not records:
        raise ArtifactVerificationError(f"{expected_name} selected file provenance is empty")
    packaged_root = root / expected_packaged_root
    recorded_paths: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ArtifactVerificationError(f"Invalid {expected_name} selected file record")
        relative_value = record.get("relative_path")
        relative_parts = _safe_relative_parts(
            relative_value, label=f"{expected_name}.selected_files.relative_path"
        )
        relative = Path(*relative_parts).as_posix()
        if relative in recorded_paths:
            raise ArtifactVerificationError(f"Duplicate {expected_name} selected file: {relative}")
        recorded_paths.add(relative)
        expected_packaged = (Path(expected_packaged_root) / relative).as_posix()
        if record.get("packaged_path") != expected_packaged:
            raise ArtifactVerificationError(f"{expected_name} packaged source path mismatch: {relative}")
        mode, blob_oid = _resolve_git_path(source_oid, trees, relative_parts)
        if mode != record.get("mode") or blob_oid != record.get("blob_oid"):
            raise ArtifactVerificationError(f"{expected_name} Git tree/blob membership mismatch: {relative}")
        path = root / expected_packaged
        if not path.is_file() or path.is_symlink():
            raise ArtifactVerificationError(f"Missing packaged {expected_name} source file: {relative}")
        source_bytes = path.read_bytes()
        if _git_object_oid("blob", source_bytes) != blob_oid:
            raise ArtifactVerificationError(f"{expected_name} packaged Git blob mismatch: {relative}")
        if hashlib.sha256(source_bytes).hexdigest() != record.get("sha256"):
            raise ArtifactVerificationError(f"{expected_name} packaged SHA256 mismatch: {relative}")
        if len(source_bytes) != record.get("size_bytes"):
            raise ArtifactVerificationError(f"{expected_name} packaged size mismatch: {relative}")
    if recorded_paths != set(eligible_committed):
        raise ArtifactVerificationError(
            f"{expected_name} selected source is incomplete relative to the authenticated "
            f"committed subtree: missing={sorted(set(eligible_committed)-recorded_paths)}, "
            f"extra={sorted(recorded_paths-set(eligible_committed))}"
        )
    for relative, (expected_mode, expected_oid) in eligible_committed.items():
        record = next(
            item for item in records if item["relative_path"] == relative
        )
        if record.get("mode") != expected_mode or record.get("blob_oid") != expected_oid:
            raise ArtifactVerificationError(
                f"{expected_name} authenticated source identity mismatch: {relative}"
            )
    actual_paths = {
        path.relative_to(packaged_root).as_posix()
        for path in packaged_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != recorded_paths:
        raise ArtifactVerificationError(
            f"{expected_name} frozen source file set mismatch: "
            f"missing={sorted(actual_paths-recorded_paths)}, stale={sorted(recorded_paths-actual_paths)}"
        )
    return dict(payload)


def _verify_source_provenance(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    payload = read_json(root / "code/source_provenance.json")
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ArtifactVerificationError("Source provenance schema mismatch")
    if set(payload) != {"schema_version", "replay_execution", "canonical_main_reference"}:
        raise ArtifactVerificationError("Source provenance arm set mismatch")
    replay = _verify_one_source_provenance(
        root,
        payload["replay_execution"],
        expected_name="replay_execution",
        expected_commit=str(manifest["execution_source_commit"]),
        expected_source_path=REPLAY_SOURCE_PATH,
        expected_packaged_root="code/execution_source",
        expected_execution_captured=True,
    )
    main = _verify_one_source_provenance(
        root,
        payload["canonical_main_reference"],
        expected_name="canonical_main_reference",
        expected_commit=str(manifest["canonical_main_source_commit"]),
        expected_source_path=MAIN_SOURCE_PATH,
        expected_packaged_root="code/main_source",
        expected_execution_captured=False,
    )
    return {"replay_execution": replay, "canonical_main_reference": main}


def _verify_source_manifest_alignment(
    input_manifest: Mapping[str, Any],
    provenance: Mapping[str, Mapping[str, Any]],
) -> None:
    groups = (
        ("frozen_execution_source_files", "replay_execution"),
        ("frozen_main_reference_source_files", "canonical_main_reference"),
    )
    for manifest_key, provenance_key in groups:
        direct_records = {
            str(record["packaged_path"]): {
                "relative_path": record["git_source_relative_path"],
                "packaged_path": record["packaged_path"],
                "mode": record["git_mode"],
                "blob_oid": record["git_blob_oid"],
                "sha256": record["source_sha256"],
                "size_bytes": record["source_size_bytes"],
            }
            for record in input_manifest[manifest_key]
        }
        provenance_records = {
            str(record["packaged_path"]): dict(record)
            for record in provenance[provenance_key]["selected_files"]
        }
        if direct_records != provenance_records:
            raise ArtifactVerificationError(
                f"Input manifest and Git provenance disagree for {manifest_key}"
            )


def all_files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.relative_to(root).parts
        and path.suffix != ".pyc"
    }


def _strict_json(path: Path) -> Any:
    def reject(value: str) -> Any:
        raise ArtifactVerificationError(f"{path}: non-standard JSON numeric token {value}")

    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    except json.JSONDecodeError as exc:
        raise ArtifactVerificationError(f"Invalid strict JSON {path}: {exc}") from exc


def _assert_no_forbidden_payloads(root: Path) -> None:
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ArtifactVerificationError(f"Symlinks are prohibited: {relative}")
        lowered = [part.lower() for part in relative.parts]
        if relative.parts and relative.parts[0] == "data" and any(
            "smoke" in part for part in lowered
        ):
            raise ArtifactVerificationError(f"Smoke payload/path is prohibited: {relative}")
        if path.is_file() and (
            path.name in FORBIDDEN_NAMES
            or path.suffix.lower() in {".pt", ".pth", ".ckpt", ".jsonl", ".log"}
        ):
            raise ArtifactVerificationError(f"Raw/checkpoint payload is prohibited: {relative}")
        if path.is_file() and path.suffix == ".json":
            _strict_json(path)


def _verify_layout(root: Path) -> None:
    files = {path.name for path in root.iterdir() if path.is_file()}
    dirs = {path.name for path in root.iterdir() if path.is_dir()}
    if files != ROOT_FILES:
        raise ArtifactVerificationError(
            f"Root file set mismatch: expected={sorted(ROOT_FILES)}, actual={sorted(files)}"
        )
    if dirs != ROOT_DIRS:
        raise ArtifactVerificationError(
            f"Root directory set mismatch: expected={sorted(ROOT_DIRS)}, actual={sorted(dirs)}"
        )
    code_files = {path.name for path in (root / "code").iterdir() if path.is_file()}
    if code_files != ANALYSIS_CODE_FILES | {"execution_metadata.json", "source_provenance.json"}:
        raise ArtifactVerificationError(f"Code file set mismatch: {sorted(code_files)}")
    if not (root / "code/execution_source").is_dir():
        raise ArtifactVerificationError("Frozen code/execution_source directory is missing")
    if not (root / "code/main_source").is_dir():
        raise ArtifactVerificationError("Frozen code/main_source directory is missing")
    test_files = {path.name for path in (root / "tests").iterdir() if path.is_file()}
    if test_files != TEST_FILES:
        raise ArtifactVerificationError(f"Tests file set mismatch: {sorted(test_files)}")
    result_files = {path.name for path in (root / "results").iterdir() if path.is_file()}
    if result_files != RESULT_FILES:
        raise ArtifactVerificationError(
            f"Results file set mismatch: expected={sorted(RESULT_FILES)}, actual={sorted(result_files)}"
        )
    data_arms = {path.name for path in (root / "data").iterdir() if path.is_dir()}
    if data_arms != {MAIN_KEY, REPLAY_KEY}:
        raise ArtifactVerificationError(f"Data method set mismatch: {sorted(data_arms)}")
    for method_key in (MAIN_KEY, REPLAY_KEY):
        method_root = root / "data" / method_key
        seed_dirs = {path.name for path in method_root.iterdir() if path.is_dir()}
        expected = {f"seed_{seed}" for seed in SEEDS}
        if seed_dirs != expected:
            raise ArtifactVerificationError(
                f"{method_key} seed directories mismatch: expected={sorted(expected)}, actual={sorted(seed_dirs)}"
            )
        for seed in SEEDS:
            seed_root = method_root / f"seed_{seed}"
            actual = {path.name for path in seed_root.iterdir() if path.is_file()}
            if actual != SEED_FILES:
                raise ArtifactVerificationError(
                    f"{method_key} seed {seed} payload mismatch: {sorted(actual)}"
                )
    execution_source = root / "code/execution_source"
    for required in (
        execution_source / "config.json",
        execution_source / "run.py",
        execution_source / "code/src/train_fetch_pick_and_place_lite_v0.py",
        execution_source / "code/src/metacog_lite_v0/weights.py",
        execution_source / "code/src/test_current_policy_legacy_parity.py",
        execution_source / "code/src/test_support_query_modes.py",
    ):
        if not required.is_file():
            raise ArtifactVerificationError(f"Frozen execution source incomplete: {required}")
    config = read_json(execution_source / "config.json")
    required_source = {
        "trainer_method": "b3_support_runtime_opt",
        "support_query_mode": "replay_action",
        "weight_scope": "critic_only",
        "weighting_mode": "batch_norm",
        "local_search_max": 512,
        "total_steps": TOTAL_STEPS,
        "seeds": list(SEEDS),
    }
    for key, expected in required_source.items():
        if config.get(key) != expected:
            raise ArtifactVerificationError(
                f"Frozen source config.{key} expected {expected!r}, got {config.get(key)!r}"
            )
    main_source = root / "code/main_source"
    for required in (
        main_source / "config.json",
        main_source / "run.py",
        main_source / "code/src/train_fetch_pick_and_place_lite_v0.py",
        main_source / "code/src/metacog_lite_v0/weights.py",
    ):
        if not required.is_file():
            raise ArtifactVerificationError(f"Frozen canonical Main source incomplete: {required}")
    main_config = read_json(main_source / "config.json")
    required_main = {
        "trainer_method": "b3_support_runtime_opt",
        "weight_scope": "critic_only",
        "weighting_mode": "batch_norm",
        "local_search_max": 512,
        "total_steps": TOTAL_STEPS,
        "seeds": list(SEEDS),
    }
    for key, expected in required_main.items():
        if main_config.get(key) != expected:
            raise ArtifactVerificationError(
                f"Frozen Main source config.{key} expected {expected!r}, got {main_config.get(key)!r}"
            )
    if main_config.get("support_query_mode", "current_policy") != "current_policy":
        raise ArtifactVerificationError("Frozen canonical Main query mode is not current_policy")
    readme = (root / "README.md").read_text(encoding="utf-8")
    limitation = (
        "Replay-action querying retains the sampled transition in the local replay pool "
        "and can therefore self-support."
    )
    if limitation not in readme:
        raise ArtifactVerificationError("README is missing the exact self-support limitation")
    provenance_caveat = (
        "historical Main run files do not capture an execution commit SHA"
    )
    if provenance_caveat not in readme:
        raise ArtifactVerificationError("README is missing the canonical Main provenance caveat")


def _verify_artifact_manifest(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "artifact_manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("artifact_id") != ARTIFACT_ID:
        raise ArtifactVerificationError("Artifact manifest identity/schema mismatch")
    source_commit = manifest.get("execution_source_commit")
    canonical_commit = manifest.get("canonical_main_source_commit")
    if not isinstance(source_commit, str) or not HEX40.fullmatch(source_commit):
        raise ArtifactVerificationError(f"Invalid execution source commit: {source_commit!r}")
    if not isinstance(canonical_commit, str) or not HEX40.fullmatch(canonical_commit):
        raise ArtifactVerificationError(f"Invalid canonical source commit: {canonical_commit!r}")
    if (root / "source_commit.txt").read_text(encoding="utf-8").strip() != source_commit:
        raise ArtifactVerificationError("source_commit.txt disagrees with artifact manifest")
    if manifest.get("method_seed_coverage") != {
        MAIN_KEY: list(SEEDS), REPLAY_KEY: list(SEEDS)
    }:
        raise ArtifactVerificationError("Manifest method/seed coverage mismatch")
    if manifest.get("evaluation") != {
        "steps": list(GRID),
        "checkpoint_count": len(GRID),
        "auc_normalization_denominator": GRID[-1] - GRID[0],
    }:
        raise ArtifactVerificationError("Manifest evaluation protocol mismatch")
    if manifest.get("bootstrap") != {
        "resamples": BOOTSTRAP_RESAMPLES, "rng_seed": BOOTSTRAP_SEED
    }:
        raise ArtifactVerificationError("Manifest bootstrap protocol mismatch")
    if manifest.get("exact_sign_flip_assignments") != 32:
        raise ArtifactVerificationError("Manifest exact sign-flip assignment count must be 32")
    if manifest.get("manifest_self_excluded") is not True or manifest.get(
        "sha256sums_self_excluded"
    ) is not True:
        raise ArtifactVerificationError("Manifest self-exclusion policy missing")

    actual = {
        name: path
        for name, path in all_files(root).items()
        if name not in {"artifact_manifest.json", "SHA256SUMS"}
    }
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ArtifactVerificationError("artifact_manifest.files must be a list")
    recorded: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ArtifactVerificationError(f"Invalid manifest entry: {entry!r}")
        name = str(entry["path"])
        if name in recorded:
            raise ArtifactVerificationError(f"Duplicate artifact manifest path: {name}")
        recorded[name] = dict(entry)
    if set(recorded) != set(actual):
        raise ArtifactVerificationError(
            f"Artifact manifest file set mismatch: missing={sorted(set(actual)-set(recorded))}, "
            f"stale={sorted(set(recorded)-set(actual))}"
        )
    for name, path in actual.items():
        entry = recorded[name]
        if entry.get("sha256") != sha256(path) or entry.get("size_bytes") != path.stat().st_size:
            raise ArtifactVerificationError(f"Artifact manifest hash/size mismatch: {name}")
    if manifest.get("file_count") != len(actual):
        raise ArtifactVerificationError("Artifact manifest file_count mismatch")
    if manifest.get("total_size_bytes") != sum(path.stat().st_size for path in actual.values()):
        raise ArtifactVerificationError("Artifact manifest total_size_bytes mismatch")
    return manifest


def _parse_sha256sums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ArtifactVerificationError(f"Malformed SHA256SUMS line {line_number}") from exc
        if not HEX64.fullmatch(digest) or not relative or relative in result:
            raise ArtifactVerificationError(f"Invalid SHA256SUMS line {line_number}: {line!r}")
        result[relative] = digest
    return result


def _verify_sha256sums(root: Path) -> None:
    actual = {name: path for name, path in all_files(root).items() if name != "SHA256SUMS"}
    recorded = _parse_sha256sums(root / "SHA256SUMS")
    if set(recorded) != set(actual):
        raise ArtifactVerificationError(
            f"SHA256SUMS file set mismatch: missing={sorted(set(actual)-set(recorded))}, "
            f"stale={sorted(set(recorded)-set(actual))}"
        )
    for name, path in actual.items():
        if recorded[name] != sha256(path):
            raise ArtifactVerificationError(f"SHA256SUMS mismatch: {name}")


def _path_value(payload: Any, expression: str) -> Any:
    if not expression.startswith("$"):
        raise ArtifactVerificationError(f"Invalid recorded JSON path {expression!r}")
    value = payload
    rest = expression[1:]
    while rest:
        if rest.startswith("/"):
            rest = rest[1:]
            stop = len(rest)
            for marker in ("/", "["):
                index = rest.find(marker)
                if index >= 0:
                    stop = min(stop, index)
            token, rest = rest[:stop], rest[stop:]
            token = token.replace("~1", "/").replace("~0", "~")
            if not isinstance(value, Mapping) or token not in value:
                raise ArtifactVerificationError(f"Recorded JSON path not found: {expression}")
            value = value[token]
        elif rest.startswith("["):
            end = rest.find("]")
            if end < 0:
                raise ArtifactVerificationError(f"Invalid array JSON path: {expression}")
            index = int(rest[1:end])
            rest = rest[end + 1 :]
            if not isinstance(value, list) or not 0 <= index < len(value):
                raise ArtifactVerificationError(f"Recorded array JSON path not found: {expression}")
            value = value[index]
        else:
            raise ArtifactVerificationError(f"Invalid recorded JSON path: {expression}")
    return value


def _verify_input_manifest(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = read_json(root / "input_manifest.json")
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": ARTIFACT_ID,
        "execution_source_commit": manifest["execution_source_commit"],
        "canonical_main_source_commit": manifest["canonical_main_source_commit"],
    }
    for key, value in expected_identity.items():
        if payload.get(key) != value:
            raise ArtifactVerificationError(f"input_manifest {key} mismatch")
    repository = payload.get("repository")
    if repository != {"id": "Hyunwoojooo/saber-rl", "branch": "main"}:
        raise ArtifactVerificationError(f"input_manifest repository mismatch: {repository!r}")
    methods = payload.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != {MAIN_KEY, REPLAY_KEY}:
        raise ArtifactVerificationError("input_manifest method set mismatch")
    for method_key, expected_mode in ((MAIN_KEY, "current_policy"), (REPLAY_KEY, "replay_action")):
        method = methods[method_key]
        if method.get("support_query_mode") != expected_mode or method.get("seeds") != list(SEEDS):
            raise ArtifactVerificationError(f"input_manifest {method_key} mode/seeds mismatch")
        runs = method.get("runs")
        if not isinstance(runs, list) or [run.get("seed") for run in runs] != list(SEEDS):
            raise ArtifactVerificationError(f"input_manifest {method_key} run coverage mismatch")
        for run in runs:
            files = run.get("files")
            if not isinstance(files, list) or {item.get("role") for item in files} != {
                "config", "env_snapshot", "summary", "metrics"
            }:
                raise ArtifactVerificationError(f"input_manifest {method_key} source roles mismatch")
            for item in files:
                packaged = item.get("packaged_path")
                if not isinstance(packaged, str) or not (root / packaged).is_file():
                    raise ArtifactVerificationError(f"Missing packaged input: {packaged!r}")
                if not HEX64.fullmatch(str(item.get("source_sha256", ""))):
                    raise ArtifactVerificationError(f"Invalid source hash record: {item!r}")
                if not isinstance(item.get("source_size_bytes"), int) or item["source_size_bytes"] <= 0:
                    raise ArtifactVerificationError(f"Invalid source size record: {item!r}")
                transform = item.get("transformations")
                if not isinstance(transform, Mapping):
                    raise ArtifactVerificationError(f"Missing transformation record: {item!r}")
                count = transform.get("nonfinite_json_numbers_replaced_with_null", 0)
                paths = transform.get("nonfinite_json_paths", [])
                if item.get("role") == "metrics":
                    if count not in (0, None) or paths not in ([], None):
                        raise ArtifactVerificationError("Metric record has invalid JSON nonfinite metadata")
                    continue
                if not isinstance(count, int) or not isinstance(paths, list) or count != len(paths):
                    raise ArtifactVerificationError(f"Invalid nonfinite path record: {item!r}")
                packaged_payload = read_json(root / packaged)
                for expression in paths:
                    if _path_value(packaged_payload, expression) is not None:
                        raise ArtifactVerificationError(
                            f"Sanitized nonfinite path is not null: {packaged}:{expression}"
                        )

    direct_groups = (
        ("frozen_execution_source_files", "code/execution_source"),
        ("frozen_main_reference_source_files", "code/main_source"),
        ("analysis_code_files", "code"),
        ("artifact_test_files", "tests"),
    )
    for key, prefix in direct_groups:
        records = payload.get(key)
        if not isinstance(records, list) or not records:
            raise ArtifactVerificationError(f"input_manifest {key} is empty")
        recorded_paths = set()
        for item in records:
            packaged = item.get("packaged_path") if isinstance(item, Mapping) else None
            if not isinstance(packaged, str) or not packaged.startswith(prefix + "/"):
                raise ArtifactVerificationError(f"Invalid {key} record: {item!r}")
            if packaged in recorded_paths:
                raise ArtifactVerificationError(f"Duplicate {key} packaged path: {packaged}")
            recorded_paths.add(packaged)
            path = root / packaged
            if not path.is_file() or item.get("source_sha256") != sha256(path):
                raise ArtifactVerificationError(f"Direct-copy source hash mismatch: {packaged}")
            if item.get("source_size_bytes") != path.stat().st_size:
                raise ArtifactVerificationError(f"Direct-copy source size mismatch: {packaged}")
            if key in {"frozen_execution_source_files", "frozen_main_reference_source_files"}:
                expected_commit = (
                    manifest["execution_source_commit"]
                    if key == "frozen_execution_source_files"
                    else manifest["canonical_main_source_commit"]
                )
                if item.get("git_commit") != expected_commit:
                    raise ArtifactVerificationError(f"Direct-copy Git commit mismatch: {packaged}")
                if item.get("git_mode") not in {"100644", "100755"} or not HEX40.fullmatch(
                    str(item.get("git_blob_oid", ""))
                ):
                    raise ArtifactVerificationError(f"Direct-copy Git blob metadata invalid: {packaged}")
                if not isinstance(item.get("git_source_relative_path"), str):
                    raise ArtifactVerificationError(f"Direct-copy Git path metadata invalid: {packaged}")
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in (root / prefix).rglob("*")
            if path.is_file()
        } if key in {"frozen_execution_source_files", "frozen_main_reference_source_files"} else {
            str(item["packaged_path"]) for item in records
        }
        if key in {"frozen_execution_source_files", "frozen_main_reference_source_files"} and recorded_paths != actual_paths:
            raise ArtifactVerificationError(f"{key} manifest file set mismatch")
    execution = payload.get("execution_metadata")
    if not isinstance(execution, Mapping) or execution.get("packaged_path") != "code/execution_metadata.json":
        raise ArtifactVerificationError("Execution metadata provenance record missing")
    source_git = payload.get("source_git_provenance")
    if source_git != {
        "packaged_path": "code/source_provenance.json",
        "replay_execution_commit_captured": True,
        "canonical_main_run_execution_commit_captured": False,
    }:
        raise ArtifactVerificationError("Source Git provenance record mismatch")
    excluded = payload.get("excluded")
    if not isinstance(excluded, Mapping) or not all(excluded.values()):
        raise ArtifactVerificationError("Input manifest exclusion record incomplete")
    return payload


def _verify_execution_metadata(
    root: Path,
    source_commit: str,
    source_provenance: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    payload = read_json(root / "code/execution_metadata.json")
    expected = {
        "experiment": "ddpg_saber_query_replay_ls512",
        "source_commit": source_commit,
        "origin_main": source_commit,
        "head_matches_origin_main": True,
        "worktree_clean": True,
        "source_tree_git_oid": source_provenance["replay_execution"]["code_src_tree_oid"],
        "config_git_blob_oid": source_provenance["replay_execution"]["config_blob_oid"],
        "steps": TOTAL_STEPS,
        "seeds": list(SEEDS),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ArtifactVerificationError(
                f"execution_metadata {key} expected {value!r}, got {payload.get(key)!r}"
            )
    if payload.get("support_query_mode") not in (None, "replay_action"):
        raise ArtifactVerificationError("Execution metadata query mode mismatch")
    return payload


def _verify_test_evidence(root: Path, *, allow_synthetic: bool) -> dict[str, Any]:
    payload = read_json(root / "tests/test_evidence.json")
    status = payload.get("status")
    if status == "synthetic-test-bypass" and allow_synthetic:
        return payload
    if status != "passed":
        raise ArtifactVerificationError(f"Prepackage test evidence status is not passed: {status!r}")
    commands = payload.get("commands")
    if not isinstance(commands, list):
        raise ArtifactVerificationError("Prepackage test evidence commands missing")
    by_name = {entry.get("test"): entry for entry in commands if isinstance(entry, Mapping)}
    for required in (
        "test_analysis.py",
        "test_artifact.py",
        "test_support_query_modes.py",
        "test_ddpg_query_loss_isolation.py",
        "test_current_policy_legacy_parity.py",
        "validate_config_parity.py",
    ):
        record = by_name.get(required)
        if not isinstance(record, Mapping) or record.get("returncode") != 0 or record.get("passed") is not True:
            raise ArtifactVerificationError(f"Required passing test evidence missing: {required}")
    return payload


def _verify_validation(
    root: Path,
    bundle: Any,
    manifest: Mapping[str, Any],
    execution: Mapping[str, Any],
    source_provenance: Mapping[str, Mapping[str, Any]],
    test_evidence: Mapping[str, Any],
) -> None:
    payload = read_json(root / "results/validation.json")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("artifact_id") != ARTIFACT_ID:
        raise ArtifactVerificationError("Validation identity/schema mismatch")
    if payload.get("status") != "passed":
        raise ArtifactVerificationError("Validation status is not passed")
    if payload.get("execution_source_commit") != manifest["execution_source_commit"]:
        raise ArtifactVerificationError("Validation execution source commit mismatch")
    if payload.get("canonical_main_source_commit") != manifest["canonical_main_source_commit"]:
        raise ArtifactVerificationError("Validation canonical source commit mismatch")
    limitation = (
        "Replay-action querying retains the sampled transition in the local replay pool "
        "and can therefore self-support."
    )
    if payload.get("self_support_limitation") != limitation:
        raise ArtifactVerificationError("Validation self-support limitation mismatch")
    gates = payload.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        raise ArtifactVerificationError("Validation gates missing")
    for name, gate in gates.items():
        if not isinstance(gate, Mapping) or gate.get("passed") is not True:
            raise ArtifactVerificationError(f"Validation gate not passed: {name}")
    if gates["seed_coverage"]["actual_by_method"] != {
        MAIN_KEY: list(SEEDS), REPLAY_KEY: list(SEEDS)
    }:
        raise ArtifactVerificationError("Validation seed coverage disagrees with recomputation")
    if gates["config_parity"]["only_allowed_resolved_differences"] != sorted(
        PAIR_CONFIG_EXEMPT_KEYS
    ):
        raise ArtifactVerificationError("Validation config parity allowlist mismatch")
    expected_differences = {
        str(seed): [
            key for key, _left, _right in config_differences(
                bundle.runs_by_method[MAIN_KEY][seed - 1].config,
                bundle.runs_by_method[REPLAY_KEY][seed - 1].config,
            )
        ]
        for seed in SEEDS
    }
    if gates["config_parity"]["actual_differences_by_seed"] != expected_differences:
        raise ArtifactVerificationError("Validation config differences disagree with recomputation")
    expected_matches = {
        str(run.seed): {
            "mean": run.diagnostics["query/query_bin_matches_replay_action_rate"].mean,
            "finite_checkpoint_count": run.diagnostics[
                "query/query_bin_matches_replay_action_rate"
            ].finite_count,
        }
        for run in bundle.runs_by_method[REPLAY_KEY]
    }
    if gates["replay_query_bin_matches_replay_action"]["actual_by_seed"] != expected_matches:
        raise ArtifactVerificationError("Validation replay-match diagnostics disagree with recomputation")
    if gates["execution_provenance"]["source_commit"] != execution["source_commit"]:
        raise ArtifactVerificationError("Validation execution provenance mismatch")
    if gates["execution_provenance"]["source_tree_git_oid"] != execution[
        "source_tree_git_oid"
    ] or gates["execution_provenance"]["config_git_blob_oid"] != execution[
        "config_git_blob_oid"
    ]:
        raise ArtifactVerificationError("Validation execution tree/blob provenance mismatch")
    canonical_gate = gates.get("canonical_main_reference_source_provenance")
    canonical = source_provenance["canonical_main_reference"]
    expected_canonical = {
        "reference_source_commit": manifest["canonical_main_source_commit"],
        "source_root_tree_oid": canonical["source_root_tree_oid"],
        "code_src_tree_oid": canonical["code_src_tree_oid"],
        "config_blob_oid": canonical["config_blob_oid"],
        "run_execution_commit_captured": False,
        "passed": True,
    }
    if canonical_gate != expected_canonical:
        raise ArtifactVerificationError("Validation canonical Main source provenance mismatch")
    if gates["current_policy_legacy_parity_evidence"]["test_suite_status"] != test_evidence["status"]:
        raise ArtifactVerificationError("Validation test evidence status mismatch")


def verify_artifact(root: Path, *, allow_synthetic_test_bypass: bool = False) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise ArtifactVerificationError(f"Artifact root does not exist: {root}")
    _assert_no_forbidden_payloads(root)
    _verify_layout(root)
    manifest = _verify_artifact_manifest(root)
    _verify_sha256sums(root)
    input_manifest = _verify_input_manifest(root, manifest)
    source_provenance = _verify_source_provenance(root, manifest)
    _verify_source_manifest_alignment(input_manifest, source_provenance)
    execution = _verify_execution_metadata(
        root, str(manifest["execution_source_commit"]), source_provenance
    )
    test_evidence = _verify_test_evidence(root, allow_synthetic=allow_synthetic_test_bypass)
    bundle = analyze_query_ablation(root / "data/main", root / "data/replay")
    verify_rendered_outputs(
        bundle, root / "results", extra_allowed_files={"validation.json"}
    )
    _verify_validation(
        root, bundle, manifest, execution, source_provenance, test_evidence
    )
    return {
        "artifact_id": ARTIFACT_ID,
        "execution_source_commit": manifest["execution_source_commit"],
        "canonical_main_source_commit": manifest["canonical_main_source_commit"],
        "seeds": list(SEEDS),
        "evaluation_grid_points": len(GRID),
        "config_parity": "verified",
        "query_replay_match_rate": "identically_1_at_all_exact_evaluations",
        "analysis_recomputation": "verified",
        "source_provenance": "commit_tree_blob_chains_verified",
        "canonical_main_run_execution_commit": "not_captured",
        "sha256_manifests": "verified",
        "forbidden_payloads_and_symlinks": "absent",
        "validation_gates": "verified",
    }


def default_artifact_root() -> Path:
    packaged = Path(__file__).resolve().parents[1]
    if (packaged / "artifact_manifest.json").is_file():
        return packaged
    current = Path(__file__).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "experiment_FPP").is_dir():
            return candidate / "experiment_FPP/paper_artifacts" / ARTIFACT_ID
    return Path.cwd()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=default_artifact_root())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = verify_artifact(args.artifact_root)
    except (AnalysisError, OSError, AssertionError) as exc:
        print(f"artifact verification failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
