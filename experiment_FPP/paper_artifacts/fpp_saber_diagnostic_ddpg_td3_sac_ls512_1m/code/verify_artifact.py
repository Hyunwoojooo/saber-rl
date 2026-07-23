#!/usr/bin/env python3
"""Fail-closed standalone verifier for the integrated FPP diagnostic artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


sys.dont_write_bytecode = True

SCHEMA_VERSION = 1
ARTIFACT_ID = "fpp_saber_diagnostic_ddpg_td3_sac_ls512_1m"
ALGORITHMS = ("ddpg", "td3", "sac")
SEEDS = (1, 2, 3, 4, 5)
DATA_FILES = (
    "config.json",
    "diagnostic_metrics.jsonl",
    "env_snapshot.json",
    "metrics_selected_10k.csv",
    "summary.json",
)
FILE_ROLES = {
    "config.json": "resolved_training_config",
    "diagnostic_metrics.jsonl": "primary_full_batch_diagnostic_input",
    "env_snapshot.json": "environment_snapshot",
    "metrics_selected_10k.csv": "exact_evaluation_grid",
    "summary.json": "completion_summary",
}
RESULT_FILES = (
    "aggregate_by_backbone.csv",
    "analysis_results.json",
    "config_summary.csv",
    "data_coverage.csv",
    "diagnostic_by_step.csv",
    "per_seed_diagnostics.csv",
    "residual_reduction.csv",
    "schema_compatibility.csv",
    "support_rate.csv",
    "td_error_enrichment.csv",
)
SOURCE_SNAPSHOT_FILES = {
    "ddpg": (
        "launcher.py",
        "launcher_config.json",
        "legacy_analyzer.py",
        "trainer.py",
    ),
    "td3": (
        "launcher.py",
        "launcher_config.json",
        "legacy_analyzer.py",
        "trainer.py",
    ),
    "sac": (
        "launcher.py",
        "launcher_config.json",
        "trainer.py",
    ),
}
ORIGINS: dict[str, dict[str, Any]] = {
    "ddpg": {
        "repository_id": "Hyunwoojooo/metacog-rl",
        "repository_url": "https://github.com/Hyunwoojooo/metacog-rl.git",
        "data_introduction_commit": "a30660506c2ca39ef1280f3fbbff2bd4ac97eaab",
        "execution_source_commit": None,
        "execution_source_status": "unverified_candidate_snapshot",
        "source_root": (
            "experiment_FPP/runs/fpp_ddpg_saber_diagnostic_ls512_1m"
        ),
    },
    "td3": {
        "repository_id": "Hyunwoojooo/metacog-rl",
        "repository_url": "https://github.com/Hyunwoojooo/metacog-rl.git",
        "data_introduction_commit": "e27438c1ea288fdb7bd6c75c49283d347fee77d9",
        "execution_source_commit": "e27438c1ea288fdb7bd6c75c49283d347fee77d9",
        "execution_source_status": "verified_from_execution_commit",
        "source_root": (
            "experiment_FPP/runs/fpp_td3_saber_diagnostic_ls512_1m"
        ),
    },
    "sac": {
        "repository_id": "Hyunwoojooo/saber-rl",
        "repository_url": "https://github.com/Hyunwoojooo/saber-rl.git",
        "data_introduction_commit": "04833bb8b7613e7e1b60e335d83ad21376e62eaa",
        "execution_source_commit": None,
        "execution_source_status": "unverified_candidate_snapshot",
        "source_root": (
            "experiment_FPP/runs/fpp_sac_saber_diagnostic_ls512_1m"
        ),
    },
}
ROOT_FILES = {
    "README.md",
    "METRICS.md",
    "artifact_manifest.json",
    "input_manifest.json",
    "validation.json",
}
ROOT_DIRS = {"code", "data", "provenance", "results"}
CODE_FILES = {"build_manifests.py", "rebuild_results.py", "verify_artifact.py"}
PROVENANCE_FILES = {"external_raw_samples.json", "source_manifest.json"}
FORBIDDEN_NAMES = {
    "best_checkpoint.pt",
    "checkpoint.pt",
    "diagnostic_samples.jsonl",
    "metrics.csv",
    "metrics.jsonl",
    "run.log",
}
FORBIDDEN_SUFFIXES = {".ckpt", ".nbc", ".nbi", ".pth", ".pt", ".pyc"}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class VerificationError(ValueError):
    """Raised on any integrity, provenance, or analysis-contract failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot load strict JSON {path}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_oid(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: object, context: str) -> Path:
    require(isinstance(value, str) and value != "", f"{context}: empty path")
    require("\\" not in value and "\x00" not in value, f"{context}: unsafe path")
    path = Path(value)
    require(not path.is_absolute(), f"{context}: absolute path is forbidden")
    require(path.as_posix() == value, f"{context}: path must be canonical POSIX")
    require(
        all(part not in {"", ".", ".."} for part in path.parts),
        f"{context}: path traversal is forbidden",
    )
    return path


def artifact_files(root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise VerificationError(f"symlink is forbidden: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            safe_relative_path(relative, "artifact path")
            output[relative] = path
        elif not path.is_dir():
            raise VerificationError(f"unsupported filesystem entry: {path}")
    return output


def exact_keys(payload: Mapping[str, Any], keys: set[str], context: str) -> None:
    require(
        set(payload) == keys,
        f"{context}: key mismatch; expected {sorted(keys)}, got {sorted(payload)}",
    )


def verify_hash_record(
    root: Path,
    record: Mapping[str, Any],
    context: str,
    *,
    expected_path: str | None = None,
) -> Path:
    for key in ("path", "size_bytes", "sha256", "git_blob_oid"):
        require(key in record, f"{context}: missing {key}")
    relative = record["path"]
    safe = safe_relative_path(relative, f"{context}.path")
    if expected_path is not None:
        require(relative == expected_path, f"{context}: path mismatch")
    path = root / safe
    require(path.is_file() and not path.is_symlink(), f"{context}: file missing")
    require(
        isinstance(record["size_bytes"], int)
        and not isinstance(record["size_bytes"], bool)
        and record["size_bytes"] == path.stat().st_size,
        f"{context}: size mismatch",
    )
    require(
        isinstance(record["sha256"], str)
        and HEX64.fullmatch(record["sha256"]) is not None
        and record["sha256"] == sha256(path),
        f"{context}: SHA256 mismatch",
    )
    require(
        isinstance(record["git_blob_oid"], str)
        and HEX40.fullmatch(record["git_blob_oid"]) is not None
        and record["git_blob_oid"] == git_blob_oid(path),
        f"{context}: Git blob ID mismatch",
    )
    return path


def verify_layout(root: Path, all_files: Mapping[str, Path]) -> None:
    actual_root_files = {path.name for path in root.iterdir() if path.is_file()}
    actual_root_dirs = {path.name for path in root.iterdir() if path.is_dir()}
    require(actual_root_files == ROOT_FILES, "root file set mismatch")
    require(actual_root_dirs == ROOT_DIRS, "root directory set mismatch")
    require(
        {path.name for path in (root / "code").iterdir() if path.is_file()}
        == CODE_FILES,
        "code file set mismatch",
    )
    require(
        {path.name for path in (root / "results").iterdir() if path.is_file()}
        == set(RESULT_FILES),
        "result file set mismatch",
    )
    provenance_root = root / "provenance"
    require(
        {path.name for path in provenance_root.iterdir() if path.is_file()}
        == PROVENANCE_FILES,
        "provenance file set mismatch",
    )
    require(
        {path.name for path in provenance_root.iterdir() if path.is_dir()}
        == {"source_snapshots"},
        "provenance directory set mismatch",
    )
    data_root = root / "data"
    require(
        {path.name for path in data_root.iterdir() if path.is_dir()}
        == set(ALGORITHMS),
        "algorithm directory coverage mismatch",
    )
    for algorithm in ALGORITHMS:
        algorithm_root = data_root / algorithm
        require(
            {path.name for path in algorithm_root.iterdir() if path.is_dir()}
            == {f"seed_{seed}" for seed in SEEDS},
            f"{algorithm}: seed directory coverage mismatch",
        )
        for seed in SEEDS:
            seed_root = algorithm_root / f"seed_{seed}"
            require(
                {path.name for path in seed_root.iterdir() if path.is_file()}
                == set(DATA_FILES),
                f"{algorithm} seed {seed}: data payload mismatch",
            )
            require(
                not any(path.is_dir() for path in seed_root.iterdir()),
                f"{algorithm} seed {seed}: nested directories forbidden",
            )
    snapshot_root = provenance_root / "source_snapshots"
    require(
        {path.name for path in snapshot_root.iterdir() if path.is_dir()}
        == set(ALGORITHMS),
        "source snapshot algorithm coverage mismatch",
    )
    for algorithm in ALGORITHMS:
        require(
            {
                path.name
                for path in (snapshot_root / algorithm).iterdir()
                if path.is_file()
            }
            == set(SOURCE_SNAPSHOT_FILES[algorithm]),
            f"{algorithm}: source snapshot payload mismatch",
        )
    for relative, path in all_files.items():
        require("__pycache__" not in path.parts, f"bytecode cache forbidden: {relative}")
        require(path.name not in FORBIDDEN_NAMES, f"forbidden payload: {relative}")
        require(path.suffix not in FORBIDDEN_SUFFIXES, f"forbidden payload: {relative}")


def verify_artifact_manifest(
    root: Path, all_files: Mapping[str, Path]
) -> dict[str, Any]:
    manifest = strict_json(root / "artifact_manifest.json")
    require(isinstance(manifest, dict), "artifact manifest must be an object")
    exact_keys(
        manifest,
        {
            "schema_version",
            "artifact_id",
            "manifest_scope",
            "self_excluded_path",
            "hash_algorithm",
            "git_object_algorithm",
            "file_count",
            "total_size_bytes",
            "files",
        },
        "artifact_manifest",
    )
    require(manifest["schema_version"] == SCHEMA_VERSION, "manifest schema mismatch")
    require(manifest["artifact_id"] == ARTIFACT_ID, "manifest identity mismatch")
    require(
        manifest["manifest_scope"] == "all_artifact_files_except_this_manifest",
        "manifest scope mismatch",
    )
    require(
        manifest["self_excluded_path"] == "artifact_manifest.json",
        "manifest self-exclusion mismatch",
    )
    require(manifest["hash_algorithm"] == "sha256", "hash algorithm mismatch")
    require(
        manifest["git_object_algorithm"] == "sha1_blob_oid",
        "Git object algorithm mismatch",
    )
    entries = manifest["files"]
    require(isinstance(entries, list), "manifest files must be a list")
    actual = {
        relative: path
        for relative, path in all_files.items()
        if relative != "artifact_manifest.json"
    }
    recorded: dict[str, Mapping[str, Any]] = {}
    ordered_paths: list[str] = []
    for index, entry in enumerate(entries):
        context = f"artifact_manifest.files[{index}]"
        require(isinstance(entry, dict), f"{context}: expected object")
        exact_keys(
            entry,
            {"path", "size_bytes", "sha256", "git_blob_oid"},
            context,
        )
        relative = entry["path"]
        safe_relative_path(relative, f"{context}.path")
        require(relative not in recorded, f"{context}: duplicate path")
        recorded[relative] = entry
        ordered_paths.append(relative)
    require(ordered_paths == sorted(ordered_paths), "manifest files not sorted")
    require(set(recorded) == set(actual), "manifest exact file set mismatch")
    for relative, path in actual.items():
        entry = recorded[relative]
        require(entry["size_bytes"] == path.stat().st_size, f"{relative}: size mismatch")
        require(entry["sha256"] == sha256(path), f"{relative}: SHA256 mismatch")
        require(
            entry["git_blob_oid"] == git_blob_oid(path),
            f"{relative}: Git blob ID mismatch",
        )
    require(manifest["file_count"] == len(actual), "manifest file count mismatch")
    require(
        manifest["total_size_bytes"]
        == sum(path.stat().st_size for path in actual.values()),
        "manifest total size mismatch",
    )
    return manifest


def load_run_names(root: Path) -> dict[tuple[str, int], str]:
    names: dict[tuple[str, int], str] = {}
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            path = root / "data" / algorithm / f"seed_{seed}" / "config.json"
            payload = strict_json(path)
            require(isinstance(payload, dict), f"{path}: expected object")
            name = payload.get("run_name")
            require(
                isinstance(name, str)
                and name
                and "/" not in name
                and "\\" not in name,
                f"{path}: invalid run_name",
            )
            names[(algorithm, seed)] = name
    return names


def verify_external_raw_samples(
    root: Path,
    descriptor: Mapping[str, Any],
    run_names: Mapping[tuple[str, int], str],
) -> None:
    context = "input_manifest.external_raw_samples"
    exact_keys(
        descriptor,
        {
            "path",
            "size_bytes",
            "sha256",
            "git_blob_oid",
            "record_count",
            "external_total_size_bytes",
            "role",
        },
        context,
    )
    path = verify_hash_record(
        root,
        descriptor,
        context,
        expected_path="provenance/external_raw_samples.json",
    )
    require(
        descriptor["role"] == "external_secondary_raw_sample_log_manifest",
        f"{context}: role mismatch",
    )
    payload = strict_json(path)
    require(isinstance(payload, dict), f"{path}: expected object")
    exact_keys(
        payload,
        {"schema_version", "record_count", "total_size_bytes", "records"},
        str(path),
    )
    require(payload["schema_version"] == 1, f"{path}: schema mismatch")
    records = payload["records"]
    require(isinstance(records, list) and len(records) == 15, f"{path}: record count")
    required_record_keys = {
        "algorithm",
        "seed",
        "source_repository_url",
        "data_introduction_commit",
        "source_path",
        "size_bytes",
        "sha256",
        "git_blob_oid",
        "inclusion_status",
        "reason",
    }
    coverage: list[tuple[str, int]] = []
    total_size = 0
    source_paths: set[str] = set()
    for index, record in enumerate(records):
        record_context = f"{path}.records[{index}]"
        require(isinstance(record, dict), f"{record_context}: expected object")
        exact_keys(record, required_record_keys, record_context)
        algorithm = record["algorithm"]
        seed = record["seed"]
        require(algorithm in ALGORITHMS, f"{record_context}: algorithm mismatch")
        require(
            isinstance(seed, int) and not isinstance(seed, bool) and seed in SEEDS,
            f"{record_context}: seed mismatch",
        )
        origin = ORIGINS[algorithm]
        require(
            record["source_repository_url"] == origin["repository_url"],
            f"{record_context}: repository mismatch",
        )
        require(
            record["data_introduction_commit"] == origin["data_introduction_commit"],
            f"{record_context}: data commit mismatch",
        )
        expected_source = (
            f"{origin['source_root']}/{run_names[(algorithm, seed)]}/"
            "diagnostic_samples.jsonl"
        )
        safe_relative_path(record["source_path"], f"{record_context}.source_path")
        require(
            record["source_path"] == expected_source,
            f"{record_context}: source path mismatch",
        )
        require(
            record["source_path"] not in source_paths,
            f"{record_context}: duplicate source path",
        )
        source_paths.add(record["source_path"])
        require(
            isinstance(record["size_bytes"], int)
            and not isinstance(record["size_bytes"], bool)
            and record["size_bytes"] > 0,
            f"{record_context}: invalid size",
        )
        require(
            isinstance(record["sha256"], str)
            and HEX64.fullmatch(record["sha256"]) is not None,
            f"{record_context}: invalid SHA256",
        )
        require(
            isinstance(record["git_blob_oid"], str)
            and HEX40.fullmatch(record["git_blob_oid"]) is not None,
            f"{record_context}: invalid Git blob ID",
        )
        require(
            record["inclusion_status"] == "excluded_secondary_raw_subset_log",
            f"{record_context}: inclusion status mismatch",
        )
        reason = record["reason"]
        require(
            isinstance(reason, str)
            and "128-of-256" in reason
            and "not a primary full-batch input" in reason,
            f"{record_context}: exclusion reason is incomplete",
        )
        coverage.append((algorithm, seed))
        total_size += record["size_bytes"]
    require(
        coverage
        == [(algorithm, seed) for algorithm in ALGORITHMS for seed in SEEDS],
        f"{path}: coverage/order mismatch",
    )
    require(payload["record_count"] == len(records), f"{path}: record_count mismatch")
    require(payload["total_size_bytes"] == total_size, f"{path}: size total mismatch")
    require(descriptor["record_count"] == len(records), f"{context}: count mismatch")
    require(
        descriptor["external_total_size_bytes"] == total_size,
        f"{context}: external size mismatch",
    )


def verify_input_manifest(root: Path) -> None:
    path = root / "input_manifest.json"
    payload = strict_json(path)
    require(isinstance(payload, dict), f"{path}: expected object")
    exact_keys(
        payload,
        {
            "schema_version",
            "artifact_id",
            "origins",
            "run_coverage",
            "included_data",
            "runs",
            "external_raw_samples",
            "analysis_input_contract",
        },
        str(path),
    )
    require(payload["schema_version"] == SCHEMA_VERSION, "input schema mismatch")
    require(payload["artifact_id"] == ARTIFACT_ID, "input identity mismatch")
    expected_origins = {
        algorithm: {
            "repository_id": ORIGINS[algorithm]["repository_id"],
            "repository_url": ORIGINS[algorithm]["repository_url"],
            "data_introduction_commit": ORIGINS[algorithm][
                "data_introduction_commit"
            ],
            "source_root": ORIGINS[algorithm]["source_root"],
        }
        for algorithm in ALGORITHMS
    }
    require(payload["origins"] == expected_origins, "input origin mapping mismatch")
    require(
        payload["run_coverage"]
        == {algorithm: list(SEEDS) for algorithm in ALGORITHMS},
        "input run coverage mismatch",
    )
    contract = payload["analysis_input_contract"]
    require(
        contract
        == {
            "headline_input": "diagnostic_metrics.jsonl",
            "headline_input_grain": "full_256_transition_minibatch",
            "secondary_sample_subset_included": False,
            "secondary_sample_subset_manifest": (
                "provenance/external_raw_samples.json"
            ),
        },
        "analysis input contract mismatch",
    )
    run_names = load_run_names(root)
    included = payload["included_data"]
    require(isinstance(included, dict), "included_data must be an object")
    exact_keys(included, {"file_count", "total_size_bytes", "files"}, "included_data")
    records = included["files"]
    require(isinstance(records, list), "included data files must be a list")
    require(len(records) == 75, "included data must contain 75 files")
    required_file_keys = {
        "path",
        "size_bytes",
        "sha256",
        "git_blob_oid",
        "algorithm",
        "seed",
        "role",
        "source_repository_url",
        "data_introduction_commit",
        "source_path",
    }
    expected_paths: list[str] = []
    total_size = 0
    for index, record in enumerate(records):
        context = f"included_data.files[{index}]"
        require(isinstance(record, dict), f"{context}: expected object")
        exact_keys(record, required_file_keys, context)
        algorithm = record["algorithm"]
        seed = record["seed"]
        require(algorithm in ALGORITHMS and seed in SEEDS, f"{context}: coverage")
        filename = Path(record["path"]).name
        require(filename in DATA_FILES, f"{context}: unexpected file")
        expected_path = f"data/{algorithm}/seed_{seed}/{filename}"
        verify_hash_record(root, record, context, expected_path=expected_path)
        origin = ORIGINS[algorithm]
        require(record["role"] == FILE_ROLES[filename], f"{context}: role mismatch")
        require(
            record["source_repository_url"] == origin["repository_url"],
            f"{context}: repository mismatch",
        )
        require(
            record["data_introduction_commit"] == origin["data_introduction_commit"],
            f"{context}: data commit mismatch",
        )
        expected_source = (
            f"{origin['source_root']}/{run_names[(algorithm, seed)]}/{filename}"
        )
        safe_relative_path(record["source_path"], f"{context}.source_path")
        require(record["source_path"] == expected_source, f"{context}: source path")
        expected_paths.append(expected_path)
        total_size += record["size_bytes"]
    canonical_order = [
        f"data/{algorithm}/seed_{seed}/{filename}"
        for algorithm in ALGORITHMS
        for seed in SEEDS
        for filename in DATA_FILES
    ]
    require(expected_paths == canonical_order, "included data record order mismatch")
    require(included["file_count"] == len(records), "included file count mismatch")
    require(included["total_size_bytes"] == total_size, "included size total mismatch")

    runs = payload["runs"]
    require(isinstance(runs, list) and len(runs) == 15, "run records mismatch")
    expected_run_keys = {
        "algorithm",
        "seed",
        "run_name",
        "packaged_root",
        "source_root",
        "files",
    }
    for index, run in enumerate(runs):
        context = f"runs[{index}]"
        require(isinstance(run, dict), f"{context}: expected object")
        exact_keys(run, expected_run_keys, context)
        algorithm = ALGORITHMS[index // len(SEEDS)]
        seed = SEEDS[index % len(SEEDS)]
        name = run_names[(algorithm, seed)]
        require(run["algorithm"] == algorithm and run["seed"] == seed, f"{context}: order")
        require(run["run_name"] == name, f"{context}: run_name mismatch")
        require(
            run["packaged_root"] == f"data/{algorithm}/seed_{seed}",
            f"{context}: packaged root mismatch",
        )
        require(
            run["source_root"] == f"{ORIGINS[algorithm]['source_root']}/{name}",
            f"{context}: source root mismatch",
        )
        require(
            isinstance(run["files"], list)
            and [record["path"] for record in run["files"]]
            == [
                f"data/{algorithm}/seed_{seed}/{filename}"
                for filename in DATA_FILES
            ],
            f"{context}: file references mismatch",
        )
    verify_external_raw_samples(root, payload["external_raw_samples"], run_names)


def verify_source_manifest(root: Path) -> None:
    path = root / "provenance" / "source_manifest.json"
    payload = strict_json(path)
    require(isinstance(payload, dict), f"{path}: expected object")
    exact_keys(
        payload,
        {
            "schema_version",
            "artifact_id",
            "provenance_status",
            "provenance_note",
            "algorithms",
            "snapshot_file_count",
            "snapshot_total_size_bytes",
        },
        str(path),
    )
    require(payload["schema_version"] == SCHEMA_VERSION, "source schema mismatch")
    require(payload["artifact_id"] == ARTIFACT_ID, "source identity mismatch")
    require(payload["provenance_status"] == "partial", "source status mismatch")
    note = payload["provenance_note"]
    require(
        isinstance(note, str)
        and "Data-introduction commits" in note
        and "Execution-source commits are a separate claim" in note
        and "DDPG and SAC" in note,
        "source provenance note incomplete",
    )
    algorithms = payload["algorithms"]
    require(
        isinstance(algorithms, dict) and set(algorithms) == set(ALGORITHMS),
        "source algorithm set mismatch",
    )
    all_records: list[Mapping[str, Any]] = []
    required_algorithm_keys = {
        "repository_id",
        "repository_url",
        "data_introduction_commit",
        "data_introduction_commit_role",
        "execution_source_commit",
        "execution_source_status",
        "execution_source_commit_verified",
        "snapshot_role",
        "snapshot_root",
        "snapshot_files",
    }
    for algorithm in ALGORITHMS:
        context = f"source_manifest.algorithms.{algorithm}"
        record = algorithms[algorithm]
        require(isinstance(record, dict), f"{context}: expected object")
        exact_keys(record, required_algorithm_keys, context)
        origin = ORIGINS[algorithm]
        expected_verified = origin["execution_source_commit"] is not None
        expected_role = (
            "verified_execution_source"
            if expected_verified
            else "unverified_candidate_execution_source"
        )
        expected_scalars = {
            "repository_id": origin["repository_id"],
            "repository_url": origin["repository_url"],
            "data_introduction_commit": origin["data_introduction_commit"],
            "data_introduction_commit_role": (
                "commit_that_introduced_the_packaged_run_data"
            ),
            "execution_source_commit": origin["execution_source_commit"],
            "execution_source_status": origin["execution_source_status"],
            "execution_source_commit_verified": expected_verified,
            "snapshot_role": expected_role,
            "snapshot_root": f"provenance/source_snapshots/{algorithm}",
        }
        for key, expected in expected_scalars.items():
            require(record[key] == expected, f"{context}.{key}: mismatch")
        snapshots = record["snapshot_files"]
        require(
            isinstance(snapshots, list)
            and len(snapshots) == len(SOURCE_SNAPSHOT_FILES[algorithm]),
            f"{context}: snapshot count mismatch",
        )
        for index, snapshot in enumerate(snapshots):
            snapshot_context = f"{context}.snapshot_files[{index}]"
            require(isinstance(snapshot, dict), f"{snapshot_context}: expected object")
            exact_keys(
                snapshot,
                {
                    "path",
                    "size_bytes",
                    "sha256",
                    "git_blob_oid",
                    "algorithm",
                    "role",
                },
                snapshot_context,
            )
            filename = SOURCE_SNAPSHOT_FILES[algorithm][index]
            verify_hash_record(
                root,
                snapshot,
                snapshot_context,
                expected_path=f"provenance/source_snapshots/{algorithm}/{filename}",
            )
            require(snapshot["algorithm"] == algorithm, f"{snapshot_context}: algorithm")
            require(snapshot["role"] == "source_snapshot", f"{snapshot_context}: role")
            all_records.append(snapshot)
    require(
        payload["snapshot_file_count"] == len(all_records),
        "source snapshot file count mismatch",
    )
    require(
        payload["snapshot_total_size_bytes"]
        == sum(record["size_bytes"] for record in all_records),
        "source snapshot size total mismatch",
    )


def require_any(text: str, alternatives: Iterable[str], context: str) -> None:
    require(
        any(alternative in text for alternative in alternatives),
        f"{context}: missing required caveat ({' | '.join(alternatives)})",
    )


def verify_documentation(root: Path) -> None:
    readme = (root / "README.md").read_text(encoding="utf-8").casefold()
    metrics = (root / "METRICS.md").read_text(encoding="utf-8").casefold()
    for alternatives, label in (
        (("same-minibatch", "same minibatch"), "same-minibatch"),
        (("in-sample", "in sample"), "in-sample"),
        (("not held-out", "not a held-out", "not held out"), "not-held-out"),
        (
            ("not causal", "does not establish causality", "not causal proof"),
            "not-causal",
        ),
        (
            (
                "cross-backbone pooling is forbidden",
                "not pooled across backbones",
                "no cross-backbone grand mean",
            ),
            "cross-backbone pooling",
        ),
        (("diagnostic_metrics.jsonl",), "primary diagnostic log"),
        (("full 256", "full-batch 256", "256-transition"), "full batch"),
        (("diagnostic_samples.jsonl",), "external sample log"),
        (("128-of-256", "128/256"), "secondary subset size"),
        (
            (
                "ddpg and sac execution-source commits are unverified",
                "ddpg/sac execution-source provenance is unverified",
            ),
            "partial execution provenance",
        ),
        (
            (
                "td3 execution-source commit is verified",
                "td3 execution source is verified",
            ),
            "TD3 execution provenance",
        ),
    ):
        require_any(readme, alternatives, f"README {label}")
    for alternatives, label in (
        (
            ("diagnostic-window support", "diagnostic window support"),
            "diagnostic support definition",
        ),
        (
            ("final-checkpoint support", "final checkpoint support"),
            "final support definition",
        ),
        (("top-k", "top k"), "top-k enrichment"),
        (("enrichment",), "enrichment"),
        (
            ("|δ_before| - |δ_after|", "|delta_before| - |delta_after|"),
            "residual formula",
        ),
        (("seed-first", "seed first"), "seed-first aggregation"),
        (
            ("sample standard deviation", "sample sd", "ddof=1", "ddof 1"),
            "sample SD",
        ),
        (
            (
                "no cross-backbone grand mean",
                "not pooled across backbones",
                "cross-backbone pooling is forbidden",
            ),
            "cross-backbone aggregation",
        ),
    ):
        require_any(metrics, alternatives, f"METRICS {label}")


def verify_validation(root: Path) -> None:
    payload = strict_json(root / "validation.json")
    require(isinstance(payload, dict), "validation must be an object")
    expected = {
        "schema_version": 1,
        "overall_status": "validated_with_partial_execution_source_provenance",
        "data_analysis_status": "passed",
        "execution_source_provenance_status": "partial",
        "expected_algorithms": list(ALGORITHMS),
        "expected_seeds": list(SEEDS),
        "run_count": 15,
        "diagnostic_batch_count": 1500,
        "full_batch_observation_count": 384000,
        "invariant_failure_count": 0,
        "undefined_primary_metric_count": 0,
        "cross_backbone_pooling": "forbidden",
    }
    for key, value in expected.items():
        require(payload.get(key) == value, f"validation.{key}: mismatch")
    contract = payload.get("completion_contract")
    require(isinstance(contract, dict), "validation completion contract missing")
    require(contract.get("total_steps") == 1_000_000, "total_steps contract mismatch")
    require(contract.get("total_updates") == 995_001, "updates contract mismatch")
    require(contract.get("diagnostic_batch_size") == 256, "batch contract mismatch")
    require(
        contract.get("logged_sample_subset_size") == 128,
        "sample subset contract mismatch",
    )
    require(
        contract.get("diagnostic_steps")
        == list(range(10_000, 1_000_001, 10_000)),
        "diagnostic grid contract mismatch",
    )


def compare_bytes(expected: Path, actual: Path, context: str) -> None:
    require(actual.is_file(), f"{context}: regenerated file missing")
    require(
        expected.read_bytes() == actual.read_bytes(),
        f"{context}: stored output is not byte-identical to regeneration",
    )


def verify_rebuild(root: Path) -> dict[str, Any]:
    script = root / "code" / "rebuild_results.py"
    require(script.is_file() and not script.is_symlink(), "rebuild script missing")
    with tempfile.TemporaryDirectory(prefix="fpp-diagnostic-verify-") as temp_name:
        temp = Path(temp_name)
        generated_results = temp / "results"
        generated_validation = temp / "validation.json"
        command = [
            sys.executable,
            "-B",
            str(script),
            "--artifact-root",
            str(root),
            "--output-dir",
            str(generated_results),
            "--validation-output",
            str(generated_validation),
        ]
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise VerificationError(f"cannot run deterministic rebuild: {exc}") from exc
        require(
            completed.returncode == 0,
            "deterministic rebuild failed:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        generated_names = {
            path.name for path in generated_results.iterdir() if path.is_file()
        }
        require(generated_names == set(RESULT_FILES), "regenerated result set mismatch")
        for filename in RESULT_FILES:
            compare_bytes(
                root / "results" / filename,
                generated_results / filename,
                f"results/{filename}",
            )
        compare_bytes(
            root / "validation.json",
            generated_validation,
            "validation.json",
        )
        try:
            rebuild_report = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise VerificationError("rebuild did not emit its JSON report") from exc
        require(
            rebuild_report.get("status") == "passed"
            and rebuild_report.get("run_count") == 15
            and rebuild_report.get("diagnostic_batch_count") == 1500,
            "rebuild report contract mismatch",
        )
        return rebuild_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the integrated FPP DDPG/TD3/SAC diagnostic artifact."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    root_arg = args.artifact_root.absolute()
    require(not root_arg.is_symlink(), f"artifact root may not be a symlink: {root_arg}")
    root = root_arg.resolve()
    require(root.is_dir(), f"artifact root does not exist: {root}")
    all_files = artifact_files(root)
    verify_layout(root, all_files)
    manifest = verify_artifact_manifest(root, all_files)
    verify_input_manifest(root)
    verify_source_manifest(root)
    verify_documentation(root)
    verify_validation(root)
    rebuild = verify_rebuild(root)
    print(
        json.dumps(
            {
                "artifact_id": ARTIFACT_ID,
                "diagnostic_batch_count": rebuild["diagnostic_batch_count"],
                "file_count": manifest["file_count"],
                "run_count": rebuild["run_count"],
                "status": "verified",
                "total_size_bytes": manifest["total_size_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
