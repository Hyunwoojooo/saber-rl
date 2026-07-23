#!/usr/bin/env python3
"""Build deterministic provenance and integrity manifests for this artifact.

This script intentionally has no Git or network dependency.  Git blob object
IDs are computed directly from the packaged bytes, while repository/commit
provenance is frozen in the constants below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
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
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """Raised when the artifact cannot be safely manifested."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot load strict JSON {path}: {exc}") from exc


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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


def safe_relative_path(value: str, context: str) -> Path:
    require(isinstance(value, str) and value != "", f"{context}: empty path")
    require("\\" not in value and "\x00" not in value, f"{context}: unsafe path")
    path = Path(value)
    require(not path.is_absolute(), f"{context}: absolute path is forbidden")
    require(path.as_posix() == value, f"{context}: path must be canonical POSIX")
    require(
        all(part not in {"", ".", ".."} for part in path.parts),
        f"{context}: traversal is forbidden",
    )
    return path


def file_record(
    artifact_root: Path,
    path: Path,
    *,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    require(path.is_file() and not path.is_symlink(), f"invalid file: {path}")
    relative = path.relative_to(artifact_root).as_posix()
    safe_relative_path(relative, "packaged path")
    record: dict[str, object] = {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "git_blob_oid": git_blob_oid(path),
    }
    if extra:
        record.update(extra)
    return record


def iter_artifact_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ManifestError(f"symlink is forbidden: {path}")
        if path.is_file():
            yield path
        elif not path.is_dir():
            raise ManifestError(f"unsupported filesystem entry: {path}")


def run_name(config_path: Path) -> str:
    payload = strict_json(config_path)
    require(isinstance(payload, dict), f"{config_path}: expected object")
    value = payload.get("run_name")
    require(isinstance(value, str) and value, f"{config_path}: missing run_name")
    require("/" not in value and "\\" not in value, f"{config_path}: unsafe run_name")
    return value


def validate_origins() -> None:
    for algorithm, origin in ORIGINS.items():
        require(
            HEX40.fullmatch(origin["data_introduction_commit"]) is not None,
            f"{algorithm}: invalid data-introduction commit",
        )
        execution = origin["execution_source_commit"]
        require(
            execution is None or HEX40.fullmatch(execution) is not None,
            f"{algorithm}: invalid execution-source commit",
        )
        safe_relative_path(origin["source_root"], f"{algorithm}.source_root")


def validate_external_raw_samples(root: Path) -> tuple[dict[str, Any], dict[str, object]]:
    path = root / "provenance" / "external_raw_samples.json"
    payload = strict_json(path)
    require(isinstance(payload, dict), f"{path}: expected object")
    records = payload.get("records")
    require(isinstance(records, list), f"{path}: records must be a list")
    require(payload.get("schema_version") == 1, f"{path}: schema mismatch")
    require(payload.get("record_count") == 15, f"{path}: record count mismatch")
    require(len(records) == 15, f"{path}: expected 15 records")
    coverage: list[tuple[str, int]] = []
    size_sum = 0
    for index, record in enumerate(records):
        context = f"{path}.records[{index}]"
        require(isinstance(record, dict), f"{context}: expected object")
        algorithm = record.get("algorithm")
        seed = record.get("seed")
        require(algorithm in ALGORITHMS, f"{context}: invalid algorithm")
        require(
            isinstance(seed, int) and not isinstance(seed, bool) and seed in SEEDS,
            f"{context}: invalid seed",
        )
        origin = ORIGINS[algorithm]
        require(
            record.get("source_repository_url") == origin["repository_url"],
            f"{context}: repository mismatch",
        )
        require(
            record.get("data_introduction_commit")
            == origin["data_introduction_commit"],
            f"{context}: data-introduction commit mismatch",
        )
        require(
            record.get("inclusion_status")
            == "excluded_secondary_raw_subset_log",
            f"{context}: inclusion status mismatch",
        )
        require(
            isinstance(record.get("reason"), str) and record["reason"],
            f"{context}: exclusion reason missing",
        )
        safe_relative_path(record.get("source_path"), f"{context}.source_path")
        require(
            record["source_path"].endswith("/diagnostic_samples.jsonl"),
            f"{context}: not a diagnostic sample log",
        )
        require(
            isinstance(record.get("size_bytes"), int)
            and not isinstance(record["size_bytes"], bool)
            and record["size_bytes"] > 0,
            f"{context}: invalid size",
        )
        require(
            isinstance(record.get("sha256"), str)
            and HEX64.fullmatch(record["sha256"]) is not None,
            f"{context}: invalid SHA256",
        )
        require(
            isinstance(record.get("git_blob_oid"), str)
            and HEX40.fullmatch(record["git_blob_oid"]) is not None,
            f"{context}: invalid Git blob ID",
        )
        coverage.append((algorithm, seed))
        size_sum += record["size_bytes"]
    require(
        coverage == [
            (algorithm, seed) for algorithm in ALGORITHMS for seed in SEEDS
        ],
        f"{path}: records must be ordered by algorithm and seed",
    )
    require(payload.get("total_size_bytes") == size_sum, f"{path}: size total mismatch")
    descriptor = file_record(
        root,
        path,
        extra={
            "record_count": len(records),
            "external_total_size_bytes": size_sum,
            "role": "external_secondary_raw_sample_log_manifest",
        },
    )
    return payload, descriptor


def build_input_manifest(root: Path) -> dict[str, object]:
    included: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    for algorithm in ALGORITHMS:
        origin = ORIGINS[algorithm]
        for seed in SEEDS:
            run_dir = root / "data" / algorithm / f"seed_{seed}"
            require(run_dir.is_dir() and not run_dir.is_symlink(), f"missing {run_dir}")
            actual = sorted(path.name for path in run_dir.iterdir() if path.is_file())
            require(actual == sorted(DATA_FILES), f"{run_dir}: payload mismatch {actual}")
            name = run_name(run_dir / "config.json")
            run_records: list[dict[str, object]] = []
            for filename in DATA_FILES:
                packaged = run_dir / filename
                source_path = f"{origin['source_root']}/{name}/{filename}"
                record = file_record(
                    root,
                    packaged,
                    extra={
                        "algorithm": algorithm,
                        "seed": seed,
                        "role": FILE_ROLES[filename],
                        "source_repository_url": origin["repository_url"],
                        "data_introduction_commit": origin[
                            "data_introduction_commit"
                        ],
                        "source_path": source_path,
                    },
                )
                run_records.append(record)
                included.append(record)
            runs.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "run_name": name,
                    "packaged_root": f"data/{algorithm}/seed_{seed}",
                    "source_root": f"{origin['source_root']}/{name}",
                    "files": run_records,
                }
            )
    _external_payload, external_descriptor = validate_external_raw_samples(root)
    origins = {
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
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": ARTIFACT_ID,
        "origins": origins,
        "run_coverage": {
            algorithm: list(SEEDS) for algorithm in ALGORITHMS
        },
        "included_data": {
            "file_count": len(included),
            "total_size_bytes": sum(record["size_bytes"] for record in included),
            "files": included,
        },
        "runs": runs,
        "external_raw_samples": external_descriptor,
        "analysis_input_contract": {
            "headline_input": "diagnostic_metrics.jsonl",
            "headline_input_grain": "full_256_transition_minibatch",
            "secondary_sample_subset_included": False,
            "secondary_sample_subset_manifest": (
                "provenance/external_raw_samples.json"
            ),
        },
    }


def build_source_manifest(root: Path) -> dict[str, object]:
    algorithms: dict[str, object] = {}
    all_records: list[dict[str, object]] = []
    for algorithm in ALGORITHMS:
        snapshot_root = root / "provenance" / "source_snapshots" / algorithm
        require(
            snapshot_root.is_dir() and not snapshot_root.is_symlink(),
            f"missing source snapshot {snapshot_root}",
        )
        actual = sorted(path.name for path in snapshot_root.iterdir() if path.is_file())
        expected = sorted(SOURCE_SNAPSHOT_FILES[algorithm])
        require(actual == expected, f"{snapshot_root}: expected {expected}, got {actual}")
        records = [
            file_record(
                root,
                snapshot_root / filename,
                extra={"algorithm": algorithm, "role": "source_snapshot"},
            )
            for filename in SOURCE_SNAPSHOT_FILES[algorithm]
        ]
        all_records.extend(records)
        origin = ORIGINS[algorithm]
        verified = origin["execution_source_commit"] is not None
        algorithms[algorithm] = {
            "repository_id": origin["repository_id"],
            "repository_url": origin["repository_url"],
            "data_introduction_commit": origin["data_introduction_commit"],
            "data_introduction_commit_role": (
                "commit_that_introduced_the_packaged_run_data"
            ),
            "execution_source_commit": origin["execution_source_commit"],
            "execution_source_status": origin["execution_source_status"],
            "execution_source_commit_verified": verified,
            "snapshot_role": (
                "verified_execution_source"
                if verified
                else "unverified_candidate_execution_source"
            ),
            "snapshot_root": f"provenance/source_snapshots/{algorithm}",
            "snapshot_files": records,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": ARTIFACT_ID,
        "provenance_status": "partial",
        "provenance_note": (
            "Data-introduction commits identify immutable input blobs. "
            "Execution-source commits are a separate claim: TD3 is verified; "
            "DDPG and SAC retain hashed candidate snapshots but have no verified "
            "execution-source commit."
        ),
        "algorithms": algorithms,
        "snapshot_file_count": len(all_records),
        "snapshot_total_size_bytes": sum(
            record["size_bytes"] for record in all_records
        ),
    }


def build_artifact_manifest(root: Path) -> dict[str, object]:
    records = [
        file_record(root, path)
        for path in iter_artifact_files(root)
        if path.relative_to(root).as_posix() != "artifact_manifest.json"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": ARTIFACT_ID,
        "manifest_scope": "all_artifact_files_except_this_manifest",
        "self_excluded_path": "artifact_manifest.json",
        "hash_algorithm": "sha256",
        "git_object_algorithm": "sha1_blob_oid",
        "file_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic manifests for the integrated FPP artifact."
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
    validate_origins()

    input_manifest = build_input_manifest(root)
    write_json(root / "input_manifest.json", input_manifest)
    source_manifest = build_source_manifest(root)
    write_json(root / "provenance" / "source_manifest.json", source_manifest)
    artifact_manifest = build_artifact_manifest(root)
    write_json(root / "artifact_manifest.json", artifact_manifest)
    print(
        json.dumps(
            {
                "artifact_id": ARTIFACT_ID,
                "file_count": artifact_manifest["file_count"],
                "status": "manifests_built",
                "total_size_bytes": artifact_manifest["total_size_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ManifestError as exc:
        print(f"manifest build failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
