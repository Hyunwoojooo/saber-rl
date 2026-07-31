#!/usr/bin/env python3
"""Verify the compact FPP SABER actor-critic paper artifact, fail closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

# The verifier is shipped inside the checksummed artifact. Prevent importing its
# sibling analysis module from mutating that artifact with ``__pycache__`` files.
sys.dont_write_bytecode = True

from scope_ablation import (
    ACTOR_KEY,
    ARTIFACT_ID,
    CRITIC_KEY,
    GRID,
    PAIR_CONFIG_EXEMPT_KEYS,
    SEEDS,
    TOTAL_STEPS,
    TOTAL_UPDATES,
    AnalysisError,
    analyze_scope_ablation,
    find_repo_root,
    read_json,
    verify_rendered_outputs,
)


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SEED_REQUIRED_FILES = frozenset(
    {"config.json", "summary.json", "env_snapshot.json", "metrics_selected_10k.csv"}
)
ROOT_REQUIRED_FILES = frozenset(
    {
        "README.md",
        "execution_metadata.json",
        "source_commit.txt",
        "source_inputs_manifest.json",
        "validation.json",
        "artifact_manifest.json",
        "SHA256SUMS",
    }
)
CODE_REQUIRED_FILES = frozenset(
    {"scope_ablation.py", "analyze_results.py", "build_artifact.py", "verify_artifact.py"}
)
ANALYSIS_REQUIRED_FILES = frozenset(
    {
        "per_seed_results.csv",
        "aggregate_results.csv",
        "paired_results.csv",
        "paired_statistics.csv",
        "analysis_results.json",
        "analysis_summary.md",
        "mean_learning_curve.csv",
    }
)
FORBIDDEN_EXACT_NAMES = frozenset(
    {
        "metrics.csv",
        "metrics.jsonl",
        "run.log",
        "checkpoint.pt",
        "best_checkpoint.pt",
    }
)
CANONICAL_CRITIC_SOURCE_COMMIT = "c43d40502b84ae713924ebf995be623e7acc7ca8"


class ArtifactVerificationError(AnalysisError):
    """Raised when a packaged artifact is incomplete or has been modified."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _assert_no_forbidden_payloads(root: Path) -> None:
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ArtifactVerificationError(f"Symlinks are prohibited in the artifact: {relative}")
        lowered_parts = [part.lower() for part in relative.parts]
        if relative.parts and relative.parts[0] == "data" and any(
            "smoke" in part for part in lowered_parts
        ):
            raise ArtifactVerificationError(f"Smoke payload/path is prohibited: {relative}")
        if path.is_file() and (
            path.name in FORBIDDEN_EXACT_NAMES or path.suffix.lower() in {".pt", ".pth", ".ckpt"}
        ):
            raise ArtifactVerificationError(f"Raw log/checkpoint payload is prohibited: {relative}")


def _verify_layout(root: Path) -> None:
    actual_root_files = {path.name for path in root.iterdir() if path.is_file()}
    actual_root_dirs = {path.name for path in root.iterdir() if path.is_dir()}
    if actual_root_files != ROOT_REQUIRED_FILES:
        raise ArtifactVerificationError(
            f"Root file-set mismatch: expected={sorted(ROOT_REQUIRED_FILES)}, "
            f"actual={sorted(actual_root_files)}"
        )
    if actual_root_dirs != {"analysis", "code", "data", "source"}:
        raise ArtifactVerificationError(
            f"Root directory-set mismatch: expected=['analysis', 'code', 'data', 'source'], "
            f"actual={sorted(actual_root_dirs)}"
        )
    actual_code = {path.name for path in (root / "code").iterdir() if path.is_file()}
    if actual_code != CODE_REQUIRED_FILES:
        raise ArtifactVerificationError(
            f"Code file-set mismatch: expected={sorted(CODE_REQUIRED_FILES)}, actual={sorted(actual_code)}"
        )
    actual_analysis = {path.name for path in (root / "analysis").iterdir() if path.is_file()}
    if actual_analysis != ANALYSIS_REQUIRED_FILES:
        raise ArtifactVerificationError(
            "Analysis file-set mismatch: "
            f"expected={sorted(ANALYSIS_REQUIRED_FILES)}, actual={sorted(actual_analysis)}"
        )
    data_dirs = {path.name for path in (root / "data").iterdir() if path.is_dir()}
    if data_dirs != {ACTOR_KEY, CRITIC_KEY}:
        raise ArtifactVerificationError(
            f"Data arm mismatch: expected={[ACTOR_KEY, CRITIC_KEY]}, actual={sorted(data_dirs)}"
        )
    for arm in (ACTOR_KEY, CRITIC_KEY):
        arm_root = root / "data" / arm
        expected_seed_dirs = {f"seed_{seed}" for seed in SEEDS}
        actual_seed_dirs = {path.name for path in arm_root.iterdir() if path.is_dir()}
        if actual_seed_dirs != expected_seed_dirs:
            raise ArtifactVerificationError(
                f"{arm} seed directories mismatch: expected={sorted(expected_seed_dirs)}, "
                f"actual={sorted(actual_seed_dirs)}"
            )
        for seed in SEEDS:
            seed_root = arm_root / f"seed_{seed}"
            actual = {path.name for path in seed_root.iterdir() if path.is_file()}
            if actual != SEED_REQUIRED_FILES:
                raise ArtifactVerificationError(
                    f"{arm} seed {seed} file-set mismatch: expected={sorted(SEED_REQUIRED_FILES)}, "
                    f"actual={sorted(actual)}"
                )
    source_root = root / "source"
    for required in (
        source_root / "config.json",
        source_root / "run.py",
        source_root / "code/src/train_fetch_pick_and_place_lite_v0.py",
    ):
        if not required.is_file():
            raise ArtifactVerificationError(f"Frozen executable source is incomplete: {required}")
    source_config = read_json(source_root / "config.json")
    expected_source = {
        "trainer_method": "b3_support_runtime_opt",
        "weight_scope": "actor_critic",
        "weighting_mode": "batch_norm",
        "local_search_max": 512,
        "total_steps": 1_000_000,
        "seeds": list(SEEDS),
    }
    for key, expected in expected_source.items():
        if source_config.get(key) != expected:
            raise ArtifactVerificationError(
                f"Frozen source config.{key} expected {expected!r}, got {source_config.get(key)!r}"
            )


def _verify_artifact_manifest(root: Path) -> dict[str, Any]:
    payload = read_json(root / "artifact_manifest.json")
    if payload.get("artifact_id") != ARTIFACT_ID or payload.get("schema_version") != 1:
        raise ArtifactVerificationError("artifact_manifest.json identity/schema mismatch")
    if payload.get("canonical_critic_source_commit") != CANONICAL_CRITIC_SOURCE_COMMIT:
        raise ArtifactVerificationError("Canonical critic source commit mismatch")
    if payload.get("manifest_self_excluded") is not True or payload.get("sha256sums_self_excluded") is not True:
        raise ArtifactVerificationError("Manifest exclusion policy is missing")
    source_commit = str(payload.get("source_commit", ""))
    if not HEX40.fullmatch(source_commit):
        raise ArtifactVerificationError(f"Invalid full source commit: {source_commit!r}")
    recorded_commit = (root / "source_commit.txt").read_text(encoding="utf-8").strip()
    if recorded_commit != source_commit:
        raise ArtifactVerificationError("source_commit.txt disagrees with artifact_manifest.json")
    if payload.get("seeds") != list(SEEDS):
        raise ArtifactVerificationError(f"Manifest seeds must be {list(SEEDS)}")
    grid = payload.get("evaluation_grid")
    if grid != {
        "start": GRID[0],
        "end": GRID[-1],
        "frequency": 10_000,
        "points": len(GRID),
    }:
        raise ArtifactVerificationError(f"Manifest evaluation grid mismatch: {grid!r}")
    if set(payload.get("required_seed_files", [])) != SEED_REQUIRED_FILES:
        raise ArtifactVerificationError("Manifest required_seed_files mismatch")

    excluded = {"artifact_manifest.json", "SHA256SUMS"}
    actual = {name: path for name, path in _files(root).items() if name not in excluded}
    entries = payload.get("files")
    if not isinstance(entries, list):
        raise ArtifactVerificationError("artifact_manifest.files must be a list")
    by_name: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ArtifactVerificationError(f"Invalid artifact manifest entry: {entry!r}")
        name = entry["path"]
        if name in by_name:
            raise ArtifactVerificationError(f"Duplicate artifact manifest entry: {name}")
        by_name[name] = entry
    if set(by_name) != set(actual):
        raise ArtifactVerificationError(
            "Artifact manifest file-set mismatch: "
            f"missing={sorted(set(actual) - set(by_name))}, stale={sorted(set(by_name) - set(actual))}"
        )
    for name, path in actual.items():
        entry = by_name[name]
        if entry.get("sha256") != sha256(path):
            raise ArtifactVerificationError(f"SHA256 mismatch for {name}")
        if entry.get("size_bytes") != path.stat().st_size:
            raise ArtifactVerificationError(f"Size mismatch for {name}")
    if payload.get("file_count") != len(actual):
        raise ArtifactVerificationError("artifact_manifest.file_count mismatch")
    if payload.get("total_size_bytes") != sum(path.stat().st_size for path in actual.values()):
        raise ArtifactVerificationError("artifact_manifest.total_size_bytes mismatch")
    return payload


def _parse_sha256sums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ArtifactVerificationError(f"Malformed SHA256SUMS line {line_number}") from exc
        if not HEX64.fullmatch(digest) or not relative or relative in entries:
            raise ArtifactVerificationError(f"Invalid SHA256SUMS line {line_number}: {line!r}")
        entries[relative] = digest
    return entries


def _verify_sha256sums(root: Path) -> None:
    actual = {name: path for name, path in _files(root).items() if name != "SHA256SUMS"}
    recorded = _parse_sha256sums(root / "SHA256SUMS")
    if set(recorded) != set(actual):
        raise ArtifactVerificationError(
            f"SHA256SUMS file-set mismatch: missing={sorted(set(actual) - set(recorded))}, "
            f"stale={sorted(set(recorded) - set(actual))}"
        )
    for name, path in actual.items():
        if recorded[name] != sha256(path):
            raise ArtifactVerificationError(f"SHA256SUMS mismatch for {name}")


def _verify_source_inputs(root: Path, source_commit: str) -> None:
    payload = read_json(root / "source_inputs_manifest.json")
    if payload.get("schema_version") != 1 or payload.get("artifact_id") != ARTIFACT_ID:
        raise ArtifactVerificationError("source_inputs_manifest identity/schema mismatch")
    if payload.get("source_commit") != source_commit:
        raise ArtifactVerificationError("source_inputs_manifest source_commit mismatch")
    if payload.get("canonical_critic_source_commit") != CANONICAL_CRITIC_SOURCE_COMMIT:
        raise ArtifactVerificationError("source_inputs_manifest canonical source commit mismatch")
    execution_record = payload.get("execution_metadata_source")
    if not isinstance(execution_record, dict):
        raise ArtifactVerificationError("source_inputs_manifest execution metadata record is missing")
    if (
        execution_record.get("role") != "launch_metadata"
        or execution_record.get("packaged_path") != "execution_metadata.json"
        or not isinstance(execution_record.get("path"), str)
        or not HEX64.fullmatch(str(execution_record.get("sha256", "")))
        or execution_record.get("size_bytes") != (root / "execution_metadata.json").stat().st_size
        or execution_record.get("sha256") != sha256(root / "execution_metadata.json")
    ):
        raise ArtifactVerificationError("source_inputs_manifest execution metadata hash/size mismatch")
    arms = payload.get("arms")
    if not isinstance(arms, dict) or set(arms) != {ACTOR_KEY, CRITIC_KEY}:
        raise ArtifactVerificationError("source_inputs_manifest arms mismatch")
    for arm in (ACTOR_KEY, CRITIC_KEY):
        entries = arms[arm]
        if not isinstance(entries, list) or len(entries) != len(SEEDS):
            raise ArtifactVerificationError(f"source input entries incomplete for {arm}")
        if {entry.get("seed") for entry in entries if isinstance(entry, dict)} != set(SEEDS):
            raise ArtifactVerificationError(f"source input seeds mismatch for {arm}")
        for entry in entries:
            files = entry.get("files") if isinstance(entry, dict) else None
            if not isinstance(files, list) or {item.get("role") for item in files} != {
                "config",
                "summary",
                "env_snapshot",
                "metrics",
            }:
                raise ArtifactVerificationError(f"source input file roles mismatch for {arm}")
            for item in files:
                if not isinstance(item.get("path"), str) or not HEX64.fullmatch(
                    str(item.get("sha256", ""))
                ):
                    raise ArtifactVerificationError(f"Invalid source input record: {item!r}")
                if not isinstance(item.get("size_bytes"), int) or item["size_bytes"] <= 0:
                    raise ArtifactVerificationError(f"Invalid source input size: {item!r}")
    source_files = payload.get("frozen_source_files")
    if not isinstance(source_files, list) or not source_files:
        raise ArtifactVerificationError("frozen_source_files manifest is empty")
    actual_source_files = {
        path.relative_to(root).as_posix(): path
        for path in (root / "source").rglob("*")
        if path.is_file()
    }
    recorded_source_files: dict[str, dict[str, Any]] = {}
    for item in source_files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ArtifactVerificationError(f"Invalid frozen source record: {item!r}")
        if not HEX64.fullmatch(str(item.get("sha256", ""))):
            raise ArtifactVerificationError(f"Invalid frozen source SHA256: {item!r}")
        packaged = item.get("packaged_path")
        if not isinstance(packaged, str) or packaged in recorded_source_files:
            raise ArtifactVerificationError(f"Invalid/duplicate packaged source path: {item!r}")
        recorded_source_files[packaged] = item
    if set(recorded_source_files) != set(actual_source_files):
        raise ArtifactVerificationError(
            "Frozen source file-set mismatch: "
            f"missing={sorted(set(actual_source_files) - set(recorded_source_files))}, "
            f"stale={sorted(set(recorded_source_files) - set(actual_source_files))}"
        )
    for relative, path in actual_source_files.items():
        item = recorded_source_files[relative]
        if item["sha256"] != sha256(path) or item.get("size_bytes") != path.stat().st_size:
            raise ArtifactVerificationError(f"Frozen source hash/size mismatch: {relative}")


def _verify_execution_metadata(root: Path, source_commit: str) -> dict[str, Any]:
    payload = read_json(root / "execution_metadata.json")
    expected = {
        "schema_version": 1,
        "experiment": "ddpg_saber_actor_critic_v4_ls512",
        "repo_head": source_commit,
        "origin_main": source_commit,
        "head_matches_origin_main": True,
        "worktree_clean": True,
        "steps": TOTAL_STEPS,
        "seeds": list(SEEDS),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ArtifactVerificationError(
                f"execution_metadata.json {key} expected {value!r}, got {payload.get(key)!r}"
            )
    timestamp = payload.get("timestamp_utc")
    if not isinstance(timestamp, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", timestamp
    ):
        raise ArtifactVerificationError("execution_metadata.json timestamp_utc is invalid")
    max_parallel = payload.get("max_parallel")
    if not isinstance(max_parallel, int) or isinstance(max_parallel, bool) or max_parallel < 1:
        raise ArtifactVerificationError("execution_metadata.json max_parallel must be positive")
    return payload


def _expected_validation_gates(
    bundle: Any, source_commit: str, execution_metadata: dict[str, Any]
) -> dict[str, Any]:
    run_map = {
        arm: {str(run.seed): run for run in bundle.runs_by_arm[arm]}
        for arm in (ACTOR_KEY, CRITIC_KEY)
    }
    actual_protocol = {
        arm: {
            seed: {"total_steps": run.config["total_steps"], "smoke": run.config["smoke"]}
            for seed, run in run_map[arm].items()
        }
        for arm in (ACTOR_KEY, CRITIC_KEY)
    }
    actual_updates = {
        arm: {seed: run.summary["total_updates"] for seed, run in run_map[arm].items()}
        for arm in (ACTOR_KEY, CRITIC_KEY)
    }
    actual_grids = {
        arm: {
            seed: [int(row["step"]) for row in run.metrics_rows]
            for seed, run in run_map[arm].items()
        }
        for arm in (ACTOR_KEY, CRITIC_KEY)
    }
    actual_differences = {}
    for seed in map(str, bundle.seeds):
        actor_config = run_map[ACTOR_KEY][seed].config
        critic_config = run_map[CRITIC_KEY][seed].config
        actual_differences[seed] = sorted(
            key
            for key in set(actor_config) | set(critic_config)
            if actor_config.get(key) != critic_config.get(key)
        )
    values = [
        value
        for arm in (ACTOR_KEY, CRITIC_KEY)
        for run in bundle.runs_by_arm[arm]
        for value in (run.auc, run.final, run.best)
    ]
    return {
        "paired_seed_set": {
            "expected": list(SEEDS), "actual": list(bundle.seeds), "passed": True,
        },
        "non_smoke_1m": {
            "expected_total_steps": TOTAL_STEPS,
            "expected_smoke": False,
            "actual_by_arm": actual_protocol,
            "passed": True,
        },
        "total_updates": {
            "expected_per_seed": TOTAL_UPDATES,
            "actual_by_arm": actual_updates,
            "passed": True,
        },
        "exact_evaluation_grid": {
            "start": GRID[0],
            "end": GRID[-1],
            "frequency": 10_000,
            "points": len(GRID),
            "expected_steps": list(GRID),
            "actual_steps_by_arm": actual_grids,
            "passed": True,
        },
        "config_parity": {
            "only_differences": sorted(PAIR_CONFIG_EXEMPT_KEYS),
            "actual_differences_by_seed": actual_differences,
            "passed": True,
        },
        "environment_snapshot_core": {
            "train_eval_identity_and_dimensions": True,
            "runs_verified_by_arm": {
                ACTOR_KEY: list(bundle.seeds), CRITIC_KEY: list(bundle.seeds),
            },
            "passed": True,
        },
        "finite_core_metrics": {
            "metrics": ["success_auc", "final_success", "best_success"],
            "value_count": len(values),
            "nan_count": sum(not math.isfinite(value) for value in values),
            "passed": True,
        },
        "source_frozen": {
            "included": True, "source_commit_recorded": source_commit, "passed": True,
        },
        "execution_metadata": {
            "repo_head": execution_metadata["repo_head"],
            "origin_main": execution_metadata["origin_main"],
            "head_matches_origin_main": execution_metadata["head_matches_origin_main"],
            "worktree_clean": execution_metadata["worktree_clean"],
            "passed": True,
        },
        "excluded_payloads": {
            "raw_metrics": True,
            "logs": True,
            "checkpoints": True,
            "smoke": True,
            "passed": True,
        },
    }


def _verify_validation(
    root: Path,
    source_commit: str,
    bundle: Any,
    execution_metadata: dict[str, Any],
) -> None:
    payload = read_json(root / "validation.json")
    if payload.get("schema_version") != 1 or payload.get("artifact_id") != ARTIFACT_ID:
        raise ArtifactVerificationError("validation.json identity/schema mismatch")
    if payload.get("status") != "passed" or payload.get("source_commit") != source_commit:
        raise ArtifactVerificationError("validation.json status/source commit mismatch")
    if payload.get("canonical_critic_source_commit") != CANONICAL_CRITIC_SOURCE_COMMIT:
        raise ArtifactVerificationError("validation.json canonical source commit mismatch")
    gates = payload.get("gates")
    expected_gates = _expected_validation_gates(bundle, source_commit, execution_metadata)
    if gates != expected_gates:
        raise ArtifactVerificationError(
            "validation.json gates do not match exact recomputation from packaged inputs"
        )


def verify_artifact(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise ArtifactVerificationError(f"Artifact directory does not exist: {root}")
    _assert_no_forbidden_payloads(root)
    _verify_layout(root)
    manifest = _verify_artifact_manifest(root)
    _verify_sha256sums(root)
    _verify_source_inputs(root, str(manifest["source_commit"]))
    execution_metadata = _verify_execution_metadata(root, str(manifest["source_commit"]))

    bundle = analyze_scope_ablation(
        root / "data" / ACTOR_KEY,
        root / "data" / CRITIC_KEY,
        SEEDS,
    )
    verify_rendered_outputs(bundle, root / "analysis")
    _verify_validation(root, str(manifest["source_commit"]), bundle, execution_metadata)
    return {
        "artifact_id": ARTIFACT_ID,
        "source_commit": manifest["source_commit"],
        "seeds": list(SEEDS),
        "evaluation_grid_points": len(GRID),
        "config_parity": "verified",
        "analysis_recomputation": "verified",
        "sha256_manifests": "verified",
        "forbidden_payloads": "absent",
        "validation_gates": "verified",
        "frozen_source": "verified",
        "execution_metadata": "verified",
    }


def _default_artifact_root() -> Path:
    packaged_candidate = Path(__file__).resolve().parents[1]
    if (packaged_candidate / "artifact_manifest.json").is_file():
        return packaged_candidate
    repo = find_repo_root(Path(__file__))
    return repo / "experiment_FPP/paper_artifacts" / ARTIFACT_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=_default_artifact_root())
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
