#!/usr/bin/env python3
"""Verify the compact FPP DDPG/TD3 HER High-RR artifact, fail closed."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

from fpp_high_rr_artifact import (
    ARTIFACT_ID,
    FINAL_QUARTER_GRID,
    SCHEMA_VERSION,
    SEEDS,
    TABLE_GRID,
    TOTAL_STEPS,
    ArtifactError,
    expected_native_grid,
    first_hit,
    mean_sd,
    normalized_auc,
    read_compact_curve,
    read_json,
    sha256,
)


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_NAMES = {
    "metrics.csv",
    "metrics.jsonl",
    "run.log",
    "checkpoint.pt",
    "best_checkpoint.pt",
}
DATA_FILES = {
    "config.json",
    "env_snapshot.json",
    "summary.json",
    "eval_curve_10k.csv",
    "runtime_segments.csv",
}
ANALYSIS_FILES = {
    "per_seed_metrics.csv",
    "aggregate_metrics.csv",
    "common_grid_curves.csv",
    "mean_common_grid_curve.csv",
}
CODE_FILES = {
    "fpp_high_rr_artifact.py",
    "build_fpp_high_rr_compact_artifact.py",
    "verify_fpp_high_rr_compact_artifact.py",
}
ROOT_FILES = {
    "README.md",
    "SHA256SUMS",
    "artifact_manifest.json",
    "provenance.json",
    "source_inputs_manifest.json",
    "validation.json",
}
EXPECTED_COMMITS = {
    "ddpg": "e27438c1ea288fdb7bd6c75c49283d347fee77d9",
    "td3": "783a34cfeddfddf03bc5a5d18700f39e92e507dd",
}
EXPECTED_NUL_PADDING = {
    ("TD3", 1): 4042,
    ("TD3", 2): 3039,
    ("TD3", 3): 3565,
    ("TD3", 4): 3250,
    ("TD3", 5): 0,
}


def files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(left: object, right: object, label: str) -> None:
    if not math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ArtifactError(f"{label}: {left!r} != {right!r}")


def verify_layout(root: Path) -> None:
    if not root.is_dir():
        raise ArtifactError(f"artifact root not found: {root}")
    root_files = {path.name for path in root.iterdir() if path.is_file()}
    root_dirs = {path.name for path in root.iterdir() if path.is_dir()}
    if root_files != ROOT_FILES:
        raise ArtifactError(
            f"root file set mismatch: {sorted(root_files)}"
        )
    if root_dirs != {"analysis", "code", "data", "source"}:
        raise ArtifactError(
            f"root directory set mismatch: {sorted(root_dirs)}"
        )
    if {
        path.name for path in (root / "analysis").iterdir()
        if path.is_file()
    } != ANALYSIS_FILES:
        raise ArtifactError("analysis file set mismatch")
    if {
        path.name for path in (root / "code").iterdir()
        if path.is_file()
    } != CODE_FILES:
        raise ArtifactError("code file set mismatch")
    if {
        path.name for path in (root / "data").iterdir()
        if path.is_dir()
    } != {"ddpg", "td3"}:
        raise ArtifactError("data method directories mismatch")
    for method in ("ddpg", "td3"):
        method_root = root / "data" / method
        expected_seeds = {f"seed_{seed}" for seed in SEEDS}
        actual_seeds = {
            path.name for path in method_root.iterdir()
            if path.is_dir()
        }
        if actual_seeds != expected_seeds:
            raise ArtifactError(f"{method}: seed directory mismatch")
        for seed in SEEDS:
            seed_root = method_root / f"seed_{seed}"
            actual = {
                path.name for path in seed_root.iterdir()
                if path.is_file()
            }
            if actual != DATA_FILES:
                raise ArtifactError(
                    f"{method} seed {seed}: data file set mismatch"
                )
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ArtifactError(
                f"artifact symlinks are prohibited: {path}"
            )
        if path.is_file() and (
            path.name in FORBIDDEN_NAMES
            or path.suffix.lower() in {".pt", ".pth", ".ckpt"}
        ):
            raise ArtifactError(f"forbidden raw payload: {path}")


def verify_manifest(root: Path) -> None:
    manifest = read_json(root / "artifact_manifest.json")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_id") != ARTIFACT_ID
    ):
        raise ArtifactError("artifact manifest identity mismatch")
    if (
        manifest.get("manifest_self_excluded") is not True
        or manifest.get("sha256sums_self_excluded") is not True
    ):
        raise ArtifactError("artifact manifest exclusion policy mismatch")
    if manifest.get("seeds") != list(SEEDS):
        raise ArtifactError("artifact manifest seeds mismatch")
    if manifest.get("backbones") != ["DDPG", "TD3"]:
        raise ArtifactError("artifact manifest backbones mismatch")
    if manifest.get("table_common_grid") != list(TABLE_GRID):
        raise ArtifactError("artifact manifest table grid mismatch")
    if manifest.get("validation_status") != "passed_with_caveats":
        raise ArtifactError("artifact validation status mismatch")

    actual = {
        name: path for name, path in files(root).items()
        if name not in {"artifact_manifest.json", "SHA256SUMS"}
    }
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ArtifactError("artifact manifest files must be a list")
    indexed = {
        str(entry.get("path")): entry
        for entry in entries
        if isinstance(entry, dict)
    }
    if len(indexed) != len(entries) or set(indexed) != set(actual):
        raise ArtifactError("artifact manifest file set mismatch")
    for name, path in actual.items():
        entry = indexed[name]
        if entry.get("sha256") != sha256(path):
            raise ArtifactError(f"manifest SHA256 mismatch: {name}")
        if entry.get("size_bytes") != path.stat().st_size:
            raise ArtifactError(f"manifest size mismatch: {name}")
    if manifest.get("file_count") != len(actual):
        raise ArtifactError("manifest file_count mismatch")
    if manifest.get("total_size_bytes") != sum(
        path.stat().st_size for path in actual.values()
    ):
        raise ArtifactError("manifest total_size_bytes mismatch")


def verify_sha256sums(root: Path) -> None:
    recorded: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(
        encoding="utf-8"
    ).splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise ArtifactError("malformed SHA256SUMS") from error
        if (
            not HEX64.fullmatch(digest)
            or not relative
            or relative in recorded
        ):
            raise ArtifactError("invalid SHA256SUMS entry")
        recorded[relative] = digest
    actual = {
        name: path for name, path in files(root).items()
        if name != "SHA256SUMS"
    }
    if set(recorded) != set(actual):
        raise ArtifactError("SHA256SUMS file set mismatch")
    for name, path in actual.items():
        if recorded[name] != sha256(path):
            raise ArtifactError(f"SHA256SUMS mismatch: {name}")


def verify_provenance(root: Path) -> None:
    provenance = read_json(root / "provenance.json")
    if (
        provenance.get("schema_version") != SCHEMA_VERSION
        or provenance.get("artifact_id") != ARTIFACT_ID
        or provenance.get("source_machine_label") != "host_a"
        or provenance.get("seeds") != list(SEEDS)
        or provenance.get("steps_per_seed") != TOTAL_STEPS
    ):
        raise ArtifactError("provenance identity/protocol mismatch")
    methods = provenance.get("methods")
    if not isinstance(methods, dict) or set(methods) != {"ddpg", "td3"}:
        raise ArtifactError("provenance methods mismatch")
    for method, commit in EXPECTED_COMMITS.items():
        record = methods[method]
        if record.get("execution_source_commit") != "not retained":
            raise ArtifactError(
                f"{method}: execution source must remain not retained"
            )
        if record.get("first_retained_results_commit") != commit:
            raise ArtifactError(f"{method}: retained commit mismatch")
    if methods["ddpg"].get("trajectory_status") != (
        "completed_uninterrupted"
    ):
        raise ArtifactError("DDPG trajectory status mismatch")
    if methods["td3"].get("trajectory_status") != "completed_resumed":
        raise ArtifactError("TD3 trajectory status mismatch")
    expected_resume = {
        "1": 800001,
        "2": 800001,
        "3": 800001,
        "4": 800001,
        "5": 630001,
    }
    if provenance.get("td3_resume_points") != expected_resume:
        raise ArtifactError("TD3 resume map mismatch")
    contract = provenance.get("metric_contract")
    if not isinstance(contract, dict):
        raise ArtifactError("metric contract missing")
    if contract.get("table_common_grid") != list(TABLE_GRID):
        raise ArtifactError("metric contract common grid mismatch")
    if contract.get("final_quarter_grid") != list(
        FINAL_QUARTER_GRID
    ):
        raise ArtifactError("metric contract final-quarter grid mismatch")
    exclusion = contract.get("table_grid_exclusion")
    if not isinstance(exclusion, dict) or exclusion.get("step") != 800000:
        raise ArtifactError("metric contract 800k exclusion missing")
    integrity = provenance.get("raw_integrity")
    if not isinstance(integrity, dict):
        raise ArtifactError("raw-integrity provenance missing")
    per_seed = integrity.get("per_seed")
    expected = {
        f"{backbone.lower()}_seed_{seed}": {
            "nul_padding_bytes": EXPECTED_NUL_PADDING.get((backbone, seed), 0),
            "affected_rows": 1 if EXPECTED_NUL_PADDING.get((backbone, seed), 0) else 0,
        }
        for backbone in ("DDPG", "TD3")
        for seed in SEEDS
    }
    if per_seed != expected:
        raise ArtifactError("raw-integrity NUL audit mismatch")


def verify_sources(root: Path) -> None:
    manifest = read_json(root / "source_inputs_manifest.json")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_id") != ARTIFACT_ID
    ):
        raise ArtifactError("source manifest identity mismatch")
    raw_runs = manifest.get("raw_runs")
    if not isinstance(raw_runs, dict) or set(raw_runs) != {
        "ddpg",
        "td3",
    }:
        raise ArtifactError("source manifest raw methods mismatch")
    for method in ("ddpg", "td3"):
        runs = raw_runs[method]
        if not isinstance(runs, list) or len(runs) != len(SEEDS):
            raise ArtifactError(f"{method}: source run count mismatch")
        if {int(run["seed"]) for run in runs} != set(SEEDS):
            raise ArtifactError(f"{method}: source seed set mismatch")
        for run in runs:
            retained = run.get("retained_history")
            if not isinstance(retained, dict):
                raise ArtifactError("retained history missing")
            if retained.get(
                "first_retained_results_commit"
            ) != EXPECTED_COMMITS[method]:
                raise ArtifactError("retained history commit mismatch")
            if retained.get(
                "raw_curve_matches_retained_compact"
            ) is not True:
                raise ArtifactError("raw/retained curve parity missing")
            blob = str(retained.get("compact_curve_blob_id", ""))
            if not HEX40.fullmatch(blob):
                raise ArtifactError("invalid retained compact blob id")
            parity = retained.get("control_file_byte_parity")
            if not isinstance(parity, dict) or not parity or not all(
                value is True for value in parity.values()
            ):
                raise ArtifactError("control blob parity failed")
            source_files = run.get("files")
            if not isinstance(source_files, list) or not source_files:
                raise ArtifactError("source file records missing")
            for record in source_files:
                logical = record.get("logical_path")
                digest = str(record.get("sha256", ""))
                size = record.get("size_bytes")
                if (
                    not isinstance(logical, str)
                    or not logical.startswith("machine_local/")
                    or not HEX64.fullmatch(digest)
                    or not isinstance(size, int)
                    or size <= 0
                ):
                    raise ArtifactError(
                        f"invalid source file record: {record!r}"
                    )
                packaged = record.get("packaged_path")
                if packaged is not None and not (
                    root / str(packaged)
                ).is_file():
                    raise ArtifactError(
                        f"missing packaged source copy: {packaged}"
                    )
    launch = manifest.get("launch_snapshots")
    if not isinstance(launch, dict) or set(launch) != {"ddpg", "td3"}:
        raise ArtifactError("launch snapshot manifest mismatch")
    for records in launch.values():
        if not isinstance(records, list) or not records:
            raise ArtifactError("launch snapshot records missing")
        for record in records:
            packaged = record.get("packaged_path")
            if not isinstance(packaged, str) or not (
                root / packaged
            ).is_file():
                raise ArtifactError("packaged launch snapshot missing")


def parse_runtime_segments(path: Path) -> list[float]:
    rows = read_csv(path)
    if not rows:
        raise ArtifactError(f"{path}: empty runtime segment table")
    indices = [int(row["segment_index"]) for row in rows]
    if indices != list(range(1, len(rows) + 1)):
        raise ArtifactError(f"{path}: runtime segment indices mismatch")
    active = [float(row["active_seconds"]) for row in rows]
    if any(not math.isfinite(value) or value <= 0.0 for value in active):
        raise ArtifactError(f"{path}: invalid active runtime")
    return active


def verify_metrics(root: Path) -> None:
    expected_rows: dict[tuple[str, int], dict[str, object]] = {}
    expected_common: list[dict[str, object]] = []
    for method, backbone in (("ddpg", "DDPG"), ("td3", "TD3")):
        for seed in SEEDS:
            data = root / "data" / method / f"seed_{seed}"
            config = read_json(data / "config.json")
            summary = read_json(data / "summary.json")
            if (
                config.get("total_steps") != TOTAL_STEPS
                or config.get("updates_per_step") != 4
                or config.get("her_ratio") != 0.8
                or config.get("weight_scope") != "none"
                or config.get("smoke") is not False
            ):
                raise ArtifactError(
                    f"{backbone} seed {seed}: config protocol mismatch"
                )
            expected_method = (
                "b0_ddpg_her"
                if backbone == "DDPG"
                else "td3_her_high_rr_x4"
            )
            expected_updates = (
                3_980_004 if backbone == "DDPG" else 3_980_000
            )
            if (
                config.get("method") != expected_method
                or summary.get("method") != expected_method
                or int(summary.get("total_steps", -1))
                != TOTAL_STEPS
                or int(summary.get("total_updates", -1))
                != expected_updates
            ):
                raise ArtifactError(
                    f"{backbone} seed {seed}: summary protocol mismatch"
                )
            if config.get("resume") is not (backbone == "TD3"):
                raise ArtifactError(
                    f"{backbone} seed {seed}: resume flag mismatch"
                )
            curve = read_compact_curve(data / "eval_curve_10k.csv")
            native_grid = tuple(sorted(curve))
            if native_grid != expected_native_grid(backbone, seed):
                raise ArtifactError(
                    f"{backbone} seed {seed}: native grid mismatch"
                )
            if any(step not in curve for step in TABLE_GRID):
                raise ArtifactError(
                    f"{backbone} seed {seed}: table grid incomplete"
                )
            native_auc = normalized_auc(curve, native_grid)
            close(
                native_auc,
                summary["eval_AUC_success"],
                f"{backbone} seed {seed} summary AUC",
            )
            segments = parse_runtime_segments(
                data / "runtime_segments.csv"
            )
            expected_segment_count = 1 if backbone == "DDPG" else 2
            if len(segments) != expected_segment_count:
                raise ArtifactError(
                    f"{backbone} seed {seed}: runtime segments mismatch"
                )
            active_runtime = sum(segments)
            resume_step = (
                None
                if backbone == "DDPG"
                else (630001 if seed == 5 else 800001)
            )
            expected_rows[(backbone, seed)] = {
                "backbone": backbone,
                "method": "HER high-RR",
                "seed": seed,
                "utd": 4,
                "trajectory_status": (
                    "uninterrupted"
                    if backbone == "DDPG"
                    else "resumed_after_reboot"
                ),
                "resume_step": resume_step,
                "native_grid_points": len(native_grid),
                "native_auc": native_auc,
                "common_grid_points": len(TABLE_GRID),
                "common_grid_auc": normalized_auc(curve, TABLE_GRID),
                "final_quarter_grid_points": len(
                    FINAL_QUARTER_GRID
                ),
                "final_quarter_auc": normalized_auc(
                    curve, FINAL_QUARTER_GRID
                ),
                "final_success": curve[TOTAL_STEPS],
                "hit90_step": first_hit(curve, TABLE_GRID),
                "active_runtime_segments": len(segments),
                "active_runtime_seconds": active_runtime,
                "recomputed_steps_per_second": (
                    TOTAL_STEPS / active_runtime
                ),
                "summary_steps_per_second": float(
                    summary["final_steps_per_sec"]
                ),
                "raw_nul_padding_bytes": EXPECTED_NUL_PADDING.get(
                    (backbone, seed), 0
                ),
                "raw_nul_padding_rows": (
                    1 if EXPECTED_NUL_PADDING.get((backbone, seed), 0) else 0
                ),
            }
            for step in TABLE_GRID:
                expected_common.append(
                    {
                        "backbone": backbone,
                        "method": "HER high-RR",
                        "seed": seed,
                        "step": step,
                        "success": curve[step],
                    }
                )

    rows = read_csv(root / "analysis" / "per_seed_metrics.csv")
    if len(rows) != 10:
        raise ArtifactError("per-seed analysis must contain ten rows")
    for row in rows:
        key = (row["backbone"], int(row["seed"]))
        expected = expected_rows.get(key)
        if expected is None:
            raise ArtifactError(f"unexpected per-seed row {key}")
        for field in (
            "backbone",
            "method",
            "trajectory_status",
        ):
            if row[field] != str(expected[field]):
                raise ArtifactError(f"{key}: {field} mismatch")
        for field in (
            "seed",
            "utd",
            "native_grid_points",
            "common_grid_points",
            "final_quarter_grid_points",
            "active_runtime_segments",
            "raw_nul_padding_bytes",
            "raw_nul_padding_rows",
        ):
            if int(row[field]) != int(expected[field]):
                raise ArtifactError(f"{key}: {field} mismatch")
        for field in (
            "native_auc",
            "common_grid_auc",
            "final_quarter_auc",
            "final_success",
            "active_runtime_seconds",
            "recomputed_steps_per_second",
            "summary_steps_per_second",
        ):
            close(row[field], expected[field], f"{key}: {field}")
        for field in ("resume_step", "hit90_step"):
            observed = int(row[field]) if row[field] else None
            if observed != expected[field]:
                raise ArtifactError(f"{key}: {field} mismatch")

    common = read_csv(
        root / "analysis" / "common_grid_curves.csv"
    )
    if len(common) != len(expected_common):
        raise ArtifactError("common-grid curve row count mismatch")
    for actual, expected in zip(common, expected_common):
        if (
            actual["backbone"] != expected["backbone"]
            or actual["method"] != expected["method"]
            or int(actual["seed"]) != expected["seed"]
            or int(actual["step"]) != expected["step"]
        ):
            raise ArtifactError("common-grid curve key/order mismatch")
        close(actual["success"], expected["success"], "common curve")

    aggregate = {
        row["backbone"]: row
        for row in read_csv(
            root / "analysis" / "aggregate_metrics.csv"
        )
    }
    if set(aggregate) != {"DDPG", "TD3"}:
        raise ArtifactError("aggregate backbones mismatch")
    for backbone, row in aggregate.items():
        selected = [
            value
            for (candidate, _), value in expected_rows.items()
            if candidate == backbone
        ]
        if int(row["n_seeds"]) != 5 or int(row["utd"]) != 4:
            raise ArtifactError(f"{backbone}: aggregate protocol mismatch")
        for field in (
            "native_auc",
            "common_grid_auc",
            "final_quarter_auc",
            "final_success",
            "recomputed_steps_per_second",
            "summary_steps_per_second",
        ):
            mean, sample_sd = mean_sd(
                [float(value[field]) for value in selected]
            )
            close(row[f"{field}_mean"], mean, f"{backbone} {field} mean")
            close(
                row[f"{field}_sample_sd"],
                sample_sd,
                f"{backbone} {field} sample SD",
            )
        reached = [
            int(value["hit90_step"])
            for value in selected
            if value["hit90_step"] is not None
        ]
        if (
            int(row["hit90_reached"]) != len(reached)
            or int(row["hit90_total"]) != 5
        ):
            raise ArtifactError(f"{backbone}: Hit90 count mismatch")


def verify_validation(root: Path) -> None:
    validation = read_json(root / "validation.json")
    if (
        validation.get("schema_version") != SCHEMA_VERSION
        or validation.get("artifact_id") != ARTIFACT_ID
        or validation.get("status") != "passed_with_caveats"
    ):
        raise ArtifactError("validation identity/status mismatch")
    gates = validation.get("gates")
    if not isinstance(gates, dict):
        raise ArtifactError("validation gates missing")
    for name in (
        "paired_seed_set",
        "completed_non_smoke_1m",
        "standard_json_and_public_path_sanitization",
    ):
        gate = gates.get(name)
        if not isinstance(gate, dict) or gate.get("passed") is not True:
            raise ArtifactError(f"validation gate failed: {name}")


def verify_public_sanitization(root: Path) -> None:
    forbidden_patterns = (
        re.compile(b"/" + b"home/" + rb"[^/\s]+/"),
        re.compile(rb"GPU-[0-9a-fA-F-]{16,}"),
        re.compile(rb"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    )
    for relative, path in files(root).items():
        if path.suffix.lower() not in {
            ".json",
            ".csv",
            ".md",
            ".txt",
            ".py",
            ".sh",
        } and path.name != "SHA256SUMS":
            continue
        payload = path.read_bytes()
        for pattern in forbidden_patterns:
            if pattern.search(payload):
                raise ArtifactError(
                    f"public sanitization failed: {relative}"
                )


def verify(root: Path) -> None:
    verify_layout(root)
    verify_manifest(root)
    verify_sha256sums(root)
    verify_provenance(root)
    verify_sources(root)
    verify_metrics(root)
    verify_validation(root)
    verify_public_sanitization(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    arguments = parser.parse_args()
    verify(arguments.artifact_root.resolve())
    print(
        f"Verified {ARTIFACT_ID}: passed_with_caveats "
        "(DDPG uninterrupted; TD3 resumed)"
    )


if __name__ == "__main__":
    main()
