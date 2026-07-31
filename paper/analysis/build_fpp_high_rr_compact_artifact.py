#!/usr/bin/env python3
"""Build the compact FPP DDPG/TD3 HER High-RR x4 paper artifact."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

from fpp_high_rr_artifact import (
    ARTIFACT_ID,
    FINAL_QUARTER_GRID,
    SCHEMA_VERSION,
    SEEDS,
    SUCCESS_FIELD,
    TABLE_GRID,
    TOTAL_STEPS,
    ArtifactError,
    expected_native_grid,
    file_record,
    first_hit,
    git_blob_oid,
    mean_sd,
    normalized_auc,
    parse_raw_metrics,
    runtime_total,
    sanitize_json,
    sha256,
    write_csv,
    write_json,
)


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "experiment_FPP"
    / "paper_artifacts"
    / ARTIFACT_ID
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
RESUME_RE = re.compile(r"resumed from (.+) at step=(\d+)")
CODE_FILES = (
    "fpp_high_rr_artifact.py",
    "build_fpp_high_rr_compact_artifact.py",
    "verify_fpp_high_rr_compact_artifact.py",
)

METHOD_SPECS = {
    "ddpg": {
        "backbone": "DDPG",
        "method": "HER high-RR",
        "trainer_method": "b0_ddpg_her",
        "updates_per_step": 4,
        "total_updates": 3_980_004,
        "run_root": "experiment_FPP/runs/B0-high-replay-ratio-x4",
        "run_name": (
            "fetch_pick_and_place_b0_high_rr_x4_s{seed}_"
            "1000000_ls1024"
        ),
        "launch_root": (
            "SABER-FetchPickAndPlace-active/FetchPickAndPlace/"
            "B0-high-replay-ratio-x4"
        ),
        "resume_log_root": None,
        "resume_steps": {},
        "resume_checkpoint_names": {},
        "first_retained_results_commit": (
            "e27438c1ea288fdb7bd6c75c49283d347fee77d9"
        ),
        "execution_source_commit": None,
        "source_retention": (
            "execution source commit not retained; current-machine launch "
            "snapshot is packaged, and result/control blobs are bound to the "
            "first retained results commit"
        ),
    },
    "td3": {
        "backbone": "TD3",
        "method": "HER high-RR",
        "trainer_method": "td3_her_high_rr_x4",
        "updates_per_step": 4,
        "total_updates": 3_980_000,
        "run_root": (
            "experiment_FPP/runs/td3_her_high_replay_ratio_x4"
        ),
        "run_name": (
            "fetch_pick_and_place_td3_her_high_rr_x4_s{seed}_"
            "1000000_ls512"
        ),
        "launch_root": "experiment_FPP/td3_her_high_replay_ratio_x4",
        "resume_log_root": (
            "experiment_FPP/logs/"
            "td3_her_high_replay_ratio_x4_resume_after_reboot"
        ),
        "resume_steps": {
            1: 800_001,
            2: 800_001,
            3: 800_001,
            4: 800_001,
            5: 630_001,
        },
        "resume_checkpoint_names": {
            1: "checkpoint.pt",
            2: "checkpoint.pt",
            3: "checkpoint.pt",
            4: "checkpoint.pt",
            5: "best_checkpoint.pt",
        },
        "first_retained_results_commit": (
            "783a34cfeddfddf03bc5a5d18700f39e92e507dd"
        ),
        "execution_source_commit": None,
        "source_retention": (
            "execution commit not retained; the first retained results "
            "commit includes the frozen launch package and byte-identical "
            "result/control blobs"
        ),
    },
}


def git(repo: Path, *arguments: str, binary: bool = False) -> str | bytes:
    process = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
        check=False,
    )
    if process.returncode:
        stderr = (
            process.stderr.decode("utf-8", "replace")
            if binary
            else process.stderr
        )
        raise ArtifactError(
            stderr.strip() or f"git {' '.join(arguments)} failed"
        )
    return process.stdout


def logical_raw_path(path: Path, raw_repo: Path) -> str:
    return (
        Path("machine_local")
        / path.resolve().relative_to(raw_repo.resolve())
    ).as_posix()


def historical_path(spec: dict[str, Any], seed: int, filename: str) -> str:
    run_name = str(spec["run_name"]).format(seed=seed)
    return f"{spec['run_root']}/{run_name}/{filename}"


def curve_from_bytes(payload: bytes, label: str) -> dict[int, float]:
    curve: dict[int, float] = {}
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    required = {"step", SUCCESS_FIELD}
    missing = required - set(reader.fieldnames or ())
    if missing:
        raise ArtifactError(f"{label}: missing columns {sorted(missing)}")
    for row in reader:
        value = row.get(SUCCESS_FIELD, "")
        try:
            success = float(value)
        except (TypeError, ValueError):
            continue
        if not (0.0 <= success <= 1.0):
            continue
        step = int(float(row["step"]))
        if step in curve:
            raise ArtifactError(f"{label}: duplicate finite eval at {step}")
        curve[step] = success
    return curve


def retained_blob(
    history_repo: Path, commit: str, path: str
) -> tuple[str, bytes]:
    resolved = str(git(history_repo, "rev-parse", f"{commit}:{path}")).strip()
    if not HEX40.fullmatch(resolved):
        raise ArtifactError(f"invalid retained blob id for {path}: {resolved}")
    payload = git(history_repo, "show", f"{commit}:{path}", binary=True)
    assert isinstance(payload, bytes)
    return resolved, payload


def sanitized_copy_json(
    source: Path,
    destination: Path,
    raw_repo: Path,
    run_dir: Path,
) -> None:
    replacements = (
        (run_dir.resolve().as_posix(), "${SOURCE_RUN}"),
        (raw_repo.resolve().as_posix(), "${RAW_REPOSITORY}"),
        (raw_repo.parent.resolve().as_posix(), "${WORKSPACE}"),
        (Path.home().as_posix(), "${HOME}"),
        (socket.gethostname(), "${HOST}"),
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    write_json(destination, sanitize_json(payload, replacements))


def sanitized_copy_text(
    source: Path, destination: Path, raw_repo: Path
) -> None:
    value = source.read_text(encoding="utf-8")
    replacements = (
        (raw_repo.resolve().as_posix(), "${RAW_REPOSITORY}"),
        (raw_repo.parent.resolve().as_posix(), "${WORKSPACE}"),
        (Path.home().as_posix(), "${HOME}"),
        (socket.gethostname(), "${HOST}"),
    )
    for before, after in replacements:
        if before:
            value = value.replace(before, after)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(value, encoding="utf-8", newline="\n")


def validate_config_summary(
    spec: dict[str, Any],
    seed: int,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    expected = {
        "method": spec["trainer_method"],
        "total_steps": TOTAL_STEPS,
        "updates_per_step": spec["updates_per_step"],
        "her_ratio": 0.8,
        "weight_scope": "none",
        "smoke": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ArtifactError(
                f"{spec['backbone']} seed {seed}: config.{key} expected "
                f"{value!r}, got {config.get(key)!r}"
            )
    seeds = config.get("seeds")
    if not isinstance(seeds, dict) or int(seeds.get("env_seed", -1)) != seed:
        raise ArtifactError(
            f"{spec['backbone']} seed {seed}: env seed mismatch"
        )
    expected_resume = spec["backbone"] == "TD3"
    if config.get("resume") is not expected_resume:
        raise ArtifactError(
            f"{spec['backbone']} seed {seed}: resume flag mismatch"
        )
    if int(summary.get("total_steps", -1)) != TOTAL_STEPS:
        raise ArtifactError(
            f"{spec['backbone']} seed {seed}: incomplete total_steps"
        )
    if int(summary.get("total_updates", -1)) != spec["total_updates"]:
        raise ArtifactError(
            f"{spec['backbone']} seed {seed}: total_updates mismatch"
        )
    if summary.get("method") != spec["trainer_method"]:
        raise ArtifactError(
            f"{spec['backbone']} seed {seed}: summary method mismatch"
        )


def parse_resume_metadata(
    spec: dict[str, Any], seed: int, raw_repo: Path
) -> tuple[dict[str, object], Path | None]:
    if spec["backbone"] == "DDPG":
        return {
            "status": "uninterrupted",
            "resume_step": None,
            "checkpoint_kind": None,
        }, None
    log = raw_repo / str(spec["resume_log_root"]) / f"seed{seed}.log"
    matches = RESUME_RE.findall(log.read_text(encoding="utf-8"))
    if len(matches) != 1:
        raise ArtifactError(f"{log}: expected one resume record")
    checkpoint_path, step_text = matches[0]
    resume_step = int(step_text)
    checkpoint_kind = Path(checkpoint_path).name
    if resume_step != spec["resume_steps"][seed]:
        raise ArtifactError(f"{log}: unexpected resume step {resume_step}")
    if checkpoint_kind != spec["resume_checkpoint_names"][seed]:
        raise ArtifactError(
            f"{log}: unexpected checkpoint kind {checkpoint_kind}"
        )
    return {
        "status": "resumed_after_reboot",
        "resume_step": resume_step,
        "checkpoint_kind": checkpoint_kind,
    }, log


def write_eval_curve(path: Path, curve: dict[int, float]) -> None:
    write_csv(
        path,
        (
            {"step": step, SUCCESS_FIELD: curve[step]}
            for step in sorted(curve)
        ),
        ("step", SUCCESS_FIELD),
    )


def aggregate_rows(per_seed: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for backbone in ("DDPG", "TD3"):
        selected = [
            row for row in per_seed if row["backbone"] == backbone
        ]
        aggregate: dict[str, object] = {
            "backbone": backbone,
            "method": "HER high-RR",
            "utd": 4,
            "n_seeds": len(selected),
        }
        for field in (
            "native_auc",
            "common_grid_auc",
            "final_quarter_auc",
            "final_success",
            "recomputed_steps_per_second",
            "summary_steps_per_second",
        ):
            values = [float(row[field]) for row in selected]
            mean, sample_sd = mean_sd(values)
            aggregate[f"{field}_mean"] = mean
            aggregate[f"{field}_sample_sd"] = sample_sd
        reached = [
            int(row["hit90_step"])
            for row in selected
            if row["hit90_step"] is not None
        ]
        aggregate["hit90_reached"] = len(reached)
        aggregate["hit90_total"] = len(selected)
        aggregate["hit90_median_reached"] = (
            int(__import__("statistics").median(reached))
            if reached
            else None
        )
        rows.append(aggregate)
    return rows


def build(
    raw_repo: Path,
    history_repo: Path,
    output: Path,
) -> None:
    raw_repo = raw_repo.resolve()
    history_repo = history_repo.resolve()
    if not (raw_repo / ".git").exists():
        raise ArtifactError(f"raw repository not found: {raw_repo}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{ARTIFACT_ID}.", dir=output.parent
    ) as temporary:
        artifact = Path(temporary) / ARTIFACT_ID
        artifact.mkdir()
        per_seed: list[dict[str, object]] = []
        common_rows: list[dict[str, object]] = []
        source_runs: dict[str, list[dict[str, object]]] = {
            "ddpg": [],
            "td3": [],
        }
        history_validation: dict[str, dict[str, object]] = {}
        grid_validation: dict[str, dict[str, object]] = {}
        runtime_validation: dict[str, dict[str, object]] = {}
        resume_validation: dict[str, dict[str, object]] = {}

        for method_key, spec in METHOD_SPECS.items():
            commit = str(spec["first_retained_results_commit"])
            resolved_commit = str(
                git(history_repo, "rev-parse", f"{commit}^{{commit}}")
            ).strip()
            if resolved_commit != commit:
                raise ArtifactError(
                    f"retained commit mismatch: {commit} -> {resolved_commit}"
                )
            for seed in SEEDS:
                run_name = str(spec["run_name"]).format(seed=seed)
                run_dir = raw_repo / str(spec["run_root"]) / run_name
                source_paths = {
                    "config": run_dir / "config.json",
                    "environment": run_dir / "env_snapshot.json",
                    "summary": run_dir / "summary.json",
                    "raw_metrics": run_dir / "metrics.csv",
                    "run_log": run_dir / "run.log",
                }
                if method_key == "ddpg":
                    source_paths["retained_compact_metrics"] = (
                        run_dir / "metrics_selected_10k.csv"
                    )
                missing = [
                    str(path)
                    for path in source_paths.values()
                    if not path.is_file()
                ]
                if missing:
                    raise ArtifactError(f"missing run sources: {missing}")

                raw_config = json.loads(
                    source_paths["config"].read_text(encoding="utf-8")
                )
                raw_summary = json.loads(
                    source_paths["summary"].read_text(encoding="utf-8")
                )
                validate_config_summary(
                    spec, seed, raw_config, raw_summary
                )
                metrics = parse_raw_metrics(source_paths["raw_metrics"])
                expected_grid = expected_native_grid(
                    str(spec["backbone"]), seed
                )
                actual_grid = tuple(sorted(metrics.curve))
                if actual_grid != expected_grid:
                    raise ArtifactError(
                        f"{spec['backbone']} seed {seed}: native grid "
                        "does not match the retained contract"
                    )
                if any(step not in metrics.curve for step in TABLE_GRID):
                    raise ArtifactError(
                        f"{spec['backbone']} seed {seed}: common table "
                        "grid is incomplete"
                    )
                summary_auc = float(raw_summary["eval_AUC_success"])
                native_auc = normalized_auc(metrics.curve, actual_grid)
                if abs(summary_auc - native_auc) > 1e-12:
                    raise ArtifactError(
                        f"{spec['backbone']} seed {seed}: summary AUC "
                        "does not match raw curve"
                    )

                resume, resume_log = parse_resume_metadata(
                    spec, seed, raw_repo
                )
                expected_segments = (
                    2 if spec["backbone"] == "TD3" else 1
                )
                if len(metrics.segments) != expected_segments:
                    raise ArtifactError(
                        f"{spec['backbone']} seed {seed}: expected "
                        f"{expected_segments} runtime segments, found "
                        f"{len(metrics.segments)}"
                    )
                active_runtime = runtime_total(metrics.segments)
                recomputed_sps = TOTAL_STEPS / active_runtime
                summary_sps = float(raw_summary["final_steps_per_sec"])

                history_file_parity: dict[str, bool] = {}
                history_blob_ids: dict[str, str] = {}
                for role, filename in (
                    ("config", "config.json"),
                    ("environment", "env_snapshot.json"),
                    ("summary", "summary.json"),
                    ("run_log", "run.log"),
                ):
                    retained_path = historical_path(
                        spec, seed, filename
                    )
                    blob_id, _ = retained_blob(
                        history_repo, commit, retained_path
                    )
                    current_oid = git_blob_oid(source_paths[role])
                    history_file_parity[role] = current_oid == blob_id
                    history_blob_ids[role] = blob_id
                    if not history_file_parity[role]:
                        raise ArtifactError(
                            f"{spec['backbone']} seed {seed}: {role} "
                            "differs from retained Git blob"
                        )

                compact_history_path = historical_path(
                    spec, seed, "metrics_selected_10k.csv"
                )
                compact_blob, compact_payload = retained_blob(
                    history_repo, commit, compact_history_path
                )
                retained_curve = curve_from_bytes(
                    compact_payload, compact_history_path
                )
                if retained_curve != metrics.curve:
                    raise ArtifactError(
                        f"{spec['backbone']} seed {seed}: raw curve "
                        "differs from retained compact Git curve"
                    )
                if method_key == "ddpg" and (
                    git_blob_oid(
                        source_paths["retained_compact_metrics"]
                    )
                    != compact_blob
                ):
                    raise ArtifactError(
                        f"DDPG seed {seed}: local compact metrics differ "
                        "from retained blob"
                    )

                data_relative = (
                    Path("data")
                    / method_key
                    / f"seed_{seed}"
                )
                data_dir = artifact / data_relative
                sanitized_copy_json(
                    source_paths["config"],
                    data_dir / "config.json",
                    raw_repo,
                    run_dir,
                )
                sanitized_copy_json(
                    source_paths["environment"],
                    data_dir / "env_snapshot.json",
                    raw_repo,
                    run_dir,
                )
                sanitized_copy_json(
                    source_paths["summary"],
                    data_dir / "summary.json",
                    raw_repo,
                    run_dir,
                )
                write_eval_curve(
                    data_dir / "eval_curve_10k.csv", metrics.curve
                )
                write_csv(
                    data_dir / "runtime_segments.csv",
                    (
                        {
                            "segment_index": segment.index,
                            "first_row": segment.first_row,
                            "last_row": segment.last_row,
                            "first_step": segment.first_step,
                            "last_step": segment.last_step,
                            "first_runtime_seconds": (
                                segment.first_runtime_seconds
                            ),
                            "active_seconds": segment.active_seconds,
                        }
                        for segment in metrics.segments
                    ),
                    (
                        "segment_index",
                        "first_row",
                        "last_row",
                        "first_step",
                        "last_step",
                        "first_runtime_seconds",
                        "active_seconds",
                    ),
                )

                seed_row: dict[str, object] = {
                    "backbone": spec["backbone"],
                    "method": spec["method"],
                    "seed": seed,
                    "utd": 4,
                    "trajectory_status": resume["status"],
                    "resume_step": resume["resume_step"],
                    "native_grid_points": len(actual_grid),
                    "native_auc": native_auc,
                    "common_grid_points": len(TABLE_GRID),
                    "common_grid_auc": normalized_auc(
                        metrics.curve, TABLE_GRID
                    ),
                    "final_quarter_grid_points": len(
                        FINAL_QUARTER_GRID
                    ),
                    "final_quarter_auc": normalized_auc(
                        metrics.curve, FINAL_QUARTER_GRID
                    ),
                    "final_success": metrics.curve[TOTAL_STEPS],
                    "hit90_step": first_hit(
                        metrics.curve, TABLE_GRID
                    ),
                    "active_runtime_segments": len(metrics.segments),
                    "active_runtime_seconds": active_runtime,
                    "recomputed_steps_per_second": recomputed_sps,
                    "summary_steps_per_second": summary_sps,
                    "raw_nul_padding_bytes": metrics.nul_padding_bytes,
                    "raw_nul_padding_rows": metrics.nul_padding_rows,
                }
                per_seed.append(seed_row)
                for step in TABLE_GRID:
                    common_rows.append(
                        {
                            "backbone": spec["backbone"],
                            "method": spec["method"],
                            "seed": seed,
                            "step": step,
                            "success": metrics.curve[step],
                        }
                    )

                source_records = [
                    file_record(
                        path,
                        logical_raw_path(path, raw_repo),
                        role=role,
                        packaged_path=(
                            (
                                data_relative
                                / {
                                    "config": "config.json",
                                    "environment": "env_snapshot.json",
                                    "summary": "summary.json",
                                }[role]
                            ).as_posix()
                            if role in {
                                "config",
                                "environment",
                                "summary",
                            }
                            else None
                        ),
                    )
                    for role, path in source_paths.items()
                ]
                if resume_log is not None:
                    source_records.append(
                        file_record(
                            resume_log,
                            logical_raw_path(resume_log, raw_repo),
                            role="resume_log",
                        )
                    )
                source_runs[method_key].append(
                    {
                        "seed": seed,
                        "run_key": run_name,
                        "files": source_records,
                        "retained_history": {
                            "first_retained_results_commit": commit,
                            "control_blob_ids": history_blob_ids,
                            "control_file_byte_parity": (
                                history_file_parity
                            ),
                            "compact_curve_path": (
                                compact_history_path
                            ),
                            "compact_curve_blob_id": compact_blob,
                            "raw_curve_matches_retained_compact": True,
                        },
                        "resume": resume,
                    }
                )
                key = f"{method_key}_seed_{seed}"
                history_validation[key] = {
                    "control_files_byte_identical": all(
                        history_file_parity.values()
                    ),
                    "raw_curve_matches_retained_compact": True,
                    "retained_commit": commit,
                }
                grid_validation[key] = {
                    "native_points": len(actual_grid),
                    "missing_native_steps": [
                        step for step in expected_native_grid(
                            str(spec["backbone"]), seed
                        )
                        if step not in metrics.curve
                    ],
                    "table_grid_points": len(TABLE_GRID),
                    "table_grid_complete": True,
                    "summary_auc_matches_raw": True,
                }
                runtime_validation[key] = {
                    "segments": len(metrics.segments),
                    "active_runtime_seconds": active_runtime,
                    "recomputed_steps_per_second": recomputed_sps,
                    "recorded_summary_steps_per_second": summary_sps,
                    "raw_nul_padding_bytes": metrics.nul_padding_bytes,
                    "raw_nul_padding_rows": metrics.nul_padding_rows,
                    "passed": True,
                }
                resume_validation[key] = {
                    **resume,
                    "runtime_segment_count": len(metrics.segments),
                    "passed": True,
                }

        aggregate = aggregate_rows(per_seed)
        write_csv(
            artifact / "analysis" / "per_seed_metrics.csv",
            per_seed,
            (
                "backbone",
                "method",
                "seed",
                "utd",
                "trajectory_status",
                "resume_step",
                "native_grid_points",
                "native_auc",
                "common_grid_points",
                "common_grid_auc",
                "final_quarter_grid_points",
                "final_quarter_auc",
                "final_success",
                "hit90_step",
                "active_runtime_segments",
                "active_runtime_seconds",
                "recomputed_steps_per_second",
                "summary_steps_per_second",
                "raw_nul_padding_bytes",
                "raw_nul_padding_rows",
            ),
        )
        aggregate_fields = (
            "backbone",
            "method",
            "utd",
            "n_seeds",
            "native_auc_mean",
            "native_auc_sample_sd",
            "common_grid_auc_mean",
            "common_grid_auc_sample_sd",
            "final_quarter_auc_mean",
            "final_quarter_auc_sample_sd",
            "final_success_mean",
            "final_success_sample_sd",
            "hit90_reached",
            "hit90_total",
            "hit90_median_reached",
            "recomputed_steps_per_second_mean",
            "recomputed_steps_per_second_sample_sd",
            "summary_steps_per_second_mean",
            "summary_steps_per_second_sample_sd",
        )
        write_csv(
            artifact / "analysis" / "aggregate_metrics.csv",
            aggregate,
            aggregate_fields,
        )
        write_csv(
            artifact / "analysis" / "common_grid_curves.csv",
            common_rows,
            ("backbone", "method", "seed", "step", "success"),
        )
        mean_curve_rows = []
        for backbone in ("DDPG", "TD3"):
            for step in TABLE_GRID:
                values = [
                    float(row["success"])
                    for row in common_rows
                    if row["backbone"] == backbone
                    and int(row["step"]) == step
                ]
                mean, sample_sd = mean_sd(values)
                mean_curve_rows.append(
                    {
                        "backbone": backbone,
                        "method": "HER high-RR",
                        "step": step,
                        "mean_success": mean,
                        "sample_sd": sample_sd,
                        "n_seeds": len(values),
                    }
                )
        write_csv(
            artifact / "analysis" / "mean_common_grid_curve.csv",
            mean_curve_rows,
            (
                "backbone",
                "method",
                "step",
                "mean_success",
                "sample_sd",
                "n_seeds",
            ),
        )

        source_root = artifact / "source"
        source_root.mkdir()
        source_notes = [
            "# Retained launch snapshots",
            "",
            "These files document the launch packages found on the execution "
            "machine. DDPG's exact execution source commit was not retained. "
            "TD3's first retained results commit includes its frozen launch "
            "package, but the exact launch commit was not recorded. These "
            "snapshots are provenance evidence, not a claim of byte-exact "
            "DDPG execution-source recovery.",
            "",
        ]
        (source_root / "README.md").write_text(
            "\n".join(source_notes), encoding="utf-8", newline="\n"
        )
        launch_records: dict[str, list[dict[str, object]]] = {}
        for method_key, spec in METHOD_SPECS.items():
            launch_root = raw_repo / str(spec["launch_root"])
            files = sorted(
                path for path in launch_root.iterdir()
                if path.is_file()
                and (
                    path.name in {"README.md", "config.json", "run.py"}
                    or path.suffix == ".sh"
                )
            )
            launch_records[method_key] = []
            for source in files:
                packaged = (
                    Path("source") / method_key / source.name
                )
                sanitized_copy_text(
                    source, artifact / packaged, raw_repo
                )
                launch_records[method_key].append(
                    file_record(
                        source,
                        logical_raw_path(source, raw_repo),
                        role="launch_snapshot",
                        packaged_path=packaged.as_posix(),
                    )
                )

        code_root = artifact / "code"
        code_root.mkdir()
        for name in CODE_FILES:
            source = HERE / name
            if not source.is_file():
                raise ArtifactError(f"missing artifact code: {source}")
            shutil.copy2(source, code_root / name)

        write_json(
            artifact / "provenance.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_id": ARTIFACT_ID,
                "source_machine_label": "host_a",
                "task": "FetchPickAndPlace-v4",
                "reward": "sparse",
                "seeds": list(SEEDS),
                "steps_per_seed": TOTAL_STEPS,
                "methods": {
                    key: {
                        "backbone": spec["backbone"],
                        "variant": "HER high-RR x4",
                        "updates_per_step": 4,
                        "trajectory_status": (
                            "completed_uninterrupted"
                            if key == "ddpg"
                            else "completed_resumed"
                        ),
                        "execution_source_commit": (
                            spec["execution_source_commit"]
                            or "not retained"
                        ),
                        "first_retained_results_commit": (
                            spec["first_retained_results_commit"]
                        ),
                        "source_retention": spec["source_retention"],
                    }
                    for key, spec in METHOD_SPECS.items()
                },
                "td3_resume_points": {
                    str(seed): METHOD_SPECS["td3"]["resume_steps"][seed]
                    for seed in SEEDS
                },
                "metric_contract": {
                    "native_auc": (
                        "normalized trapezoidal AUC on each retained exact "
                        "10k evaluation grid"
                    ),
                    "table_common_grid": list(TABLE_GRID),
                    "table_grid_exclusion": {
                        "step": 800_000,
                        "reason": (
                            "TD3 High-RR seeds 1, 3, and 4 did not retain "
                            "an 800k evaluation; no value was interpolated"
                        ),
                    },
                    "final_quarter_grid": list(
                        FINAL_QUARTER_GRID
                    ),
                    "steps_per_second": (
                        "1,000,000 divided by the sum of monotonic "
                        "runtime/total_sec segment maxima; reboot downtime "
                        "and unretained discarded work are excluded"
                    ),
                },
                "raw_integrity": {
                    "nul_padding_policy": (
                        "NUL bytes are accepted only as leading padding in "
                        "the step field, removed before numeric parsing, and "
                        "recorded per seed; any NUL in another field fails"
                    ),
                    "per_seed": {
                        f"{str(row['backbone']).lower()}_seed_{row['seed']}": {
                            "nul_padding_bytes": row["raw_nul_padding_bytes"],
                            "affected_rows": row["raw_nul_padding_rows"],
                        }
                        for row in per_seed
                    },
                },
                "limitations": [
                    "DDPG execution source commit was not retained.",
                    (
                        "TD3 used checkpoint continuation; equivalence to "
                        "an uninterrupted trajectory is not claimed."
                    ),
                    (
                        "TD3 seeds 1, 3, and 4 lack an 800k evaluation; "
                        "the shared Table 16 grid excludes 800k for every "
                        "method and seed."
                    ),
                    (
                        "Recomputed throughput sums retained active-runtime "
                        "segments and excludes reboot downtime and any "
                        "discarded work not present in the retained metrics."
                    ),
                ],
            },
        )
        write_json(
            artifact / "source_inputs_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_id": ARTIFACT_ID,
                "path_policy": (
                    "machine-local paths are anonymized and repository "
                    "relative; no hostname, username, IP, UUID, serial, or "
                    "absolute home path is retained"
                ),
                "raw_runs": source_runs,
                "launch_snapshots": launch_records,
            },
        )
        write_json(
            artifact / "validation.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_id": ARTIFACT_ID,
                "status": "passed_with_caveats",
                "gates": {
                    "paired_seed_set": {
                        "expected": list(SEEDS),
                        "actual_by_backbone": {
                            backbone: [
                                int(row["seed"])
                                for row in per_seed
                                if row["backbone"] == backbone
                            ]
                            for backbone in ("DDPG", "TD3")
                        },
                        "passed": True,
                    },
                    "completed_non_smoke_1m": {
                        "steps": TOTAL_STEPS,
                        "seeds_per_backbone": 5,
                        "passed": True,
                    },
                    "retained_git_history_parity": history_validation,
                    "exact_evaluation_grids": grid_validation,
                    "runtime_recalculation": runtime_validation,
                    "resume_provenance": resume_validation,
                    "standard_json_and_public_path_sanitization": {
                        "passed": True
                    },
                },
                "caveats": [
                    "DDPG execution source commit not retained.",
                    (
                        "TD3 completed via checkpoint continuation; "
                        "uninterrupted-trajectory equivalence is uncertain."
                    ),
                    (
                        "The common table grid excludes 800k because that "
                        "evaluation is absent for TD3 seeds 1, 3, and 4."
                    ),
                ],
            },
        )
        (artifact / "README.md").write_text(
            "# FPP DDPG/TD3 HER High-RR x4 (1M)\n\n"
            "Compact, self-verifying seeds 1–5 artifact for the FPP DDPG "
            "and TD3 HER update-ratio-four runs. It contains sanitized "
            "configurations, environment snapshots, summaries, exact "
            "evaluation curves, runtime segments, recomputed FQ-AUC and "
            "throughput, retained launch snapshots, and cryptographic "
            "source bindings. Raw metrics, logs, checkpoints, replay "
            "buffers, and model weights are excluded.\n\n"
            "DDPG completed without resume. TD3 resumed after reboot at "
            "step 800001 for seeds 1–4 and at step 630001 from "
            "`best_checkpoint.pt` for seed 5. Exact equivalence to an "
            "uninterrupted TD3 trajectory is not claimed. TD3 seeds 1, 3, "
            "and 4 lack an 800k evaluation, so the Table 16 common grid is "
            "the exact 50k–1M grid with 800k removed for every row; no "
            "interpolation, smoothing, extrapolation, or forward-fill is "
            "used.\n\n"
            "The prior TD3 summary throughput used only the post-resume "
            "runtime counter. `analysis/per_seed_metrics.csv` instead "
            "computes steps/s from the sum of retained monotonic runtime "
            "segments. Reboot downtime and unretained discarded work are "
            "not estimated.\n\n"
            "Verify from this directory with "
            "`python3 code/verify_fpp_high_rr_compact_artifact.py "
            "--artifact-root .`.\n",
            encoding="utf-8",
            newline="\n",
        )

        manifest_files = sorted(
            path for path in artifact.rglob("*")
            if path.is_file()
            and path.name not in {"artifact_manifest.json", "SHA256SUMS"}
        )
        write_json(
            artifact / "artifact_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_id": ARTIFACT_ID,
                "manifest_self_excluded": True,
                "sha256sums_self_excluded": True,
                "seeds": list(SEEDS),
                "backbones": ["DDPG", "TD3"],
                "table_common_grid": list(TABLE_GRID),
                "validation_status": "passed_with_caveats",
                "file_count": len(manifest_files),
                "total_size_bytes": sum(
                    path.stat().st_size for path in manifest_files
                ),
                "files": [
                    {
                        "path": path.relative_to(artifact).as_posix(),
                        "sha256": sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in manifest_files
                ],
            },
        )
        checksum_files = sorted(
            path for path in artifact.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        )
        (artifact / "SHA256SUMS").write_text(
            "".join(
                f"{sha256(path)}  {path.relative_to(artifact).as_posix()}\n"
                for path in checksum_files
            ),
            encoding="utf-8",
            newline="\n",
        )

        if output.exists():
            shutil.rmtree(output)
        os.replace(artifact, output)

    print(f"Built {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-repo",
        type=Path,
        default=REPOSITORY_ROOT.parent,
    )
    parser.add_argument(
        "--history-repo",
        type=Path,
        default=None,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    history_repo = arguments.history_repo or arguments.raw_repo
    build(arguments.raw_repo, history_repo, arguments.output)


if __name__ == "__main__":
    main()
