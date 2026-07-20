#!/usr/bin/env python3
"""Build the compact FPP DDPG SABER Replay-Action Query paper artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from query_ablation import (
    ARTIFACT_ID,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DIAGNOSTIC_FIELDS,
    EVAL_FREQ,
    GRID,
    MAIN_KEY,
    PAIR_CONFIG_EXEMPT_KEYS,
    REPLAY_KEY,
    SCHEMA_VERSION,
    SEEDS,
    TOTAL_STEPS,
    TOTAL_UPDATES,
    AnalysisBundle,
    AnalysisError,
    RunResult,
    analyze_query_ablation,
    config_differences,
    find_repo_root,
    read_json,
    write_outputs,
)


HEX40 = re.compile(r"^[0-9a-f]{40}$")
EXPERIMENT_REL = Path("experiment_FPP/ddpg_saber_query_replay_ls512")
CANONICAL_SOURCE_REL = Path("experiment_FPP/b3_critic_only_runtime_opt_v4_ls512")
ARTIFACT_REL = Path("experiment_FPP/paper_artifacts") / ARTIFACT_ID
EXECUTION_METADATA_REL = Path("_execution/launch_metadata.json")
ANALYSIS_FILES = (
    "query_ablation.py",
    "analyze_results.py",
    "build_artifact.py",
    "verify_artifact.py",
)
TEST_FILES = ("test_analysis.py", "test_artifact.py")
SEED_FILES = frozenset(
    {"config.json", "env_snapshot.json", "summary.json", "metrics_selected_10k.csv"}
)

COMPACT_METRIC_COLUMNS = (
    "step",
    "eval/return_mean",
    "eval/success_mean",
    "eval/goal_success_mean",
    "eval/env_success_mean",
    "eval/episode_length_mean",
    "eval/final_success",
    "eval/best_success",
    "eval/AUC_success",
    "eval/first_hit_0.90",
    "config/weight_scope",
    "config/support_query_mode",
    "runtime/steps_per_sec",
    "runtime/total_sec",
    *DIAGNOSTIC_FIELDS,
    "lite/psi_q_rank_rate",
    "lite/mean_w_P",
    "lite/std_w_P",
    "lite/ESS_w",
    "b3/support_by_phase_approach",
    "b3/support_by_phase_grasp",
    "b3/support_by_phase_lift",
    "b3/support_by_phase_place",
)

FORBIDDEN_SOURCE_PARTS = frozenset({"__pycache__", ".pytest_cache"})


class ArtifactBuildError(AnalysisError):
    """Raised when provenance or artifact construction fails closed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise ArtifactBuildError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def source_commit_for_path(repo: Path, relative: Path) -> str:
    commit = _git(repo, "log", "-1", "--format=%H", "--", relative.as_posix())
    if not HEX40.fullmatch(commit):
        raise ArtifactBuildError(f"Could not resolve source commit for {relative}: {commit!r}")
    return commit


def resolve_execution_source_commit(
    repo: Path, supplied: str | None, *, require_clean: bool
) -> str:
    commit = supplied or _git(repo, "rev-parse", "HEAD")
    if not HEX40.fullmatch(commit):
        raise ArtifactBuildError(f"Execution source commit must be a full SHA: {commit!r}")
    tracked = _git(repo, "ls-tree", "-r", "--name-only", commit, "--", EXPERIMENT_REL.as_posix())
    if not tracked:
        raise ArtifactBuildError(f"Experiment source is absent from commit {commit}")
    if require_clean:
        dirty = _git(repo, "status", "--porcelain", "--", EXPERIMENT_REL.as_posix())
        if dirty:
            raise ArtifactBuildError(f"Experiment source has uncommitted changes:\n{dirty}")
    return commit


def logical_path(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def source_record(
    path: Path,
    repo: Path,
    *,
    role: str,
    packaged_path: str | None = None,
    transformations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": role,
        "source_path": logical_path(path, repo),
        "source_sha256": sha256(path),
        "source_size_bytes": path.stat().st_size,
    }
    if packaged_path:
        result["packaged_path"] = packaged_path
    if transformations is not None:
        result["transformations"] = dict(transformations)
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _json_path(parent: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{parent}[{key}]"
    escaped = key.replace("~", "~0").replace("/", "~1")
    return f"{parent}/{escaped}"


def sanitize_json(
    value: Any,
    *,
    replacements: Sequence[tuple[str, str]],
    path: str = "$",
    nonfinite_paths: list[str] | None = None,
) -> Any:
    if nonfinite_paths is None:
        nonfinite_paths = []
    if isinstance(value, Mapping):
        return {
            str(key): sanitize_json(
                item,
                replacements=replacements,
                path=_json_path(path, str(key)),
                nonfinite_paths=nonfinite_paths,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            sanitize_json(
                item,
                replacements=replacements,
                path=_json_path(path, index),
                nonfinite_paths=nonfinite_paths,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite_paths.append(path)
        return None
    if isinstance(value, str):
        for before, after in replacements:
            if before:
                value = value.replace(before, after)
    return value


def copy_sanitized_json(
    source: Path,
    destination: Path,
    repo: Path,
    run_dir: Path,
) -> dict[str, Any]:
    payload = read_json(source)
    paths: list[str] = []
    replacements = (
        (run_dir.resolve().as_posix(), "${SOURCE_RUN}"),
        (repo.resolve().as_posix(), "${REPO_ROOT}"),
        (Path.home().resolve().as_posix(), "${HOME}"),
        (socket.gethostname(), "${HOST}"),
    )
    sanitized = sanitize_json(
        payload,
        replacements=replacements,
        nonfinite_paths=paths,
    )
    write_json(destination, sanitized)
    return {
        "machine_specific_strings_sanitized": True,
        "nonfinite_json_numbers_replaced_with_null": len(paths),
        "nonfinite_json_paths": sorted(paths),
    }


def copy_compact_metrics(run: RunResult, destination: Path) -> dict[str, Any]:
    available = set(run.metrics_rows[0])
    fields = [field for field in COMPACT_METRIC_COLUMNS if field in available]
    if "step" not in fields or not ({"eval/env_success_mean", "eval/success_mean"} & set(fields)):
        raise ArtifactBuildError(f"Required compact metric columns absent from {run.metrics_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    optional_nonfinite_count = 0
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in run.metrics_rows:
            output: dict[str, str] = {}
            for field in fields:
                value = row.get(field, "")
                if value not in (None, ""):
                    try:
                        parsed = float(value)
                    except (TypeError, ValueError):
                        parsed = None
                    if parsed is not None and not math.isfinite(parsed):
                        value = ""
                        optional_nonfinite_count += 1
                output[field] = "" if value is None else str(value)
            writer.writerow(output)
    return {
        "exact_evaluation_rows_selected": len(run.metrics_rows),
        "output_columns": fields,
        "optional_nonfinite_csv_cells_replaced_with_blank": optional_nonfinite_count,
        "numeric_values_otherwise_unchanged": True,
    }


def package_run(
    run: RunResult,
    destination: Path,
    packaged_rel: Path,
    repo: Path,
) -> dict[str, Any]:
    source_paths = {
        "config": run.run_dir / "config.json",
        "env_snapshot": run.run_dir / "env_snapshot.json",
        "summary": run.run_dir / "summary.json",
        "metrics": run.metrics_path,
    }
    records: list[dict[str, Any]] = []
    for role in ("config", "env_snapshot", "summary"):
        packaged_path = packaged_rel / f"{role}.json"
        transformations = copy_sanitized_json(
            source_paths[role], destination / f"{role}.json", repo, run.run_dir
        )
        records.append(
            source_record(
                source_paths[role],
                repo,
                role=role,
                packaged_path=packaged_path.as_posix(),
                transformations=transformations,
            )
        )
    metric_rel = packaged_rel / "metrics_selected_10k.csv"
    metric_transform = copy_compact_metrics(run, destination / metric_rel.name)
    records.append(
        source_record(
            source_paths["metrics"],
            repo,
            role="metrics",
            packaged_path=metric_rel.as_posix(),
            transformations=metric_transform,
        )
    )
    actual = {path.name for path in destination.iterdir() if path.is_file()}
    if actual != SEED_FILES:
        raise ArtifactBuildError(
            f"Seed payload mismatch at {destination}: expected={sorted(SEED_FILES)}, actual={sorted(actual)}"
        )
    return {
        "seed": run.seed,
        "support_query_mode": run.query_mode,
        "source_run_dir": logical_path(run.run_dir, repo),
        "packaged_run_dir": packaged_rel.as_posix(),
        "source_metrics_kind": run.metrics_path.name,
        "files": records,
    }


def frozen_source_files(root: Path) -> list[Path]:
    result: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if "analysis" in relative.parts or any(part in FORBIDDEN_SOURCE_PARTS for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pt", ".pth", ".ckpt"}:
            continue
        result.append(path)
    required = {
        root / "config.json",
        root / "run.py",
        root / "code/src/train_fetch_pick_and_place_lite_v0.py",
        root / "code/src/metacog_lite_v0/weights.py",
    }
    if not required.issubset(set(result)):
        raise ArtifactBuildError(f"Frozen execution source missing required files: {required - set(result)}")
    return result


def copy_frozen_source(root: Path, destination: Path, repo: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in frozen_source_files(root):
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            source_record(
                source,
                repo,
                role="execution_source",
                packaged_path=(Path("code/execution_source") / relative).as_posix(),
            )
        )
    return records


def copy_analysis_and_tests(
    analysis_root: Path, stage: Path, repo: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    analysis_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []
    for name in ANALYSIS_FILES:
        source = analysis_root / name
        if not source.is_file():
            raise ArtifactBuildError(f"Missing analysis tool {source}")
        target = stage / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        analysis_records.append(
            source_record(source, repo, role="analysis_code", packaged_path=f"code/{name}")
        )
    for name in TEST_FILES:
        source = analysis_root / name
        if not source.is_file():
            raise ArtifactBuildError(f"Missing analysis/artifact test {source}")
        target = stage / "tests" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        test_records.append(
            source_record(source, repo, role="artifact_test", packaged_path=f"tests/{name}")
        )
    return analysis_records, test_records


def validate_execution_metadata(
    replay_root: Path, expected_commit: str
) -> tuple[Path, dict[str, Any]]:
    path = replay_root / EXECUTION_METADATA_REL
    payload = read_json(path)
    expected = {
        "experiment": "ddpg_saber_query_replay_ls512",
        "source_commit": expected_commit,
        "origin_main": expected_commit,
        "head_matches_origin_main": True,
        "worktree_clean": True,
        "steps": TOTAL_STEPS,
        "seeds": list(SEEDS),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ArtifactBuildError(
                f"{path}: {key} expected {value!r}, got {payload.get(key)!r}"
            )
    if payload.get("support_query_mode") not in (None, "replay_action"):
        raise ArtifactBuildError(f"{path}: support_query_mode is not replay_action")
    return path, payload


def run_prepackage_tests(
    experiment_root: Path,
    analysis_root: Path,
    *,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"schema_version": 1, "status": "synthetic-test-bypass", "commands": []}
    candidates = [
        analysis_root / "test_analysis.py",
        analysis_root / "test_artifact.py",
        experiment_root / "code/src/test_support_query_modes.py",
        experiment_root / "code/src/test_ddpg_query_loss_isolation.py",
        experiment_root / "code/src/test_current_policy_legacy_parity.py",
        experiment_root / "validate_config_parity.py",
        experiment_root / "code/src/test_numba_support_scan_fpp.py",
        experiment_root / "code/src/test_numba_runtime_opt_fpp.py",
    ]
    required = candidates[:6]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise ArtifactBuildError(f"Required prepackage tests are missing: {missing}")
    commands: list[dict[str, Any]] = []
    replacements = (
        (experiment_root.resolve().as_posix(), "${EXPERIMENT_ROOT}"),
        (experiment_root.parents[1].resolve().as_posix(), "${REPO_ROOT}"),
        (Path.home().resolve().as_posix(), "${HOME}"),
        (socket.gethostname(), "${HOST}"),
    )
    for path in candidates:
        if not path.is_file():
            continue
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=experiment_root.parents[1],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        stdout, stderr = result.stdout, result.stderr
        for before, after in replacements:
            stdout = stdout.replace(before, after)
            stderr = stderr.replace(before, after)
        record = {
            "test": path.name,
            "command": ["python", logical_path(path, experiment_root.parents[1])],
            "returncode": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "passed": result.returncode == 0,
        }
        commands.append(record)
        if result.returncode:
            raise ArtifactBuildError(
                f"Prepackage test failed: {path}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )
    return {"schema_version": 1, "status": "passed", "commands": commands}


def validation_payload(
    bundle: AnalysisBundle,
    execution_source_commit: str,
    canonical_source_commit: str,
    execution_metadata: Mapping[str, Any],
    test_evidence: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    run_map = {
        key: {str(run.seed): run for run in bundle.runs_by_method[key]}
        for key in (MAIN_KEY, REPLAY_KEY)
    }
    actual_differences = {
        str(seed): [key for key, _left, _right in config_differences(
            run_map[MAIN_KEY][str(seed)].config,
            run_map[REPLAY_KEY][str(seed)].config,
        )]
        for seed in bundle.seeds
    }
    replay_matches = {
        str(run.seed): {
            "mean": run.diagnostics["query/query_bin_matches_replay_action_rate"].mean,
            "finite_checkpoint_count": run.diagnostics[
                "query/query_bin_matches_replay_action_rate"
            ].finite_count,
        }
        for run in bundle.runs_by_method[REPLAY_KEY]
    }
    nonfinite_replacements = 0
    nonfinite_paths: dict[str, list[str]] = {}
    methods = input_manifest["methods"]
    for method_key in (MAIN_KEY, REPLAY_KEY):
        for run in methods[method_key]["runs"]:
            for record in run["files"]:
                transform = record.get("transformations", {})
                count = int(transform.get("nonfinite_json_numbers_replaced_with_null", 0))
                nonfinite_replacements += count
                if count:
                    nonfinite_paths[record["packaged_path"]] = list(
                        transform["nonfinite_json_paths"]
                    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": ARTIFACT_ID,
        "status": "passed",
        "execution_source_commit": execution_source_commit,
        "canonical_main_source_commit": canonical_source_commit,
        "gates": {
            "seed_coverage": {
                "expected": list(SEEDS),
                "actual_by_method": {MAIN_KEY: list(bundle.seeds), REPLAY_KEY: list(bundle.seeds)},
                "passed": True,
            },
            "completion": {
                "expected_total_steps": TOTAL_STEPS,
                "expected_total_updates": TOTAL_UPDATES,
                "actual_total_steps_by_method_seed": {
                    key: {seed: run.config["total_steps"] for seed, run in runs.items()}
                    for key, runs in run_map.items()
                },
                "actual_total_updates_by_method_seed": {
                    key: {seed: run.summary["total_updates"] for seed, run in runs.items()}
                    for key, runs in run_map.items()
                },
                "passed": True,
            },
            "exact_evaluation_grid": {
                "expected_steps": list(GRID),
                "points": len(GRID),
                "actual_steps_by_method_seed": {
                    key: {
                        seed: [int(row["step"]) for row in run.metrics_rows]
                        for seed, run in runs.items()
                    }
                    for key, runs in run_map.items()
                },
                "interpolation": False,
                "smoothing": False,
                "extrapolation": False,
                "forward_fill": False,
                "passed": True,
            },
            "query_modes": {
                "expected": {MAIN_KEY: "current_policy", REPLAY_KEY: "replay_action"},
                "actual": {
                    key: {seed: run.query_mode for seed, run in runs.items()}
                    for key, runs in run_map.items()
                },
                "passed": True,
            },
            "config_parity": {
                "only_allowed_resolved_differences": sorted(PAIR_CONFIG_EXEMPT_KEYS),
                "actual_differences_by_seed": actual_differences,
                "passed": True,
            },
            "replay_query_bin_matches_replay_action": {
                "expected_rate": 1.0,
                "expected_checkpoint_count": len(GRID),
                "actual_by_seed": replay_matches,
                "passed": True,
            },
            "execution_provenance": {
                "source_commit": execution_metadata["source_commit"],
                "origin_main": execution_metadata["origin_main"],
                "head_matches_origin_main": execution_metadata["head_matches_origin_main"],
                "worktree_clean": execution_metadata["worktree_clean"],
                "passed": True,
            },
            "current_policy_legacy_parity_evidence": {
                "test": "test_current_policy_legacy_parity.py",
                "test_suite_status": test_evidence["status"],
                "passed": test_evidence["status"] in {"passed", "synthetic-test-bypass"},
            },
            "json_sanitization": {
                "optional_nonfinite_values_replaced_with_null": nonfinite_replacements,
                "paths_by_file": nonfinite_paths,
                "required_fields_validated_before_sanitization": True,
                "passed": True,
            },
            "excluded_payloads": {
                "checkpoints": True,
                "full_per_update_metrics": True,
                "jsonl": True,
                "logs": True,
                "smoke": True,
                "passed": True,
            },
        },
        "self_support_limitation": (
            "Replay-action querying retains the sampled transition in the local replay pool "
            "and can therefore self-support."
        ),
    }


def artifact_readme(
    execution_source_commit: str,
    canonical_source_commit: str,
    bundle: AnalysisBundle,
) -> str:
    inference = bundle.inference_rows[0]
    return f"""# FPP DDPG SABER Replay-Action Query LS512 (1M)

Compact matched seeds 1-5 paper artifact.

- Execution source commit: `{execution_source_commit}`
- Canonical Main source commit: `{canonical_source_commit}`
- Scientific factor: `support_query_mode: current_policy -> replay_action`
- Backbone/scope: DDPG, HER=0.8, SABER critic-only, LS512

The primary endpoint is normalized trapezoidal environment-success AUC on the
exact observed 10k, 20k, ..., 1M grid (100 points; denominator 990,000). The
fallback success column is used only when the environment-success value is
absent/non-finite. There is no interpolation, smoothing, extrapolation, or
forward filling.

Main minus Replay mean AUC delta is `{inference['mean_delta']:+.6f}` with paired
percentile-bootstrap 95% CI `[{inference['bootstrap_ci95_low']:.6f},
{inference['bootstrap_ci95_high']:.6f}]` (10,000 resamples, RNG seed 27027).
The exact two-sided sign-flip test enumerates all 32 assignments.

Replay-action querying retains the sampled transition in the local replay pool and can therefore self-support.

## Contents

- `code/`: frozen execution source plus analysis/build/verification code.
- `data/main/`, `data/replay/`: sanitized per-seed config, environment snapshot,
  summary, and exact-grid compact metrics only.
- `results/`: exact analysis tables, query diagnostics, validation, and report.
- `tests/`: analysis/artifact tests and captured prepackage test evidence.
- `input_manifest.json`: input provenance, source hashes, and sanitization record.
- `artifact_manifest.json` and `SHA256SUMS`: complete packaged integrity records.

Large checkpoints, full per-update metrics, JSONL, raw logs, and smoke outputs are
excluded.

Verify from a copied artifact, without the repository:

```bash
python3 code/verify_artifact.py --artifact-root .
```

Regenerate analysis from packaged inputs:

```bash
python3 code/analyze_results.py --main-root data/main --replay-root data/replay --output-dir /tmp/query-results
```
"""


def all_files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def write_manifests(
    root: Path,
    execution_source_commit: str,
    canonical_source_commit: str,
) -> None:
    excluded = {"artifact_manifest.json", "SHA256SUMS"}
    files = {
        name: path for name, path in all_files(root).items() if name not in excluded
    }
    entries = [
        {"path": name, "sha256": sha256(path), "size_bytes": path.stat().st_size}
        for name, path in files.items()
    ]
    write_json(
        root / "artifact_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_id": ARTIFACT_ID,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution_source_commit": execution_source_commit,
            "canonical_main_source_commit": canonical_source_commit,
            "method_seed_coverage": {MAIN_KEY: list(SEEDS), REPLAY_KEY: list(SEEDS)},
            "evaluation": {
                "steps": list(GRID),
                "checkpoint_count": len(GRID),
                "auc_normalization_denominator": GRID[-1] - GRID[0],
            },
            "bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES,
                "rng_seed": BOOTSTRAP_SEED,
            },
            "exact_sign_flip_assignments": 32,
            "manifest_self_excluded": True,
            "sha256sums_self_excluded": True,
            "file_count": len(entries),
            "total_size_bytes": sum(entry["size_bytes"] for entry in entries),
            "files": entries,
        },
    )
    sums = {
        name: path for name, path in all_files(root).items() if name != "SHA256SUMS"
    }
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {name}\n" for name, path in sums.items()),
        encoding="utf-8",
        newline="\n",
    )


def build_artifact(
    *,
    repo: Path,
    main_root: Path,
    replay_root: Path,
    artifact_root: Path,
    experiment_root: Path,
    analysis_root: Path,
    execution_source_commit: str,
    canonical_source_commit: str,
    replace: bool = False,
    run_tests: bool = True,
    require_execution_metadata: bool = True,
    allow_synthetic_verification: bool = False,
) -> dict[str, Any]:
    repo = repo.resolve()
    artifact_root = artifact_root.resolve()
    if artifact_root.exists() and not replace:
        raise ArtifactBuildError(f"Artifact already exists: {artifact_root}; use --replace explicitly")
    bundle = analyze_query_ablation(main_root, replay_root)
    if require_execution_metadata:
        execution_path, execution_metadata = validate_execution_metadata(
            replay_root, execution_source_commit
        )
    else:
        execution_path = replay_root / EXECUTION_METADATA_REL
        execution_metadata = read_json(execution_path)
    test_evidence = run_prepackage_tests(
        experiment_root, analysis_root, enabled=run_tests
    )
    artifact_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{ARTIFACT_ID}.tmp-", dir=artifact_root.parent))
    try:
        method_records: dict[str, Any] = {}
        for method_key in (MAIN_KEY, REPLAY_KEY):
            run_records = []
            for run in bundle.runs_by_method[method_key]:
                relative = Path("data") / method_key / f"seed_{run.seed}"
                run_records.append(package_run(run, stage / relative, relative, repo))
            method_records[method_key] = {
                "label": bundle.runs_by_method[method_key][0].method,
                "support_query_mode": bundle.runs_by_method[method_key][0].query_mode,
                "source_commit": (
                    canonical_source_commit if method_key == MAIN_KEY else execution_source_commit
                ),
                "seeds": list(SEEDS),
                "runs": run_records,
            }

        packaged_bundle = analyze_query_ablation(
            stage / "data/main", stage / "data/replay"
        )
        write_outputs(packaged_bundle, stage / "results")
        analysis_records, test_records = copy_analysis_and_tests(analysis_root, stage, repo)
        execution_records = copy_frozen_source(
            experiment_root, stage / "code/execution_source", repo
        )
        execution_transform = copy_sanitized_json(
            execution_path,
            stage / "code/execution_metadata.json",
            repo,
            replay_root,
        )
        execution_record = source_record(
            execution_path,
            repo,
            role="execution_metadata",
            packaged_path="code/execution_metadata.json",
            transformations=execution_transform,
        )
        write_json(stage / "tests/test_evidence.json", test_evidence)
        (stage / "source_commit.txt").write_text(
            execution_source_commit + "\n", encoding="utf-8", newline="\n"
        )
        input_manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_id": ARTIFACT_ID,
            "repository": {"id": "Hyunwoojooo/saber-rl", "branch": "main"},
            "execution_source_commit": execution_source_commit,
            "canonical_main_source_commit": canonical_source_commit,
            "methods": method_records,
            "execution_metadata": execution_record,
            "frozen_execution_source_files": execution_records,
            "analysis_code_files": analysis_records,
            "artifact_test_files": test_records,
            "transforms": {
                "json": (
                    "Machine-specific strings are tokenized; optional NaN/Infinity values are "
                    "replaced by null and every path is recorded per input file."
                ),
                "metrics": (
                    "Only the exact 100 evaluation rows and a deterministic compact column set "
                    "are retained; optional non-finite cells become blank."
                ),
                "numeric_core_values": "Validated before sanitization and unchanged.",
            },
            "excluded": {
                "checkpoints": True,
                "full_per_update_metrics": True,
                "jsonl": True,
                "raw_logs": True,
                "smoke_outputs": True,
            },
        }
        write_json(stage / "input_manifest.json", input_manifest)
        write_json(
            stage / "results/validation.json",
            validation_payload(
                packaged_bundle,
                execution_source_commit,
                canonical_source_commit,
                execution_metadata,
                test_evidence,
                input_manifest,
            ),
        )
        (stage / "README.md").write_text(
            artifact_readme(execution_source_commit, canonical_source_commit, packaged_bundle),
            encoding="utf-8",
            newline="\n",
        )
        write_manifests(stage, execution_source_commit, canonical_source_commit)

        # Import only after the staged verifier has been copied.  Its public
        # function is shared with the command-line standalone entry point.
        sys.path.insert(0, str(analysis_root))
        try:
            from verify_artifact import verify_artifact

            result = verify_artifact(
                stage, allow_synthetic_test_bypass=allow_synthetic_verification
            )
        finally:
            sys.path.pop(0)
        if artifact_root.exists():
            shutil.rmtree(artifact_root)
        os.replace(stage, artifact_root)
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    repo = find_repo_root(Path(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-root",
        type=Path,
        default=repo / "experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512",
    )
    parser.add_argument(
        "--replay-root",
        type=Path,
        default=repo / "experiment_FPP/runs/ddpg_saber_query_replay_ls512",
    )
    parser.add_argument("--artifact-root", type=Path, default=repo / ARTIFACT_REL)
    parser.add_argument("--source-commit")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        repo = find_repo_root(Path(__file__))
        source_commit = resolve_execution_source_commit(
            repo, args.source_commit, require_clean=True
        )
        canonical_commit = source_commit_for_path(repo, CANONICAL_SOURCE_REL)
        result = build_artifact(
            repo=repo,
            main_root=args.main_root,
            replay_root=args.replay_root,
            artifact_root=args.artifact_root,
            experiment_root=repo / EXPERIMENT_REL,
            analysis_root=Path(__file__).resolve().parent,
            execution_source_commit=source_commit,
            canonical_source_commit=canonical_commit,
            replace=args.replace,
        )
    except (AnalysisError, OSError, subprocess.SubprocessError) as exc:
        print(f"artifact build failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
