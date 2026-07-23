#!/usr/bin/env python3
"""Build the compact, fail-closed actor-only SABER paper artifact."""

from __future__ import annotations

import argparse
import csv
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
from typing import Any, Sequence

from scope_ablation import (
    ACTOR_KEY,
    ARTIFACT_ID,
    CRITIC_KEY,
    GRID,
    SEEDS,
    TOTAL_STEPS,
    TOTAL_UPDATES,
    AnalysisBundle,
    AnalysisError,
    RunResult,
    analyze_scope_ablation,
    find_repo_root,
    read_json,
    write_outputs,
)
from verify_artifact import CANONICAL_CRITIC_SOURCE_COMMIT, sha256, verify_artifact


HEX40 = re.compile(r"^[0-9a-f]{40}$")
EXPERIMENT_REL = Path("experiment_FPP/ddpg_saber_actor_only_v4_ls512")
AUDIT_CODE = ("scope_ablation.py", "analyze_results.py", "build_artifact.py", "verify_artifact.py")
SEED_FILES = {"config.json", "summary.json", "env_snapshot.json", "metrics_selected_10k.csv"}
EXECUTION_METADATA_REL = Path("_execution/launch_metadata.json")
COMPACT_COLUMNS = (
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
    "runtime/steps_per_sec",
    "runtime/total_sec",
)


class ArtifactBuildError(AnalysisError):
    pass


def validate_execution_metadata(actor_root: Path, source_commit: str) -> tuple[Path, dict[str, Any]]:
    """Bind the result rows to the exact clean origin/main commit used at launch."""
    path = actor_root / EXECUTION_METADATA_REL
    payload = read_json(path)
    expected = {
        "schema_version": 1,
        "experiment": "ddpg_saber_actor_only_v4_ls512",
        "repo_head": source_commit,
        "origin_main": source_commit,
        "head_matches_origin_main": True,
        "worktree_clean": True,
        "steps": TOTAL_STEPS,
        "seeds": list(SEEDS),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ArtifactBuildError(
                f"{path}: execution metadata {key} expected {value!r}, got {payload.get(key)!r}"
            )
    timestamp = payload.get("timestamp_utc")
    if not isinstance(timestamp, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", timestamp
    ):
        raise ArtifactBuildError(f"{path}: invalid timestamp_utc {timestamp!r}")
    max_parallel = payload.get("max_parallel")
    if not isinstance(max_parallel, int) or isinstance(max_parallel, bool) or max_parallel < 1:
        raise ArtifactBuildError(f"{path}: max_parallel must be a positive integer")
    return path, payload


def _git(repo: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", *args], cwd=repo, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if process.returncode:
        raise ArtifactBuildError(process.stderr.strip() or f"git {' '.join(args)} failed")
    return process.stdout.strip()


def resolve_source_commit(repo: Path, supplied: str | None, *, require_clean: bool) -> str:
    commit = supplied or _git(repo, "rev-parse", "HEAD")
    if not HEX40.fullmatch(commit):
        raise ArtifactBuildError(f"source commit must be a full 40-character SHA: {commit!r}")
    if require_clean:
        tracked = _git(repo, "ls-tree", "-r", "--name-only", commit, "--", str(EXPERIMENT_REL))
        if not tracked:
            raise ArtifactBuildError(f"experiment source is absent from commit {commit}")
        dirty = _git(repo, "status", "--porcelain", "--", str(EXPERIMENT_REL))
        if dirty:
            raise ArtifactBuildError(f"experiment source has uncommitted changes:\n{dirty}")
    return commit


def logical_path(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def record(path: Path, repo: Path, *, role: str | None = None, packaged: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": logical_path(path, repo), "sha256": sha256(path), "size_bytes": path.stat().st_size,
    }
    if role:
        value["role"] = role
    if packaged:
        value["packaged_path"] = packaged
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )


def sanitize(value: Any, replacements: Sequence[tuple[str, str]]) -> Any:
    if isinstance(value, dict):
        return {key: sanitize(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, replacements) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        # Raw summaries may contain undefined auxiliary diagnostics encoded as
        # non-standard JSON NaN/Infinity.  Keep the source hash in the input
        # manifest and normalize only the compact copy to standards-valid null.
        return None
    if isinstance(value, str):
        for before, after in replacements:
            if before:
                value = value.replace(before, after)
    return value


def copy_json(source: Path, destination: Path, repo: Path, run_dir: Path) -> None:
    replacements = (
        (run_dir.resolve().as_posix(), "${SOURCE_RUN}"),
        (repo.resolve().as_posix(), "${REPO_ROOT}"),
        (Path.home().as_posix(), "${HOME}"),
        (socket.gethostname(), "${HOST}"),
    )
    write_json(destination, sanitize(json.loads(source.read_text(encoding="utf-8")), replacements))


def copy_metrics(run: RunResult, destination: Path) -> None:
    fields = [name for name in COMPACT_COLUMNS if name in run.metrics_rows[0]]
    if not {"step", "eval/success_mean"}.issubset(fields):
        raise ArtifactBuildError(f"missing compact metric fields in {run.metrics_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in run.metrics_rows:
            writer.writerow({name: row.get(name, "") for name in fields})


def package_run(run: RunResult, destination: Path, repo: Path, packaged_rel: Path) -> dict[str, Any]:
    sources = {
        "config": run.run_dir / "config.json",
        "summary": run.run_dir / "summary.json",
        "env_snapshot": run.run_dir / "env_snapshot.json",
        "metrics": run.metrics_path,
    }
    for role in ("config", "summary", "env_snapshot"):
        copy_json(sources[role], destination / f"{role}.json", repo, run.run_dir)
    copy_metrics(run, destination / "metrics_selected_10k.csv")
    actual = {path.name for path in destination.iterdir() if path.is_file()}
    if actual != SEED_FILES:
        raise ArtifactBuildError(f"compact seed payload mismatch: {destination}: {sorted(actual)}")
    return {
        "seed": run.seed,
        "source_run_dir": logical_path(run.run_dir, repo),
        "packaged_run_dir": packaged_rel.as_posix(),
        "source_metrics_kind": run.metrics_path.name,
        "files": [record(sources[role], repo, role=role) for role in sources],
    }


def frozen_source_files(source_root: Path) -> list[Path]:
    result = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source_root)
        if "analysis" in relative.parts or "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        result.append(path)
    required = {source_root / "config.json", source_root / "run.py"}
    if not required.issubset(result) or not any(
        path.relative_to(source_root).as_posix().startswith("code/src/") for path in result
    ):
        raise ArtifactBuildError("frozen source must include config.json, run.py, and code/src/**")
    return result


def copy_frozen_source(source_root: Path, destination: Path, repo: Path) -> list[dict[str, Any]]:
    records = []
    for source in frozen_source_files(source_root):
        relative = source.relative_to(source_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(record(source, repo, packaged=(Path("source") / relative).as_posix()))
    return records


def copy_audit_code(source_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in AUDIT_CODE:
        source = source_root / name
        if not source.is_file():
            raise ArtifactBuildError(f"missing analysis tool {source}")
        shutil.copy2(source, destination / name)


def validation_payload(
    bundle: AnalysisBundle, source_commit: str, execution_metadata: dict[str, Any]
) -> dict[str, Any]:
    run_map = {
        arm: {str(run.seed): run for run in bundle.runs_by_arm[arm]}
        for arm in (ACTOR_KEY, CRITIC_KEY)
    }
    actual_protocol = {
        arm: {
            seed: {
                "total_steps": run.config["total_steps"],
                "smoke": run.config["smoke"],
            }
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
    core_values = [
        value
        for arm in (ACTOR_KEY, CRITIC_KEY)
        for run in bundle.runs_by_arm[arm]
        for value in (run.auc, run.final, run.best)
    ]
    return {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "status": "passed",
        "source_commit": source_commit,
        "canonical_critic_source_commit": CANONICAL_CRITIC_SOURCE_COMMIT,
        "gates": {
            "paired_seed_set": {"expected": list(SEEDS), "actual": list(bundle.seeds), "passed": True},
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
                "start": GRID[0], "end": GRID[-1], "frequency": 10_000,
                "points": len(GRID),
                "expected_steps": list(GRID),
                "actual_steps_by_arm": actual_grids,
                "passed": True,
            },
            "config_parity": {
                "only_differences": [
                    "output_root", "run_name", "thread_hint", "weight_scope",
                ],
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
                "value_count": len(core_values),
                "nan_count": sum(not math.isfinite(value) for value in core_values),
                "passed": True,
            },
            "source_frozen": {
                "included": True,
                "source_commit_recorded": source_commit,
                "passed": True,
            },
            "execution_metadata": {
                "repo_head": execution_metadata["repo_head"],
                "origin_main": execution_metadata["origin_main"],
                "head_matches_origin_main": execution_metadata["head_matches_origin_main"],
                "worktree_clean": execution_metadata["worktree_clean"],
                "passed": True,
            },
            "excluded_payloads": {
                "raw_metrics": True, "logs": True, "checkpoints": True, "smoke": True,
                "passed": True,
            },
        },
    }


def readme(source_commit: str) -> str:
    return f"""# FPP DDPG+HER+SABER Actor-Only V4 LS512 (1M)

Compact paired seeds 1–5 artifact. Source commit: `{source_commit}`.
Canonical critic source commit: `{CANONICAL_CRITIC_SOURCE_COMMIT}`.

The primary endpoint is normalized trapezoidal success AUC on the exact observed
10k–1M grid (100 points, denominator 990k). There is no interpolation, smoothing,
or forward-fill. `analysis/mean_learning_curve.csv` contains only exact-grid
per-arm means and sample SDs. `execution_metadata.json` binds the actor results
to a clean `origin/main` source commit. `source/` is the frozen executable
experiment package; `code/` is the reproducible analysis/build/verification
tooling. Raw run metrics, logs, checkpoints, and smoke result payloads are excluded.
Undefined non-core summary diagnostics are normalized from non-standard
NaN/Infinity values to JSON `null`; primary performance metrics remain subject to
strict finite-value validation.

Verify with `python3 -B code/verify_artifact.py --artifact-root .` so Python does
not add bytecode-cache files to the manifest-locked artifact directory.
"""


def manifest_entries(root: Path, excluded: set[str]) -> list[dict[str, Any]]:
    return [
        {"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    ]


def write_manifests(root: Path, source_commit: str) -> None:
    entries = manifest_entries(root, {"artifact_manifest.json", "SHA256SUMS"})
    write_json(root / "artifact_manifest.json", {
        "schema_version": 1, "artifact_id": ARTIFACT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_commit": source_commit,
        "canonical_critic_source_commit": CANONICAL_CRITIC_SOURCE_COMMIT,
        "seeds": list(SEEDS),
        "evaluation_grid": {"start": GRID[0], "end": GRID[-1], "frequency": 10_000, "points": len(GRID)},
        "required_seed_files": sorted(SEED_FILES),
        "manifest_self_excluded": True, "sha256sums_self_excluded": True,
        "file_count": len(entries), "total_size_bytes": sum(item["size_bytes"] for item in entries),
        "files": entries,
    })
    sums = manifest_entries(root, {"SHA256SUMS"})
    (root / "SHA256SUMS").write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in sums),
        encoding="utf-8", newline="\n",
    )


def build_artifact(
    *, repo: Path, actor_root: Path, critic_root: Path, artifact_root: Path,
    source_commit: str, experiment_source_root: Path, analysis_source_root: Path,
    replace: bool = False,
) -> dict[str, Any]:
    repo, artifact_root = repo.resolve(), artifact_root.resolve()
    if artifact_root.exists() and not replace:
        raise ArtifactBuildError(f"artifact exists: {artifact_root}; pass --replace explicitly")
    execution_path, execution_metadata = validate_execution_metadata(actor_root, source_commit)
    bundle = analyze_scope_ablation(actor_root, critic_root, SEEDS)
    artifact_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{ARTIFACT_ID}.tmp-", dir=artifact_root.parent))
    try:
        arm_records: dict[str, list[dict[str, Any]]] = {ACTOR_KEY: [], CRITIC_KEY: []}
        for arm in (ACTOR_KEY, CRITIC_KEY):
            for run in bundle.runs_by_arm[arm]:
                relative = Path("data") / arm / f"seed_{run.seed}"
                arm_records[arm].append(package_run(run, stage / relative, repo, relative))
        packaged = analyze_scope_ablation(stage / "data" / ACTOR_KEY, stage / "data" / CRITIC_KEY, SEEDS)
        write_outputs(packaged, stage / "analysis")
        copy_audit_code(analysis_source_root, stage / "code")
        source_records = copy_frozen_source(experiment_source_root, stage / "source", repo)
        shutil.copy2(execution_path, stage / "execution_metadata.json")
        (stage / "README.md").write_text(readme(source_commit), encoding="utf-8", newline="\n")
        (stage / "source_commit.txt").write_text(source_commit + "\n", encoding="utf-8", newline="\n")
        write_json(
            stage / "validation.json",
            validation_payload(packaged, source_commit, execution_metadata),
        )
        write_json(stage / "source_inputs_manifest.json", {
            "schema_version": 1, "artifact_id": ARTIFACT_ID, "source_commit": source_commit,
            "canonical_critic_source_commit": CANONICAL_CRITIC_SOURCE_COMMIT,
            "canonical_raw_roots": {
                ACTOR_KEY: "experiment_FPP/runs/ddpg_saber_actor_only_v4_ls512",
                CRITIC_KEY: "experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512",
            },
            "arms": arm_records,
            "execution_metadata_source": record(
                execution_path,
                repo,
                role="launch_metadata",
                packaged="execution_metadata.json",
            ),
            "frozen_source_files": source_records,
        })
        write_manifests(stage, source_commit)
        result = verify_artifact(stage)
        if artifact_root.exists():
            shutil.rmtree(artifact_root)
        os.replace(stage, artifact_root)
        return result
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    repo = find_repo_root(Path(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-root", type=Path, default=repo / "experiment_FPP/runs/ddpg_saber_actor_only_v4_ls512")
    parser.add_argument("--critic-root", type=Path, default=repo / "experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512")
    parser.add_argument("--artifact-root", type=Path, default=repo / "experiment_FPP/paper_artifacts" / ARTIFACT_ID)
    parser.add_argument("--source-commit")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        repo = find_repo_root(Path(__file__))
        commit = resolve_source_commit(repo, args.source_commit, require_clean=True)
        result = build_artifact(
            repo=repo, actor_root=args.actor_root, critic_root=args.critic_root,
            artifact_root=args.artifact_root, source_commit=commit,
            experiment_source_root=repo / EXPERIMENT_REL,
            analysis_source_root=Path(__file__).resolve().parent, replace=args.replace,
        )
    except (AnalysisError, OSError, subprocess.SubprocessError) as exc:
        print(f"artifact build failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
