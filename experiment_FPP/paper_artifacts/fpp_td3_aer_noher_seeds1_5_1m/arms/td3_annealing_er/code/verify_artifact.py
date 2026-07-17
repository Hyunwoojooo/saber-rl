#!/usr/bin/env python3
"""Independently verify the compact FPP TD3-AnnealingER no-HER paper artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any


ARTIFACT_ID = "fpp_td3_annealing_er_noher_seeds1_5_1m_arm"
SEEDS = (1, 2, 3, 4, 5)
TOTAL_STEPS = 1_000_000
EVAL_GRID = tuple(range(10_000, TOTAL_STEPS + 1, 10_000))
RUN_TEMPLATE = "fetch_pick_and_place_td3_annealing_er_noher_s{seed}_1000000"
JSON_FILES = (
    "summary.json",
    "config.json",
    "env_snapshot.json",
    "final_evaluation_100.json",
    "best_checkpoint.json",
)
SELECTED_COLUMNS = (
    "step",
    "episode",
    "train/replay_size",
    "train/her_relabel_ratio",
    "aer/enabled",
    "aer/lambda",
    "aer/candidate_count",
    "aer/selected_count",
    "aer/candidate_similarity_mean",
    "aer/candidate_similarity_std",
    "aer/candidate_similarity_min",
    "aer/candidate_similarity_max",
    "aer/selected_similarity_mean",
    "aer/selected_similarity_std",
    "aer/selected_similarity_min",
    "aer/selected_similarity_max",
    "aer/similarity_uplift",
    "aer/candidate_duplicate_rate",
    "aer/selected_duplicate_rate",
    "aer/candidate_age_mean",
    "aer/candidate_age_p90",
    "aer/selected_age_mean",
    "aer/selected_age_p90",
    "aer/candidate_success_fraction",
    "aer/selected_success_fraction",
    "aer/buffer_success_fraction",
    "aer/effective_actor_lr",
    "aer/effective_critic_lr",
    "aer/sampling_seconds",
    "aer/selected_phase_approach_fraction",
    "aer/selected_phase_grasp_fraction",
    "aer/selected_phase_lift_fraction",
    "aer/selected_phase_place_fraction",
    "eval/return_mean",
    "eval/success_mean",
    "eval/goal_success_mean",
    "eval/env_success_mean",
    "eval/goal_success_ever_mean",
    "eval/goal_success_final_mean",
    "eval/env_success_ever_mean",
    "eval/env_success_final_mean",
    "eval/episode_length_mean",
    "eval/final_success",
    "eval/best_success",
    "eval/AUC_success",
    "eval/first_hit_0.25",
    "eval/first_hit_0.50",
    "eval/first_hit_0.75",
    "eval/first_hit_0.90",
    "runtime/steps_per_sec",
    "runtime/total_sec",
    "loss/critic",
    "loss/actor",
    "loss/td_error_abs_mean",
    "loss/td_error_abs_p90",
)
PER_SEED_FIELDS = (
    "artifact_id",
    "seed",
    "run_name",
    "method",
    "total_steps",
    "episodes",
    "total_updates",
    "best_eval_step",
    "best_eval_success",
    "final_eval_success",
    "eval_AUC_success",
    "first_hit_0.25",
    "first_hit_0.50",
    "first_hit_0.75",
    "first_hit_0.90",
    "fixed_final_eval_return",
    "fixed_final_eval_goal_success",
    "fixed_final_eval_env_success",
    "runtime_total_sec",
    "final_steps_per_sec",
)
THRESHOLDS = (0.25, 0.50, 0.75, 0.90)
AGGREGATE_FIELDS = (
    "best_eval_success",
    "final_eval_success",
    "eval_AUC_success",
    "fixed_final_eval_return",
    "fixed_final_eval_goal_success",
    "fixed_final_eval_env_success",
    "runtime_total_sec",
    "final_steps_per_sec",
)
AGGREGATE_HEADER = ("metric", "n", "mean", "sample_std", "min", "max")
FORBIDDEN_NAMES = {
    "checkpoint.pt",
    "best_checkpoint.pt",
    "metrics.csv",
    "metrics.jsonl",
    "run.log",
}


class VerificationError(ValueError):
    """Raised on any integrity or semantic failure."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path, nargs="?", default=Path("."))
    parser.add_argument(
        "--experiment-source-root",
        type=Path,
        help="Verify the copied frozen source manifest against this directory.",
    )
    parser.add_argument(
        "--result-source-root",
        type=Path,
        help="Also verify source_artifacts hashes against the original result root.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise VerificationError(f"missing or empty file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"expected JSON object: {path}")
    return value


def finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"{label}: expected finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise VerificationError(f"{label}: expected finite number, got {value!r}")
    return number


def rate(value: Any, label: str) -> float:
    number = finite(value, label)
    if not 0.0 <= number <= 1.0:
        raise VerificationError(f"{label}: expected rate in [0,1], got {number}")
    return number


def same_number(left: Any, right: Any, label: str) -> None:
    a = finite(left, f"{label} (left)")
    b = finite(right, f"{label} (right)")
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12):
        raise VerificationError(f"{label}: {a!r} != {b!r}")


def recompute_curve(rows: list[dict[str, str]], seed: int) -> tuple[float, float, int, dict[float, str]]:
    successes = [rate(row["eval/success_mean"], f"seed {seed} success") for row in rows]
    area = sum(
        (EVAL_GRID[index + 1] - EVAL_GRID[index])
        * (successes[index] + successes[index + 1])
        / 2.0
        for index in range(len(EVAL_GRID) - 1)
    )
    auc = area / float(EVAL_GRID[-1] - EVAL_GRID[0])
    best = max(successes)
    best_step = next(step for step, value in zip(EVAL_GRID, successes) if value == best)
    hits = {
        threshold: next(
            (str(step) for step, value in zip(EVAL_GRID, successes) if value >= threshold),
            "",
        )
        for threshold in THRESHOLDS
    }
    return auc, best, best_step, hits


def normalized_hit(value: Any, label: str) -> str:
    if value is None or str(value).strip() == "":
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"{label}: invalid first-hit value {value!r}") from exc


def safe_relative(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise VerificationError(f"unsafe manifest path: {name!r}")
    return path


def parse_sha256sums(path: Path) -> dict[str, str]:
    if not path.is_file() or path.stat().st_size == 0:
        raise VerificationError(f"missing SHA256SUMS: {path}")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise VerificationError(f"invalid SHA256SUMS line {line_number}: {line!r}")
        name = match.group(2)
        safe_relative(name)
        if name in entries:
            raise VerificationError(f"duplicate SHA256SUMS path: {name}")
        entries[name] = match.group(1)
    if not entries:
        raise VerificationError("SHA256SUMS is empty")
    return entries


def verify_inventory(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise VerificationError(f"symlinks are not allowed in artifact: {path}")
        if path.is_file() and (path.name in FORBIDDEN_NAMES or path.suffix == ".log"):
            raise VerificationError(f"forbidden raw/large artifact included: {path}")
    hashes = parse_sha256sums(root / "SHA256SUMS")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(hashes) != actual:
        raise VerificationError(
            f"SHA256SUMS inventory mismatch: missing={sorted(actual - set(hashes))}, "
            f"extra={sorted(set(hashes) - actual)}"
        )
    for name, expected in hashes.items():
        path = root / safe_relative(name)
        actual_hash = sha256(path)
        if actual_hash != expected:
            raise VerificationError(f"SHA256 mismatch: {name}")


def read_frozen_manifest(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    entries: list[tuple[str, str]] = []
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise VerificationError(f"invalid frozen source manifest: {exc}") from exc
        candidates: Any = value
        if isinstance(value, dict):
            candidates = value.get("files", value.get("source_files", value))
        if isinstance(candidates, dict):
            entries = [(str(name), str(digest)) for name, digest in candidates.items()]
        elif isinstance(candidates, list):
            for item in candidates:
                if not isinstance(item, dict):
                    raise VerificationError("invalid JSON source manifest entry")
                entries.append((str(item.get("path", "")), str(item.get("sha256", ""))))
    else:
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line.strip())
            if not match:
                raise VerificationError(f"invalid source manifest line {line_number}")
            entries.append((match.group(2), match.group(1).lower()))
    if not entries:
        raise VerificationError("frozen source manifest has no entries")
    seen: set[str] = set()
    for name, digest in entries:
        safe_relative(name)
        if name in seen or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise VerificationError(f"duplicate path or bad source digest: {name!r}")
        seen.add(name)
    return [(name, digest.lower()) for name, digest in entries]


def verify_frozen_sources(manifest: Path, source_root: Path) -> None:
    source_root = source_root.resolve()
    for name, expected in read_frozen_manifest(manifest):
        candidate = (source_root / safe_relative(name)).resolve()
        try:
            candidate.relative_to(source_root)
        except ValueError as exc:
            raise VerificationError(f"source path escapes root: {name}") from exc
        if not candidate.is_file():
            raise VerificationError(f"missing frozen source file: {candidate}")
        if sha256(candidate) != expected:
            raise VerificationError(f"frozen source SHA256 mismatch: {name}")


def read_metrics(path: Path, seed: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SELECTED_COLUMNS:
            raise VerificationError(
                f"seed {seed}: unexpected selected metrics columns: {reader.fieldnames}"
            )
        rows = list(reader)
    if len(rows) != len(EVAL_GRID):
        raise VerificationError(f"seed {seed}: expected 100 selected rows, got {len(rows)}")
    try:
        steps = tuple(int(row["step"]) for row in rows)
    except ValueError as exc:
        raise VerificationError(f"seed {seed}: non-integer metric step") from exc
    if steps != EVAL_GRID:
        raise VerificationError(f"seed {seed}: selected metrics are not the exact 10k grid")
    for row in rows:
        for field in (
            "eval/success_mean",
            "eval/goal_success_mean",
            "eval/env_success_mean",
            "eval/AUC_success",
        ):
            rate(row[field], f"seed {seed} {field} step {row['step']}")
        if finite(row["train/her_relabel_ratio"], "HER ratio") != 0.0:
            raise VerificationError(f"seed {seed}: HER ratio is nonzero at {row['step']}")
    return rows


def validate_seed(root: Path, seed: int) -> dict[str, Any]:
    run_name = RUN_TEMPLATE.format(seed=seed)
    run_dir = root / "data" / "td3_annealing_er" / str(seed)
    expected_files = set(JSON_FILES) | {"metrics_selected_10k.csv"}
    if not run_dir.is_dir():
        raise VerificationError(f"seed {seed}: missing data directory")
    actual_files = {path.name for path in run_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise VerificationError(
            f"seed {seed}: data file set mismatch: expected={sorted(expected_files)}, "
            f"actual={sorted(actual_files)}"
        )
    config = load_json(run_dir / "config.json")
    summary = load_json(run_dir / "summary.json")
    env = load_json(run_dir / "env_snapshot.json")
    final_eval = load_json(run_dir / "final_evaluation_100.json")
    best = load_json(run_dir / "best_checkpoint.json")
    rows = read_metrics(run_dir / "metrics_selected_10k.csv", seed)
    for key, expected in {
        "run_name": run_name,
        "method": "td3_annealing_er",
        "task_name": "fetch_pick_and_place",
        "env_id": "FetchPickAndPlace-v4",
        "reward_type": "sparse",
        "total_steps": TOTAL_STEPS,
        "max_episode_steps": 50,
        "her_ratio": 0.0,
        "eval_freq": 10_000,
        "eval_episodes": 20,
        "final_eval_episodes": 100,
        "td3_policy_delay": 2,
    }.items():
        if config.get(key) != expected:
            raise VerificationError(f"seed {seed}: config.{key} mismatch")
    if config.get("seeds", {}).get("env_seed") != seed:
        raise VerificationError(f"seed {seed}: config seed mapping mismatch")
    for key, expected in {
        "method": "td3_annealing_er",
        "total_steps": TOTAL_STEPS,
        "episodes": 20_000,
        "total_updates": 995_001,
        "fixed_final_eval_policy": "final",
        "fixed_final_eval_episodes": 100,
        "fixed_final_eval_seed": 100_010_000 + seed,
    }.items():
        if summary.get(key) != expected:
            raise VerificationError(f"seed {seed}: summary.{key} mismatch")
    last = rows[-1]
    recomputed_auc, recomputed_best, recomputed_best_step, recomputed_hits = recompute_curve(rows, seed)
    same_number(summary.get("eval_AUC_success"), recomputed_auc, f"seed {seed} recomputed AUC")
    same_number(last.get("eval/AUC_success"), recomputed_auc, f"seed {seed} selected-row AUC")
    same_number(summary.get("best_eval_success"), recomputed_best, f"seed {seed} recomputed best")
    same_number(last.get("eval/best_success"), recomputed_best, f"seed {seed} selected-row best")
    same_number(last.get("eval/final_success"), last.get("eval/success_mean"), f"seed {seed} selected-row final")
    if int(summary.get("best_eval_step", -1)) != recomputed_best_step:
        raise VerificationError(f"seed {seed}: recomputed first best step mismatch")
    for threshold, expected_hit in recomputed_hits.items():
        label = f"first_hit_{threshold:.2f}"
        if normalized_hit(summary.get(label), f"seed {seed} summary {label}") != expected_hit:
            raise VerificationError(f"seed {seed}: recomputed summary {label} mismatch")
        if normalized_hit(last.get(f"eval/{label}"), f"seed {seed} selected {label}") != expected_hit:
            raise VerificationError(f"seed {seed}: recomputed selected-row {label} mismatch")
    for summary_key, metric_key in {
        "final_eval_return": "eval/return_mean",
        "final_eval_success": "eval/success_mean",
        "final_eval_goal_success": "eval/goal_success_mean",
        "final_eval_env_success": "eval/env_success_mean",
        "eval_AUC_success": "eval/AUC_success",
    }.items():
        same_number(summary.get(summary_key), last[metric_key], f"seed {seed} {summary_key}")
    rate(summary.get("best_eval_success"), f"seed {seed} best success")
    rate(summary.get("eval_AUC_success"), f"seed {seed} AUC")
    if int(best.get("step", -1)) != int(summary.get("best_eval_step", -2)):
        raise VerificationError(f"seed {seed}: best step parity failure")
    same_number(best.get("eval_success_mean"), summary.get("best_eval_success"), "best success")
    for final_key, summary_key in {
        "eval_return_mean": "fixed_final_eval_return",
        "eval_goal_success_mean": "fixed_final_eval_goal_success",
        "eval_env_success_mean": "fixed_final_eval_env_success",
    }.items():
        same_number(final_eval.get(final_key), summary.get(summary_key), f"seed {seed} fixed final")
    for key, expected in {
        "step": TOTAL_STEPS,
        "episodes": 100,
        "base_seed": 100_010_000 + seed,
        "policy": "final",
    }.items():
        if final_eval.get(key) != expected:
            raise VerificationError(f"seed {seed}: final evaluation {key} mismatch")
    for env_name in ("train_env", "eval_env"):
        item = env.get(env_name)
        if not isinstance(item, dict):
            raise VerificationError(f"seed {seed}: missing {env_name}")
        for key, expected in {
            "env_id": "FetchPickAndPlace-v4",
            "obs_dim": 25,
            "goal_dim": 3,
            "state_dim": 28,
            "action_dim": 4,
            "max_episode_steps": 50,
        }.items():
            if item.get(key) != expected:
                raise VerificationError(f"seed {seed}: {env_name}.{key} mismatch")
    result = {
        "artifact_id": ARTIFACT_ID,
        "seed": seed,
        "run_name": run_name,
        "method": "td3_annealing_er",
        "total_steps": TOTAL_STEPS,
        "episodes": summary["episodes"],
        "total_updates": summary["total_updates"],
        "best_eval_step": recomputed_best_step,
        "best_eval_success": recomputed_best,
        "final_eval_success": finite(summary["final_eval_success"], "final success"),
        "eval_AUC_success": recomputed_auc,
        "first_hit_0.25": recomputed_hits[0.25],
        "first_hit_0.50": recomputed_hits[0.50],
        "first_hit_0.75": recomputed_hits[0.75],
        "first_hit_0.90": recomputed_hits[0.90],
        "fixed_final_eval_return": finite(summary["fixed_final_eval_return"], "fixed return"),
        "fixed_final_eval_goal_success": finite(summary["fixed_final_eval_goal_success"], "fixed goal"),
        "fixed_final_eval_env_success": finite(summary["fixed_final_eval_env_success"], "fixed env"),
        "runtime_total_sec": finite(summary["runtime_total_sec"], "runtime"),
        "final_steps_per_sec": finite(summary["final_steps_per_sec"], "throughput"),
    }
    if result["runtime_total_sec"] <= 0 or result["final_steps_per_sec"] <= 0:
        raise VerificationError(f"seed {seed}: invalid runtime")
    return result


def read_csv_exact(path: Path, header: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != header:
            raise VerificationError(f"unexpected CSV header in {path}: {reader.fieldnames}")
        return list(reader)


def validate_tables(root: Path, expected: list[dict[str, Any]]) -> None:
    rows = read_csv_exact(root / "per_seed_results.csv", PER_SEED_FIELDS)
    if len(rows) != len(SEEDS):
        raise VerificationError("per_seed_results.csv must contain exactly five rows")
    for actual, wanted in zip(rows, expected):
        for field in ("artifact_id", "run_name", "method"):
            if actual[field] != str(wanted[field]):
                raise VerificationError(f"per-seed table mismatch: seed={wanted['seed']} field={field}")
        hit_fields = {f"first_hit_{threshold:.2f}" for threshold in THRESHOLDS}
        for field in hit_fields:
            if actual[field] != str(wanted[field]):
                raise VerificationError(f"per-seed first-hit mismatch: seed={wanted['seed']} field={field}")
        for field in set(PER_SEED_FIELDS).difference({"artifact_id", "run_name", "method"} | hit_fields):
            same_number(actual[field], wanted[field], f"per-seed seed={wanted['seed']} {field}")

    aggregate = read_csv_exact(root / "aggregate_results.csv", AGGREGATE_HEADER)
    if [row["metric"] for row in aggregate] != list(AGGREGATE_FIELDS):
        raise VerificationError("aggregate_results.csv metric ordering/content mismatch")
    for row, field in zip(aggregate, AGGREGATE_FIELDS):
        values = [finite(item[field], field) for item in expected]
        wanted = {
            "n": len(values),
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
        }
        for key, value in wanted.items():
            same_number(row[key], value, f"aggregate {field}.{key}")


def validate_manifest(root: Path, args: argparse.Namespace) -> None:
    manifest = load_json(root / "artifact_manifest.json")
    if manifest.get("schema_version") != "1.0" or manifest.get("artifact_id") != ARTIFACT_ID:
        raise VerificationError("artifact manifest identity/schema mismatch")
    if manifest.get("sha256sums_scope") != "all artifact files except SHA256SUMS itself":
        raise VerificationError("artifact manifest SHA256SUMS scope mismatch")
    commit = manifest.get("source_commit_full")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise VerificationError("manifest source_commit_full is not a full commit")
    commit_file = root / "provenance" / "source_commit.txt"
    if commit_file.read_text(encoding="utf-8").strip() != commit:
        raise VerificationError("source commit provenance mismatch")
    completion = manifest.get("completion", {})
    if completion.get("seeds") != list(SEEDS) or completion.get("required_final_step") != TOTAL_STEPS:
        raise VerificationError("manifest completion block mismatch")
    if completion.get("all_runs_complete") is not True:
        raise VerificationError("manifest does not assert all runs complete")
    completion_file = load_json(root / "completion_validation.json")
    if completion_file.get("artifact_id") != ARTIFACT_ID or completion_file.get("all_runs_complete") is not True:
        raise VerificationError("completion_validation.json identity/status mismatch")
    grid = completion_file.get("evaluation_grid", {})
    if grid != {"start": EVAL_GRID[0], "stop": EVAL_GRID[-1], "stride": 10_000, "count": len(EVAL_GRID)}:
        raise VerificationError("completion_validation.json grid mismatch")
    completed_seeds = completion_file.get("seeds")
    if not isinstance(completed_seeds, list) or [item.get("seed") for item in completed_seeds if isinstance(item, dict)] != list(SEEDS):
        raise VerificationError("completion_validation.json seed set mismatch")
    if not all(item.get("summary_parity") is True and item.get("fixed_final_evaluation_parity") is True for item in completed_seeds):
        raise VerificationError("completion_validation.json parity status mismatch")
    frozen_rel = manifest.get("frozen_source_manifest")
    if not isinstance(frozen_rel, str):
        raise VerificationError("manifest frozen_source_manifest is missing")
    frozen_path = root / safe_relative(frozen_rel)
    read_frozen_manifest(frozen_path)
    if args.experiment_source_root:
        verify_frozen_sources(frozen_path, args.experiment_source_root)
    sources = manifest.get("source_artifacts")
    if not isinstance(sources, list) or len(sources) != len(SEEDS) * 6:
        raise VerificationError("manifest must record six source artifacts per seed")
    if args.result_source_root:
        result_root = args.result_source_root.resolve()
        for item in sources:
            if not isinstance(item, dict):
                raise VerificationError("invalid source_artifacts entry")
            rel = safe_relative(str(item.get("source_path", "")))
            path = (result_root / rel).resolve()
            try:
                path.relative_to(result_root)
            except ValueError as exc:
                raise VerificationError(f"result source path escapes root: {rel}") from exc
            if not path.is_file() or sha256(path) != item.get("source_sha256"):
                raise VerificationError(f"result source hash mismatch: {rel}")


def verify(root: Path, args: argparse.Namespace) -> None:
    root = root.resolve()
    if not root.is_dir():
        raise VerificationError(f"artifact root is not a directory: {root}")
    verify_inventory(root)
    validate_manifest(root, args)
    data_root = root / "data"
    top_dirs = {path.name for path in data_root.iterdir() if path.is_dir()}
    if top_dirs != {"td3_annealing_er"}:
        raise VerificationError(f"data method directory set mismatch: {sorted(top_dirs)}")
    seed_dirs = {path.name for path in (data_root / "td3_annealing_er").iterdir() if path.is_dir()}
    if seed_dirs != {str(seed) for seed in SEEDS}:
        raise VerificationError(f"data seed directory set mismatch: {sorted(seed_dirs)}")
    expected = [validate_seed(root, seed) for seed in SEEDS]
    validate_tables(root, expected)


def main() -> int:
    args = parse_args()
    try:
        verify(args.artifact_root, args)
    except (VerificationError, OSError, csv.Error) as exc:
        print(f"[artifact-verify-fail] {exc}", file=sys.stderr)
        return 1
    print(f"[artifact-verify-pass] {args.artifact_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
