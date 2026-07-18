#!/usr/bin/env python3
"""Package and verify the completed FPP DDPG no-HER three-arm experiment.

The committed artifact is intentionally compact.  Raw checkpoints, replay
state, full-resolution metrics, JSONL streams, and logs are validated in place
but are not copied into the artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = REPO_ROOT / "experiment_FPP"
EXPERIMENT_DIR = EXPERIMENT_ROOT / "ddpg_aer_noher_1m"
VALIDATOR = EXPERIMENT_DIR / "validate_main_arm_gate.py"
ANALYZER = EXPERIMENT_DIR / "analysis" / "analyze_ddpg_aer_noher.py"
DEFAULT_RUNS_ROOT = EXPERIMENT_ROOT / "runs" / "fpp_ddpg_aer_noher_1m"
DEFAULT_LOG_DIR = EXPERIMENT_ROOT / "logs" / "fpp_ddpg_aer_noher_1m"
QUEUE_PROVENANCE_DIR = (
    EXPERIMENT_ROOT / "logs" / "fpp_ddpg_aer_three_arm_after_sac_queue"
)
FROZEN_SOURCE_MANIFEST = QUEUE_PROVENANCE_DIR / "frozen_source.sha256"
SOURCE_BASE_COMMIT_FULL = "a5436ef2fd107a24deae0d92468f27e17893dddd"
DEFAULT_ARTIFACT_ROOT = (
    EXPERIMENT_ROOT
    / "paper_artifacts"
    / "fpp_ddpg_aer_noher_seeds1_5_1m"
)

METHODS = ("ddpg_er", "ddpg_annealing_er", "ddpg_aer")
METHOD_LABELS = {
    "ddpg_er": "DDPG ER",
    "ddpg_annealing_er": "DDPG Annealing ER",
    "ddpg_aer": "DDPG AER",
}
SEEDS = (1, 2, 3, 4, 5)
TOTAL_STEPS = 1_000_000
EXPECTED_EPISODES = 20_000
EXPECTED_UPDATES = 995_001
EXPECTED_REPLAY_SIZE = 300_000
EVAL_FREQUENCY = 10_000
EVAL_GRID = tuple(range(EVAL_FREQUENCY, TOTAL_STEPS + 1, EVAL_FREQUENCY))
EXPECTED_EVALUATIONS = len(EVAL_GRID)
FINAL_EVAL_EPISODES = 100
BASE_ACTOR_LR = 3e-4
BASE_CRITIC_LR = 3e-4
AER_LAMBDA0 = 2.5
AER_ALPHA = 2.0 / 3.0
BATCH_SIZE = 256

JSON_FILES = (
    "config.json",
    "env_snapshot.json",
    "summary.json",
    "best_checkpoint.json",
    "final_evaluation_100.json",
)

# This is the established compact AER schema used by the TD3 three-arm
# artifact.  It retains the exact learning curve plus the diagnostics needed
# to distinguish fixed ER, annealing ER, and attentive replay.
SELECTED_METRIC_COLUMNS = (
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

ANALYSIS_OUTPUTS = (
    "per_seed_results.csv",
    "method_summary.csv",
    "pairwise_summary.csv",
    "mean_curves.csv",
    "analysis_results.json",
    "analysis_summary.md",
)

FIXED_FINAL_COLUMNS = (
    "method",
    "seed",
    "episodes",
    "base_seed",
    "success_mean",
    "success_std",
    "goal_success_mean",
    "goal_success_std",
    "env_success_mean",
    "env_success_std",
    "return_mean",
    "return_std",
    "episode_length_mean",
)

NONFINITE_TOKEN = re.compile(
    r"(?<![A-Za-z])(?:nan|[+-]?infinity)(?![A-Za-z])", re.IGNORECASE
)
RAW_ABSOLUTE_PREFIX = re.compile(
    r"(?:^|[\s\"'=:,(\[])/(?:home|tmp|root|usr|var|mnt|opt|Users)/"
)
FROZEN_LINE = re.compile(r"^([0-9a-f]{64})\s+(.+)$")
FORBIDDEN_SUFFIXES = {".pt", ".jsonl", ".log", ".pyc"}


class ArtifactError(ValueError):
    """Raised when raw inputs or the compact artifact fail closed checks."""


def fail(message: str) -> NoReturn:
    raise ArtifactError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        fail(f"missing or empty required file: {path}")
    return path


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ArtifactError(f"input must be inside the repository: {path}") from exc


def file_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(relative_to.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(require_file(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        fail(f"expected a JSON object: {path}")
    return value


def load_json_strict(path: Path) -> dict[str, Any]:
    def reject_constant(token: str) -> NoReturn:
        fail(f"{path}: forbidden non-finite JSON constant {token}")

    try:
        value = json.loads(
            require_file(path).read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"invalid strict JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        fail(f"expected a JSON object: {path}")
    return value


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def require_finite(value: Any, label: str) -> float:
    parsed = finite_float(value)
    if parsed is None:
        fail(f"{label}: expected a finite number, got {value!r}")
    return parsed


def require_rate(value: Any, label: str) -> float:
    parsed = require_finite(value, label)
    if not 0.0 <= parsed <= 1.0:
        fail(f"{label}: expected a rate in [0, 1], got {parsed!r}")
    return parsed


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        fail(f"{label}: expected {expected!r}, got {actual!r}")


def require_close(
    actual: Any,
    expected: float,
    label: str,
    *,
    tolerance: float = 1e-9,
) -> None:
    parsed = require_finite(actual, label)
    if not math.isclose(parsed, expected, rel_tol=tolerance, abs_tol=tolerance):
        fail(f"{label}: expected {expected!r}, got {parsed!r}")


def sanitize_string(value: str, *, artifact_root: Path | None = None) -> str:
    sanitized = value
    if artifact_root is not None:
        sanitized = sanitized.replace(str(artifact_root.resolve()), "${ARTIFACT_ROOT}")
    sanitized = sanitized.replace(str(REPO_ROOT.resolve()), "${REPO_ROOT}")
    if sanitized.startswith("/"):
        sanitized = f"${{ABSOLUTE_PATH_REDACTED}}/{Path(sanitized).name}"
    return sanitized


def sanitize(value: Any, *, artifact_root: Path | None = None) -> Any:
    if isinstance(value, dict):
        return {
            str(key): sanitize(item, artifact_root=artifact_root)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize(item, artifact_root=artifact_root) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        return sanitize_string(value, artifact_root=artifact_root)
    return value


def write_json(path: Path, value: Any, *, artifact_root: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            sanitize(value, artifact_root=artifact_root),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def csv_value(value: Any, *, artifact_root: Path | None = None) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else ""
    if isinstance(value, str):
        if NONFINITE_TOKEN.fullmatch(value.strip()):
            return ""
        return sanitize_string(value, artifact_root=artifact_root)
    return value


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, Any]],
    *,
    artifact_root: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: csv_value(row.get(field, ""), artifact_root=artifact_root)
                    for field in fieldnames
                }
            )


def git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def experiment_git_status() -> list[str]:
    result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            repo_relative(EXPERIMENT_DIR),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def python_executable() -> Path:
    repository_python = REPO_ROOT / ".venv" / "bin" / "python"
    if repository_python.is_file():
        # Keep the venv entry-point path intact. Resolving this symlink invokes
        # the base interpreter directly and drops the venv site-packages.
        return repository_python
    return Path(sys.executable)


def run_name(method: str, seed: int) -> str:
    return f"fetch_pick_and_place_{method}_noher_s{seed}_{TOTAL_STEPS}"


def parse_frozen_source_manifest() -> list[dict[str, Any]]:
    manifest = require_file(FROZEN_SOURCE_MANIFEST)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        match = FROZEN_LINE.fullmatch(line.strip())
        if match is None:
            fail(f"invalid frozen source manifest line {line_number}: {line!r}")
        expected_digest, relative = match.groups()
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            fail(f"unsafe frozen source path on line {line_number}: {relative!r}")
        normalized = relative_path.as_posix()
        if normalized in seen:
            fail(f"duplicate frozen source path: {normalized}")
        seen.add(normalized)
        source = require_file(EXPERIMENT_DIR / relative_path)
        actual_digest = sha256(source)
        if actual_digest != expected_digest:
            fail(
                "executed source no longer matches the queue-frozen snapshot: "
                f"{normalized}: expected {expected_digest}, got {actual_digest}"
            )
        records.append(
            {
                "execution_relative_path": normalized,
                "repository_path": repo_relative(source),
                "size_bytes": source.stat().st_size,
                "sha256": actual_digest,
            }
        )
    if not records:
        fail("frozen source manifest contained no files")
    return records


def run_raw_completion_gates(
    *, runs_root: Path, log_dir: Path, python: Path
) -> list[dict[str, Any]]:
    require_file(VALIDATOR)
    records: list[dict[str, Any]] = []
    for method in METHODS:
        command = [
            str(python),
            str(VALIDATOR.resolve()),
            "--arm",
            method,
            "--seeds",
            " ".join(str(seed) for seed in SEEDS),
            "--output-root",
            str(runs_root.resolve()),
            "--log-dir",
            str(log_dir.resolve()),
        ]
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        record = {
            "arm": method,
            "validated_at_utc": utc_now(),
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        records.append(record)
        if result.returncode != 0:
            detail = (result.stdout + "\n" + result.stderr).strip()
            fail(f"strict raw completion gate failed for {method}:\n{detail[-8000:]}")
        expected_marker = (
            f"[gate-pass] arm={method} completed cleanly for {len(SEEDS)} seed(s)"
        )
        if expected_marker not in result.stdout:
            fail(f"strict gate for {method} omitted marker {expected_marker!r}")
    return records


def expected_lambda(method: str, step: int) -> float:
    if method == "ddpg_er":
        return 1.0
    progress = max(0.0, float(step)) / (AER_ALPHA * float(TOTAL_STEPS))
    return 1.0 + (AER_LAMBDA0 - 1.0) * max(1.0 - progress, 0.0)


def select_metrics(
    path: Path, *, method: str, seed: int
) -> list[dict[str, str]]:
    by_step: dict[int, dict[str, str]] = {}
    with require_file(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = sorted(set(SELECTED_METRIC_COLUMNS).difference(fieldnames))
        if missing:
            fail(f"{path}: missing compact metric columns {missing}")
        for row_number, row in enumerate(reader, start=2):
            raw_step = finite_float(row.get("step"))
            if raw_step is None or not raw_step.is_integer():
                continue
            step = int(raw_step)
            success = finite_float(row.get("eval/success_mean"))
            if step not in EVAL_GRID or success is None:
                continue
            if step in by_step:
                fail(f"{path}:{row_number}: duplicate evaluation row at step {step}")

            for field in (
                "eval/success_mean",
                "eval/goal_success_mean",
                "eval/env_success_mean",
                "eval/goal_success_ever_mean",
                "eval/goal_success_final_mean",
                "eval/env_success_ever_mean",
                "eval/env_success_final_mean",
                "eval/final_success",
                "eval/best_success",
                "eval/AUC_success",
            ):
                require_rate(row.get(field), f"{method}/seed{seed} step={step} {field}")
            require_finite(
                row.get("eval/return_mean"),
                f"{method}/seed{seed} step={step} eval/return_mean",
            )
            require_finite(
                row.get("eval/episode_length_mean"),
                f"{method}/seed{seed} step={step} eval/episode_length_mean",
            )
            native = require_rate(
                row.get("eval/success_mean"),
                f"{method}/seed{seed} step={step} native success",
            )
            goal = require_rate(
                row.get("eval/goal_success_mean"),
                f"{method}/seed{seed} step={step} goal success",
            )
            env = require_rate(
                row.get("eval/env_success_mean"),
                f"{method}/seed{seed} step={step} env success",
            )
            if not (
                math.isclose(native, goal, rel_tol=0.0, abs_tol=1e-12)
                and math.isclose(native, env, rel_tol=0.0, abs_tol=1e-12)
            ):
                fail(f"{method}/seed{seed} step={step}: success aliases disagree")

            require_close(
                row.get("train/her_relabel_ratio"),
                0.0,
                f"{method}/seed{seed} step={step} HER ratio",
                tolerance=1e-12,
            )
            require_close(
                row.get("train/replay_size"),
                float(min(step, EXPECTED_REPLAY_SIZE)),
                f"{method}/seed{seed} step={step} replay size",
                tolerance=1e-12,
            )
            lambda_value = expected_lambda(method, step)
            require_close(
                row.get("aer/lambda"),
                lambda_value,
                f"{method}/seed{seed} step={step} AER lambda",
            )
            enabled = 1.0 if method == "ddpg_aer" else 0.0
            require_close(
                row.get("aer/enabled"),
                enabled,
                f"{method}/seed{seed} step={step} AER enabled",
                tolerance=1e-12,
            )
            candidate_count = (
                float(math.ceil(lambda_value * BATCH_SIZE))
                if method == "ddpg_aer"
                else float(BATCH_SIZE)
            )
            require_close(
                row.get("aer/candidate_count"),
                candidate_count,
                f"{method}/seed{seed} step={step} candidate count",
                tolerance=1e-12,
            )
            require_close(
                row.get("aer/selected_count"),
                float(BATCH_SIZE),
                f"{method}/seed{seed} step={step} selected count",
                tolerance=1e-12,
            )
            lr_multiplier = lambda_value if method != "ddpg_er" else 1.0
            require_close(
                row.get("aer/effective_actor_lr"),
                BASE_ACTOR_LR * lr_multiplier,
                f"{method}/seed{seed} step={step} actor LR",
                tolerance=1e-10,
            )
            require_close(
                row.get("aer/effective_critic_lr"),
                BASE_CRITIC_LR * lr_multiplier,
                f"{method}/seed{seed} step={step} critic LR",
                tolerance=1e-10,
            )
            if method == "ddpg_aer":
                for field in (
                    "aer/candidate_similarity_mean",
                    "aer/selected_similarity_mean",
                    "aer/similarity_uplift",
                ):
                    require_finite(
                        row.get(field), f"{method}/seed{seed} step={step} {field}"
                    )

            by_step[step] = {
                field: str(row.get(field, "")) for field in SELECTED_METRIC_COLUMNS
            }

    actual_grid = tuple(sorted(by_step))
    if actual_grid != EVAL_GRID:
        missing = sorted(set(EVAL_GRID).difference(actual_grid))
        extra = sorted(set(actual_grid).difference(EVAL_GRID))
        fail(f"{path}: evaluation grid mismatch; missing={missing}, extra={extra}")
    return [by_step[step] for step in EVAL_GRID]


def normalized_auc(steps: Sequence[int], values: Sequence[float]) -> float:
    if tuple(steps) != EVAL_GRID or len(values) != EXPECTED_EVALUATIONS:
        fail("AUC requires the exact 100-point 10k..1M grid")
    area = sum(
        (steps[index] - steps[index - 1])
        * (values[index] + values[index - 1])
        / 2.0
        for index in range(1, len(steps))
    )
    return area / float(EVAL_GRID[-1] - EVAL_GRID[0])


def canonical_hit(value: Any, label: str) -> int | str:
    if value in (None, ""):
        return ""
    parsed = require_finite(value, label)
    if not parsed.is_integer():
        fail(f"{label}: first-hit value is not an integer: {value!r}")
    return int(parsed)


def validate_run(
    *, runs_root: Path, method: str, seed: int
) -> tuple[
    dict[str, Any],
    list[dict[str, str]],
    dict[str, Any],
    dict[str, Any],
    list[Path],
]:
    name = run_name(method, seed)
    run_dir = runs_root / name
    if not run_dir.is_dir():
        fail(f"missing run directory: {run_dir}")
    required = {filename: require_file(run_dir / filename) for filename in JSON_FILES}
    metrics_path = require_file(run_dir / "metrics.csv")
    raw_inputs = [*required.values(), metrics_path]

    config = load_json(required["config.json"])
    env_snapshot = load_json(required["env_snapshot.json"])
    summary = load_json(required["summary.json"])
    best = load_json(required["best_checkpoint.json"])
    final_eval = load_json(required["final_evaluation_100.json"])

    expected_config = {
        "run_name": name,
        "method": method,
        "task_name": "fetch_pick_and_place",
        "env_id": "FetchPickAndPlace-v4",
        "resolved_env_id": "FetchPickAndPlace-v4",
        "reward_type": "sparse",
        "total_steps": TOTAL_STEPS,
        "max_episode_steps": 50,
        "start_steps": 10_000,
        "warmup_steps": 5_000,
        "updates_per_step": 1,
        "batch_size": BATCH_SIZE,
        "buffer_capacity": EXPECTED_REPLAY_SIZE,
        "her_ratio": 0.0,
        "gamma": 0.98,
        "tau": 0.005,
        "actor_lr": BASE_ACTOR_LR,
        "critic_lr": BASE_CRITIC_LR,
        "hidden_dim": 256,
        "noise_std": 0.2,
        "eval_freq": EVAL_FREQUENCY,
        "eval_episodes": 20,
        "final_eval_episodes": FINAL_EVAL_EPISODES,
        "checkpoint_every_steps": 50_000,
        "success_terminates": False,
        "timeout_bootstrap": True,
        "lambda_support": 0.0,
        "weight_scope": "none",
        "support_cache_enabled": False,
        "local_scan_cache_enabled": False,
        "support_short_circuit": False,
        "numba_support_scan": False,
        "support_predicate_early_exit": False,
        "aer_lambda0": AER_LAMBDA0,
        "aer_alpha": AER_ALPHA,
        "aer_similarity": "cosine",
        "aer_similarity_eps": 1e-8,
        "aer_candidate_rounding": "ceil",
        "smoke": False,
        "resume": False,
    }
    for key, expected in expected_config.items():
        require_equal(config.get(key), expected, f"{method}/seed{seed} config.{key}")
    require_equal(
        config.get("seeds"),
        {
            "env_seed": seed,
            "torch_seed": seed,
            "numpy_seed": seed,
            "action_noise_seed": seed,
            "weight_seed": 20_000 + seed,
            "eval_seed": 10_000 + seed,
        },
        f"{method}/seed{seed} seed mapping",
    )
    for key in (
        "td3_policy_delay",
        "td3_target_policy_noise",
        "td3_target_noise_clip",
    ):
        if key in config:
            fail(f"{method}/seed{seed}: DDPG config contains TD3 field {key}")

    expected_summary = {
        "method": method,
        "total_steps": TOTAL_STEPS,
        "episodes": EXPECTED_EPISODES,
        "total_updates": EXPECTED_UPDATES,
        "final_replay_size": EXPECTED_REPLAY_SIZE,
        "fixed_final_eval_policy": "final",
        "fixed_final_eval_episodes": FINAL_EVAL_EPISODES,
        "fixed_final_eval_seed": 100_010_000 + seed,
        "selection_metric": "eval_goal_success_mean",
    }
    for key, expected in expected_summary.items():
        require_equal(summary.get(key), expected, f"{method}/seed{seed} summary.{key}")
    if require_finite(summary.get("runtime_total_sec"), "runtime") <= 0.0:
        fail(f"{method}/seed{seed}: runtime must be positive")
    if require_finite(summary.get("final_steps_per_sec"), "throughput") <= 0.0:
        fail(f"{method}/seed{seed}: throughput must be positive")

    rows = select_metrics(metrics_path, method=method, seed=seed)
    steps = [int(float(row["step"])) for row in rows]
    successes = [float(row["eval/success_mean"]) for row in rows]
    returns = [float(row["eval/return_mean"]) for row in rows]
    auc = normalized_auc(steps, successes)
    best_success = max(successes)
    tied = [
        index
        for index, success in enumerate(successes)
        if math.isclose(success, best_success, rel_tol=1e-5, abs_tol=1e-8)
    ]
    best_index = max(tied, key=returns.__getitem__)
    best_step = steps[best_index]
    best_return = returns[best_index]
    first_hits = {
        f"first_hit_{threshold:.2f}": next(
            (
                step
                for step, success in zip(steps, successes)
                if success >= threshold
            ),
            "",
        )
        for threshold in (0.25, 0.50, 0.75, 0.90)
    }

    require_close(summary.get("eval_AUC_success"), auc, "summary AUC")
    require_close(rows[-1].get("eval/AUC_success"), auc, "selected-row AUC")
    require_close(summary.get("final_eval_success"), successes[-1], "final success")
    require_close(
        summary.get("final_eval_goal_success"), successes[-1], "final goal success"
    )
    require_close(
        summary.get("final_eval_env_success"), successes[-1], "final env success"
    )
    require_close(summary.get("final_eval_return"), returns[-1], "final return")
    require_close(summary.get("best_eval_success"), best_success, "best success")
    require_close(summary.get("best_eval_goal_success"), best_success, "best goal")
    require_close(summary.get("best_eval_env_success"), best_success, "best env")
    require_close(summary.get("best_eval_return"), best_return, "best return")
    require_equal(summary.get("best_eval_step"), best_step, "best step")
    for label, expected in first_hits.items():
        require_equal(canonical_hit(summary.get(label), label), expected, label)

    require_equal(best.get("step"), best_step, f"{method}/seed{seed} best metadata step")
    require_equal(
        best.get("selection_metric"),
        "eval_goal_success_mean",
        f"{method}/seed{seed} best selection metric",
    )
    require_close(best.get("eval_success_mean"), best_success, "best metadata success")
    require_close(best.get("eval_goal_success_mean"), best_success, "best metadata goal")
    require_close(best.get("eval_env_success_mean"), best_success, "best metadata env")
    require_close(best.get("eval_return_mean"), best_return, "best metadata return")

    for key, expected in {
        "policy": "final",
        "step": TOTAL_STEPS,
        "episodes": FINAL_EVAL_EPISODES,
        "base_seed": 100_010_000 + seed,
    }.items():
        require_equal(final_eval.get(key), expected, f"{method}/seed{seed} final.{key}")
    for key in (
        "eval_success_mean",
        "eval_goal_success_mean",
        "eval_env_success_mean",
    ):
        require_rate(final_eval.get(key), f"{method}/seed{seed} final.{key}")
    fixed_success = require_rate(
        final_eval.get("eval_success_mean"), f"{method}/seed{seed} fixed success"
    )
    fixed_goal = require_rate(
        final_eval.get("eval_goal_success_mean"), f"{method}/seed{seed} fixed goal"
    )
    fixed_env = require_rate(
        final_eval.get("eval_env_success_mean"), f"{method}/seed{seed} fixed env"
    )
    if not (
        math.isclose(fixed_success, fixed_goal, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(fixed_success, fixed_env, rel_tol=0.0, abs_tol=1e-12)
    ):
        fail(f"{method}/seed{seed}: fixed final success aliases disagree")
    require_close(
        summary.get("fixed_final_eval_goal_success"), fixed_goal, "fixed summary goal"
    )
    require_close(
        summary.get("fixed_final_eval_env_success"), fixed_env, "fixed summary env"
    )
    require_close(
        summary.get("fixed_final_eval_return"),
        require_finite(final_eval.get("eval_return_mean"), "fixed return"),
        "fixed summary return",
    )

    for env_name in ("train_env", "eval_env"):
        snapshot = env_snapshot.get(env_name)
        if not isinstance(snapshot, dict):
            fail(f"{method}/seed{seed}: missing {env_name} snapshot")
        for key, expected in {
            "env_id": "FetchPickAndPlace-v4",
            "obs_dim": 25,
            "goal_dim": 3,
            "state_dim": 28,
            "action_dim": 4,
            "max_episode_steps": 50,
        }.items():
            require_equal(
                snapshot.get(key), expected, f"{method}/seed{seed} {env_name}.{key}"
            )
        require_equal(
            snapshot.get("spec_kwargs", {}).get("reward_type"),
            "sparse",
            f"{method}/seed{seed} {env_name} reward type",
        )

    fixed_row = {
        "method": method,
        "seed": seed,
        "episodes": final_eval["episodes"],
        "base_seed": final_eval["base_seed"],
        "success_mean": fixed_success,
        "success_std": require_finite(final_eval.get("eval_success_std"), "fixed std"),
        "goal_success_mean": fixed_goal,
        "goal_success_std": require_finite(
            final_eval.get("eval_goal_success_std"), "fixed goal std"
        ),
        "env_success_mean": fixed_env,
        "env_success_std": require_finite(
            final_eval.get("eval_env_success_std"), "fixed env std"
        ),
        "return_mean": require_finite(final_eval.get("eval_return_mean"), "fixed return"),
        "return_std": require_finite(final_eval.get("eval_return_std"), "fixed return std"),
        "episode_length_mean": require_finite(
            final_eval.get("eval_episode_length_mean"), "fixed episode length"
        ),
    }
    validation = {
        "method": method,
        "seed": seed,
        "run_name": name,
        "evaluation_points": EXPECTED_EVALUATIONS,
        "recomputed": {
            "auc": auc,
            "final_success": successes[-1],
            "best_success": best_success,
            "best_step": best_step,
            "best_return": best_return,
            **first_hits,
            "fixed_final_100_success": fixed_success,
            "fixed_final_100_return": fixed_row["return_mean"],
        },
        "checks": {
            "strict_config_contract": True,
            "exact_evaluation_grid": True,
            "success_aliases": True,
            "curve_recomputed": True,
            "best_checkpoint_metadata": True,
            "fixed_final_100": True,
            "arm_sampling_and_lr_behavior": True,
            "environment_snapshot": True,
        },
    }
    packaged_json = {
        "config.json": config,
        "env_snapshot.json": env_snapshot,
        "summary.json": summary,
        "best_checkpoint.json": best,
        "final_evaluation_100.json": final_eval,
    }
    return packaged_json, rows, fixed_row, validation, raw_inputs


def run_analyzer(
    *, data_root: Path, output_dir: Path, python: Path
) -> dict[str, Any]:
    require_file(ANALYZER)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(ANALYZER.resolve()),
        "--runs-root",
        str(data_root.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--seeds",
        "1-5",
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stdout + "\n" + result.stderr).strip()
        fail(f"DDPG paired analyzer failed:\n{detail[-8000:]}")
    for filename in ANALYSIS_OUTPUTS:
        require_file(output_dir / filename)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
        "stderr": result.stderr,
    }


def normalize_analysis_outputs(output_dir: Path, *, artifact_root: Path) -> None:
    json_path = output_dir / "analysis_results.json"
    write_json(json_path, load_json(json_path), artifact_root=artifact_root)
    for filename in (
        "per_seed_results.csv",
        "method_summary.csv",
        "pairwise_summary.csv",
        "mean_curves.csv",
    ):
        path = output_dir / filename
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
        if not fieldnames or not rows:
            fail(f"analyzer emitted an empty CSV: {path}")
        write_csv(path, fieldnames, rows, artifact_root=artifact_root)
    markdown = output_dir / "analysis_summary.md"
    text = markdown.read_text(encoding="utf-8")
    text = sanitize_string(text, artifact_root=artifact_root)
    markdown.write_text(text, encoding="utf-8", newline="\n")


def verify_analysis_semantics(artifact_root: Path, *, python: Path) -> None:
    results_dir = artifact_root / "results"
    analysis = load_json_strict(results_dir / "analysis_results.json")
    require_equal(analysis.get("methods"), list(METHODS), "analysis methods")
    require_equal(analysis.get("seeds"), list(SEEDS), "analysis seeds")
    require_equal(analysis.get("n_pairs"), len(SEEDS), "analysis pair count")
    require_equal(analysis.get("evaluation_grid"), list(EVAL_GRID), "analysis grid")
    require_equal(
        analysis.get("auc_denominator_steps"),
        EVAL_GRID[-1] - EVAL_GRID[0],
        "analysis AUC denominator",
    )
    aggregates = analysis.get("method_aggregates")
    comparisons = analysis.get("comparisons")
    if not isinstance(aggregates, list) or [row.get("method") for row in aggregates] != list(
        METHODS
    ):
        fail("analysis method aggregates are incomplete or out of order")
    expected_comparisons = [
        "ddpg_aer_vs_ddpg_annealing_er",
        "ddpg_aer_vs_ddpg_er",
    ]
    if not isinstance(comparisons, list) or [
        row.get("comparison") for row in comparisons
    ] != expected_comparisons:
        fail("analysis paired comparisons are incomplete or out of order")

    with tempfile.TemporaryDirectory(prefix="fpp-ddpg-aer-analysis-") as temp_name:
        rerun_dir = Path(temp_name)
        run_analyzer(
            data_root=artifact_root / "data", output_dir=rerun_dir, python=python
        )
        normalize_analysis_outputs(rerun_dir, artifact_root=artifact_root)
        for filename in ANALYSIS_OUTPUTS:
            expected = (results_dir / filename).read_bytes()
            actual = (rerun_dir / filename).read_bytes()
            if actual != expected:
                fail(f"packaged analyzer output is not reproducible: {filename}")


def fixed_final_summary(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for method in METHODS:
        values = [
            float(row["success_mean"]) for row in rows if row["method"] == method
        ]
        if len(values) != len(SEEDS):
            fail(f"fixed final summary requires five rows for {method}")
        result[method] = (statistics.mean(values), statistics.stdev(values))
    return result


def make_readme(
    analysis: dict[str, Any], fixed_rows: Sequence[dict[str, Any]]
) -> str:
    aggregates = {row["method"]: row for row in analysis["method_aggregates"]}
    fixed = fixed_final_summary(fixed_rows)
    lines = [
        "# FPP DDPG no-HER AER three-arm results",
        "",
        "This compact artifact contains the completed paired seeds 1-5 runs for",
        "uniform ER, uniform ER with the AER learning-rate schedule, and attentive",
        "experience replay with the same schedule.",
        "",
        "| Method | Success AUC mean +/- SD | 20-episode final mean +/- SD | Fixed final-100 mean +/- SD |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        aggregate = aggregates[method]
        fixed_mean, fixed_std = fixed[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {aggregate['auc_mean']:.6f} +/- "
            f"{aggregate['auc_sample_std']:.6f} | {aggregate['final_mean']:.6f} +/- "
            f"{aggregate['final_sample_std']:.6f} | {fixed_mean:.6f} +/- "
            f"{fixed_std:.6f} |"
        )
    lines.extend(
        [
            "",
            "The primary paired comparison is `ddpg_aer - ddpg_annealing_er`;",
            "`ddpg_aer - ddpg_er` is the secondary full-recipe comparison.",
            "",
            "## Contents",
            "",
            "- `data/`: sanitized per-run config, environment, summary, best-checkpoint",
            "  metadata, fixed 100-episode evaluation, and exact 100-point metrics.",
            "- `results/`: canonical paired analyzer outputs, fixed-final table, and",
            "  machine-readable validation evidence.",
            "- `provenance/`: the queue-frozen source manifest and the verified hashes of",
            "  the exact source files used by the completed run.",
            "- `input_manifest.json` and `artifact_manifest.json`: SHA256 inventories.",
            "",
            "Online curve evaluations use 20 deterministic episodes at 10k, 20k, ...,",
            "1M steps. The success AUC is trapezoidal integration divided by 990k. The",
            "fixed final evaluation uses the final policy for 100 deterministic episodes",
            "and is reported separately from the final point on the learning curve.",
            "",
            "Raw model checkpoints, replay state, full metrics, JSONL streams, and logs",
            "are intentionally excluded. They were checked by the strict completion gate",
            "before this artifact was built.",
            "",
            "Regenerate from the repository root with:",
            "",
            "```bash",
            ".venv/bin/python experiment_FPP/package_ddpg_aer_results.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def assert_no_absolute_strings(value: Any, *, context: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_no_absolute_strings(item, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_no_absolute_strings(item, context=f"{context}[{index}]")
    elif isinstance(value, str):
        if str(REPO_ROOT.resolve()) in value or RAW_ABSOLUTE_PREFIX.search(value):
            fail(f"absolute path survived sanitization at {context}: {value!r}")


def verify_input_manifest(artifact_root: Path) -> None:
    manifest = load_json_strict(artifact_root / "input_manifest.json")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        fail("input manifest has no inputs")
    seen: set[str] = set()
    for record in inputs:
        if not isinstance(record, dict):
            fail("input manifest contains a non-object record")
        relative = record.get("path")
        if not isinstance(relative, str) or relative in seen:
            fail(f"invalid or duplicate input manifest path: {relative!r}")
        seen.add(relative)
        source = require_file(REPO_ROOT / relative)
        if (
            source.stat().st_size != record.get("size_bytes")
            or sha256(source) != record.get("sha256")
        ):
            fail(f"input manifest digest mismatch: {relative}")


def expected_artifact_files() -> set[str]:
    files = {
        "README.md",
        "input_manifest.json",
        "artifact_manifest.json",
        "provenance/frozen_source.sha256",
        "provenance/executed_source_manifest.json",
        "results/fixed_final_100.csv",
        "results/validation.json",
    }
    files.update(f"results/{name}" for name in ANALYSIS_OUTPUTS)
    for method in METHODS:
        for seed in SEEDS:
            prefix = f"data/{method}/seed{seed}"
            files.update(f"{prefix}/{name}" for name in JSON_FILES)
            files.add(f"{prefix}/metrics_selected_10k.csv")
    return files


def verify_artifact(artifact_root: Path) -> None:
    expected = expected_artifact_files()
    actual = {
        path.relative_to(artifact_root).as_posix()
        for path in artifact_root.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        fail(
            "artifact file inventory mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )

    for method in METHODS:
        for seed in SEEDS:
            seed_dir = artifact_root / "data" / method / f"seed{seed}"
            metrics = seed_dir / "metrics_selected_10k.csv"
            with metrics.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            require_equal(reader.fieldnames, list(SELECTED_METRIC_COLUMNS), "metric schema")
            require_equal(len(rows), EXPECTED_EVALUATIONS, "metric row count")
            require_equal(
                [int(row["step"]) for row in rows], list(EVAL_GRID), "metric grid"
            )

    for path in sorted(artifact_root.rglob("*.json")):
        value = load_json_strict(path)
        assert_no_absolute_strings(
            value, context=path.relative_to(artifact_root).as_posix()
        )

    for path in artifact_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(artifact_root).as_posix()
        if path.suffix in FORBIDDEN_SUFFIXES or path.name == "metrics.csv":
            fail(f"forbidden raw artifact was packaged: {relative}")
        if "checkpoint" in path.name.lower() and path.name != "best_checkpoint.json":
            fail(f"forbidden checkpoint payload was packaged: {relative}")
        if path.suffix in {".json", ".md", ".csv", ".sha256"}:
            text = path.read_text(encoding="utf-8")
            if (
                str(REPO_ROOT.resolve()) in text
                or RAW_ABSOLUTE_PREFIX.search(text)
                or NONFINITE_TOKEN.search(text)
            ):
                fail(f"unsanitized path or non-finite token in {relative}")

    fixed_path = artifact_root / "results" / "fixed_final_100.csv"
    with fixed_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fixed_rows = list(reader)
    require_equal(reader.fieldnames, list(FIXED_FINAL_COLUMNS), "fixed final schema")
    require_equal(len(fixed_rows), len(METHODS) * len(SEEDS), "fixed final row count")
    for row in fixed_rows:
        method = row["method"]
        seed = int(row["seed"])
        final_eval = load_json_strict(
            artifact_root
            / "data"
            / method
            / f"seed{seed}"
            / "final_evaluation_100.json"
        )
        require_close(
            row["success_mean"],
            require_rate(final_eval.get("eval_success_mean"), "fixed packaged success"),
            "fixed CSV success",
        )

    validation = load_json_strict(artifact_root / "results" / "validation.json")
    require_equal(validation.get("passed"), True, "validation status")
    gates = validation.get("raw_completion_gates")
    runs = validation.get("runs")
    if not isinstance(gates, list) or len(gates) != len(METHODS):
        fail("validation does not contain three raw completion gates")
    if not isinstance(runs, list) or len(runs) != len(METHODS) * len(SEEDS):
        fail("validation does not contain fifteen compact run records")
    if any(record.get("returncode") != 0 for record in gates):
        fail("validation contains a failed raw completion gate")

    verify_input_manifest(artifact_root)
    manifest_path = artifact_root / "artifact_manifest.json"
    manifest = load_json_strict(manifest_path)
    listed = {record["path"]: record for record in manifest.get("files", [])}
    files_without_manifest = {
        path.relative_to(artifact_root).as_posix(): path
        for path in artifact_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(listed) != set(files_without_manifest):
        fail("artifact SHA manifest inventory mismatch")
    for relative, path in files_without_manifest.items():
        record = listed[relative]
        if (
            path.stat().st_size != record.get("size_bytes")
            or sha256(path) != record.get("sha256")
        ):
            fail(f"artifact SHA manifest digest mismatch: {relative}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_root = args.runs_root.resolve()
    log_dir = args.log_dir.resolve()
    artifact_argument = args.artifact_root.absolute()
    expected_artifact_root = DEFAULT_ARTIFACT_ROOT.absolute()
    if artifact_argument != expected_artifact_root:
        fail(
            "artifact root is fixed for destructive replacement safety: "
            f"expected {expected_artifact_root}, got {artifact_argument}"
        )
    if artifact_argument.is_symlink():
        fail(f"artifact root must not be a symlink: {artifact_argument}")
    artifact_root = artifact_argument.resolve()
    repo_relative(runs_root)
    repo_relative(log_dir)
    repo_relative(artifact_root)
    if artifact_root.parent != (EXPERIMENT_ROOT / "paper_artifacts").resolve():
        fail("artifact root must be a direct child of experiment_FPP/paper_artifacts")

    frozen_records = parse_frozen_source_manifest()
    python = python_executable()
    gate_records = run_raw_completion_gates(
        runs_root=runs_root, log_dir=log_dir, python=python
    )

    artifact_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(
            prefix=f".{artifact_root.name}.tmp-", dir=str(artifact_root.parent)
        )
    )
    try:
        raw_inputs: list[Path] = []
        fixed_rows: list[dict[str, Any]] = []
        validation_runs: list[dict[str, Any]] = []

        for method in METHODS:
            for seed in SEEDS:
                packaged_json, rows, fixed_row, validation, inputs = validate_run(
                    runs_root=runs_root, method=method, seed=seed
                )
                raw_inputs.extend(inputs)
                fixed_rows.append(fixed_row)
                validation_runs.append(validation)
                destination = temp_root / "data" / method / f"seed{seed}"
                for filename, value in packaged_json.items():
                    write_json(
                        destination / filename, value, artifact_root=temp_root
                    )
                write_csv(
                    destination / "metrics_selected_10k.csv",
                    SELECTED_METRIC_COLUMNS,
                    rows,
                    artifact_root=temp_root,
                )

        provenance_dir = temp_root / "provenance"
        provenance_dir.mkdir(parents=True, exist_ok=True)
        frozen_text = require_file(FROZEN_SOURCE_MANIFEST).read_text(encoding="utf-8")
        if not frozen_text.endswith("\n"):
            frozen_text += "\n"
        (provenance_dir / "frozen_source.sha256").write_text(
            frozen_text, encoding="utf-8", newline="\n"
        )
        write_json(
            provenance_dir / "executed_source_manifest.json",
            {
                "schema_version": 1,
                "description": "Queue-frozen worktree snapshot used by the completed run",
                "frozen_manifest_repository_path": repo_relative(
                    FROZEN_SOURCE_MANIFEST
                ),
                "frozen_manifest_sha256": sha256(FROZEN_SOURCE_MANIFEST),
                "source_base_commit_full_recorded_by_queue": SOURCE_BASE_COMMIT_FULL,
                "source_identity_note": (
                    "The base commit predates local run.py and queue-script changes; "
                    "the per-file frozen SHA256 records define the exact executed source."
                ),
                "repository_head_observed_at_packaging": git_head(),
                "experiment_git_status_observed_at_packaging": experiment_git_status(),
                "all_current_files_match_frozen_execution_hashes": True,
                "files": frozen_records,
            },
            artifact_root=temp_root,
        )

        analyzer_record = run_analyzer(
            data_root=temp_root / "data",
            output_dir=temp_root / "results",
            python=python,
        )
        normalize_analysis_outputs(temp_root / "results", artifact_root=temp_root)
        write_csv(
            temp_root / "results" / "fixed_final_100.csv",
            FIXED_FINAL_COLUMNS,
            fixed_rows,
            artifact_root=temp_root,
        )
        verify_analysis_semantics(temp_root, python=python)

        source_files = [EXPERIMENT_DIR / record["execution_relative_path"] for record in frozen_records]
        all_inputs = sorted(
            set(
                raw_inputs
                + source_files
                + [
                    Path(__file__).resolve(),
                    ANALYZER.resolve(),
                    VALIDATOR.resolve(),
                    FROZEN_SOURCE_MANIFEST.resolve(),
                ]
            )
        )
        write_json(
            temp_root / "input_manifest.json",
            {
                "schema_version": 1,
                "repository_head_observed_at_packaging": git_head(),
                "raw_run_count": len(METHODS) * len(SEEDS),
                "frozen_source_manifest_sha256": sha256(FROZEN_SOURCE_MANIFEST),
                "inputs": [
                    file_record(path, relative_to=REPO_ROOT) for path in all_inputs
                ],
            },
            artifact_root=temp_root,
        )

        analysis = load_json_strict(temp_root / "results" / "analysis_results.json")
        write_json(
            temp_root / "results" / "validation.json",
            {
                "schema_version": 1,
                "passed": True,
                "validated_at_utc": utc_now(),
                "methods": list(METHODS),
                "seeds": list(SEEDS),
                "expected_steps": TOTAL_STEPS,
                "expected_episodes": EXPECTED_EPISODES,
                "expected_updates": EXPECTED_UPDATES,
                "expected_replay_size": EXPECTED_REPLAY_SIZE,
                "evaluation_grid": list(EVAL_GRID),
                "fixed_final_eval_episodes": FINAL_EVAL_EPISODES,
                "raw_completion_gates": gate_records,
                "runs": validation_runs,
                "paired_analysis": {
                    "verified_by_deterministic_rerun": True,
                    "analyzer_repository_path": repo_relative(ANALYZER),
                    "analyzer_sha256": sha256(ANALYZER),
                    "invocation": analyzer_record,
                    "primary_comparison": "ddpg_aer_vs_ddpg_annealing_er",
                    "secondary_comparison": "ddpg_aer_vs_ddpg_er",
                },
                "executed_source": {
                    "queue_frozen_manifest_verified": True,
                    "file_count": len(frozen_records),
                    "manifest_sha256": sha256(FROZEN_SOURCE_MANIFEST),
                },
            },
            artifact_root=temp_root,
        )
        (temp_root / "README.md").write_text(
            make_readme(analysis, fixed_rows), encoding="utf-8", newline="\n"
        )

        packaged_files = sorted(
            path
            for path in temp_root.rglob("*")
            if path.is_file() and path.name != "artifact_manifest.json"
        )
        write_json(
            temp_root / "artifact_manifest.json",
            {
                "schema_version": 1,
                "generated_at_utc": utc_now(),
                "repository_head_observed_at_packaging": git_head(),
                "coverage": {
                    "environment": "FetchPickAndPlace-v4",
                    "methods": list(METHODS),
                    "seeds": list(SEEDS),
                    "steps_per_run": TOTAL_STEPS,
                    "evaluations_per_run": EXPECTED_EVALUATIONS,
                    "fixed_final_eval_episodes": FINAL_EVAL_EPISODES,
                    "run_count": len(METHODS) * len(SEEDS),
                },
                "excluded": [
                    "model checkpoints and replay state",
                    "full metrics.csv",
                    "metrics.jsonl",
                    "run and launcher logs",
                    "bytecode caches",
                ],
                "files": [
                    file_record(path, relative_to=temp_root) for path in packaged_files
                ],
            },
            artifact_root=temp_root,
        )

        verify_artifact(temp_root)
        if artifact_root.is_symlink():
            fail(f"refusing to replace a symlinked artifact root: {artifact_root}")
        if artifact_root.exists() and not artifact_root.is_dir():
            fail(f"artifact root exists but is not a directory: {artifact_root}")
        if artifact_root.exists():
            shutil.rmtree(artifact_root)
        temp_root.replace(artifact_root)
        print(
            "Packaged and verified 15 DDPG AER three-arm runs at "
            f"{artifact_root.relative_to(REPO_ROOT)}"
        )
        return 0
    except Exception:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
