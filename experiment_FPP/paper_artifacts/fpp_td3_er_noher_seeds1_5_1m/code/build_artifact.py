#!/usr/bin/env python3
"""Build the compact, auditable five-seed FPP TD3-ER paper artifact.

Only Python's standard library is used.  The command is fail-closed: every seed
must contain a complete 1M run on the exact 10k evaluation grid before the
destination is created.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ARTIFACT_ID = "fpp_td3_er_noher_seeds1_5_1m"
SEEDS = (1, 2, 3, 4, 5)
TOTAL_STEPS = 1_000_000
EVAL_GRID = tuple(range(10_000, TOTAL_STEPS + 1, 10_000))
RUN_TEMPLATE = "fetch_pick_and_place_td3_er_noher_s{seed}_1000000"
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
REQUIRED_METRIC_COLUMNS = {
    "step",
    "eval/return_mean",
    "eval/success_mean",
    "eval/goal_success_mean",
    "eval/env_success_mean",
    "eval/AUC_success",
}
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
HEX40 = re.compile(r"^[0-9a-f]{40}$")


class ArtifactError(ValueError):
    """Raised when source data does not satisfy the frozen protocol."""


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    repo = here.parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=repo / "experiment_FPP/runs/fpp_td3_aer_noher_1m",
        help="Directory containing the five TD3-ER run directories.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=repo / f"experiment_FPP/paper_artifacts/{ARTIFACT_ID}",
        help="Exact destination directory (not merely its parent).",
    )
    parser.add_argument(
        "--frozen-source-manifest",
        type=Path,
        required=True,
        help="SHA256 manifest for the result-producing experiment source.",
    )
    parser.add_argument(
        "--source-commit-full",
        required=True,
        help="Full 40-character Git commit containing the frozen source.",
    )
    parser.add_argument(
        "--experiment-source-root",
        type=Path,
        help="If supplied, verify the frozen source manifest before packaging.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing exact artifact directory after validation succeeds.",
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
        raise ArtifactError(f"missing or empty required file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"expected JSON object: {path}")
    return value


def finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"{label}: expected finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ArtifactError(f"{label}: expected finite number, got {value!r}")
    return number


def rate(value: Any, label: str) -> float:
    number = finite(value, label)
    if not 0.0 <= number <= 1.0:
        raise ArtifactError(f"{label}: expected rate in [0,1], got {number}")
    return number


def equal_number(left: Any, right: Any, label: str) -> None:
    a = finite(left, f"{label} (left)")
    b = finite(right, f"{label} (right)")
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12):
        raise ArtifactError(f"{label}: {a!r} != {b!r}")


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
        raise ArtifactError(f"{label}: invalid first-hit value {value!r}") from exc


def sanitize(value: Any, source_root: Path) -> Any:
    """Remove machine-specific absolute source paths without changing numbers."""
    if isinstance(value, dict):
        return {key: sanitize(item, source_root) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, source_root) for item in value]
    if isinstance(value, str):
        prefix = str(source_root.resolve())
        return value.replace(prefix, "${SOURCE_ROOT}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def select_metrics(path: Path, seed: int) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ArtifactError(f"seed {seed}: missing metrics.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ArtifactError(f"seed {seed}: metrics.csv has no header")
        missing = REQUIRED_METRIC_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ArtifactError(f"seed {seed}: missing metric columns {sorted(missing)}")
        selected_columns = [name for name in SELECTED_COLUMNS if name in reader.fieldnames]
        by_step: dict[int, dict[str, str]] = {}
        for row_number, row in enumerate(reader, start=2):
            raw_step = row.get("step", "")
            try:
                step = int(raw_step)
            except (TypeError, ValueError):
                continue
            if step not in EVAL_GRID or not row.get("eval/success_mean", "").strip():
                continue
            if step in by_step:
                raise ArtifactError(f"seed {seed}: duplicate eval row at step {step}")
            for field in (
                "eval/success_mean",
                "eval/goal_success_mean",
                "eval/env_success_mean",
                "eval/AUC_success",
            ):
                rate(row.get(field), f"seed {seed} {field} line {row_number}")
            by_step[step] = {name: row.get(name, "") for name in selected_columns}
    if tuple(sorted(by_step)) != EVAL_GRID:
        missing = sorted(set(EVAL_GRID).difference(by_step))
        extra = sorted(set(by_step).difference(EVAL_GRID))
        raise ArtifactError(
            f"seed {seed}: evaluation grid mismatch; missing={missing}, extra={extra}"
        )
    return [by_step[step] for step in EVAL_GRID]


def validate_run(run_dir: Path, seed: int) -> tuple[dict[str, Any], list[dict[str, str]]]:
    for filename in JSON_FILES:
        if not (run_dir / filename).is_file():
            raise ArtifactError(f"seed {seed}: missing {run_dir / filename}")
    config = load_json(run_dir / "config.json")
    summary = load_json(run_dir / "summary.json")
    env = load_json(run_dir / "env_snapshot.json")
    final_eval = load_json(run_dir / "final_evaluation_100.json")
    best = load_json(run_dir / "best_checkpoint.json")
    expected_name = RUN_TEMPLATE.format(seed=seed)
    expected_config = {
        "run_name": expected_name,
        "method": "td3_er",
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
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ArtifactError(
                f"seed {seed}: config.{key} expected {expected!r}, got {config.get(key)!r}"
            )
    if config.get("seeds", {}).get("env_seed") != seed:
        raise ArtifactError(f"seed {seed}: config seed mapping is inconsistent")
    for key, expected in {
        "method": "td3_er",
        "total_steps": TOTAL_STEPS,
        "episodes": 20_000,
        "total_updates": 995_001,
        "fixed_final_eval_policy": "final",
        "fixed_final_eval_episodes": 100,
        "fixed_final_eval_seed": 100_010_000 + seed,
    }.items():
        if summary.get(key) != expected:
            raise ArtifactError(
                f"seed {seed}: summary.{key} expected {expected!r}, got {summary.get(key)!r}"
            )
    rows = select_metrics(run_dir / "metrics.csv", seed)
    last = rows[-1]
    recomputed_auc, recomputed_best, recomputed_best_step, recomputed_hits = recompute_curve(rows, seed)
    equal_number(summary.get("eval_AUC_success"), recomputed_auc, f"seed {seed} recomputed AUC")
    equal_number(last.get("eval/AUC_success"), recomputed_auc, f"seed {seed} selected-row AUC")
    equal_number(summary.get("best_eval_success"), recomputed_best, f"seed {seed} recomputed best")
    equal_number(last.get("eval/best_success"), recomputed_best, f"seed {seed} selected-row best")
    equal_number(last.get("eval/final_success"), last.get("eval/success_mean"), f"seed {seed} selected-row final")
    if int(summary.get("best_eval_step", -1)) != recomputed_best_step:
        raise ArtifactError(f"seed {seed}: recomputed first best step mismatch")
    for threshold, expected_hit in recomputed_hits.items():
        label = f"first_hit_{threshold:.2f}"
        if normalized_hit(summary.get(label), f"seed {seed} summary {label}") != expected_hit:
            raise ArtifactError(f"seed {seed}: recomputed summary {label} mismatch")
        if normalized_hit(last.get(f"eval/{label}"), f"seed {seed} selected {label}") != expected_hit:
            raise ArtifactError(f"seed {seed}: recomputed selected-row {label} mismatch")
    for summary_key, metric_key in {
        "final_eval_return": "eval/return_mean",
        "final_eval_success": "eval/success_mean",
        "final_eval_goal_success": "eval/goal_success_mean",
        "final_eval_env_success": "eval/env_success_mean",
        "eval_AUC_success": "eval/AUC_success",
    }.items():
        equal_number(summary.get(summary_key), last.get(metric_key), f"seed {seed} {summary_key}")
    for key in ("best_eval_success", "eval_AUC_success"):
        rate(summary.get(key), f"seed {seed} summary.{key}")
    if int(best.get("step", -1)) != int(summary.get("best_eval_step", -2)):
        raise ArtifactError(f"seed {seed}: best checkpoint step disagrees with summary")
    equal_number(
        best.get("eval_success_mean"),
        summary.get("best_eval_success"),
        f"seed {seed} best success",
    )
    for final_key, summary_key in {
        "eval_return_mean": "fixed_final_eval_return",
        "eval_goal_success_mean": "fixed_final_eval_goal_success",
        "eval_env_success_mean": "fixed_final_eval_env_success",
    }.items():
        equal_number(final_eval.get(final_key), summary.get(summary_key), f"seed {seed} final {final_key}")
    for key, expected in {
        "step": TOTAL_STEPS,
        "episodes": 100,
        "base_seed": 100_010_000 + seed,
        "policy": "final",
    }.items():
        if final_eval.get(key) != expected:
            raise ArtifactError(f"seed {seed}: final evaluation {key} mismatch")
    for env_name in ("train_env", "eval_env"):
        item = env.get(env_name)
        if not isinstance(item, dict):
            raise ArtifactError(f"seed {seed}: missing {env_name} snapshot")
        for key, expected in {
            "env_id": "FetchPickAndPlace-v4",
            "obs_dim": 25,
            "goal_dim": 3,
            "state_dim": 28,
            "action_dim": 4,
            "max_episode_steps": 50,
        }.items():
            if item.get(key) != expected:
                raise ArtifactError(f"seed {seed}: {env_name}.{key} mismatch")
    result = {
        "artifact_id": ARTIFACT_ID,
        "seed": seed,
        "run_name": expected_name,
        "method": "td3_er",
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
        raise ArtifactError(f"seed {seed}: runtime and throughput must be positive")
    return result, rows


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    fields = list(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def aggregate_rows(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in AGGREGATE_FIELDS:
        values = [finite(row[field], field) for row in per_seed]
        rows.append(
            {
                "metric": field,
                "n": len(values),
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values),
                "min": min(values),
                "max": max(values),
            }
        )
    return rows


def read_frozen_manifest(path: Path) -> list[tuple[str, str]]:
    """Read common JSON or ``sha256  relative/path`` source manifests."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ArtifactError(f"missing frozen source manifest: {path}")
    text = path.read_text(encoding="utf-8")
    entries: list[tuple[str, str]] = []
    if path.suffix.lower() == ".json":
        value = json.loads(text)
        candidates: Any = value
        if isinstance(value, dict):
            candidates = value.get("files", value.get("source_files", value))
        if isinstance(candidates, dict):
            entries = [(str(name), str(digest)) for name, digest in candidates.items()]
        elif isinstance(candidates, list):
            for item in candidates:
                if not isinstance(item, dict):
                    raise ArtifactError("invalid JSON frozen source manifest entry")
                entries.append((str(item.get("path", "")), str(item.get("sha256", ""))))
    else:
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line.strip())
            if not match:
                raise ArtifactError(f"invalid source manifest line {line_number}: {line!r}")
            entries.append((match.group(2), match.group(1).lower()))
    if not entries:
        raise ArtifactError("frozen source manifest has no file entries")
    seen: set[str] = set()
    normalized: list[tuple[str, str]] = []
    for name, digest in entries:
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or name in seen:
            raise ArtifactError(f"unsafe or duplicate source manifest path: {name!r}")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ArtifactError(f"invalid SHA256 for {name!r}: {digest!r}")
        seen.add(name)
        normalized.append((name, digest.lower()))
    return normalized


def verify_frozen_sources(manifest: Path, root: Path) -> None:
    root = root.resolve()
    for name, expected in read_frozen_manifest(manifest):
        candidate = (root / name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ArtifactError(f"source manifest escapes root: {name}") from exc
        if not candidate.is_file():
            raise ArtifactError(f"frozen source file missing: {candidate}")
        actual = sha256(candidate)
        if actual != expected:
            raise ArtifactError(f"frozen source hash mismatch: {name}")


def readme_text(source_commit: str) -> str:
    return f"""# FPP TD3-ER no-HER, seeds 1-5, 1M steps

This is the compact paper artifact for the uniform-replay TD3 control in
`FetchPickAndPlace-v4` with sparse rewards and HER disabled.  It contains five
paired seeds, the exact 10k-step evaluation grid, fixed 100-episode final
evaluations, aggregate tables, and provenance needed for numerical review.

## Frozen protocol

- Artifact ID: `{ARTIFACT_ID}`
- Method: `td3_er` (uniform replay, no learning-rate annealing)
- Seeds: 1, 2, 3, 4, 5
- Training: 1,000,000 environment steps
- Evaluation grid: 10,000 through 1,000,000, every 10,000 steps
- Online evaluation: 20 deterministic episodes per grid point
- Fixed final evaluation: 100 deterministic episodes
- HER ratio: 0.0
- Source commit: `{source_commit}`

Each seed directory under `data/td3_er/{{1..5}}/` contains `summary.json`, `config.json`,
`env_snapshot.json`, `final_evaluation_100.json`, `best_checkpoint.json`, and
the reduced `metrics_selected_10k.csv`.  Model checkpoints, full-resolution
metrics, JSONL streams, and logs are intentionally excluded.
SHA256SUMS covers every artifact file except SHA256SUMS itself.

## Verification

From this directory:

```bash
python code/verify_artifact.py .
```

To additionally check the result-producing source against the frozen source
manifest copied into `provenance/`:

```bash
python code/verify_artifact.py . --experiment-source-root /path/to/experiment/source
```
"""


def build(args: argparse.Namespace) -> Path:
    source_root = args.source_root.resolve()
    artifact_root = args.artifact_root.resolve()
    frozen_manifest = args.frozen_source_manifest.resolve()
    source_commit = args.source_commit_full.lower()
    if not HEX40.fullmatch(source_commit):
        raise ArtifactError("--source-commit-full must be a full 40-character lowercase SHA-1")
    read_frozen_manifest(frozen_manifest)
    if args.experiment_source_root:
        verify_frozen_sources(frozen_manifest, args.experiment_source_root)
    if artifact_root == source_root or source_root in artifact_root.parents:
        raise ArtifactError("artifact destination must not be inside the result source root")
    if artifact_root.exists() and not args.replace:
        raise ArtifactError(f"artifact destination already exists: {artifact_root}")

    validated: list[tuple[int, Path, dict[str, Any], list[dict[str, str]]]] = []
    for seed in SEEDS:
        run_dir = source_root / RUN_TEMPLATE.format(seed=seed)
        if not run_dir.is_dir():
            raise ArtifactError(f"missing seed {seed} run directory: {run_dir}")
        result, rows = validate_run(run_dir, seed)
        validated.append((seed, run_dir, result, rows))

    artifact_root.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{ARTIFACT_ID}.", dir=artifact_root.parent))
    source_artifacts: list[dict[str, Any]] = []
    try:
        (temp / "data").mkdir()
        (temp / "code").mkdir()
        (temp / "provenance").mkdir()
        per_seed: list[dict[str, Any]] = []
        for seed, run_dir, result, rows in validated:
            packaged_dir = temp / "data" / "td3_er" / str(seed)
            packaged_dir.mkdir(parents=True)
            for filename in JSON_FILES:
                source = run_dir / filename
                destination = packaged_dir / filename
                write_json(destination, sanitize(load_json(source), source_root))
                source_artifacts.append(
                    {
                        "source_path": str(source.relative_to(source_root)),
                        "source_sha256": sha256(source),
                        "packaged_path": str(destination.relative_to(temp)),
                        "transformation": "JSON parsed; absolute source-root prefix replaced",
                    }
                )
            metrics_out = packaged_dir / "metrics_selected_10k.csv"
            write_csv(metrics_out, rows[0].keys(), rows)
            source_artifacts.append(
                {
                    "source_path": str((run_dir / "metrics.csv").relative_to(source_root)),
                    "source_sha256": sha256(run_dir / "metrics.csv"),
                    "packaged_path": str(metrics_out.relative_to(temp)),
                    "transformation": "Exact 10k evaluation rows and selected columns",
                }
            )
            per_seed.append(result)

        write_csv(temp / "per_seed_results.csv", per_seed[0].keys(), per_seed)
        write_csv(
            temp / "aggregate_results.csv",
            ("metric", "n", "mean", "sample_std", "min", "max"),
            aggregate_rows(per_seed),
        )
        (temp / "README.md").write_text(readme_text(source_commit), encoding="utf-8", newline="\n")
        shutil.copy2(Path(__file__).resolve(), temp / "code" / "build_artifact.py")
        write_json(
            temp / "completion_validation.json",
            {
                "artifact_id": ARTIFACT_ID,
                "all_runs_complete": True,
                "evaluation_grid": {
                    "start": EVAL_GRID[0],
                    "stop": EVAL_GRID[-1],
                    "stride": 10_000,
                    "count": len(EVAL_GRID),
                },
                "required_files_per_seed": [*JSON_FILES, "metrics_selected_10k.csv"],
                "seeds": [
                    {
                        "seed": row["seed"],
                        "run_name": row["run_name"],
                        "total_steps": row["total_steps"],
                        "evaluation_points": len(EVAL_GRID),
                        "summary_parity": True,
                        "fixed_final_evaluation_parity": True,
                    }
                    for row in per_seed
                ],
            },
        )
        verifier_source = Path(__file__).with_name("verify_artifact.py").resolve()
        if not verifier_source.is_file():
            raise ArtifactError(f"missing sibling verifier: {verifier_source}")
        shutil.copy2(verifier_source, temp / "code" / "verify_artifact.py")
        frozen_name = "frozen_source_manifest" + (frozen_manifest.suffix or ".sha256")
        frozen_out = temp / "provenance" / frozen_name
        shutil.copy2(frozen_manifest, frozen_out)
        (temp / "provenance" / "source_commit.txt").write_text(
            source_commit + "\n", encoding="utf-8", newline="\n"
        )

        pre_manifest_files = sorted(
            path for path in temp.rglob("*") if path.is_file()
        )
        manifest = {
            "schema_version": "1.0",
            "sha256sums_scope": "all artifact files except SHA256SUMS itself",
            "artifact_id": ARTIFACT_ID,
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "source_commit_full": source_commit,
            "frozen_source_manifest": str(frozen_out.relative_to(temp)),
            "completion": {
                "seeds": list(SEEDS),
                "required_final_step": TOTAL_STEPS,
                "evaluation_grid_start": EVAL_GRID[0],
                "evaluation_grid_stop": EVAL_GRID[-1],
                "evaluation_grid_stride": 10_000,
                "all_runs_complete": True,
            },
            "source_artifacts": source_artifacts,
            "excluded": [
                {"pattern": "checkpoint*.pt", "reason": "large model checkpoints"},
                {"pattern": "best_checkpoint.pt", "reason": "large model checkpoint"},
                {"pattern": "metrics.csv", "reason": "full-resolution metrics"},
                {"pattern": "metrics.jsonl", "reason": "raw event stream"},
                {"pattern": "*.log", "reason": "runtime logs"},
            ],
            "packaged_files_before_manifests": [
                {
                    "path": str(path.relative_to(temp)),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in pre_manifest_files
            ],
        }
        write_json(temp / "artifact_manifest.json", manifest)
        inventory = sorted(path for path in temp.rglob("*") if path.is_file())
        with (temp / "SHA256SUMS").open("w", encoding="utf-8", newline="\n") as handle:
            for path in inventory:
                handle.write(f"{sha256(path)}  {path.relative_to(temp).as_posix()}\n")

        subprocess.run(
            [sys.executable, str(temp / "code" / "verify_artifact.py"), str(temp)],
            check=True,
        )
        if artifact_root.exists():
            shutil.rmtree(artifact_root)
        os.replace(temp, artifact_root)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return artifact_root


def main() -> int:
    try:
        output = build(parse_args())
    except (ArtifactError, OSError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"[artifact-build-fail] {exc}", file=sys.stderr)
        return 1
    print(f"[artifact-build-pass] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
