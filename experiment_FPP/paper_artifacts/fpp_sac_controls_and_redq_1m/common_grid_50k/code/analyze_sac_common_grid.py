from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


COMMON_GRID_STEPS = tuple(range(50_000, 1_000_001, 50_000))
REQUIRED_SEEDS = (1, 2, 3, 4, 5)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 27_027
PRIMARY_SUCCESS_FIELD = "eval/env_success_mean"
FALLBACK_SUCCESS_FIELD = "eval/success_mean"


class AnalysisError(RuntimeError):
    pass


class ArtifactValidationError(AnalysisError):
    pass


class GridValidationError(AnalysisError):
    pass


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    label: str
    root: Path
    diagnostic: bool = False


@dataclass(frozen=True)
class RunArtifact:
    condition_key: str
    method: str
    seed: int
    run_dir: Path
    metrics_path: Path
    summary_path: Path


@dataclass(frozen=True)
class EvaluationGrid:
    values: dict[int, float]
    source_fields: dict[int, str]
    duplicate_counts: dict[int, int]
    source_field_counts: dict[str, int]


@dataclass(frozen=True)
class ArtifactAudit:
    runs: tuple[RunArtifact, ...]
    missing_artifacts: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_artifacts and not self.errors


@dataclass(frozen=True)
class ComparisonSpec:
    name: str
    left_key: str
    right_key: str


COMPARISONS = (
    ComparisonSpec("SABER - HER", "saber", "her"),
    ComparisonSpec("SABER - Random-Support", "saber", "random_support"),
    ComparisonSpec(
        "SABER - Random-Fixed-Count",
        "saber",
        "random_fixed_count",
    ),
    ComparisonSpec("SABER - high-RR x4", "saber", "high_rr_x4"),
    ComparisonSpec("REDQ-HER TQB - SABER", "redq_her_tqb", "saber"),
)


PER_SEED_FIELDS = [
    "method_key",
    "method",
    "seed",
    "run_name",
    "run_dir",
    "metrics_path",
    "summary_path",
    "common_grid_auc",
    "common_grid_final_success",
    "common_grid_hit90",
    "native_auc",
    "native_minus_common_auc",
    "common_checkpoint_count",
    "common_checkpoints",
]

AGGREGATE_FIELDS = [
    "method_key",
    "method",
    "n",
    "seeds",
    "common_grid_auc_mean",
    "common_grid_auc_std",
    "common_grid_final_success_mean",
    "common_grid_final_success_std",
    "native_auc_mean",
    "native_auc_std",
    "native_minus_common_auc_mean",
    "native_minus_common_auc_std",
    "common_grid_hit90_reached_seed_count",
    "common_grid_hit90_median_reached",
    "common_grid_hit90_mean_reached",
    "common_grid_hit90_std_reached",
]

PAIRED_FIELDS = [
    "comparison",
    "left_method",
    "right_method",
    "seed",
    "left_common_grid_auc",
    "right_common_grid_auc",
    "common_grid_auc_difference",
    "left_common_grid_final_success",
    "right_common_grid_final_success",
    "common_grid_final_success_difference",
    "left_common_grid_hit90",
    "right_common_grid_hit90",
    "common_grid_hit90_difference",
    "left_native_auc",
    "right_native_auc",
    "native_auc_difference",
]

BOOTSTRAP_FIELDS = [
    "comparison",
    "n_seed_pairs",
    "paired_difference_mean",
    "bootstrap_mean",
    "ci_lower_95",
    "ci_upper_95",
    "resamples",
    "rng_seed",
    "seed_differences",
]


def build_condition_specs(
    repo_root: Path,
    redq_root: Path,
    include_diagnostic: bool = False,
    artifact_root: Path | None = None,
) -> list[ConditionSpec]:
    repository_runs_root = repo_root / "experiment_FPP" / "runs"
    runs_root = artifact_root if artifact_root is not None else repository_runs_root
    if artifact_root is not None:
        redq_root = artifact_root / "redq_her_utd4_n10_m2_tqbound_ls512_1m"
    specs = [
        ConditionSpec("her", "HER", runs_root / "sac_her_baseline_ls512"),
        ConditionSpec(
            "saber",
            "HER+SABER",
            runs_root / "sac_saber_critic_only_v4_ls512",
        ),
        ConditionSpec(
            "random_support",
            "Random-Support",
            runs_root / "sac_support_random_critic_only_ls512",
        ),
        ConditionSpec(
            "random_fixed_count",
            "Random-Fixed-Count",
            runs_root / "sac_random_fixed_count_k76_critic_only_ls512",
        ),
        ConditionSpec(
            "high_rr_x4",
            "HER high-RR x4",
            runs_root / "sac_her_high_replay_ratio_x4",
        ),
        ConditionSpec("redq_her_tqb", "REDQ-HER TQB", redq_root),
    ]
    if include_diagnostic:
        specs.append(
            ConditionSpec(
                "diagnostic",
                "SABER diagnostic",
                repository_runs_root / "fpp_sac_saber_diagnostic_ls512_1m",
                diagnostic=True,
            )
        )
    return specs


def _finite_float(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_step(value: object, path: Path, line_number: int) -> int:
    number = _finite_float(value)
    if number is None or not number.is_integer():
        raise GridValidationError(
            f"{path}: invalid evaluation step {value!r} at CSV line {line_number}"
        )
    return int(number)


def _seed_from_run_dir(run_dir: Path) -> int:
    config_seed: int | None = None
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactValidationError(f"Cannot parse {config_path}: {exc}") from exc
        seeds = config.get("seeds", {})
        candidates = [
            seeds.get("env_seed") if isinstance(seeds, dict) else None,
            seeds.get("torch_seed") if isinstance(seeds, dict) else None,
            config.get("seed"),
        ]
        parsed = {int(value) for value in candidates if value not in (None, "")}
        if len(parsed) > 1:
            raise ArtifactValidationError(
                f"Conflicting seed values in {config_path}: {sorted(parsed)}"
            )
        if parsed:
            config_seed = parsed.pop()

    match = re.search(r"(?:^|_)s(\d+)(?:_|$)", run_dir.name)
    name_seed = int(match.group(1)) if match else None
    if config_seed is not None and name_seed is not None and config_seed != name_seed:
        raise ArtifactValidationError(
            f"Seed mismatch for {run_dir}: config={config_seed}, run_name={name_seed}"
        )
    seed = config_seed if config_seed is not None else name_seed
    if seed is None:
        raise ArtifactValidationError(
            f"Cannot determine seed from config.json or run directory name: {run_dir}"
        )
    return seed


def discover_condition_runs(
    spec: ConditionSpec,
    required_seeds: Sequence[int] = REQUIRED_SEEDS,
) -> tuple[list[RunArtifact], list[str], list[str]]:
    runs: list[RunArtifact] = []
    missing: list[str] = []
    errors: list[str] = []
    expected_metrics = spec.root / "*" / "metrics_selected_10k.csv"
    expected_summaries = spec.root / "*" / "summary.json"
    if not spec.root.is_dir():
        missing.extend([str(expected_metrics), str(expected_summaries)])
        return runs, missing, errors

    metrics_paths = sorted(spec.root.glob("*/metrics_selected_10k.csv"))
    if not metrics_paths:
        missing.extend([str(expected_metrics), str(expected_summaries)])
        return runs, missing, errors

    by_seed: dict[int, list[RunArtifact]] = defaultdict(list)
    for metrics_path in metrics_paths:
        run_dir = metrics_path.parent
        summary_path = run_dir / "summary.json"
        try:
            seed = _seed_from_run_dir(run_dir)
        except ArtifactValidationError as exc:
            errors.append(str(exc))
            continue
        artifact = RunArtifact(
            condition_key=spec.key,
            method=spec.label,
            seed=seed,
            run_dir=run_dir,
            metrics_path=metrics_path,
            summary_path=summary_path,
        )
        runs.append(artifact)
        by_seed[seed].append(artifact)
        if not summary_path.is_file():
            missing.append(str(summary_path))

    for seed in required_seeds:
        matches = by_seed.get(seed, [])
        if not matches:
            missing.extend(
                [
                    str(spec.root / f"*_s{seed}_*" / "metrics_selected_10k.csv"),
                    str(spec.root / f"*_s{seed}_*" / "summary.json"),
                ]
            )
        elif len(matches) > 1:
            paths = ", ".join(str(run.run_dir) for run in matches)
            errors.append(
                f"{spec.label}: expected one run for seed {seed}, found {len(matches)}: "
                f"{paths}"
            )

    unexpected_seeds = sorted(set(by_seed) - set(required_seeds))
    if unexpected_seeds:
        errors.append(
            f"{spec.label}: unexpected seeds discovered: {unexpected_seeds}; "
            f"required seeds are {list(required_seeds)}"
        )
    return sorted(runs, key=lambda run: (run.seed, run.run_dir.name)), missing, errors


def audit_artifacts(
    specs: Sequence[ConditionSpec],
    required_seeds: Sequence[int] = REQUIRED_SEEDS,
) -> ArtifactAudit:
    runs: list[RunArtifact] = []
    missing: list[str] = []
    errors: list[str] = []
    for spec in specs:
        spec_runs, spec_missing, spec_errors = discover_condition_runs(
            spec, required_seeds
        )
        runs.extend(spec_runs)
        missing.extend(spec_missing)
        errors.extend(spec_errors)
    return ArtifactAudit(
        runs=tuple(runs),
        missing_artifacts=tuple(dict.fromkeys(missing)),
        errors=tuple(dict.fromkeys(errors)),
    )


def read_evaluation_grid(
    metrics_path: Path,
    required_steps: Sequence[int] = COMMON_GRID_STEPS,
) -> EvaluationGrid:
    values: dict[int, float] = {}
    source_fields: dict[int, str] = {}
    duplicate_counts: dict[int, int] = {}
    source_counts: Counter[str] = Counter()
    try:
        handle = metrics_path.open("r", newline="", encoding="utf-8")
    except OSError as exc:
        raise ArtifactValidationError(f"Cannot read {metrics_path}: {exc}") from exc

    with handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if "step" not in fields:
            raise GridValidationError(f"{metrics_path}: missing required CSV column 'step'")
        if not fields.intersection({PRIMARY_SUCCESS_FIELD, FALLBACK_SUCCESS_FIELD}):
            raise GridValidationError(
                f"{metrics_path}: missing both evaluation success columns "
                f"'{PRIMARY_SUCCESS_FIELD}' and '{FALLBACK_SUCCESS_FIELD}'"
            )

        for row in reader:
            success = _finite_float(row.get(PRIMARY_SUCCESS_FIELD))
            source_field = PRIMARY_SUCCESS_FIELD
            if success is None:
                success = _finite_float(row.get(FALLBACK_SUCCESS_FIELD))
                source_field = FALLBACK_SUCCESS_FIELD
            if success is None:
                continue

            step = _parse_step(row.get("step"), metrics_path, reader.line_num)
            if step in values:
                duplicate_counts[step] = duplicate_counts.get(step, 1) + 1
                if not math.isclose(
                    values[step], success, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise GridValidationError(
                        f"{metrics_path}: conflicting duplicate evaluation values at "
                        f"step {step}: {values[step]} vs {success}"
                    )
            else:
                values[step] = success
                source_fields[step] = source_field
                duplicate_counts[step] = 1
            source_counts[source_field] += 1

    missing_steps = [int(step) for step in required_steps if int(step) not in values]
    if missing_steps:
        available = ", ".join(str(step) for step in sorted(values)) or "none"
        missing = ", ".join(str(step) for step in missing_steps)
        raise GridValidationError(
            f"{metrics_path}: missing required common-grid checkpoints: {missing}. "
            f"Available finite evaluation steps: {available}"
        )
    return EvaluationGrid(
        values=values,
        source_fields=source_fields,
        duplicate_counts=duplicate_counts,
        source_field_counts=dict(sorted(source_counts.items())),
    )


def normalized_trapezoidal_auc(
    steps: Sequence[int], successes: Sequence[float]
) -> float:
    if len(steps) != len(successes):
        raise ValueError("steps and successes must have the same length")
    if len(steps) < 2:
        raise ValueError("at least two grid points are required for trapezoidal AUC")
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError("steps must be strictly increasing")
    area = math.fsum(
        (right_step - left_step) * (left_value + right_value) / 2.0
        for left_step, right_step, left_value, right_value in zip(
            steps, steps[1:], successes, successes[1:]
        )
    )
    return area / float(steps[-1] - steps[0])


def first_hit90(
    steps: Sequence[int], successes: Sequence[float]
) -> int | None:
    for step, success in zip(steps, successes):
        if success >= 0.90:
            return int(step)
    return None


def _load_native_auc(summary_path: Path) -> float:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Cannot parse {summary_path}: {exc}") from exc
    native_auc = _finite_float(summary.get("eval_AUC_success"))
    if native_auc is None:
        raise ArtifactValidationError(
            f"{summary_path}: missing finite 'eval_AUC_success' for native AUC audit"
        )
    return native_auc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_run(
    artifact: RunArtifact,
    common_steps: Sequence[int] = COMMON_GRID_STEPS,
) -> dict:
    grid = read_evaluation_grid(artifact.metrics_path, common_steps)
    successes = [grid.values[int(step)] for step in common_steps]
    common_auc = normalized_trapezoidal_auc(common_steps, successes)
    native_auc = _load_native_auc(artifact.summary_path)
    return {
        "method_key": artifact.condition_key,
        "method": artifact.method,
        "seed": artifact.seed,
        "run_name": artifact.run_dir.name,
        "run_dir": str(artifact.run_dir),
        "metrics_path": str(artifact.metrics_path),
        "summary_path": str(artifact.summary_path),
        "common_grid_auc": common_auc,
        "common_grid_final_success": successes[-1],
        "common_grid_hit90": first_hit90(common_steps, successes),
        "native_auc": native_auc,
        "native_minus_common_auc": native_auc - common_auc,
        "common_checkpoint_count": len(common_steps),
        "common_checkpoints": ";".join(str(step) for step in common_steps),
        "_common_grid_successes": successes,
        "_all_evaluation_steps": sorted(grid.values),
        "_duplicate_evaluation_steps": {
            str(step): count
            for step, count in sorted(grid.duplicate_counts.items())
            if count > 1
        },
        "_success_source_field_counts": grid.source_field_counts,
        "_metrics_sha256": _sha256(artifact.metrics_path),
        "_summary_sha256": _sha256(artifact.summary_path),
    }


def analyze_runs(
    specs: Sequence[ConditionSpec],
    runs: Sequence[RunArtifact],
    common_steps: Sequence[int] = COMMON_GRID_STEPS,
) -> list[dict]:
    spec_order = {spec.key: index for index, spec in enumerate(specs)}
    ordered_runs = sorted(
        runs,
        key=lambda run: (spec_order[run.condition_key], run.seed, run.run_dir.name),
    )
    return [analyze_run(run, common_steps) for run in ordered_runs]


def _sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else math.nan


def aggregate_rows(per_seed_rows: Sequence[Mapping[str, object]]) -> list[dict]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    labels: dict[str, str] = {}
    order: list[str] = []
    for row in per_seed_rows:
        key = str(row["method_key"])
        if key not in grouped:
            order.append(key)
        grouped[key].append(row)
        labels[key] = str(row["method"])

    output: list[dict] = []
    for key in order:
        group = sorted(grouped[key], key=lambda row: int(row["seed"]))
        record: dict[str, object] = {
            "method_key": key,
            "method": labels[key],
            "n": len(group),
            "seeds": ";".join(str(int(row["seed"])) for row in group),
        }
        for metric in (
            "common_grid_auc",
            "common_grid_final_success",
            "native_auc",
            "native_minus_common_auc",
        ):
            values = [float(row[metric]) for row in group]
            record[f"{metric}_mean"] = statistics.mean(values)
            record[f"{metric}_std"] = _sample_std(values)

        reached = [
            int(row["common_grid_hit90"])
            for row in group
            if row.get("common_grid_hit90") is not None
        ]
        record["common_grid_hit90_reached_seed_count"] = len(reached)
        record["common_grid_hit90_median_reached"] = (
            statistics.median(reached) if reached else None
        )
        record["common_grid_hit90_mean_reached"] = (
            statistics.mean(reached) if reached else None
        )
        record["common_grid_hit90_std_reached"] = (
            _sample_std(reached) if reached else None
        )
        output.append(record)
    return output


def build_paired_rows(
    per_seed_rows: Sequence[Mapping[str, object]],
    comparisons: Sequence[ComparisonSpec] = COMPARISONS,
    required_seeds: Sequence[int] = REQUIRED_SEEDS,
) -> list[dict]:
    indexed: dict[tuple[str, int], Mapping[str, object]] = {}
    for row in per_seed_rows:
        key = (str(row["method_key"]), int(row["seed"]))
        if key in indexed:
            raise ArtifactValidationError(
                f"Duplicate per-seed result for method={key[0]}, seed={key[1]}"
            )
        indexed[key] = row

    paired: list[dict] = []
    for comparison in comparisons:
        for seed in required_seeds:
            left = indexed.get((comparison.left_key, int(seed)))
            right = indexed.get((comparison.right_key, int(seed)))
            if left is None or right is None:
                missing = []
                if left is None:
                    missing.append(f"{comparison.left_key}/seed={seed}")
                if right is None:
                    missing.append(f"{comparison.right_key}/seed={seed}")
                raise ArtifactValidationError(
                    f"Cannot pair {comparison.name}: missing {', '.join(missing)}"
                )
            left_hit = left.get("common_grid_hit90")
            right_hit = right.get("common_grid_hit90")
            paired.append(
                {
                    "comparison": comparison.name,
                    "left_method": left["method"],
                    "right_method": right["method"],
                    "seed": seed,
                    "left_common_grid_auc": left["common_grid_auc"],
                    "right_common_grid_auc": right["common_grid_auc"],
                    "common_grid_auc_difference": float(left["common_grid_auc"])
                    - float(right["common_grid_auc"]),
                    "left_common_grid_final_success": left[
                        "common_grid_final_success"
                    ],
                    "right_common_grid_final_success": right[
                        "common_grid_final_success"
                    ],
                    "common_grid_final_success_difference": float(
                        left["common_grid_final_success"]
                    )
                    - float(right["common_grid_final_success"]),
                    "left_common_grid_hit90": left_hit,
                    "right_common_grid_hit90": right_hit,
                    "common_grid_hit90_difference": (
                        int(left_hit) - int(right_hit)
                        if left_hit is not None and right_hit is not None
                        else None
                    ),
                    "left_native_auc": left["native_auc"],
                    "right_native_auc": right["native_auc"],
                    "native_auc_difference": float(left["native_auc"])
                    - float(right["native_auc"]),
                }
            )
    return paired


def paired_percentile_bootstrap(
    differences: Sequence[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    rng_seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired bootstrap requires at least one difference")
    if not np.isfinite(values).all():
        raise ValueError("paired bootstrap differences must all be finite")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        "n_seed_pairs": int(values.size),
        "paired_difference_mean": float(values.mean()),
        "bootstrap_mean": float(bootstrap_means.mean()),
        "ci_lower_95": float(lower),
        "ci_upper_95": float(upper),
        "resamples": int(resamples),
        "rng_seed": int(rng_seed),
    }


def build_bootstrap_rows(
    paired_rows: Sequence[Mapping[str, object]],
    comparisons: Sequence[ComparisonSpec] = COMPARISONS,
    resamples: int = BOOTSTRAP_RESAMPLES,
    rng_seed: int = BOOTSTRAP_SEED,
) -> list[dict]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in paired_rows:
        grouped[str(row["comparison"])].append(row)

    output: list[dict] = []
    for comparison in comparisons:
        group = sorted(grouped[comparison.name], key=lambda row: int(row["seed"]))
        differences = [float(row["common_grid_auc_difference"]) for row in group]
        result = paired_percentile_bootstrap(differences, resamples, rng_seed)
        output.append(
            {
                "comparison": comparison.name,
                **result,
                "seed_differences": ";".join(
                    f"{int(row['seed'])}:{float(row['common_grid_auc_difference']):.12g}"
                    for row in group
                ),
            }
        )
    return output


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fields: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _display_path(path: Path, repo_root: Path) -> str:
    return Path(os.path.relpath(path, repo_root)).as_posix()


def build_grid_validation(
    specs: Sequence[ConditionSpec],
    rows: Sequence[Mapping[str, object]],
    repo_root: Path,
    common_steps: Sequence[int] = COMMON_GRID_STEPS,
) -> dict:
    rows_by_key: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_key[str(row["method_key"])].append(row)
    conditions = []
    for spec in specs:
        discovered = []
        for row in sorted(rows_by_key[spec.key], key=lambda item: int(item["seed"])):
            discovered.append(
                {
                    "seed": int(row["seed"]),
                    "run_name": row["run_name"],
                    "run_dir": _display_path(Path(str(row["run_dir"])), repo_root),
                    "metrics_path": _display_path(
                        Path(str(row["metrics_path"])), repo_root
                    ),
                    "summary_path": _display_path(
                        Path(str(row["summary_path"])), repo_root
                    ),
                    "metrics_sha256": row["_metrics_sha256"],
                    "summary_sha256": row["_summary_sha256"],
                    "all_finite_evaluation_steps": row["_all_evaluation_steps"],
                    "common_checkpoint_count": row["common_checkpoint_count"],
                    "common_checkpoints": [int(step) for step in common_steps],
                    "duplicate_evaluation_steps": row[
                        "_duplicate_evaluation_steps"
                    ],
                    "success_source_field_counts": row[
                        "_success_source_field_counts"
                    ],
                }
            )
        conditions.append(
            {
                "method_key": spec.key,
                "method": spec.label,
                "root": _display_path(spec.root, repo_root),
                "diagnostic": spec.diagnostic,
                "expected_seeds": list(REQUIRED_SEEDS),
                "discovered_runs": discovered,
            }
        )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_passed": True,
        "missing_artifacts": [],
        "common_grid_steps": [int(step) for step in common_steps],
        "common_grid_point_count": len(common_steps),
        "success_field_preference": [
            PRIMARY_SUCCESS_FIELD,
            FALLBACK_SUCCESS_FIELD,
        ],
        "interpolation": False,
        "forward_fill": False,
        "smoothing": False,
        "extrapolation": False,
        "conditions": conditions,
    }


def _format_float(value: object, digits: int = 6) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def _ranking(aggregate: Sequence[Mapping[str, object]], metric: str) -> list[str]:
    indexed = list(enumerate(aggregate))
    ranked = sorted(
        indexed,
        key=lambda item: (-float(item[1][metric]), item[0]),
    )
    return [str(row["method"]) for _, row in ranked]


def render_report(
    specs: Sequence[ConditionSpec],
    per_seed: Sequence[Mapping[str, object]],
    aggregate: Sequence[Mapping[str, object]],
    bootstrap: Sequence[Mapping[str, object]],
    analysis_command: str,
) -> str:
    native_ranking = _ranking(aggregate, "native_auc_mean")
    common_ranking = _ranking(aggregate, "common_grid_auc_mean")
    ranking_changed = native_ranking != common_ranking
    lines = [
        "# SAC Common Evaluation Grid Reanalysis",
        "",
        "All metrics in this report are CPU-only reanalysis results. No RL training "
        "is run by this script.",
        "",
        "The shared evaluation grid is `50k, 100k, ..., 1M` (20 exact points). "
        "No interpolation, forward filling, smoothing, or extrapolation is used.",
        "",
        "## Artifact Validation",
        "",
        "Missing artifacts: none.",
        "",
        "| Method | Seed | Run | Metrics | Summary |",
        "|---|---:|---|---|---|",
    ]
    for row in per_seed:
        lines.append(
            f"| {row['method']} | {row['seed']} | `{row['run_name']}` | "
            f"`{row['metrics_path']}` | `{row['summary_path']}` |"
        )

    lines.extend(
        [
            "",
            "## Native Versus Common-Grid AUC",
            "",
            "| Method | Seed | Native AUC | Common-grid AUC | Native - Common | "
            "Final success | Hit90 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_seed:
        lines.append(
            f"| {row['method']} | {row['seed']} | "
            f"{_format_float(row['native_auc'])} | "
            f"{_format_float(row['common_grid_auc'])} | "
            f"{_format_float(row['native_minus_common_auc'])} | "
            f"{_format_float(row['common_grid_final_success'])} | "
            f"{row['common_grid_hit90'] or ''} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "Standard deviations are sample standard deviations (`ddof=1`).",
            "",
            "| Method | Seeds | Common AUC mean +/- std | Native AUC mean +/- std | "
            "Final mean +/- std | Hit90 reached | Median Hit90 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate:
        lines.append(
            f"| {row['method']} | {row['seeds']} | "
            f"{_format_float(row['common_grid_auc_mean'])} +/- "
            f"{_format_float(row['common_grid_auc_std'])} | "
            f"{_format_float(row['native_auc_mean'])} +/- "
            f"{_format_float(row['native_auc_std'])} | "
            f"{_format_float(row['common_grid_final_success_mean'])} +/- "
            f"{_format_float(row['common_grid_final_success_std'])} | "
            f"{row['common_grid_hit90_reached_seed_count']}/{row['n']} | "
            f"{_format_float(row['common_grid_hit90_median_reached'], 0)} |"
        )

    lines.extend(
        [
            "",
            f"Native AUC ranking: {' > '.join(native_ranking)}.",
            "",
            f"Common-grid AUC ranking: {' > '.join(common_ranking)}.",
            "",
            f"Method ranking changed after common-grid reanalysis: "
            f"{'yes' if ranking_changed else 'no'}.",
            "",
            "## Paired AUC Differences",
            "",
            "The intervals below are 95% paired percentile bootstrap CIs from "
            f"{BOOTSTRAP_RESAMPLES:,} resamples with RNG seed {BOOTSTRAP_SEED}. "
            "Seeds are resampled as paired AUC differences; methods are not "
            "resampled independently.",
            "",
            "| Comparison | Mean paired difference | 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for row in bootstrap:
        lines.append(
            f"| {row['comparison']} | "
            f"{_format_float(row['paired_difference_mean'])} | "
            f"[{_format_float(row['ci_lower_95'])}, "
            f"{_format_float(row['ci_upper_95'])}] |"
        )

    lines.extend(
        [
            "",
            "These intervals are reported directly; no broader statistical "
            "significance claim is made.",
            "",
            "## Reproducibility",
            "",
            f"Analysis command: `{analysis_command}`",
            "",
            "Test command: `python -m unittest "
            "experiment_FPP.shared.test_analyze_sac_common_grid`",
            "",
            "Generated files: `per_seed_common_grid.csv`, "
            "`aggregate_common_grid.csv`, `paired_seed_differences.csv`, "
            "`paired_bootstrap_ci.csv`, `grid_validation.json`, and `report.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_plot(
    output_dir: Path,
    per_seed: Sequence[Mapping[str, object]],
    common_steps: Sequence[int] = COMMON_GRID_STEPS,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise AnalysisError(
            "--plot requires matplotlib, but it is not installed"
        ) from exc

    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in per_seed:
        grouped[str(row["method"])].append(row)
    figure, axis = plt.subplots(figsize=(9.0, 5.5))
    x = np.asarray(common_steps, dtype=np.float64)
    for method, rows in grouped.items():
        curves = np.asarray(
            [row["_common_grid_successes"] for row in rows], dtype=np.float64
        )
        curve_mean = curves.mean(axis=0)
        curve_std = curves.std(axis=0, ddof=1)
        axis.plot(x, curve_mean, linewidth=1.8, label=method)
        axis.fill_between(
            x,
            curve_mean - curve_std,
            curve_mean + curve_std,
            alpha=0.12,
        )
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Evaluation success")
    axis.set_xlim(common_steps[0], common_steps[-1])
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "sac_common_grid_curve.pdf")
    figure.savefig(output_dir / "sac_common_grid_curve.png", dpi=180)
    plt.close(figure)


def write_outputs(
    output_dir: Path,
    specs: Sequence[ConditionSpec],
    per_seed: Sequence[Mapping[str, object]],
    aggregate: Sequence[Mapping[str, object]],
    paired: Sequence[Mapping[str, object]],
    bootstrap: Sequence[Mapping[str, object]],
    repo_root: Path,
    analysis_command: str,
    plot: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    display_rows = []
    for row in per_seed:
        display_row = dict(row)
        for field in ("run_dir", "metrics_path", "summary_path"):
            display_row[field] = _display_path(Path(str(row[field])), repo_root)
        display_rows.append(display_row)
    write_csv(
        output_dir / "per_seed_common_grid.csv", display_rows, PER_SEED_FIELDS
    )
    write_csv(
        output_dir / "aggregate_common_grid.csv", aggregate, AGGREGATE_FIELDS
    )
    write_csv(
        output_dir / "paired_seed_differences.csv", paired, PAIRED_FIELDS
    )
    write_csv(
        output_dir / "paired_bootstrap_ci.csv", bootstrap, BOOTSTRAP_FIELDS
    )
    validation = build_grid_validation(specs, per_seed, repo_root)
    (output_dir / "grid_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        render_report(specs, display_rows, aggregate, bootstrap, analysis_command),
        encoding="utf-8",
    )
    if plot:
        write_plot(output_dir, per_seed)


def format_audit_failure(audit: ArtifactAudit) -> str:
    lines = ["Artifact validation failed.", "", "Discovered runs:"]
    if audit.runs:
        for run in audit.runs:
            lines.append(
                f"- {run.method}: seed={run.seed}, run={run.run_dir}, "
                f"metrics={run.metrics_path}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "Missing artifacts:"])
    if audit.missing_artifacts:
        lines.extend(f"- {path}" for path in audit.missing_artifacts)
    else:
        lines.append("- none")
    if audit.errors:
        lines.extend(["", "Validation errors:"])
        lines.extend(f"- {error}" for error in audit.errors)
    return "\n".join(lines)


def _resolve_from_repo(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reanalyze FetchPickAndPlace SAC runs on the exact 50k-to-1M "
            "common evaluation grid. This script never runs RL training."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Read all six required conditions from a compact artifact bundle.",
    )
    parser.add_argument(
        "--redq-root",
        type=Path,
        default=Path(
            "../metacog-rl/experiment_FPP/runs/"
            "redq_her_utd4_n10_m2_tqbound_ls512_1m"
        ),
    )
    parser.add_argument("--include-diagnostic", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_FPP/reports/sac_common_grid_50k"),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also write sac_common_grid_curve.pdf and .png.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    redq_root = _resolve_from_repo(repo_root, args.redq_root)
    artifact_root = (
        _resolve_from_repo(repo_root, args.artifact_root)
        if args.artifact_root is not None
        else None
    )
    output_dir = _resolve_from_repo(repo_root, args.output_dir)
    specs = build_condition_specs(
        repo_root,
        redq_root,
        include_diagnostic=args.include_diagnostic,
        artifact_root=artifact_root,
    )
    audit = audit_artifacts(specs)
    if not audit.ok:
        print(format_audit_failure(audit), file=sys.stderr)
        return 2

    try:
        per_seed = analyze_runs(specs, audit.runs)
        aggregate = aggregate_rows(per_seed)
        paired = build_paired_rows(per_seed)
        bootstrap = build_bootstrap_rows(paired)
        command_argv = [
            "python", _display_path(Path(__file__).resolve(), repo_root)
        ]
        command_argv.extend(sys.argv[1:] if argv is None else argv)
        write_outputs(
            output_dir=output_dir,
            specs=specs,
            per_seed=per_seed,
            aggregate=aggregate,
            paired=paired,
            bootstrap=bootstrap,
            repo_root=repo_root,
            analysis_command=shlex.join(command_argv),
            plot=args.plot,
        )
    except AnalysisError as exc:
        print(f"Common-grid reanalysis failed: {exc}", file=sys.stderr)
        return 2

    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
