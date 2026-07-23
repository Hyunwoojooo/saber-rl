#!/usr/bin/env python3
"""Rebuild the FPP SABER three-backbone diagnostic tables.

The paper-facing input is ``diagnostic_metrics.jsonl``.  Each JSONL row
summarizes the full 256-transition training minibatch at one exact 10k-step
checkpoint.  ``diagnostic_samples.jsonl`` is deliberately not used: it is an
isolated-RNG 128/256 subset and cannot reproduce the full-batch headline
statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ALGORITHMS = ("ddpg", "td3", "sac")
SEEDS = (1, 2, 3, 4, 5)
STEPS = tuple(range(10_000, 1_000_001, 10_000))
BATCH_SIZE = 256
LOGGED_SAMPLE_COUNT = 128
TOTAL_STEPS = 1_000_000
TOTAL_UPDATES = 995_001
CRITIC_FAMILY = {
    "ddpg": "ddpg_single",
    "td3": "td3_twin",
    "sac": "sac_twin",
}
METHOD = {
    "ddpg": "b3_support_runtime_opt",
    "td3": "td3_saber_runtime_opt",
    "sac": "sac_saber",
}
PRIMARY_BATCH_FIELDS = (
    "support_positive_rate",
    "psi_td_abs_pearson",
    "psi_td_abs_spearman",
    "top10_td_support_overlap",
    "top20_td_support_overlap",
    "top10_td_support_enrichment",
    "top20_td_support_enrichment",
    "support_positive_td_abs_before_mean",
    "support_negative_td_abs_before_mean",
    "support_positive_td_abs_after_mean",
    "support_negative_td_abs_after_mean",
    "support_positive_td_abs_reduction_mean",
    "support_negative_td_abs_reduction_mean",
    "support_positive_td_abs_reduction_over_omega_mean",
    "support_negative_td_abs_reduction_over_omega_mean",
    "support_positive_td_abs_after_over_before_mean",
    "support_negative_td_abs_after_over_before_mean",
    "target_variance_support_positive",
    "target_variance_support_negative",
    "w_raw_mean",
    "omega_critic_mean",
)
COMMON_ROW_FIELDS = (
    "type",
    "run_name",
    "step",
    "update_index",
    "seed",
    "batch_size",
    "logged_sample_count",
    "td_target_fixed_from_pre_update",
    "diagnostic_rng_isolated",
    "support_positive_count",
    "phase_counts",
    "support_positive_phase_counts",
    *PRIMARY_BATCH_FIELDS,
)
TWIN_ROW_FIELDS = (
    "critic_family",
    "td_abs_before_q1_mean",
    "td_abs_before_q2_mean",
    "td_abs_before_mean",
    "td_abs_after_q1_mean",
    "td_abs_after_q2_mean",
    "td_abs_after_mean",
    "td_abs_reduction_mean",
)
BY_STEP_FIELDS = (
    "algorithm",
    "critic_family",
    "seed",
    "step",
    "update_index",
    "batch_size",
    "support_positive_count",
    "support_negative_count",
    *PRIMARY_BATCH_FIELDS,
    "td_abs_before_q1_mean",
    "td_abs_before_q2_mean",
    "td_abs_before_mean",
    "td_abs_after_q1_mean",
    "td_abs_after_q2_mean",
    "td_abs_after_mean",
    "td_abs_reduction_mean",
)
RESULT_FILES = (
    "diagnostic_by_step.csv",
    "support_rate.csv",
    "td_error_enrichment.csv",
    "residual_reduction.csv",
    "per_seed_diagnostics.csv",
    "aggregate_by_backbone.csv",
    "config_summary.csv",
    "data_coverage.csv",
    "schema_compatibility.csv",
    "analysis_results.json",
)


class ContractError(ValueError):
    """Raised when an input violates the frozen diagnostic contract."""


@dataclass(frozen=True)
class RunData:
    algorithm: str
    critic_family: str
    seed: int
    run_name: str
    config: dict[str, Any]
    summary: dict[str, Any]
    batch_rows: tuple[dict[str, Any], ...]
    eval_point_count: int
    invariant_errors: dict[str, float]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def number(value: object, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{context}: expected a number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ContractError(f"{context}: expected a finite number, got {value!r}")
    return result


def optional_number(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value: object, context: str) -> int:
    result = number(value, context)
    parsed = int(result)
    require(float(parsed) == result, f"{context}: expected an integer, got {value!r}")
    return parsed


def close(
    actual: float,
    expected: float,
    context: str,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> float:
    error = abs(actual - expected)
    if not math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol):
        raise ContractError(
            f"{context}: actual={actual:.17g}, expected={expected:.17g}, "
            f"absolute_error={error:.3g}"
        )
    return error


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load JSON {path}: {exc}") from exc
    require(isinstance(payload, dict), f"{path}: expected a JSON object")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ContractError(f"{path}:{line_number}: {exc}") from exc
                require(
                    isinstance(row, dict),
                    f"{path}:{line_number}: expected a JSON object",
                )
                rows.append(row)
    except OSError as exc:
        raise ContractError(f"cannot read {path}: {exc}") from exc
    return rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".17g")
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return str(value)


def write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_cell(row.get(field)) for field in fields})


def finite_mean(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.mean(finite) if finite else None


def sample_std(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return statistics.stdev(finite) if len(finite) > 1 else 0.0


def weighted_mean(
    values: Iterable[tuple[float | None, int]],
) -> tuple[float | None, int]:
    numerator = 0.0
    denominator = 0
    for value, weight in values:
        if value is None or not math.isfinite(value) or weight <= 0:
            continue
        numerator += value * weight
        denominator += weight
    return (numerator / denominator if denominator else None, denominator)


def seed_from_config(config: Mapping[str, Any], context: str) -> int:
    seeds = config.get("seeds")
    require(isinstance(seeds, dict), f"{context}: missing seeds object")
    return integer(seeds.get("env_seed"), f"{context}.seeds.env_seed")


def validate_eval_grid(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            require(reader.fieldnames is not None, f"{path}: missing CSV header")
            rows = list(reader)
    except OSError as exc:
        raise ContractError(f"cannot read {path}: {exc}") from exc
    eval_steps: list[int] = []
    for row in rows:
        success = optional_number(row.get("eval/env_success_mean"))
        if success is None:
            continue
        eval_steps.append(integer(row.get("step"), f"{path}.step"))
    require(
        tuple(eval_steps) == STEPS,
        f"{path}: expected exact 10k..1M evaluation grid, got "
        f"{len(eval_steps)} points from {eval_steps[:3]} to {eval_steps[-3:]}",
    )
    return len(eval_steps)


def validate_common_config(
    algorithm: str,
    seed: int,
    config: Mapping[str, Any],
    context: str,
) -> None:
    expected: dict[str, object] = {
        "total_steps": TOTAL_STEPS,
        "batch_size": BATCH_SIZE,
        "buffer_capacity": 300_000,
        "start_steps": 10_000,
        "warmup_steps": 5_000,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "gamma": 0.98,
        "tau": 0.005,
        "her_ratio": 0.8,
        "updates_per_step": 1,
        "eval_freq": 10_000,
        "eval_episodes": 20,
        "diagnostic_freq": 10_000,
        "diagnostic_log_samples": True,
        "diagnostic_samples_per_log": LOGGED_SAMPLE_COUNT,
        "local_search_max": 512,
        "lambda_support": 1.0,
        "weight_scope": "critic_only",
        "weighting_mode": "batch_norm",
        "policy_support_mode": "bin",
    }
    for key, wanted in expected.items():
        actual = config.get(key)
        if isinstance(wanted, float):
            close(number(actual, f"{context}.{key}"), wanted, f"{context}.{key}")
        else:
            require(
                actual == wanted,
                f"{context}.{key}: expected {wanted!r}, got {actual!r}",
            )
    require(
        config.get("method") == METHOD[algorithm],
        f"{context}.method: expected {METHOD[algorithm]!r}, got {config.get('method')!r}",
    )
    require(seed_from_config(config, context) == seed, f"{context}: seed mismatch")
    seeds = config["seeds"]
    seed_expectations = {
        "env_seed": seed,
        "torch_seed": seed,
        "numpy_seed": seed,
        "action_noise_seed": seed,
        "weight_seed": 20_000 + seed,
        "eval_seed": 10_000 + seed,
    }
    for key, wanted in seed_expectations.items():
        if key in seeds:
            require(
                integer(seeds[key], f"{context}.seeds.{key}") == wanted,
                f"{context}.seeds.{key}: expected {wanted}, got {seeds[key]!r}",
            )


def validate_batch_row(
    algorithm: str,
    seed: int,
    expected_step: int,
    row: Mapping[str, Any],
    context: str,
    errors: dict[str, float],
) -> None:
    missing = [field for field in COMMON_ROW_FIELDS if field not in row]
    require(not missing, f"{context}: missing common fields {missing}")
    if algorithm != "ddpg":
        missing_twin = [field for field in TWIN_ROW_FIELDS if field not in row]
        require(not missing_twin, f"{context}: missing twin-critic fields {missing_twin}")
        require(
            row.get("critic_family") == CRITIC_FAMILY[algorithm],
            f"{context}.critic_family: expected {CRITIC_FAMILY[algorithm]!r}, "
            f"got {row.get('critic_family')!r}",
        )
    require(row["type"] == "diagnostic_batch", f"{context}: invalid row type")
    require(integer(row["seed"], f"{context}.seed") == seed, f"{context}: seed mismatch")
    step = integer(row["step"], f"{context}.step")
    require(step == expected_step, f"{context}: expected step {expected_step}, got {step}")
    require(
        integer(row["update_index"], f"{context}.update_index") == step - 4_999,
        f"{context}: update_index must equal step - 4,999",
    )
    batch_size = integer(row["batch_size"], f"{context}.batch_size")
    require(batch_size == BATCH_SIZE, f"{context}: batch_size must be {BATCH_SIZE}")
    require(
        integer(row["logged_sample_count"], f"{context}.logged_sample_count")
        == LOGGED_SAMPLE_COUNT,
        f"{context}: logged_sample_count must be {LOGGED_SAMPLE_COUNT}",
    )
    require(
        row["td_target_fixed_from_pre_update"] is True,
        f"{context}: target must be fixed from the pre-update value",
    )
    require(
        row["diagnostic_rng_isolated"] is True,
        f"{context}: diagnostic RNG must be isolated",
    )

    positive = integer(row["support_positive_count"], f"{context}.support_positive_count")
    require(0 <= positive <= batch_size, f"{context}: invalid support-positive count")
    negative = batch_size - positive
    support_rate = number(row["support_positive_rate"], f"{context}.support_positive_rate")
    errors["support_rate"] = max(
        errors["support_rate"],
        close(
            support_rate,
            positive / batch_size,
            f"{context}.support_positive_rate",
            atol=1e-12,
            rtol=1e-12,
        ),
    )

    for key in PRIMARY_BATCH_FIELDS:
        number(row[key], f"{context}.{key}")

    for label, fraction in (("top10", 0.10), ("top20", 0.20)):
        overlap = number(row[f"{label}_td_support_overlap"], f"{context}.{label}.overlap")
        enrichment = number(
            row[f"{label}_td_support_enrichment"],
            f"{context}.{label}.enrichment",
        )
        require(support_rate > 0.0, f"{context}: current frozen data requires support > 0")
        errors["enrichment"] = max(
            errors["enrichment"],
            close(
                enrichment,
                overlap / support_rate,
                f"{context}.{label}.enrichment",
                atol=1e-10,
                rtol=1e-10,
            ),
        )
        expected_k = math.ceil(fraction * batch_size)
        require(
            expected_k in (26, 52),
            f"{context}: unexpected frozen top-k convention {expected_k}",
        )

    errors["raw_weight"] = max(
        errors["raw_weight"],
        close(
            number(row["w_raw_mean"], f"{context}.w_raw_mean"),
            1.0 + support_rate,
            f"{context}.w_raw_mean",
            atol=1e-7,
            rtol=1e-7,
        ),
    )
    omega_error = abs(number(row["omega_critic_mean"], f"{context}.omega_critic_mean") - 1.0)
    require(omega_error <= 1e-6, f"{context}: omega_critic_mean is not one")
    errors["omega_mean"] = max(errors["omega_mean"], omega_error)

    phase_counts = row["phase_counts"]
    support_phase_counts = row["support_positive_phase_counts"]
    require(isinstance(phase_counts, dict), f"{context}.phase_counts must be an object")
    require(
        isinstance(support_phase_counts, dict),
        f"{context}.support_positive_phase_counts must be an object",
    )
    require(
        sum(integer(value, f"{context}.phase_counts") for value in phase_counts.values())
        == batch_size,
        f"{context}: phase counts do not sum to batch size",
    )
    require(
        sum(
            integer(value, f"{context}.support_positive_phase_counts")
            for value in support_phase_counts.values()
        )
        == positive,
        f"{context}: support phase counts do not sum to support-positive count",
    )

    for group, count in (("positive", positive), ("negative", negative)):
        require(count > 0, f"{context}: frozen input unexpectedly has empty {group} group")
        before = number(
            row[f"support_{group}_td_abs_before_mean"],
            f"{context}.support_{group}.before",
        )
        after = number(
            row[f"support_{group}_td_abs_after_mean"],
            f"{context}.support_{group}.after",
        )
        reduction = number(
            row[f"support_{group}_td_abs_reduction_mean"],
            f"{context}.support_{group}.reduction",
        )
        errors["residual_identity"] = max(
            errors["residual_identity"],
            close(
                before - after,
                reduction,
                f"{context}.support_{group}.before_minus_after",
                atol=2e-7,
                rtol=2e-7,
            ),
        )

    if algorithm != "ddpg":
        before = number(row["td_abs_before_mean"], f"{context}.td_abs_before_mean")
        after = number(row["td_abs_after_mean"], f"{context}.td_abs_after_mean")
        reduction = number(row["td_abs_reduction_mean"], f"{context}.td_abs_reduction_mean")
        errors["residual_identity"] = max(
            errors["residual_identity"],
            close(
                before - after,
                reduction,
                f"{context}.overall.before_minus_after",
                atol=2e-7,
                rtol=2e-7,
            ),
        )
        weighted_before = (
            positive * number(
                row["support_positive_td_abs_before_mean"],
                f"{context}.positive.before",
            )
            + negative
            * number(
                row["support_negative_td_abs_before_mean"],
                f"{context}.negative.before",
            )
        ) / batch_size
        errors["group_weighting"] = max(
            errors["group_weighting"],
            close(
                before,
                weighted_before,
                f"{context}.overall.group_weighted_before",
                atol=2e-7,
                rtol=2e-7,
            ),
        )


def load_run(artifact_root: Path, algorithm: str, seed: int) -> RunData:
    run_dir = artifact_root / "data" / algorithm / f"seed_{seed}"
    require(run_dir.is_dir(), f"missing run directory {run_dir}")
    expected_files = {
        "config.json",
        "env_snapshot.json",
        "summary.json",
        "metrics_selected_10k.csv",
        "diagnostic_metrics.jsonl",
    }
    actual_files = {path.name for path in run_dir.iterdir() if path.is_file()}
    require(
        actual_files == expected_files,
        f"{run_dir}: expected files {sorted(expected_files)}, got {sorted(actual_files)}",
    )
    require(
        not any(path.is_symlink() for path in run_dir.iterdir()),
        f"{run_dir}: symlinks are forbidden",
    )
    config = load_json(run_dir / "config.json")
    summary = load_json(run_dir / "summary.json")
    validate_common_config(algorithm, seed, config, str(run_dir / "config.json"))
    require(
        integer(summary.get("total_steps"), f"{run_dir}/summary.total_steps")
        == TOTAL_STEPS,
        f"{run_dir}: incomplete total_steps",
    )
    require(
        integer(summary.get("total_updates"), f"{run_dir}/summary.total_updates")
        == TOTAL_UPDATES,
        f"{run_dir}: incomplete total_updates",
    )
    eval_count = validate_eval_grid(run_dir / "metrics_selected_10k.csv")
    rows = load_jsonl(run_dir / "diagnostic_metrics.jsonl")
    require(len(rows) == len(STEPS), f"{run_dir}: expected 100 diagnostic rows")
    errors = {
        "support_rate": 0.0,
        "enrichment": 0.0,
        "raw_weight": 0.0,
        "omega_mean": 0.0,
        "residual_identity": 0.0,
        "group_weighting": 0.0,
    }
    for index, (step, row) in enumerate(zip(STEPS, rows), start=1):
        validate_batch_row(
            algorithm,
            seed,
            step,
            row,
            f"{run_dir}/diagnostic_metrics.jsonl:{index}",
            errors,
        )
    run_name = str(config.get("run_name", ""))
    require(run_name, f"{run_dir}/config.json: missing run_name")
    require(
        all(str(row.get("run_name")) == run_name for row in rows),
        f"{run_dir}: diagnostic run_name mismatch",
    )
    return RunData(
        algorithm=algorithm,
        critic_family=CRITIC_FAMILY[algorithm],
        seed=seed,
        run_name=run_name,
        config=config,
        summary=summary,
        batch_rows=tuple(rows),
        eval_point_count=eval_count,
        invariant_errors=dict(errors),
    )


def group_count(row: Mapping[str, Any], group: str) -> int:
    positive = integer(row["support_positive_count"], "support_positive_count")
    return positive if group == "positive" else BATCH_SIZE - positive


def group_field(group: str, metric: str) -> str:
    return f"support_{group}_td_abs_{metric}_mean"


def build_tables(runs: Sequence[RunData]) -> tuple[dict[str, list[dict[str, object]]], dict[str, Any]]:
    tables: dict[str, list[dict[str, object]]] = {
        "diagnostic_by_step": [],
        "support_rate": [],
        "td_error_enrichment": [],
        "residual_reduction": [],
        "per_seed_diagnostics": [],
        "aggregate_by_backbone": [],
        "config_summary": [],
        "data_coverage": [],
        "schema_compatibility": [],
    }
    per_seed_numeric: dict[tuple[str, int], dict[str, float]] = {}
    field_sets: dict[str, set[str]] = {algorithm: set() for algorithm in ALGORITHMS}
    invariant_max = {
        "support_rate": 0.0,
        "enrichment": 0.0,
        "raw_weight": 0.0,
        "omega_mean": 0.0,
        "residual_identity": 0.0,
        "group_weighting": 0.0,
    }

    for run in runs:
        require(
            set(run.invariant_errors) == set(invariant_max),
            f"{run.algorithm} seed {run.seed}: unexpected invariant-error fields "
            f"{sorted(run.invariant_errors)}",
        )
        for name, error in run.invariant_errors.items():
            invariant_max[name] = max(invariant_max[name], error)

        rows = run.batch_rows
        for row in rows:
            field_sets[run.algorithm].update(row)
            by_step = {
                "algorithm": run.algorithm,
                "critic_family": run.critic_family,
                "seed": run.seed,
                "support_negative_count": BATCH_SIZE
                - integer(row["support_positive_count"], "support_positive_count"),
            }
            for field in BY_STEP_FIELDS:
                if field in by_step:
                    continue
                by_step[field] = row.get(field)
            tables["diagnostic_by_step"].append(by_step)

        total_positive = sum(
            integer(row["support_positive_count"], "support_positive_count")
            for row in rows
        )
        total_samples = sum(integer(row["batch_size"], "batch_size") for row in rows)
        support_rate = total_positive / total_samples
        support_row = {
            "algorithm": run.algorithm,
            "critic_family": run.critic_family,
            "seed": run.seed,
            "batch_count": len(rows),
            "total_batch_samples": total_samples,
            "support_positive_count": total_positive,
            "diagnostic_support_positive_rate": support_rate,
        }
        tables["support_rate"].append(support_row)

        seed_values: dict[str, float] = {
            "diagnostic_support_positive_rate": support_rate,
            "psi_td_abs_pearson_equal_update_mean": float(
                finite_mean(
                    number(row["psi_td_abs_pearson"], "psi_td_abs_pearson")
                    for row in rows
                )
            ),
            "psi_td_abs_spearman_equal_update_mean": float(
                finite_mean(
                    number(row["psi_td_abs_spearman"], "psi_td_abs_spearman")
                    for row in rows
                )
            ),
        }

        for label, fraction in (("top10", 0.10), ("top20", 0.20)):
            overlap_values = [
                optional_number(row[f"{label}_td_support_overlap"]) for row in rows
            ]
            enrichment_values = [
                optional_number(row[f"{label}_td_support_enrichment"]) for row in rows
            ]
            valid = sum(value is not None for value in enrichment_values)
            enrichment_row = {
                "algorithm": run.algorithm,
                "critic_family": run.critic_family,
                "seed": run.seed,
                "top_fraction": fraction,
                "top_k_per_batch": math.ceil(fraction * BATCH_SIZE),
                "valid_batch_count": valid,
                "undefined_batch_count": len(rows) - valid,
                "top_td_support_overlap_equal_update_mean": finite_mean(overlap_values),
                "top_td_support_enrichment_equal_update_mean": finite_mean(
                    enrichment_values
                ),
            }
            tables["td_error_enrichment"].append(enrichment_row)
            seed_values[f"{label}_td_support_overlap_equal_update_mean"] = float(
                enrichment_row["top_td_support_overlap_equal_update_mean"]
            )
            seed_values[f"{label}_td_support_enrichment_equal_update_mean"] = float(
                enrichment_row["top_td_support_enrichment_equal_update_mean"]
            )

        residual_fields = (
            ("before", "before"),
            ("after", "after"),
            ("reduction", "reduction"),
            ("reduction_over_omega", "reduction_over_omega"),
            ("after_over_before", "after_over_before"),
        )
        for group in ("positive", "negative"):
            counts = [group_count(row, group) for row in rows]
            valid_batch_count = sum(count > 0 for count in counts)
            undefined_batch_count = len(rows) - valid_batch_count
            for aggregation in ("equal_update", "sample_weighted"):
                residual_row: dict[str, object] = {
                    "algorithm": run.algorithm,
                    "critic_family": run.critic_family,
                    "seed": run.seed,
                    "support_group": group,
                    "aggregation": aggregation,
                    "valid_batch_count": valid_batch_count,
                    "undefined_batch_count": undefined_batch_count,
                    "observation_count": sum(counts),
                }
                for output_name, source_name in residual_fields:
                    source_field = group_field(group, source_name)
                    values = [optional_number(row[source_field]) for row in rows]
                    if aggregation == "equal_update":
                        value = finite_mean(values)
                    else:
                        value, _weight = weighted_mean(zip(values, counts))
                    residual_row[f"td_abs_{output_name}_mean"] = value
                tables["residual_reduction"].append(residual_row)
                prefix = f"{group}_{aggregation}"
                seed_values[f"{prefix}_td_abs_before_mean"] = float(
                    residual_row["td_abs_before_mean"]
                )
                seed_values[f"{prefix}_td_abs_after_mean"] = float(
                    residual_row["td_abs_after_mean"]
                )
                seed_values[f"{prefix}_td_abs_reduction_mean"] = float(
                    residual_row["td_abs_reduction_mean"]
                )
                seed_values[f"{prefix}_td_abs_reduction_over_omega_mean"] = float(
                    residual_row["td_abs_reduction_over_omega_mean"]
                )
                seed_values[f"{prefix}_td_abs_after_over_before_mean"] = float(
                    residual_row["td_abs_after_over_before_mean"]
                )

        per_seed_numeric[(run.algorithm, run.seed)] = seed_values
        tables["per_seed_diagnostics"].append(
            {
                "algorithm": run.algorithm,
                "critic_family": run.critic_family,
                "seed": run.seed,
                **seed_values,
            }
        )
        config = run.config
        tables["config_summary"].append(
            {
                "algorithm": run.algorithm,
                "critic_family": run.critic_family,
                "seed": run.seed,
                "run_name": run.run_name,
                "method": config["method"],
                "total_steps": config["total_steps"],
                "batch_size": config["batch_size"],
                "buffer_capacity": config["buffer_capacity"],
                "start_steps": config["start_steps"],
                "warmup_steps": config["warmup_steps"],
                "actor_lr": config["actor_lr"],
                "critic_lr": config["critic_lr"],
                "gamma": config["gamma"],
                "tau": config["tau"],
                "her_ratio": config["her_ratio"],
                "updates_per_step": config["updates_per_step"],
                "local_search_max": config["local_search_max"],
                "lambda_support": config["lambda_support"],
                "weight_scope": config["weight_scope"],
                "weighting_mode": config["weighting_mode"],
                "diagnostic_freq": config["diagnostic_freq"],
                "diagnostic_samples_per_log": config["diagnostic_samples_per_log"],
            }
        )
        tables["data_coverage"].append(
            {
                "algorithm": run.algorithm,
                "critic_family": run.critic_family,
                "seed": run.seed,
                "run_name": run.run_name,
                "total_steps": run.summary["total_steps"],
                "total_updates": run.summary["total_updates"],
                "diagnostic_batch_count": len(rows),
                "full_batch_observation_count": len(rows) * BATCH_SIZE,
                "first_diagnostic_step": rows[0]["step"],
                "last_diagnostic_step": rows[-1]["step"],
                "evaluation_point_count": run.eval_point_count,
                "target_fixed_from_pre_update": all(
                    row["td_target_fixed_from_pre_update"] is True for row in rows
                ),
                "diagnostic_rng_isolated": all(
                    row["diagnostic_rng_isolated"] is True for row in rows
                ),
            }
        )

    metric_specs: list[tuple[str, str, str, float | None, str]] = [
        (
            "diagnostic_support_positive_rate",
            "sample_weighted",
            "",
            None,
            "diagnostic_support_positive_rate",
        ),
        (
            "psi_td_abs_pearson",
            "equal_update",
            "",
            None,
            "psi_td_abs_pearson_equal_update_mean",
        ),
        (
            "psi_td_abs_spearman",
            "equal_update",
            "",
            None,
            "psi_td_abs_spearman_equal_update_mean",
        ),
    ]
    for label, fraction in (("top10", 0.10), ("top20", 0.20)):
        metric_specs.extend(
            [
                (
                    "top_td_support_overlap",
                    "equal_update",
                    "",
                    fraction,
                    f"{label}_td_support_overlap_equal_update_mean",
                ),
                (
                    "top_td_support_enrichment",
                    "equal_update",
                    "",
                    fraction,
                    f"{label}_td_support_enrichment_equal_update_mean",
                ),
            ]
        )
    for group in ("positive", "negative"):
        for aggregation in ("equal_update", "sample_weighted"):
            for metric in (
                "td_abs_before",
                "td_abs_after",
                "td_abs_reduction",
                "td_abs_reduction_over_omega",
                "td_abs_after_over_before",
            ):
                metric_specs.append(
                    (
                        metric,
                        aggregation,
                        group,
                        None,
                        f"{group}_{aggregation}_{metric}_mean",
                    )
                )

    for algorithm in ALGORITHMS:
        for metric, aggregation, support_group, top_fraction, source_key in metric_specs:
            values = [
                per_seed_numeric[(algorithm, seed)][source_key] for seed in SEEDS
            ]
            tables["aggregate_by_backbone"].append(
                {
                    "algorithm": algorithm,
                    "critic_family": CRITIC_FAMILY[algorithm],
                    "metric": metric,
                    "aggregation": aggregation,
                    "support_group": support_group,
                    "top_fraction": top_fraction,
                    "n_seeds": len(values),
                    "mean": statistics.mean(values),
                    "sample_sd": statistics.stdev(values),
                }
            )

    union_fields = sorted(set().union(*(field_sets[algorithm] for algorithm in ALGORITHMS)))
    for field in union_fields:
        presence = {
            algorithm: field in field_sets[algorithm] for algorithm in ALGORITHMS
        }
        present_count = sum(presence.values())
        if present_count == len(ALGORITHMS):
            classification = "common"
        elif (
            not presence["ddpg"]
            and presence["td3"]
            and presence["sac"]
        ):
            classification = "twin_critic_only"
        else:
            classification = "algorithm_specific"
        tables["schema_compatibility"].append(
            {
                "field": field,
                "ddpg_present": presence["ddpg"],
                "td3_present": presence["td3"],
                "sac_present": presence["sac"],
                "classification": classification,
            }
        )

    analysis_results: dict[str, Any] = {
        "schema_version": 1,
        "analysis_unit": "seed",
        "primary_within_seed_aggregation": "equal_update",
        "between_seed_variability": "sample_standard_deviation_ddof_1",
        "cross_backbone_grand_mean_produced": False,
        "batch_input": "full_256_transition_diagnostic_metrics_jsonl",
        "sample_subset_used_for_headline_metrics": False,
        "algorithms": {},
    }
    for algorithm in ALGORITHMS:
        rows = [
            row
            for row in tables["aggregate_by_backbone"]
            if row["algorithm"] == algorithm
        ]
        analysis_results["algorithms"][algorithm] = {
            "critic_family": CRITIC_FAMILY[algorithm],
            "n_seeds": len(SEEDS),
            "aggregate_metrics": rows,
        }

    validation = {
        "schema_version": 1,
        "overall_status": "validated_with_partial_execution_source_provenance",
        "data_analysis_status": "passed",
        "execution_source_provenance_status": "partial",
        "execution_source_provenance_note": (
            "TD3 has a verified execution-source commit. DDPG and SAC preserve "
            "candidate source snapshots with SHA256 but their execution commit is unknown."
        ),
        "expected_algorithms": list(ALGORITHMS),
        "expected_seeds": list(SEEDS),
        "run_count": len(runs),
        "diagnostic_batch_count": sum(len(run.batch_rows) for run in runs),
        "full_batch_observation_count": sum(
            len(run.batch_rows) * BATCH_SIZE for run in runs
        ),
        "completion_contract": {
            "total_steps": TOTAL_STEPS,
            "total_updates": TOTAL_UPDATES,
            "diagnostic_steps": list(STEPS),
            "diagnostic_batch_size": BATCH_SIZE,
            "logged_sample_subset_size": LOGGED_SAMPLE_COUNT,
        },
        "invariant_failure_count": 0,
        "maximum_absolute_invariant_errors": invariant_max,
        "undefined_primary_metric_count": 0,
        "cross_backbone_pooling": "forbidden",
    }
    return tables, {"analysis_results": analysis_results, "validation": validation}


def write_outputs(
    output_dir: Path,
    validation_path: Path,
    tables: Mapping[str, Sequence[Mapping[str, object]]],
    payloads: Mapping[str, Any],
) -> None:
    write_csv(output_dir / "diagnostic_by_step.csv", BY_STEP_FIELDS, tables["diagnostic_by_step"])
    write_csv(
        output_dir / "support_rate.csv",
        (
            "algorithm",
            "critic_family",
            "seed",
            "batch_count",
            "total_batch_samples",
            "support_positive_count",
            "diagnostic_support_positive_rate",
        ),
        tables["support_rate"],
    )
    write_csv(
        output_dir / "td_error_enrichment.csv",
        (
            "algorithm",
            "critic_family",
            "seed",
            "top_fraction",
            "top_k_per_batch",
            "valid_batch_count",
            "undefined_batch_count",
            "top_td_support_overlap_equal_update_mean",
            "top_td_support_enrichment_equal_update_mean",
        ),
        tables["td_error_enrichment"],
    )
    write_csv(
        output_dir / "residual_reduction.csv",
        (
            "algorithm",
            "critic_family",
            "seed",
            "support_group",
            "aggregation",
            "valid_batch_count",
            "undefined_batch_count",
            "observation_count",
            "td_abs_before_mean",
            "td_abs_after_mean",
            "td_abs_reduction_mean",
            "td_abs_reduction_over_omega_mean",
            "td_abs_after_over_before_mean",
        ),
        tables["residual_reduction"],
    )
    per_seed_fields = (
        "algorithm",
        "critic_family",
        "seed",
        "diagnostic_support_positive_rate",
        "psi_td_abs_pearson_equal_update_mean",
        "psi_td_abs_spearman_equal_update_mean",
        "top10_td_support_overlap_equal_update_mean",
        "top10_td_support_enrichment_equal_update_mean",
        "top20_td_support_overlap_equal_update_mean",
        "top20_td_support_enrichment_equal_update_mean",
        "positive_equal_update_td_abs_before_mean",
        "positive_equal_update_td_abs_after_mean",
        "positive_equal_update_td_abs_reduction_mean",
        "positive_equal_update_td_abs_reduction_over_omega_mean",
        "positive_equal_update_td_abs_after_over_before_mean",
        "negative_equal_update_td_abs_before_mean",
        "negative_equal_update_td_abs_after_mean",
        "negative_equal_update_td_abs_reduction_mean",
        "negative_equal_update_td_abs_reduction_over_omega_mean",
        "negative_equal_update_td_abs_after_over_before_mean",
        "positive_sample_weighted_td_abs_before_mean",
        "positive_sample_weighted_td_abs_after_mean",
        "positive_sample_weighted_td_abs_reduction_mean",
        "positive_sample_weighted_td_abs_reduction_over_omega_mean",
        "positive_sample_weighted_td_abs_after_over_before_mean",
        "negative_sample_weighted_td_abs_before_mean",
        "negative_sample_weighted_td_abs_after_mean",
        "negative_sample_weighted_td_abs_reduction_mean",
        "negative_sample_weighted_td_abs_reduction_over_omega_mean",
        "negative_sample_weighted_td_abs_after_over_before_mean",
    )
    write_csv(
        output_dir / "per_seed_diagnostics.csv",
        per_seed_fields,
        tables["per_seed_diagnostics"],
    )
    write_csv(
        output_dir / "aggregate_by_backbone.csv",
        (
            "algorithm",
            "critic_family",
            "metric",
            "aggregation",
            "support_group",
            "top_fraction",
            "n_seeds",
            "mean",
            "sample_sd",
        ),
        tables["aggregate_by_backbone"],
    )
    write_csv(
        output_dir / "config_summary.csv",
        (
            "algorithm",
            "critic_family",
            "seed",
            "run_name",
            "method",
            "total_steps",
            "batch_size",
            "buffer_capacity",
            "start_steps",
            "warmup_steps",
            "actor_lr",
            "critic_lr",
            "gamma",
            "tau",
            "her_ratio",
            "updates_per_step",
            "local_search_max",
            "lambda_support",
            "weight_scope",
            "weighting_mode",
            "diagnostic_freq",
            "diagnostic_samples_per_log",
        ),
        tables["config_summary"],
    )
    write_csv(
        output_dir / "data_coverage.csv",
        (
            "algorithm",
            "critic_family",
            "seed",
            "run_name",
            "total_steps",
            "total_updates",
            "diagnostic_batch_count",
            "full_batch_observation_count",
            "first_diagnostic_step",
            "last_diagnostic_step",
            "evaluation_point_count",
            "target_fixed_from_pre_update",
            "diagnostic_rng_isolated",
        ),
        tables["data_coverage"],
    )
    write_csv(
        output_dir / "schema_compatibility.csv",
        (
            "field",
            "ddpg_present",
            "td3_present",
            "sac_present",
            "classification",
        ),
        tables["schema_compatibility"],
    )
    write_json(output_dir / "analysis_results.json", payloads["analysis_results"])
    write_json(validation_path, payloads["validation"])


def main() -> int:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Rebuild strict FPP DDPG/TD3/SAC SABER diagnostic tables."
    )
    parser.add_argument("--artifact-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--validation-output", type=Path, default=None)
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else artifact_root / "results"
    )
    validation_output = (
        args.validation_output.resolve()
        if args.validation_output is not None
        else artifact_root / "validation.json"
    )
    runs = [
        load_run(artifact_root, algorithm, seed)
        for algorithm in ALGORITHMS
        for seed in SEEDS
    ]
    tables, payloads = build_tables(runs)
    write_outputs(output_dir, validation_output, tables, payloads)
    print(
        json.dumps(
            {
                "status": "passed",
                "run_count": len(runs),
                "diagnostic_batch_count": sum(len(run.batch_rows) for run in runs),
                "output_dir": str(output_dir),
                "validation": str(validation_output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
