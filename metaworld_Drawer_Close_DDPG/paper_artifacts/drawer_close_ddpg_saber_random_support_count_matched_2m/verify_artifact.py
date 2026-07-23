#!/usr/bin/env python3
"""Standalone verifier for the DrawerClose Random-Support compact artifact."""

from __future__ import annotations

import sys

# Keep the README's plain ``python3 verify_artifact.py`` command read-only.
# This is set before any possible local-module import.
sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import median
from typing import Mapping, Sequence


ARTIFACT_ID = "drawer_close_ddpg_saber_random_support_count_matched_2m"
RAW_RESULT_COMMIT = "ae6d93de0d1dad5bf8328fd3334addb00caec205"
COMPARATOR_ID = "drawer_close_ddpg_saber_context_eligible_count_matched_2m"
COMPARATOR_COMMIT = "841eb706b1dcdc12b59db23ec545099d39d3c619"
METHODS = ("her", "main", "random")
SEEDS = (1, 2, 3, 4, 5)
STEPS = tuple(range(10_000, 2_000_001, 10_000))
PRIMARY = "eval/env_success_mean"
CONTRASTS = (
    ("main_minus_random", "main", "random"),
    ("random_minus_her", "random", "her"),
    ("main_minus_her", "main", "her"),
)
TOL = 1e-10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"{path}: expected object")
    return payload


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        result = list(csv.DictReader(handle))
    require(bool(result), f"{path}: empty CSV")
    return result


def require_json_csv_equivalence(
    payload_rows: object,
    csv_rows: Sequence[Mapping[str, str]],
    label: str,
) -> None:
    require(isinstance(payload_rows, list), f"{label}: expected JSON list")
    require(len(payload_rows) == len(csv_rows), f"{label}: row count")
    for row_number, (json_row, csv_row) in enumerate(
        zip(payload_rows, csv_rows), start=1
    ):
        require(isinstance(json_row, dict), f"{label}: malformed JSON row")
        require(
            set(json_row) == set(csv_row),
            f"{label}: schema mismatch at row {row_number}",
        )
        for key, raw_value in csv_row.items():
            json_value = json_row[key]
            if raw_value == "":
                require(
                    json_value is None,
                    f"{label}: expected null at row {row_number}/{key}",
                )
            elif isinstance(json_value, str):
                require(
                    json_value == raw_value,
                    f"{label}: text mismatch at row {row_number}/{key}",
                )
            elif isinstance(json_value, (int, float)) and not isinstance(
                json_value, bool
            ):
                close(
                    raw_value,
                    float(json_value),
                    f"{label}: numeric mismatch at row {row_number}/{key}",
                )
            else:
                raise ValueError(
                    f"{label}: unsupported value at row {row_number}/{key}"
                )


def finite(value: object, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: nonnumeric {value!r}") from exc
    require(math.isfinite(parsed), f"{label}: non-finite")
    return parsed


def close(actual: object, expected: float, label: str) -> None:
    value = finite(actual, label)
    require(
        math.isclose(value, expected, rel_tol=0.0, abs_tol=TOL),
        f"{label}: {value} != {expected}",
    )


def mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(
        math.fsum((value - center) ** 2 for value in values)
        / (len(values) - 1)
    )


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def manifest_files(root: Path) -> int:
    manifest = read_json(root / "artifact_manifest.json")
    require(manifest.get("artifact_id") == ARTIFACT_ID, "artifact id")
    require(
        manifest.get("raw_result_commit") == RAW_RESULT_COMMIT,
        "raw result commit",
    )
    require(
        manifest.get("comparator_artifact_id") == COMPARATOR_ID
        and manifest.get("comparator_artifact_commit") == COMPARATOR_COMMIT,
        "comparator identity",
    )
    require(
        manifest.get("execution_binding_status")
        == "reconstructed_not_cryptographically_bound",
        "execution-binding caveat missing",
    )
    require(
        manifest.get("manifest_hash_policy")
        == "artifact_manifest.json is excluded to avoid recursive self-hash",
        "manifest self-hash policy",
    )
    evaluation = manifest.get("evaluation")
    require(isinstance(evaluation, dict), "manifest evaluation contract")
    require(
        evaluation.get("field") == PRIMARY
        and evaluation.get("steps") == list(STEPS)
        and int(evaluation.get("point_count_per_seed", -1)) == 200
        and evaluation.get("fallback_field") is None
        and evaluation.get("smoothing") is False
        and evaluation.get("interpolation") is False
        and evaluation.get("forward_fill") is False
        and evaluation.get("extrapolation") is False
        and int(evaluation.get("auc_normalization_span", -1)) == 1_990_000,
        "manifest exact-grid/no-fill contract",
    )
    bootstrap = manifest.get("bootstrap")
    sign_flip = manifest.get("exact_sign_flip")
    require(
        isinstance(bootstrap, dict)
        and bootstrap.get("scheme") == "exhaustive_ordered_5_power_5"
        and int(bootstrap.get("resample_count", -1)) == 3_125,
        "manifest bootstrap contract",
    )
    require(
        isinstance(sign_flip, dict)
        and int(sign_flip.get("assignment_count", -1)) == 32,
        "manifest sign-flip contract",
    )
    require(
        manifest.get("method_seed_coverage")
        == {method: list(SEEDS) for method in METHODS},
        "manifest method/seed coverage",
    )
    packaged = manifest.get("packaged_files")
    require(isinstance(packaged, list), "packaged file rows")
    listed: dict[str, Mapping[str, object]] = {}
    for row in packaged:
        require(isinstance(row, dict), "malformed manifest row")
        relative = str(row.get("path", ""))
        path = Path(relative)
        require(
            relative
            and not path.is_absolute()
            and ".." not in path.parts
            and relative not in listed,
            f"unsafe/duplicate manifest path {relative!r}",
        )
        listed[relative] = row
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    require(set(actual) == set(listed), "artifact file-set mismatch")
    for relative, path in actual.items():
        row = listed[relative]
        require(
            path.stat().st_size == int(row["size_bytes"])
            and sha256(path) == row["sha256"],
            f"hash/size mismatch {relative}",
        )
    digest = hashlib.sha256()
    for relative in sorted(listed):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(listed[relative]["sha256"]).encode())
        digest.update(b"\n")
    require(
        digest.hexdigest()
        == manifest.get("package_content_sha256_excluding_manifest"),
        "package content digest",
    )
    require(
        int(manifest.get("packaged_file_count_excluding_manifest", -1))
        == len(actual),
        "file count",
    )
    return len(actual)


def forbidden_payloads(root: Path) -> None:
    failures: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if (
            path.suffix.lower() in {".pt", ".pth", ".ckpt", ".jsonl", ".log"}
            or path.name == "metrics.csv"
            or any(
                part.lower()
                in {
                    "results",
                    "results_2m",
                    "logs",
                    "logs_2m",
                    "checkpoints",
                    "__pycache__",
                }
                for part in relative.parts
            )
        ):
            failures.append(relative.as_posix())
    require(not failures, f"forbidden payloads {failures}")


def curve(path: Path) -> list[float]:
    table = rows(path)
    actual_steps: list[int] = []
    values: list[float] = []
    for row in table:
        step_value = finite(row.get("step"), f"{path}/step")
        step = int(step_value)
        require(step_value == step, f"{path}: nonintegral step")
        value = finite(row.get(PRIMARY), f"{path}/{PRIMARY}")
        require(0.0 <= value <= 1.0, f"{path}: success range")
        actual_steps.append(step)
        values.append(value)
    require(actual_steps == list(STEPS), f"{path}: exact grid")
    return values


def auc(values: Sequence[float]) -> float:
    area = math.fsum(
        (right - left) * (left_y + right_y) / 2.0
        for left, right, left_y, right_y in zip(
            STEPS, STEPS[1:], values, values[1:]
        )
    )
    return area / (STEPS[-1] - STEPS[0])


def run_metrics(values: Sequence[float]) -> dict[str, object]:
    best = max(values)
    hit = next(
        (step for step, value in zip(STEPS, values) if value >= 0.9), None
    )
    return {
        "auc": auc(values),
        "final_success": values[-1],
        "best_success": best,
        "best_to_final_drop": best - values[-1],
        "hit90": hit,
    }


def verify_statistics(root: Path) -> dict[tuple[str, int], dict[str, object]]:
    calculated: dict[tuple[str, int], dict[str, object]] = {}
    curves: dict[tuple[str, int], list[float]] = {}
    for method in METHODS:
        for seed in SEEDS:
            values = curve(
                root
                / "data"
                / method
                / f"seed_{seed}"
                / "metrics_selected_10k.csv"
            )
            curves[(method, seed)] = values
            calculated[(method, seed)] = run_metrics(values)

    per_seed_path = root / "per_seed_results.csv"
    expected_per_seed_header = [
        "method_key",
        "method",
        "seed",
        "run_dir",
        "metrics_path",
        "evaluation_points",
        "auc",
        "final_success",
        "best_success",
        "best_to_final_drop",
        "hit90",
        "summary_auc",
        "summary_auc_difference",
        "summary_final_difference",
        "summary_best_difference",
    ]
    with per_seed_path.open(newline="", encoding="utf-8") as handle:
        actual_per_seed_header = next(csv.reader(handle))
    require(
        actual_per_seed_header == expected_per_seed_header,
        "per-seed schema/order mismatch",
    )
    per_seed = rows(per_seed_path)
    require(len(per_seed) == 15, "per-seed row count")
    index = {
        (row["method_key"], int(row["seed"])): row for row in per_seed
    }
    require(set(index) == set(calculated), "per-seed coverage")
    for key, expected in calculated.items():
        row = index[key]
        require(int(row["evaluation_points"]) == 200, f"{key}: point count")
        for field in ("auc", "final_success", "best_success", "best_to_final_drop"):
            close(row[field], float(expected[field]), f"{key}/{field}")
        native_hit = None if row["hit90"] == "" else int(float(row["hit90"]))
        require(native_hit == expected["hit90"], f"{key}: Hit90")
        summary = read_json(
            root
            / "data"
            / key[0]
            / f"seed_{key[1]}"
            / "summary.json"
        )
        require(
            summary.get("run_dir") == f"data/{key[0]}/seed_{key[1]}",
            f"{key}: portable summary run_dir",
        )
        summary_auc = finite(summary.get("eval_AUC_success"), f"{key}/summary AUC")
        summary_final = finite(
            summary.get("final_eval_env_success"), f"{key}/summary final"
        )
        summary_best = finite(
            summary.get("best_eval_env_success"), f"{key}/summary best"
        )
        close(row["summary_auc"], summary_auc, f"{key}/summary_auc")
        close(
            row["summary_auc_difference"],
            summary_auc - float(expected["auc"]),
            f"{key}/summary_auc_difference",
        )
        close(
            row["summary_final_difference"],
            summary_final - float(expected["final_success"]),
            f"{key}/summary_final_difference",
        )
        close(
            row["summary_best_difference"],
            summary_best - float(expected["best_success"]),
            f"{key}/summary_best_difference",
        )

    aggregate = {
        row["method_key"]: row for row in rows(root / "aggregate_results.csv")
    }
    require(set(aggregate) == set(METHODS), "aggregate coverage")
    for method in METHODS:
        row = aggregate[method]
        require(int(row["seed_count"]) == 5, f"{method}: seed count")
        for field in ("auc", "final_success", "best_success", "best_to_final_drop"):
            values = [
                float(calculated[(method, seed)][field]) for seed in SEEDS
            ]
            close(row[f"{field}_mean"], mean(values), f"{method}/{field}/mean")
            close(
                row[f"{field}_sample_sd"],
                sample_sd(values),
                f"{method}/{field}/sd",
            )
        hits = [
            int(calculated[(method, seed)]["hit90"])
            for seed in SEEDS
            if calculated[(method, seed)]["hit90"] is not None
        ]
        require(int(row["hit90_seed_count"]) == len(hits), "Hit90 count")
        if hits:
            close(row["hit90_median_reached"], float(median(hits)), "Hit90 median")
            close(row["hit90_mean_reached"], mean(hits), "Hit90 mean")

    paired_path = root / "paired_comparisons.csv"
    expected_paired_header = [
        "contrast_key",
        "contrast",
        "left_method",
        "right_method",
        "paired_seed_count",
        "mean_auc_difference",
        "sample_sd_auc_difference",
        "bootstrap_ci95_lower",
        "bootstrap_ci95_upper",
        "bootstrap_resample_count",
        "exact_two_sided_sign_flip_pvalue",
        "sign_flip_assignment_count",
        "positive_seed_count",
        "zero_seed_count",
        "negative_seed_count",
        "leave_one_seed_out_mean_min",
        "leave_one_seed_out_mean_max",
        "seed_1_auc_difference",
        "seed_2_auc_difference",
        "seed_3_auc_difference",
        "seed_4_auc_difference",
        "seed_5_auc_difference",
    ]
    with paired_path.open(newline="", encoding="utf-8") as handle:
        actual_paired_header = next(csv.reader(handle))
    require(
        actual_paired_header == expected_paired_header,
        "paired schema/order mismatch",
    )
    paired = {
        row["contrast_key"]: row
        for row in rows(paired_path)
    }
    require(set(paired) == {item[0] for item in CONTRASTS}, "contrast coverage")
    for key, left, right in CONTRASTS:
        differences = [
            float(calculated[(left, seed)]["auc"])
            - float(calculated[(right, seed)]["auc"])
            for seed in SEEDS
        ]
        bootstrap = [
            mean([differences[index] for index in selection])
            for selection in itertools.product(range(5), repeat=5)
        ]
        observed = abs(mean(differences))
        sign_statistics = [
            abs(
                mean(
                    [
                        sign * difference
                        for sign, difference in zip(signs, differences)
                    ]
                )
            )
            for signs in itertools.product((-1.0, 1.0), repeat=5)
        ]
        row = paired[key]
        require(
            row["left_method"] == left and row["right_method"] == right,
            f"{key}: paired method keys",
        )
        close(row["mean_auc_difference"], mean(differences), f"{key}/mean")
        close(
            row["sample_sd_auc_difference"],
            sample_sd(differences),
            f"{key}/sd",
        )
        close(
            row["bootstrap_ci95_lower"],
            percentile(bootstrap, 0.025),
            f"{key}/ci lower",
        )
        close(
            row["bootstrap_ci95_upper"],
            percentile(bootstrap, 0.975),
            f"{key}/ci upper",
        )
        require(int(row["bootstrap_resample_count"]) == 3125, "bootstrap count")
        require(int(row["sign_flip_assignment_count"]) == 32, "sign count")
        expected_p = (
            sum(value >= observed - 1e-15 for value in sign_statistics) / 32
        )
        close(
            row["exact_two_sided_sign_flip_pvalue"],
            expected_p,
            f"{key}/sign p",
        )

    mean_curve_path = root / "exact_grid_mean_curves.csv"
    expected_mean_curve_header = ["step"]
    for method in METHODS:
        expected_mean_curve_header.extend(
            (
                f"{method}_mean_success",
                f"{method}_sample_sd_success",
                f"{method}_method",
            )
        )
    with mean_curve_path.open(newline="", encoding="utf-8") as handle:
        actual_mean_curve_header = next(csv.reader(handle))
    require(
        actual_mean_curve_header == expected_mean_curve_header,
        "mean-curve schema/order mismatch",
    )
    mean_curve = rows(mean_curve_path)
    require(len(mean_curve) == 200, "mean curve rows")
    expected_labels = {
        "her": "DrawerClose DDPG HER",
        "main": "DrawerClose DDPG HER + SABER Main",
        "random": "DrawerClose DDPG Random-Support Count-Matched",
    }
    for point, row in enumerate(mean_curve):
        require(int(row["step"]) == STEPS[point], "mean curve step")
        for method in METHODS:
            values = [curves[(method, seed)][point] for seed in SEEDS]
            close(row[f"{method}_mean_success"], mean(values), "curve mean")
            close(
                row[f"{method}_sample_sd_success"],
                sample_sd(values),
                "curve sd",
            )
            require(
                row[f"{method}_method"] == expected_labels[method],
                "curve method label",
            )

    analysis_results = read_json(root / "analysis_results.json")
    require(
        set(analysis_results)
        == {
            "aggregate",
            "auc_convention",
            "bootstrap",
            "evaluation_field",
            "evaluation_steps",
            "extrapolation",
            "forward_fill",
            "interpolation",
            "mean_curves",
            "paired_comparisons",
            "per_seed",
            "schema_version",
            "seeds",
            "smoothing",
        },
        "analysis-results top-level schema mismatch",
    )
    require(int(analysis_results["schema_version"]) == 1, "analysis schema")
    require(
        analysis_results["auc_convention"]
        == (
            "observed-only trapezoid normalized by "
            "(2,000,000 - 10,000)"
        )
        and analysis_results["bootstrap"]
        == (
            "ordered exhaustive paired bootstrap, "
            "5^5=3125 index sequences"
        )
        and analysis_results["evaluation_field"] == PRIMARY
        and analysis_results["evaluation_steps"] == list(STEPS)
        and analysis_results["seeds"] == list(SEEDS)
        and analysis_results["smoothing"] is False
        and analysis_results["interpolation"] is False
        and analysis_results["forward_fill"] is False
        and analysis_results["extrapolation"] is False,
        "analysis-results exact-grid/statistical contract",
    )
    require_json_csv_equivalence(
        analysis_results["per_seed"],
        per_seed,
        "analysis-results/per_seed",
    )
    require_json_csv_equivalence(
        analysis_results["aggregate"],
        rows(root / "aggregate_results.csv"),
        "analysis-results/aggregate",
    )
    require_json_csv_equivalence(
        analysis_results["paired_comparisons"],
        rows(root / "paired_comparisons.csv"),
        "analysis-results/paired",
    )
    require_json_csv_equivalence(
        analysis_results["mean_curves"],
        mean_curve,
        "analysis-results/mean_curves",
    )
    return calculated


def verify_count_traces(root: Path) -> int:
    total_rows = 0
    rollup_records: list[dict[str, object]] = []
    for seed in SEEDS:
        base = root / "data" / "random" / f"seed_{seed}"
        trace = rows(base / "assignment_count_trace.csv")
        require(len(trace) == 10_176, f"seed {seed}: trace rows")
        min_count = 256
        max_count = 0
        for row_number, row in enumerate(trace, start=2):
            ref_rate = finite(
                row["reference_high_weight_rate"],
                f"seed {seed} trace {row_number}/ref",
            )
            assigned_rate = finite(
                row["assigned_high_weight_rate"],
                f"seed {seed} trace {row_number}/assigned",
            )
            ref_count = int(row["reference_high_weight_count"])
            assigned_count = int(row["assigned_high_weight_count"])
            require(ref_count == assigned_count, "count mismatch")
            close(ref_rate, assigned_rate, "rate mismatch")
            close(ref_rate * 256.0, float(ref_count), "reference count formula")
            close(
                assigned_rate * 256.0,
                float(assigned_count),
                "assigned count formula",
            )
            close(
                row["reference_raw_weight_mean"],
                1.0 + ref_rate,
                "reference raw mean",
            )
            close(
                row["assigned_raw_weight_mean"],
                1.0 + assigned_rate,
                "assigned raw mean",
            )
            close(
                row["canonical_support_positive_rate"],
                ref_rate,
                "canonical support rate",
            )
            min_count = min(min_count, ref_count)
            max_count = max(max_count, ref_count)
        audit = read_json(base / "positive_count_invariant.json")
        require(audit.get("passed") is True, "count audit pass")
        require(
            int(audit["serialized_post_warmup_rows_audited"]) == len(trace),
            "count audit row total",
        )
        require(
            int(audit["minimum_positive_count"]) == min_count
            and int(audit["maximum_positive_count"]) == max_count,
            "count extrema",
        )
        invariants = audit.get("invariants")
        require(isinstance(invariants, dict), "count invariant object")
        for key in (
            "reference_and_assigned_positive_counts_equal",
            "positive_counts_integral",
            "reference_rate_matches_canonical_support_rate",
            "raw_weight_histograms_count_matched",
        ):
            require(invariants.get(key) is True, f"count invariant {key}")
        deviations = audit.get("max_absolute_deviations")
        require(isinstance(deviations, dict), "count deviation object")
        max_difference = finite(
            deviations.get("reference_minus_assigned_rate"),
            f"seed {seed}: reference/assigned max difference",
        )
        rollup_records.append(
            {
                "seed": seed,
                "rows_checked": len(trace),
                "mismatch_count": 0,
                "max_abs_difference": max_difference,
                "passed": True,
            }
        )
        total_rows += len(trace)

    rollup = read_json(root / "positive_count_invariant_audit.json")
    require(
        set(rollup)
        == {
            "definition",
            "max_abs_difference",
            "passed",
            "records",
            "schema_version",
            "tolerance",
            "total_mismatch_count",
            "total_rows_checked",
        },
        "positive-count rollup schema",
    )
    require(
        int(rollup["schema_version"]) == 1
        and rollup["definition"]
        == (
            "reference_high_weight_rate equals assigned_high_weight_rate "
            "for every finite logged Random-Support row"
        )
        and math.isclose(
            finite(rollup["tolerance"], "count audit tolerance"),
            1e-12,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and rollup["passed"] is True
        and int(rollup["total_rows_checked"]) == total_rows
        and int(rollup["total_mismatch_count"]) == 0,
        "positive-count rollup contract",
    )
    records = rollup.get("records")
    require(
        isinstance(records, list) and len(records) == len(rollup_records),
        "positive-count rollup coverage",
    )
    for expected, actual in zip(rollup_records, records):
        require(isinstance(actual, dict), "malformed count rollup record")
        require(
            int(actual["seed"]) == int(expected["seed"])
            and int(actual["rows_checked"]) == int(expected["rows_checked"])
            and int(actual["mismatch_count"]) == 0
            and actual["passed"] is True,
            "positive-count rollup record",
        )
        close(
            actual["max_abs_difference"],
            float(expected["max_abs_difference"]),
            "positive-count rollup deviation",
        )
    close(
        rollup["max_abs_difference"],
        max(float(record["max_abs_difference"]) for record in rollup_records),
        "positive-count rollup maximum",
    )
    return total_rows


def verify_completion_and_provenance(
    root: Path,
    calculated: Mapping[tuple[str, int], Mapping[str, object]],
) -> None:
    completion = read_json(root / "completion_validation.json")
    require(completion.get("passed") is True, "completion status")
    require(
        completion.get("evaluation_field") == PRIMARY
        and completion.get("fallback_field") is None,
        "evaluation field/fallback",
    )
    require(completion.get("evaluation_grid") == list(STEPS), "completion grid")
    records = completion.get("records")
    require(isinstance(records, list) and len(records) == 5, "completion records")
    for seed, record in zip(SEEDS, records):
        require(isinstance(record, dict) and int(record["seed"]) == seed, "record seed")
        require(record.get("passed") is True, "record pass")
        require(
            int(record["total_steps"]) == 2_000_000
            and int(record["total_updates"]) == 1_995_001
            and int(record["evaluation_points"]) == 200,
            "completion counters",
        )
        summary = record.get("summary")
        require(isinstance(summary, dict), "record summary")
        close(
            summary["auc"],
            float(calculated[("random", seed)]["auc"]),
            "completion AUC",
        )
        for key in (
            "original_config_available",
            "original_environment_snapshot_available",
            "checkpoint_available",
            "training_log_available",
            "execution_git_sha_serialized",
        ):
            require(record.get(key) is False, f"legacy caveat {key}")
        require(
            record.get("raw_summary_run_dir_sanitized") is True,
            "raw summary run_dir sanitization flag",
        )
        packaged_summary = read_json(
            root / "data" / "random" / f"seed_{seed}" / "summary.json"
        )
        expected_run_name = (
            "drawer_close_DDPG-HER-B3-Random-Support-2M_"
            f"s{seed}_2m_cuda"
        )
        require(
            record.get("raw_summary_run_dir_basename") == expected_run_name,
            f"seed {seed}: original run_dir basename provenance",
        )
        digest = record.get("raw_summary_run_dir_sha256")
        require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"seed {seed}: original run_dir digest provenance",
        )
        packaged_run_dir = packaged_summary.get("run_dir")
        require(
            packaged_run_dir == f"data/random/seed_{seed}",
            f"seed {seed}: portable summary run_dir",
        )
        config = read_json(
            root / "data" / "random" / f"seed_{seed}" / "config.json"
        )
        env = read_json(
            root / "data" / "random" / f"seed_{seed}" / "env_snapshot.json"
        )
        require(
            config.get("not_original_run_config") is True
            and env.get("not_original_run_env_snapshot") is True,
            "proxy labels",
        )
        source_fields = config.get("source_derived_fields")
        require(isinstance(source_fields, dict), "source-derived config fields")
        require(
            source_fields.get("support_query_mode") == "current_policy"
            and source_fields.get("weighting_mode") == "batch_norm"
            and source_fields.get("weight_scope") == "critic_only"
            and source_fields.get("support_assignment_mode")
            == "batch_wide_count_matched_random"
            and source_fields.get("random_assignment_without_replacement")
            is True,
            "effective Random-Support config contract",
        )

    source = read_json(root / "provenance" / "source_provenance.json")
    require(source.get("passed") is True, "source provenance")
    require(source.get("source_tree_unchanged") is True, "source consistency")
    require(
        source.get("execution_source_commit_serialized_by_raw_run") is False
        and source.get("execution_binding_status")
        == "reconstructed_not_cryptographically_bound",
        "source caveat",
    )
    config_provenance = read_json(
        root / "provenance" / "config_provenance.json"
    )
    require(
        config_provenance.get("original_random_run_config_available") is False,
        "config provenance caveat",
    )
    input_manifest = read_json(root / "input_manifest.json")
    require(input_manifest.get("artifact_id") == ARTIFACT_ID, "input artifact id")
    inputs = input_manifest.get("inputs")
    require(isinstance(inputs, list), "input rows")
    roles = {str(row.get("role")) for row in inputs if isinstance(row, dict)}
    for seed in SEEDS:
        require(
            f"random.seed_{seed}.metrics" in roles
            and f"random.seed_{seed}.summary" in roles,
            f"raw input provenance seed {seed}",
        )

    checksums = {}
    for line in (root / "frozen_source.sha256").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, relative = line.split("  ", 1)
        checksums[relative] = digest
    source_files = {
        path.relative_to(root).as_posix(): path
        for path in (root / "source_snapshot").rglob("*")
        if path.is_file()
    }
    require(set(checksums) == set(source_files), "source checksum file set")
    for relative, path in source_files.items():
        require(sha256(path) == checksums[relative], f"source hash {relative}")
    weights = (
        root / "source_snapshot" / "code" / "src" / "metacog_lite_v0" / "weights.py"
    ).read_text(encoding="utf-8")
    require(
        "k_positive = int(np.sum(psi_q_rank > 0.5))" in weights
        and "rng.choice(b, size=k_positive, replace=False)" in weights,
        "frozen random-assignment definition",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.artifact_dir.resolve()
    require(root.is_dir(), f"missing artifact {root}")
    file_count = manifest_files(root)
    forbidden_payloads(root)
    calculated = verify_statistics(root)
    trace_rows = verify_count_traces(root)
    verify_completion_and_provenance(root, calculated)
    print(
        json.dumps(
            {
                "artifact": str(root),
                "status": "passed",
                "files_verified": file_count,
                "seed_runs_verified": 15,
                "random_completion_seeds_verified": 5,
                "evaluation_points_per_seed": 200,
                "assignment_trace_rows_verified": trace_rows,
                "bootstrap_resamples": 3125,
                "sign_flip_assignments": 32,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
