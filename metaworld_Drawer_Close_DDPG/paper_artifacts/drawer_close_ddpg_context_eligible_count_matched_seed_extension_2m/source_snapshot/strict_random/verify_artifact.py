#!/usr/bin/env python3
"""Independently verify the compact DrawerClose strict-control artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Sequence


SEEDS = (1, 2, 3, 4, 5)
STEPS = tuple(range(10_000, 2_000_001, 10_000))
PRIMARY = "eval/env_success_mean"
METHODS = ("her", "main", "strict")
CONTRASTS = (
    ("main_minus_strict", "main", "strict"),
    ("strict_minus_her", "strict", "her"),
    ("main_minus_her", "main", "her"),
)
DEFAULT_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "drawer_close_ddpg_saber_context_eligible_count_matched_2m"
)
TOL = 1e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        result = list(csv.DictReader(handle))
    if not result:
        raise ValueError(f"{path}: empty CSV")
    return result


def mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(math.fsum((value - center) ** 2 for value in values) / (len(values) - 1))


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def close(actual: object, expected: float, tolerance: float = TOL) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"not numeric: {actual!r}") from exc
    if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"numeric mismatch: {value!r} != {expected!r}")


def curve(path: Path) -> list[float]:
    table = rows(path)
    actual_steps: list[int] = []
    values: list[float] = []
    for row in table:
        raw = row.get(PRIMARY, "")
        if raw == "":
            raise ValueError(f"{path}: selected file contains a non-eval row")
        step = int(float(row["step"]))
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{path}: invalid success at {step}")
        actual_steps.append(step)
        values.append(value)
    if actual_steps != list(STEPS):
        raise ValueError(f"{path}: exact evaluation grid mismatch")
    return values


def auc(values: Sequence[float]) -> float:
    area = math.fsum(
        (right - left) * (y_left + y_right) / 2.0
        for left, right, y_left, y_right in zip(STEPS, STEPS[1:], values, values[1:])
    )
    return area / (STEPS[-1] - STEPS[0])


def run_metrics(values: Sequence[float]) -> dict[str, object]:
    hit = next((step for step, value in zip(STEPS, values) if value >= 0.90), None)
    best = max(values)
    return {
        "auc": auc(values),
        "final_success": values[-1],
        "best_success": best,
        "best_to_final_drop": best - values[-1],
        "hit90": hit,
    }


def manifest_files(root: Path, manifest: Mapping[str, object]) -> None:
    listed_rows = manifest.get("packaged_files")
    if not isinstance(listed_rows, list):
        raise ValueError("artifact manifest missing packaged_files")
    listed: dict[str, Mapping[str, object]] = {}
    for row in listed_rows:
        if not isinstance(row, dict):
            raise ValueError("malformed manifest row")
        relative = str(row.get("path", ""))
        if not relative or relative in listed or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"unsafe/duplicate manifest path {relative!r}")
        listed[relative] = row
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    if set(listed) != set(actual):
        raise ValueError(
            f"artifact file-set mismatch missing={sorted(set(listed)-set(actual))} "
            f"extra={sorted(set(actual)-set(listed))}"
        )
    for relative, row in listed.items():
        path = actual[relative]
        if path.stat().st_size != int(row["size_bytes"]) or sha256(path) != row["sha256"]:
            raise ValueError(f"artifact hash/size mismatch: {relative}")
    digest = hashlib.sha256()
    for relative in sorted(listed):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(listed[relative]["sha256"]).encode())
        digest.update(b"\n")
    if digest.hexdigest() != manifest.get("package_content_sha256_excluding_manifest"):
        raise ValueError("artifact content digest mismatch")
    if int(manifest.get("packaged_file_count_excluding_manifest", -1)) != len(actual):
        raise ValueError("artifact file count mismatch")


def forbidden_files(root: Path) -> None:
    failures = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        lowered = relative.as_posix().lower()
        if (
            path.suffix.lower() in {".pt", ".pth", ".jsonl", ".log"}
            or path.name == "metrics.csv"
            or any(part in {"runs", "logs", "smoke", "pilot", "correctness"} for part in relative.parts)
            or "/smoke_" in lowered
            or "/pilot_" in lowered
        ):
            failures.append(relative.as_posix())
    if failures:
        raise ValueError(f"forbidden artifact payloads: {failures}")


def verify_statistics(root: Path) -> dict[tuple[str, int], dict[str, object]]:
    calculated: dict[tuple[str, int], dict[str, object]] = {}
    curves: dict[tuple[str, int], list[float]] = {}
    for method in METHODS:
        for seed in SEEDS:
            values = curve(root / "data" / method / f"seed_{seed}" / "metrics_selected_10k.csv")
            curves[(method, seed)] = values
            calculated[(method, seed)] = run_metrics(values)

    per_seed = rows(root / "per_seed_results.csv")
    if len(per_seed) != 15:
        raise ValueError("per_seed_results must contain 15 rows")
    indexed = {(row["method_key"], int(row["seed"])): row for row in per_seed}
    if set(indexed) != set(calculated):
        raise ValueError("per-seed method/seed coverage mismatch")
    for key, expected in calculated.items():
        row = indexed[key]
        for field in ("auc", "final_success", "best_success", "best_to_final_drop"):
            close(row[field], float(expected[field]))
        native_hit = None if row["hit90"] == "" else int(float(row["hit90"]))
        if native_hit != expected["hit90"]:
            raise ValueError(f"Hit90 mismatch for {key}")

    aggregate = {row["method_key"]: row for row in rows(root / "aggregate_results.csv")}
    if set(aggregate) != set(METHODS):
        raise ValueError("aggregate method coverage mismatch")
    for method in METHODS:
        row = aggregate[method]
        for field in ("auc", "final_success", "best_success", "best_to_final_drop"):
            values = [float(calculated[(method, seed)][field]) for seed in SEEDS]
            close(row[f"{field}_mean"], mean(values))
            close(row[f"{field}_sample_sd"], sample_sd(values))
        hits = [
            int(calculated[(method, seed)]["hit90"])
            for seed in SEEDS
            if calculated[(method, seed)]["hit90"] is not None
        ]
        if int(row["hit90_seed_count"]) != len(hits):
            raise ValueError(f"Hit90 count mismatch for {method}")
        expected_median = "" if not hits else str(float(median(hits)))
        if (row["hit90_median_reached"] == "") != (expected_median == ""):
            raise ValueError(f"Hit90 median presence mismatch for {method}")
        if hits:
            close(row["hit90_median_reached"], float(median(hits)))

    paired = {row["contrast_key"]: row for row in rows(root / "paired_comparisons.csv")}
    if set(paired) != {item[0] for item in CONTRASTS}:
        raise ValueError("paired contrast coverage mismatch")
    for key, left, right in CONTRASTS:
        differences = [
            float(calculated[(left, seed)]["auc"]) - float(calculated[(right, seed)]["auc"])
            for seed in SEEDS
        ]
        boot_means = [
            mean([differences[index] for index in indices])
            for indices in itertools.product(range(5), repeat=5)
        ]
        observed = abs(mean(differences))
        flipped = [
            abs(mean([sign * value for sign, value in zip(signs, differences)]))
            for signs in itertools.product((-1.0, 1.0), repeat=5)
        ]
        row = paired[key]
        close(row["mean_auc_difference"], mean(differences))
        close(row["sample_sd_auc_difference"], sample_sd(differences))
        close(row["bootstrap_ci95_lower"], percentile(boot_means, 0.025))
        close(row["bootstrap_ci95_upper"], percentile(boot_means, 0.975))
        if int(row["bootstrap_resample_count"]) != 3125:
            raise ValueError("bootstrap resample count mismatch")
        pvalue = sum(value >= observed - 1e-15 for value in flipped) / 32.0
        close(row["exact_two_sided_sign_flip_pvalue"], pvalue)
        if int(row["sign_flip_assignment_count"]) != 32:
            raise ValueError("sign-flip assignment count mismatch")
        signs = (
            sum(value > 1e-12 for value in differences),
            sum(abs(value) <= 1e-12 for value in differences),
            sum(value < -1e-12 for value in differences),
        )
        if signs != (
            int(row["positive_seed_count"]),
            int(row["zero_seed_count"]),
            int(row["negative_seed_count"]),
        ):
            raise ValueError(f"paired sign counts mismatch for {key}")
        loso = [
            mean([value for index, value in enumerate(differences) if index != omitted])
            for omitted in range(5)
        ]
        close(row["leave_one_seed_out_mean_min"], min(loso))
        close(row["leave_one_seed_out_mean_max"], max(loso))
        for seed, value in zip(SEEDS, differences):
            close(row[f"seed_{seed}_auc_difference"], value)

    means = rows(root / "exact_grid_mean_curves.csv")
    if [int(row["step"]) for row in means] != list(STEPS):
        raise ValueError("mean-curve grid mismatch")
    for index, row in enumerate(means):
        for method in METHODS:
            values = [curves[(method, seed)][index] for seed in SEEDS]
            close(row[f"{method}_mean_success"], mean(values))
            close(row[f"{method}_sample_sd_success"], sample_sd(values))
    return calculated


def verify_strict(root: Path) -> None:
    completion = load_json(root / "completion_validation.json")
    if completion.get("passed") is not True or completion.get("seed_count") != 5:
        raise ValueError("completion gate did not pass for five seeds")
    records = completion.get("records")
    if not isinstance(records, list) or len(records) != 5:
        raise ValueError("completion records incomplete")
    for record in records:
        if not isinstance(record, dict) or record.get("passed") is not True:
            raise ValueError("failed completion record")
        if (
            record.get("total_steps") != 2_000_000
            or record.get("final_replay_size") != 300_000
            or record.get("evaluation_points") != 200
            or record.get("evaluation_episodes_per_point") != 20
        ):
            raise ValueError("completion contract mismatch")
        if record.get("config_parity", {}).get("passed") is not True:
            raise ValueError("config parity failed")
        if record.get("checkpoint_resume_audit", {}).get("passed") is not True:
            raise ValueError("checkpoint/resume audit failed")

    for seed in SEEDS:
        path = root / "data" / "strict" / f"seed_{seed}" / "strict_diagnostic_trace.csv"
        table = rows(path)
        expected_start = 1
        total = 0
        for row in table:
            for key, raw in row.items():
                if raw == "" or not math.isfinite(float(raw)):
                    raise ValueError(f"{path}: non-finite trace field {key}")
            start = int(float(row["update_start"]))
            end = int(float(row["update_end"]))
            count = int(float(row["updates"]))
            if start != expected_start or end != start + count - 1:
                raise ValueError(f"{path}: non-contiguous diagnostic trace")
            expected_start = end + 1
            total += count
            for field in (
                "count_equality_violation_sum",
                "control_selected_outside_eligible_sum",
                "raw_histogram_mismatch_sum",
                "fallback_sampling_used_sum",
                "replacement_sampling_used_sum",
            ):
                if float(row[field]) != 0.0:
                    raise ValueError(f"{path}: strict invariant failed: {field}")
            close(row["normalized_weight_mean_min"], 1.0, 2e-6)
            close(row["normalized_weight_mean_max"], 1.0, 2e-6)
            if float(row["normalized_weight_min_min"]) <= 0.0:
                raise ValueError(f"{path}: non-positive weight")
        if total != 1_995_001:
            raise ValueError(f"{path}: update coverage {total} != 1995001")


def verify_provenance(root: Path, manifest: Mapping[str, object]) -> None:
    commit = str(manifest.get("source_commit", ""))
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit.lower()):
        raise ValueError("invalid source commit")
    input_manifest = load_json(root / "input_manifest.json")
    completion = load_json(root / "completion_validation.json")
    freeze = load_json(root / "provenance/freeze_manifest.json")
    if input_manifest.get("source_commit") != commit or completion.get("source_commit") != commit or freeze.get("source_commit") != commit:
        raise ValueError("source commit disagreement")
    frozen = freeze.get("files")
    if not isinstance(frozen, list) or not frozen:
        raise ValueError("empty frozen source")
    expected_lines = []
    for row in frozen:
        if not isinstance(row, dict):
            raise ValueError("malformed source freeze row")
        relative = str(row["path"])
        source = root / "source_snapshot" / relative
        if source.stat().st_size != int(row["size"]) or sha256(source) != row["sha256"]:
            raise ValueError(f"frozen source mismatch: {relative}")
        expected_lines.append(f"{row['sha256']}  source_snapshot/{relative}")
    actual_lines = (root / "frozen_source.sha256").read_text(encoding="utf-8").splitlines()
    if actual_lines != expected_lines:
        raise ValueError("frozen_source.sha256 mismatch")

    parity = load_json(root / "provenance/config_parity.json")
    if parity.get("passed") is not True or parity.get("nonallowed_differences") != []:
        raise ValueError("config parity report failed")
    allowed = set(parity.get("allowed_paths", []))
    seed_rows = parity.get("seeds")
    if not isinstance(seed_rows, list) or [row.get("seed") for row in seed_rows] != list(SEEDS):
        raise ValueError("config parity seed coverage mismatch")
    for seed_row in seed_rows:
        if seed_row.get("passed") is not True or seed_row.get("nonallowed_differences") != []:
            raise ValueError("seed config parity failed")
        for difference in seed_row.get("differences", []):
            if difference.get("allowed") is not True or difference.get("path") not in allowed:
                raise ValueError("non-allowlisted config difference")
    for seed in SEEDS:
        config = load_json(root / "data" / "strict" / f"seed_{seed}" / "config.json")
        if (
            config.get("task_name") != "drawer_close"
            or config.get("metaworld_env_name") != "drawer-close-v3"
            or config.get("method") != "b3_support_runtime_opt"
            or config.get("support_assignment_mode") != "context_eligible_count_matched"
            or config.get("weight_scope") != "critic_only"
            or config.get("total_steps") != 2_000_000
            or config.get("eval_freq") != 10_000
            or config.get("eval_episodes") != 20
            or config.get("seeds", {}).get("assignment_seed") != 30_000 + seed
        ):
            raise ValueError(f"strict config contract mismatch seed {seed}")

    source_parity = load_json(root / "provenance/source_parity.json")
    if source_parity.get("passed") is not True or source_parity.get("failures") != []:
        raise ValueError("source parity failed")
    frozen_index = {str(row["path"]): row for row in frozen}
    for row in source_parity.get("exact_files", []):
        relative = f"code/src/{row['path']}"
        if row.get("byte_identical") is not True or row.get("original_sha256") != row.get("control_sha256"):
            raise ValueError(f"source parity mismatch {relative}")
        if relative not in frozen_index or frozen_index[relative]["sha256"] != row["control_sha256"]:
            raise ValueError(f"source parity/freeze disagreement {relative}")

    resume = load_json(root / "provenance/checkpoint_resume_audit.json")
    if resume.get("passed") is not True or resume.get("seed_coverage") != list(SEEDS):
        raise ValueError("checkpoint/resume provenance failed")


def verify(root: Path) -> dict[str, object]:
    root = root.resolve()
    manifest = load_json(root / "artifact_manifest.json")
    if manifest.get("artifact_id") != "drawer_close_ddpg_saber_context_eligible_count_matched_2m":
        raise ValueError("wrong artifact id")
    manifest_files(root, manifest)
    forbidden_files(root)
    calculated = verify_statistics(root)
    verify_strict(root)
    verify_provenance(root, manifest)
    return {
        "status": "passed",
        "artifact": root.as_posix(),
        "files_verified": manifest["packaged_file_count_excluding_manifest"],
        "seed_runs_verified": len(calculated),
        "evaluation_points_per_seed": len(STEPS),
        "bootstrap_resamples": 3125,
        "sign_flip_assignments": 32,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = verify(parse_args(argv).artifact_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
