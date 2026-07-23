#!/usr/bin/env python3
"""Independently verify the compact DrawerClose replay-action-query artifact.

This verifier intentionally uses only the Python standard library and fails
closed. It recomputes all paper statistics from the packaged exact-grid CSVs,
checks the packaged analysis implementation against that independent
recomputation, and verifies completion/provenance claims without the original
repository or raw run directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Sequence


SEEDS = (1, 2, 3, 4, 5)
STEPS = tuple(range(10_000, 2_000_001, 10_000))
PRIMARY = "eval/env_success_mean"
METHODS = ("her", "main", "replay")
CONTRASTS = (
    ("main_minus_replay", "main", "replay"),
    ("replay_minus_her", "replay", "her"),
    ("main_minus_her", "main", "her"),
)
DEFAULT_ARTIFACT = Path(".")
TOL = 1e-10
ARTIFACT_ID = "drawer_close_ddpg_saber_replay_action_query_2m"
EXECUTION_COMMIT = "cb871323fed51c6c7d5f692d189ebd766f282503"
EXPERIMENT_TREE = "1eb92f2850dcd492b265d2964218c37ff5eee7ff"
COMPARATOR_ARTIFACT_ID = "drawer_close_ddpg_saber_context_eligible_count_matched_2m"
COMPARATOR_ARTIFACT_COMMIT = "841eb706b1dcdc12b59db23ec545099d39d3c619"
COMPARATOR_PACKAGING_COMMIT = "de13e3a0a14d3dc6984d3c592452517f3e32c109"


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


def safe_relative(value: object, label: str) -> Path:
    relative = Path(str(value))
    if not str(value) or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}: unsafe relative path {value!r}")
    return relative


def is_passed(report: Mapping[str, object], label: str) -> None:
    evidence = False
    for key in ("passed", "pass"):
        if key in report:
            evidence = True
            if report[key] is not True:
                raise ValueError(f"{label}: {key} is not true")
    if "status" in report:
        evidence = True
        if str(report["status"]).lower() not in {"pass", "passed", "complete", "completed"}:
            raise ValueError(f"{label}: non-passing status {report['status']!r}")
    if not evidence:
        raise ValueError(f"{label}: missing explicit pass/status field")


def alias_value(
    report: Mapping[str, object],
    names: Sequence[str],
    label: str,
    *,
    required: bool = True,
) -> object | None:
    found = [(name, report[name]) for name in names if name in report]
    if not found:
        if required:
            raise ValueError(f"{label}: missing one of {tuple(names)!r}")
        return None
    first = found[0][1]
    if any(value != first for _, value in found[1:]):
        raise ValueError(f"{label}: conflicting aliases {found!r}")
    return first


def nested_alias(
    report: Mapping[str, object], names: Sequence[str], label: str
) -> object:
    value = alias_value(report, names, label, required=False)
    if value is not None:
        return value
    for container_name in ("metrics", "checks", "query", "query_diagnostics", "completion"):
        container = report.get(container_name)
        if isinstance(container, dict):
            value = alias_value(container, names, label, required=False)
            if value is not None:
                return value
    raise ValueError(f"{label}: missing one of {tuple(names)!r}")


def integer(actual: object, expected: int, label: str) -> None:
    if isinstance(actual, bool):
        raise ValueError(f"{label}: boolean is not an integer count")
    try:
        numeric = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: not numeric: {actual!r}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or int(numeric) != expected:
        raise ValueError(f"{label}: {actual!r} != {expected}")


def hex_id(value: object, lengths: Sequence[int], label: str) -> str:
    text = str(value)
    if len(text) not in lengths or any(character not in "0123456789abcdef" for character in text.lower()):
        raise ValueError(f"{label}: invalid hexadecimal identifier {text!r}")
    return text


def query_match_is_one(value: object, label: str) -> None:
    if value is True:
        return
    if isinstance(value, dict):
        if "all_one" in value and value["all_one"] is not True:
            raise ValueError(f"{label}: all_one is not true")
        checked = False
        for key in ("value", "rate", "min", "max", "mean"):
            if key in value:
                close(value[key], 1.0, 1e-12)
                checked = True
        if checked:
            return
    close(value, 1.0, 1e-12)


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
            path.suffix.lower() in {".pt", ".pth", ".ckpt", ".jsonl", ".log", ".out"}
            or path.name in {"metrics.csv", "metrics.jsonl"}
            or any(
                part.lower() in {
                    "runs", "runs_2m", "results", "results_2m", "logs",
                    "logs_2m", "checkpoints", "smoke", "pilot", "correctness",
                    "__pycache__", ".pytest_cache",
                }
                for part in relative.parts
            )
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


def verify_packaged_analyzer(
    root: Path,
    calculated: Mapping[tuple[str, int], Mapping[str, object]],
) -> None:
    """Cross-check the packaged analyzer without writing bytecode/output files."""

    analyzer_path = root / "analyze_results.py"
    spec = importlib.util.spec_from_file_location(
        "_drawerclose_replay_artifact_analyzer", analyzer_path
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load packaged analyze_results.py")
    module = importlib.util.module_from_spec(spec)
    old_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
        sys.dont_write_bytecode = old_dont_write
    if (
        getattr(module, "PRIMARY_SUCCESS_FIELD", None) != PRIMARY
        or tuple(getattr(module, "SEEDS", ())) != SEEDS
        or tuple(getattr(module, "EVAL_STEPS", ())) != STEPS
        or tuple(getattr(module, "METHOD_ORDER", ())) != METHODS
    ):
        raise ValueError("packaged analyzer primary field/cohort/grid mismatch")
    for method in METHODS:
        for seed in SEEDS:
            path = root / "data" / method / f"seed_{seed}" / "metrics_selected_10k.csv"
            steps, values = module.load_eval_curve(path)
            if list(steps) != list(STEPS) or len(values) != 200:
                raise ValueError(f"packaged analyzer grid mismatch for {method} seed {seed}")
            expected = calculated[(method, seed)]
            close(module.normalized_auc(steps, values), float(expected["auc"]))
            close(values[-1], float(expected["final_success"]))
            if module.first_hit90(steps, values) != expected["hit90"]:
                raise ValueError(f"packaged analyzer Hit90 mismatch for {method} seed {seed}")


def verify_replay(root: Path) -> None:
    completion = load_json(root / "completion_validation.json")
    is_passed(completion, "completion_validation")
    integer(alias_value(completion, ("seed_count", "replay_seed_count"), "completion seed count"), 5, "completion seed count")
    if (
        completion.get("seeds") != list(SEEDS)
        or completion.get("evaluation_grid") != list(STEPS)
        or completion.get("evaluation_field") != PRIMARY
    ):
        raise ValueError("completion seed/evaluation coverage mismatch")
    contract = completion.get("query_contract")
    if contract is not None:
        if not isinstance(contract, dict):
            raise ValueError("completion query_contract must be an object")
        if contract.get("support_query_mode") != "replay_action" or contract.get("weight_scope") != "critic_only":
            raise ValueError("completion query mode/scope mismatch")
        if (
            contract.get("support_assignment_mode_serialized") is not None
            or contract.get("effective_support_assignment_mode") != "support"
            or contract.get("support_assignment_mode") not in (None, "support")
        ):
            raise ValueError("completion assignment serialization/effective mode mismatch")
        if contract.get("sampled_transition_removed_from_local_pool") is not False:
            raise ValueError("completion must state that sampled transitions were not removed")
        if contract.get("self_support_possible") is not True:
            raise ValueError("completion must state that replay self-support is possible")
        query_match_is_one(contract.get("replay_action_bin_match"), "completion replay query match")
    records = completion.get("replay_seeds", completion.get("records"))
    if not isinstance(records, list) or len(records) != 5:
        raise ValueError("replay completion records incomplete")
    indexed = {int(record["seed"]): record for record in records if isinstance(record, dict)}
    if set(indexed) != set(SEEDS):
        raise ValueError("replay completion seed coverage mismatch")
    for seed in SEEDS:
        record = indexed[seed]
        is_passed(record, f"completion seed {seed}")
        if (
            record.get("support_assignment_mode_serialized") is not None
            or record.get("effective_support_assignment_mode") != "support"
        ):
            raise ValueError(f"seed {seed}: assignment serialization/effective mode mismatch")
        integer(nested_alias(record, ("total_steps",), f"seed {seed} total_steps"), 2_000_000, f"seed {seed} total_steps")
        integer(nested_alias(record, ("total_updates",), f"seed {seed} total_updates"), 1_995_001, f"seed {seed} total_updates")
        integer(nested_alias(record, ("final_replay_size", "replay_size"), f"seed {seed} replay size"), 300_000, f"seed {seed} replay size")
        integer(nested_alias(record, ("eval_points", "evaluation_points"), f"seed {seed} eval points"), 200, f"seed {seed} eval points")
        episodes = alias_value(record, ("evaluation_episodes_per_point", "eval_episodes"), f"seed {seed} eval episodes", required=False)
        if episodes is not None:
            integer(episodes, 20, f"seed {seed} eval episodes")
        mode = alias_value(record, ("support_query_mode", "query_mode"), f"seed {seed} query mode", required=False)
        if mode is not None and mode != "replay_action":
            raise ValueError(f"seed {seed}: wrong query mode")
        scope = alias_value(record, ("weight_scope",), f"seed {seed} weight scope", required=False)
        if scope is not None and scope != "critic_only":
            raise ValueError(f"seed {seed}: wrong weight scope")
        if record.get("single_critic_ddpg") is not True:
            raise ValueError(f"seed {seed}: single-critic DDPG gate failed")
        record_diagnostic = record.get("query_diagnostics")
        if isinstance(record_diagnostic, dict) and isinstance(record_diagnostic.get("fields"), dict):
            record_match = record_diagnostic["fields"].get(
                "query/query_bin_matches_replay_action_rate"
            )
        else:
            record_match = nested_alias(
                record,
                ("query_match", "query_match_rate", "replay_action_bin_match", "replay_action_bin_match_min"),
                f"seed {seed} replay query match",
            )
        query_match_is_one(record_match, f"seed {seed} replay query match")
        finite = alias_value(record.get("checks", {}) if isinstance(record.get("checks"), dict) else record, ("all_required_fields_finite", "all_finite", "finite"), f"seed {seed} finite gate", required=False)
        if finite is not None and finite is not True:
            raise ValueError(f"seed {seed}: finite gate failed")

        seed_root = root / "data" / "replay" / f"seed_{seed}"
        seed_completion = load_json(seed_root / "completion_validation.json")
        if seed_completion != record:
            raise ValueError(f"packaged completion record disagrees with top-level seed {seed}")
        is_passed(seed_completion, f"packaged completion seed {seed}")
        integer(nested_alias(seed_completion, ("total_steps",), f"packaged seed {seed} total_steps"), 2_000_000, f"packaged seed {seed} total_steps")
        integer(nested_alias(seed_completion, ("total_updates",), f"packaged seed {seed} total_updates"), 1_995_001, f"packaged seed {seed} total_updates")
        integer(nested_alias(seed_completion, ("final_replay_size", "replay_size"), f"packaged seed {seed} replay size"), 300_000, f"packaged seed {seed} replay size")
        integer(nested_alias(seed_completion, ("eval_points", "evaluation_points"), f"packaged seed {seed} eval points"), 200, f"packaged seed {seed} eval points")

        config = load_json(root / "data" / "replay" / f"seed_{seed}" / "config.json")
        if (
            config.get("task_name") != "drawer_close"
            or config.get("metaworld_env_name") != "drawer-close-v3"
            or config.get("method") != "b3_support_runtime_opt"
            or config.get("support_query_mode") != "replay_action"
            or config.get("weight_scope") != "critic_only"
            or config.get("weighting_mode") != "batch_norm"
            or config.get("total_steps") != 2_000_000
            or config.get("eval_freq") != 10_000
            or config.get("eval_episodes") != 20
            or config.get("save_step_checkpoints") is not False
            or config.get("seeds", {}).get("env_seed") != seed
        ):
            raise ValueError(f"seed {seed}: packaged replay config mismatch")
        if "support_assignment_mode" in config:
            raise ValueError(f"seed {seed}: legacy config falsely serializes assignment mode")
        load_json(seed_root / "env_snapshot.json")
        summary = load_json(seed_root / "summary.json")
        if (
            summary.get("method") != "b3_support_runtime_opt"
            or summary.get("support_query_mode") != "replay_action"
            or summary.get("total_steps") != 2_000_000
            or summary.get("total_updates") != 1_995_001
            or summary.get("final_replay_size") != 300_000
            or "support_assignment_mode" in summary
        ):
            raise ValueError(f"seed {seed}: packaged Replay summary contract mismatch")
        diagnostic = load_json(root / "data" / "replay" / f"seed_{seed}" / "query_diagnostic_summary.json")
        if diagnostic.get("support_query_mode") != "replay_action" or diagnostic.get("weight_scope", "critic_only") != "critic_only" or diagnostic.get("sampled_transition_removed_from_local_pool") is not False or diagnostic.get("self_support_possible") is not True:
            raise ValueError(f"seed {seed}: packaged query definition mismatch")
        if (
            diagnostic.get("support_assignment_mode_serialized") is not None
            or diagnostic.get("effective_support_assignment_mode") != "support"
            or diagnostic.get("support_assignment_mode") not in (None, "support", "implicit_support_not_serialized")
        ):
            raise ValueError(f"seed {seed}: diagnostic assignment mode mismatch")
        invariants = diagnostic.get("invariants", {})
        if isinstance(invariants, dict):
            invariant_aliases = (
                ("all_required_fields_finite", "all_finite"),
                ("replay_action_bin_match_all_one",),
                ("normalized_weight_mean_all_one_within_1e-6", "normalized_weight_mean_all_one"),
            )
            for aliases in invariant_aliases:
                present = [invariants[key] for key in aliases if key in invariants]
                if not present or any(value is not True for value in present):
                    raise ValueError(f"seed {seed}: missing/failed query invariant {aliases}")
        fields = diagnostic.get("fields")
        if not isinstance(fields, dict) or "query/query_bin_matches_replay_action_rate" not in fields:
            raise ValueError(f"seed {seed}: missing replay-action query diagnostic")
        query_match_is_one(fields["query/query_bin_matches_replay_action_rate"], f"seed {seed} packaged replay query match")
    for method in ("her", "main"):
        for seed in SEEDS:
            config = load_json(root / "data" / method / f"seed_{seed}" / "config.json")
            env = load_json(root / "data" / method / f"seed_{seed}" / "env_snapshot.json")
            config_marker = " ".join(str(config.get(key, "")) for key in ("record_type", "provenance_status", "note")).lower()
            env_marker = " ".join(str(env.get(key, "")) for key in ("record_type", "provenance_status", "note")).lower()
            if config.get("not_original_run_config") is not True or "reconstruct" not in config_marker:
                raise ValueError(f"{method} seed {seed}: comparator config is mislabelled")
            if env.get("not_original_run_env_snapshot") is not True or not any(word in env_marker for word in ("proxy", "reconstruct", "not_original")):
                raise ValueError(f"{method} seed {seed}: environment proxy is mislabelled")

def verify_comparator_provenance(
    root: Path,
    manifest: Mapping[str, object],
    input_manifest: Mapping[str, object],
) -> None:
    packaging_commit = str(
        manifest.get("comparator_packaging_source_commit", "")
    )
    if packaging_commit != COMPARATOR_PACKAGING_COMMIT:
        raise ValueError("wrong comparator packaging source commit")
    comparator = input_manifest.get("comparator")
    if not isinstance(comparator, dict):
        raise ValueError("missing comparator provenance")
    if (
        comparator.get("artifact_id") != COMPARATOR_ARTIFACT_ID
        or comparator.get("artifact_git_commit") != COMPARATOR_ARTIFACT_COMMIT
        or comparator.get("packaging_artifact_source_commit")
        != packaging_commit
        or comparator.get("original_training_source_commit") is not None
        or comparator.get("original_training_source_commit_status")
        != "unknown_not_recorded"
        or comparator.get("arms_reused") != ["her", "main"]
        or comparator.get("retrained") is not False
        or comparator.get("original_comparator_config_env_available") is not False
    ):
        raise ValueError("comparator provenance semantics mismatch")
    relative = comparator.get("upstream_input_manifest_path")
    if relative != "provenance/comparator_input_manifest.json":
        raise ValueError("unexpected comparator input-manifest path")
    upstream_path = root / str(relative)
    expected_sha = comparator.get("upstream_input_manifest_sha256")
    if not upstream_path.is_file() or sha256(upstream_path) != expected_sha:
        raise ValueError("comparator input-manifest SHA mismatch")
    upstream = load_json(upstream_path)
    if upstream.get("source_commit") != packaging_commit:
        raise ValueError("upstream packaging source commit mismatch")
    upstream_rows = upstream.get("inputs")
    if (
        not isinstance(upstream_rows, list)
        or int(upstream.get("input_file_count", -1)) != len(upstream_rows)
    ):
        raise ValueError("upstream comparator input manifest is malformed")
    upstream_index = {
        str(row.get("role", "")): row
        for row in upstream_rows
        if isinstance(row, dict)
    }
    local_rows = input_manifest.get("inputs")
    if not isinstance(local_rows, list):
        raise ValueError("local input manifest is malformed")
    local_index = {
        str(row.get("role", "")): row
        for row in local_rows
        if isinstance(row, dict)
    }
    copied_record = local_index.get("provenance.comparator_input_manifest")
    if (
        not isinstance(copied_record, dict)
        or copied_record.get("sha256") != expected_sha
        or copied_record.get("packaging_artifact_source_commit")
        != packaging_commit
    ):
        raise ValueError("copied comparator input-manifest record mismatch")
    for method in ("her", "main"):
        for seed in SEEDS:
            for suffix in ("metrics", "summary"):
                role = f"{method}.seed_{seed}.{suffix}"
                upstream_row = upstream_index.get(role)
                local_row = local_index.get(role)
                if not isinstance(upstream_row, dict) or not isinstance(local_row, dict):
                    raise ValueError(f"missing comparator provenance chain: {role}")
                if local_row.get("upstream_input_provenance") != upstream_row:
                    raise ValueError(f"upstream path/hash chain mismatch: {role}")
                if (
                    local_row.get("packaging_artifact_source_commit")
                    != packaging_commit
                    or local_row.get("original_training_source_commit") is not None
                    or local_row.get("original_training_source_commit_status")
                    != "unknown_not_recorded"
                ):
                    raise ValueError(f"training/packaging commit semantics mismatch: {role}")
                source_path = str(upstream_row.get("source_path", ""))
                source_sha = str(upstream_row.get("sha256", ""))
                if (
                    not source_path
                    or len(source_sha) != 64
                    or int(upstream_row.get("size_bytes", 0)) <= 0
                ):
                    raise ValueError(f"malformed upstream path/hash provenance: {role}")


def verify_frozen_snapshot(root: Path, freeze: Mapping[str, object]) -> None:
    if freeze.get("schema_version") != 1:
        raise ValueError("unsupported frozen-source schema")
    if freeze.get("source_commit") != EXECUTION_COMMIT:
        raise ValueError("frozen source execution commit mismatch")
    if freeze.get("source_tree_oid") != EXPERIMENT_TREE:
        raise ValueError("frozen source experiment tree mismatch")
    if freeze.get("head_binding_verified") is not True or freeze.get("tracked_source_clean") is not True:
        raise ValueError("frozen source lacks clean HEAD binding")
    expected_root = (
        "metaworld_Drawer_Close_DDPG/experiments/"
        "DDPG-HER-B3-Replay-Action-Query-2M"
    )
    if freeze.get("source_root") != expected_root:
        raise ValueError("frozen source root mismatch")
    rows = freeze.get("files")
    if not isinstance(rows, list) or len(rows) != 24 or freeze.get("file_count") != 24:
        raise ValueError("frozen source must contain exactly 24 files")
    expected_lines: list[str] = []
    seen_paths: set[str] = set()
    seen_repo_paths: set[str] = set()
    indexed: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed source freeze row")
        relative = str(row.get("path", ""))
        repo_path = str(row.get("repo_path", ""))
        digest = str(row.get("sha256", ""))
        oid = str(row.get("git_blob_oid", ""))
        if (
            not relative
            or relative in seen_paths
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not repo_path
            or repo_path in seen_repo_paths
            or Path(repo_path).is_absolute()
            or repo_path != f"{expected_root}/{relative}"
            or row.get("head_bound") is not True
            or len(digest) != 64
            or len(oid) not in (40, 64)
            or any(char not in "0123456789abcdef" for char in (digest + oid).lower())
        ):
            raise ValueError(f"invalid HEAD-bound source metadata: {relative}")
        seen_paths.add(relative)
        seen_repo_paths.add(repo_path)
        indexed[relative] = row
        source = root / "source_snapshot" / relative
        if (
            not source.is_file()
            or source.stat().st_size != int(row.get("size_bytes", -1))
            or sha256(source) != digest
        ):
            raise ValueError(f"frozen source mismatch: {relative}")
        expected_lines.append(f"{digest}  source_snapshot/{relative}")
    actual_snapshot = {
        path.relative_to(root / "source_snapshot").as_posix()
        for path in (root / "source_snapshot").rglob("*")
        if path.is_file()
    }
    if actual_snapshot != seen_paths:
        raise ValueError("source_snapshot file-set does not match freeze manifest")
    launcher_name = "DDPG-HER-B3-Replay-Action-Query-2M.sh"
    launcher = freeze.get("entrypoint")
    if not isinstance(launcher, dict) or launcher != indexed.get(launcher_name):
        raise ValueError("frozen source entrypoint mismatch")
    actual_lines = (root / "frozen_source.sha256").read_text(
        encoding="utf-8"
    ).splitlines()
    if actual_lines != expected_lines:
        raise ValueError("frozen_source.sha256 mismatch")


def verify_provenance(root: Path, manifest: Mapping[str, object]) -> None:
    commit = hex_id(manifest.get("source_commit", ""), (40,), "replay source commit")
    if commit != EXECUTION_COMMIT:
        raise ValueError("wrong Replay execution commit")
    if manifest.get("source_tree_oid") != EXPERIMENT_TREE:
        raise ValueError("wrong Replay execution source tree")
    if manifest.get("comparator_artifact_commit") != COMPARATOR_ARTIFACT_COMMIT:
        raise ValueError("wrong comparator artifact commit")
    if manifest.get("comparator_packaging_source_commit") != COMPARATOR_PACKAGING_COMMIT:
        raise ValueError("wrong comparator packaging source commit")
    if "comparator_source_commit" in manifest:
        raise ValueError("ambiguous comparator_source_commit field is forbidden")
    if manifest.get("method_seed_coverage") != {key: list(SEEDS) for key in METHODS}:
        raise ValueError("manifest 15-run coverage mismatch")
    evaluation = manifest.get("evaluation", {})
    if not isinstance(evaluation, dict) or evaluation.get("field") != PRIMARY or evaluation.get("steps") != list(STEPS) or evaluation.get("points_per_seed") != 200 or evaluation.get("episodes_per_point") != 20 or any(evaluation.get(key) is not False for key in ("smoothing", "interpolation", "forward_fill", "extrapolation")):
        raise ValueError("manifest evaluation convention mismatch")
    if manifest.get("bootstrap", {}).get("resamples") != 3125 or manifest.get("exact_sign_flip", {}).get("assignments") != 32:
        raise ValueError("manifest paired-statistic enumeration mismatch")
    if (
        manifest.get("support_assignment_mode_serialized") is not None
        or manifest.get("effective_support_assignment_mode") != "support"
        or manifest.get("resume_reproducibility_status") != "not_empirically_audited"
        or manifest.get("completion_passed") is not True
        or manifest.get("config_parity_passed") is not True
        or manifest.get("source_parity_passed") is not True
    ):
        raise ValueError("manifest assignment/resume/completion honesty mismatch")
    input_manifest = load_json(root / "input_manifest.json")
    completion = load_json(root / "completion_validation.json")
    freeze = load_json(root / "provenance/freeze_manifest.json")
    if (
        input_manifest.get("artifact_id") != ARTIFACT_ID
        or input_manifest.get("source_commit") != commit
        or input_manifest.get("source_tree_oid") != EXPERIMENT_TREE
        or completion.get("source_commit") != commit
        or freeze.get("source_commit") != commit
        or freeze.get("source_tree_oid") != EXPERIMENT_TREE
    ):
        raise ValueError("replay source commit disagreement")
    verify_comparator_provenance(root, manifest, input_manifest)
    inputs = input_manifest.get("inputs")
    if not isinstance(inputs, list) or int(input_manifest.get("input_file_count", -1)) != len(inputs):
        raise ValueError("input manifest count mismatch")
    roles: set[str] = set()
    for row in inputs:
        if not isinstance(row, dict):
            raise ValueError("malformed input-manifest row")
        role = str(row.get("role", ""))
        if not role or role in roles:
            raise ValueError(f"unsafe/duplicate input role {role!r}")
        roles.add(role)
        hex_id(row.get("sha256", ""), (64,), f"input {role} SHA256")
        if int(row.get("size_bytes", 0)) <= 0 or not str(row.get("source_path", "")):
            raise ValueError(f"input {role}: missing size/source path")
        if not isinstance(row.get("included_in_artifact"), bool):
            raise ValueError(f"input {role}: missing inclusion decision")
    for method in ("her", "main"):
        for seed in SEEDS:
            for suffix in ("metrics", "summary"):
                if f"{method}.seed_{seed}.{suffix}" not in roles:
                    raise ValueError(f"missing input provenance: {method} seed {seed} {suffix}")
            for suffix in ("reconstructed_config_proxy", "environment_proxy"):
                if f"{method}.seed_{seed}.{suffix}" not in roles:
                    raise ValueError(f"missing comparator proxy provenance: {method} seed {seed} {suffix}")
    for seed in SEEDS:
        for suffix in (
            "config", "env_snapshot", "summary", "metrics",
            "completion_checkpoint_excluded", "external_log_excluded",
        ):
            if f"replay.seed_{seed}.{suffix}" not in roles:
                raise ValueError(f"missing Replay input provenance: seed {seed} {suffix}")
        for excluded in ("completion_checkpoint_excluded", "external_log_excluded"):
            record = next(row for row in inputs if row.get("role") == f"replay.seed_{seed}.{excluded}")
            if record.get("included_in_artifact") is not False:
                raise ValueError(f"seed {seed}: raw checkpoint/log was marked included")
    transform = input_manifest.get("transform")
    if not isinstance(transform, dict) or any(
        transform.get(key) is not False
        for key in (
            "checkpoints_copied", "best_checkpoints_copied", "raw_metrics_copied",
            "jsonl_copied", "logs_copied", "smoke_or_pilot_copied",
        )
    ):
        raise ValueError("input transform does not explicitly exclude raw payloads")
    verify_frozen_snapshot(root, freeze)
    parity = load_json(root / "provenance/config_parity.json")
    if (
        parity.get("passed") is not True
        or parity.get("nonallowed_differences") != []
        or parity.get("allowed_scientific_difference") != "support_query_mode"
        or parity.get("method") != "b3_support_runtime_opt"
        or parity.get("weight_scope") != "critic_only"
        or parity.get("resolved_replay_seed_coverage") != list(SEEDS)
        or parity.get("scientific_argument_differences")
        != {
            "--support_query_mode": {
                "main": ["current_policy"],
                "replay": ["replay_action"],
            }
        }
    ):
        raise ValueError("config parity report failed")
    source_parity = load_json(root / "provenance/source_parity.json")
    if (
        source_parity.get("passed") is not True
        or source_parity.get("execution_source_commit") != EXECUTION_COMMIT
        or source_parity.get("execution_source_tree_oid") != EXPERIMENT_TREE
        or source_parity.get("tracked_source_clean") is not True
        or source_parity.get("current_execution_worktree_matches_commit") is not True
        or source_parity.get("source_file_count") != 24
        or source_parity.get("failures", []) != []
    ):
        raise ValueError("source parity failed")
    checkpoint_path = root / "provenance/checkpoint_schema_audit.json"
    if not checkpoint_path.is_file():
        checkpoint_path = root / "provenance/checkpoint_contract_audit.json"
    checkpoint = load_json(checkpoint_path)
    if (
        checkpoint.get("status") != "passed_available_fields_only"
        or checkpoint.get("seed_coverage") != list(SEEDS)
        or checkpoint.get("resume_reproducibility_status") != "not_empirically_audited"
        or checkpoint.get("resume_reproducibility_claimed") is not False
    ):
        raise ValueError("checkpoint audit overclaims resume reproducibility")
    checkpoint_rows = checkpoint.get("records")
    if not isinstance(checkpoint_rows, list) or len(checkpoint_rows) != 5:
        raise ValueError("checkpoint audit seed records incomplete")
    checkpoint_index = {int(row.get("seed", -1)): row for row in checkpoint_rows if isinstance(row, dict)}
    if set(checkpoint_index) != set(SEEDS):
        raise ValueError("checkpoint audit seed coverage mismatch")
    expected_rng = {"noise_rng_state", "sample_rng_state", "weight_rng_state"}
    for seed, row in checkpoint_index.items():
        if (
            row.get("available_field_audit_passed") is not True
            or row.get("global_step") != 2_000_000
            or row.get("total_updates") != 1_995_001
            or row.get("replay_size") != 300_000
            or row.get("single_critic_ddpg") is not True
            or row.get("support_query_mode_preserved") is not True
            or row.get("available_semantic_config_match") is not True
            or row.get("config_json_derived_fields_not_in_checkpoint")
            != ["terminal_convention", "thread_hint"]
            or row.get("support_assignment_mode_serialized") is not None
            or row.get("effective_support_assignment_mode") != "support"
            or set(row.get("serialized_rng_states_present", [])) != expected_rng
            or row.get("resume_reproducibility_status") != "not_empirically_audited"
            or row.get("resume_reproducibility_claimed") is not False
        ):
            raise ValueError(f"checkpoint available-field audit mismatch seed {seed}")
    resume_contract = completion.get("resume_contract")
    if (
        not isinstance(resume_contract, dict)
        or resume_contract.get("status") != "not_empirically_audited"
        or resume_contract.get("claim_made") is not False
        or not resume_contract.get("full_resume_fields_not_serialized")
    ):
        raise ValueError("completion resume contract is not honest")

def crosscheck_json_csv_rows(
    json_rows: object,
    csv_rows: Sequence[Mapping[str, str]],
    key_fields: Sequence[str],
    label: str,
) -> None:
    if not isinstance(json_rows, list) or not json_rows or not all(
        isinstance(row, dict) for row in json_rows
    ):
        raise ValueError(f"{label}: analysis JSON rows are missing/malformed")
    json_index = {
        tuple(str(row[field]) for field in key_fields): row for row in json_rows
    }
    csv_index = {
        tuple(str(row[field]) for field in key_fields): row for row in csv_rows
    }
    if len(json_index) != len(json_rows) or len(csv_index) != len(csv_rows):
        raise ValueError(f"{label}: duplicate row key")
    if set(json_index) != set(csv_index):
        raise ValueError(f"{label}: JSON/CSV row coverage mismatch")
    csv_fields = set(csv_rows[0])
    for key in sorted(csv_index):
        json_row = json_index[key]
        csv_row = csv_index[key]
        if set(json_row) != csv_fields:
            raise ValueError(f"{label}: JSON/CSV field-set mismatch for {key}")
        for field in csv_fields:
            expected = json_row[field]
            raw = csv_row[field]
            if raw == "":
                if expected is not None:
                    raise ValueError(
                        f"{label}: JSON/CSV null mismatch {key} {field}"
                    )
            elif isinstance(expected, bool):
                if raw != str(expected):
                    raise ValueError(
                        f"{label}: JSON/CSV boolean mismatch {key} {field}"
                    )
            elif isinstance(expected, (int, float)):
                close(raw, float(expected))
            elif raw != str(expected):
                raise ValueError(
                    f"{label}: JSON/CSV value mismatch {key} {field}"
                )


def verify_readme_statistics(
    readme: str,
    aggregate: Mapping[str, Mapping[str, str]],
    paired: Mapping[str, Mapping[str, str]],
) -> None:
    for method in METHODS:
        row = aggregate[method]
        expected_row = (
            f"| {row['method']} | "
            f"{float(row['auc_mean']):.6f} ± "
            f"{float(row['auc_sample_sd']):.6f} | "
            f"{float(row['final_success_mean']):.6f} ± "
            f"{float(row['final_success_sample_sd']):.6f} |"
        )
        if expected_row not in readme:
            raise ValueError(f"README AUC/Final table mismatch for {method}")
    primary = paired["main_minus_replay"]
    expected_primary = (
        "Primary Main − Replay AUC difference: "
        f"{float(primary['mean_auc_difference']):.6f}; "
        "exhaustive paired-bootstrap 95% CI "
        f"[{float(primary['bootstrap_ci95_lower']):.6f}, "
        f"{float(primary['bootstrap_ci95_upper']):.6f}]; "
        "exact two-sided sign-flip p="
        f"{float(primary['exact_two_sided_sign_flip_pvalue']):.6f}."
    )
    if expected_primary not in readme:
        raise ValueError("README primary contrast mismatch")


def verify_summary_record(path: Path, expected: Mapping[str, object]) -> None:
    summary = load_json(path)
    for field, metric in (("eval_AUC_success", "auc"), ("final_eval_env_success", "final_success"), ("best_eval_env_success", "best_success")):
        if field not in summary:
            raise ValueError(f"{path}: missing exact summary field {field}")
        close(summary[field], float(expected[metric]))
    native_hit = summary.get("first_hit_0.90")
    if native_hit in (None, ""):
        hit = None
    else:
        try:
            hit = int(native_hit)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid first_hit_0.90") from exc
    if hit != expected["hit90"]:
        raise ValueError(f"{path}: first_hit_0.90 mismatch")


def verify_all_summaries(root: Path, calculated: Mapping[tuple[str, int], Mapping[str, object]]) -> None:
    for method in METHODS:
        for seed in SEEDS:
            verify_summary_record(root / "data" / method / f"seed_{seed}" / "summary.json", calculated[(method, seed)])


def verify_report_consistency(
    root: Path,
    calculated: Mapping[tuple[str, int], Mapping[str, object]],
) -> None:
    verify_all_summaries(root, calculated)
    result = load_json(root / "analysis_results.json")
    if (
        result.get("task") != "drawer_close"
        or
        result.get("seeds") != list(SEEDS)
        or result.get("evaluation_steps") != list(STEPS)
        or result.get("evaluation_field") != PRIMARY
        or result.get("smoothing") is not False
        or result.get("interpolation") is not False
        or result.get("forward_fill") is not False
        or result.get("extrapolation") is not False
    ):
        raise ValueError("analysis JSON convention mismatch")

    per_seed_csv = rows(root / "per_seed_results.csv")
    aggregate_csv = rows(root / "aggregate_results.csv")
    paired_csv = rows(root / "paired_comparisons.csv")
    mean_curves_csv = rows(root / "exact_grid_mean_curves.csv")
    crosscheck_json_csv_rows(
        result.get("per_seed"),
        per_seed_csv,
        ("method_key", "seed"),
        "per_seed_results",
    )
    crosscheck_json_csv_rows(
        result.get("aggregate"),
        aggregate_csv,
        ("method_key",),
        "aggregate_results",
    )
    crosscheck_json_csv_rows(
        result.get("paired_comparisons"),
        paired_csv,
        ("contrast_key",),
        "paired_comparisons",
    )
    crosscheck_json_csv_rows(
        result.get("mean_curves"),
        mean_curves_csv,
        ("step",),
        "exact_grid_mean_curves",
    )

    # verify_statistics independently recomputes every reported curve/statistic
    # from exact-grid inputs and checks each CSV. The full JSON<->CSV checks above
    # therefore establish calculated<->CSV<->analysis_results.json parity.
    json_per_seed = {
        (str(row["method_key"]), int(row["seed"])): row
        for row in result["per_seed"]
    }
    if set(json_per_seed) != set(calculated):
        raise ValueError("analysis JSON 15-run coverage mismatch")
    for key, expected in calculated.items():
        for field in ("auc", "final_success", "best_success", "best_to_final_drop"):
            close(json_per_seed[key][field], float(expected[field]))
        native_hit = json_per_seed[key]["hit90"]
        if native_hit != expected["hit90"]:
            raise ValueError(f"analysis JSON Hit90 mismatch for {key}")

    aggregate = {row["method_key"]: row for row in aggregate_csv}
    paired = {row["contrast_key"]: row for row in paired_csv}
    readme = (root / "README.md").read_text(encoding="utf-8")
    required_phrases = (
        "Replay-action query does not remove the sampled transition",
        "self-support",
        "matches the intended FPP replay-query definition",
        "Actor objective/update and the Bellman target are unchanged",
        "All five seeds",
        "CI includes zero",
        "reconstructed metadata",
        "same-seed Strict",
        "support_assignment_mode",
        "not_empirically_audited",
        "eval/env_success_mean",
    )
    if any(phrase not in readme for phrase in required_phrases):
        raise ValueError(
            "README is missing a required replay-query/provenance statement"
        )
    verify_readme_statistics(readme, aggregate, paired)

def verify(root: Path) -> dict[str, object]:
    root = root.resolve()
    manifest = load_json(root / "artifact_manifest.json")
    if manifest.get("artifact_id") != ARTIFACT_ID:
        raise ValueError("wrong artifact id")
    manifest_files(root, manifest)
    forbidden_files(root)
    calculated = verify_statistics(root)
    verify_packaged_analyzer(root, calculated)
    verify_replay(root)
    verify_provenance(root, manifest)
    verify_report_consistency(root, calculated)
    return {
        "status": "passed",
        "artifact": root.as_posix(),
        "files_verified": manifest["packaged_file_count_excluding_manifest"],
        "seed_runs_verified": len(calculated),
        "replay_completion_seeds_verified": 5,
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
