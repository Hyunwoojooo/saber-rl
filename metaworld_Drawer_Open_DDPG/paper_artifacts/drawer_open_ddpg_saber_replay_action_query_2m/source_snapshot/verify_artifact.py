#!/usr/bin/env python3
"""Independently verify the compact DrawerOpen replay-action-query artifact."""

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
METHODS = ("her", "main", "replay")
CONTRASTS = (
    ("main_minus_replay", "main", "replay"),
    ("replay_minus_her", "replay", "her"),
    ("main_minus_her", "main", "her"),
)
DEFAULT_ARTIFACT = Path(".")
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


def verify_replay(root: Path) -> None:
    completion = load_json(root / "completion_validation.json")
    if completion.get("passed") is not True or completion.get("seed_count") != 5:
        raise ValueError("replay completion gate did not pass for five seeds")
    contract = completion.get("query_contract")
    if not isinstance(contract, dict) or contract.get("support_query_mode") != "replay_action" or contract.get("support_assignment_mode") != "support" or contract.get("weight_scope") != "critic_only" or contract.get("sampled_transition_removed_from_local_pool") is not False:
        raise ValueError("completion query contract mismatch")
    records = completion.get("records")
    if not isinstance(records, list) or len(records) != 5:
        raise ValueError("replay completion records incomplete")
    indexed = {int(record["seed"]): record for record in records if isinstance(record, dict)}
    if set(indexed) != set(SEEDS):
        raise ValueError("replay completion seed coverage mismatch")
    for seed in SEEDS:
        record = indexed[seed]
        if record.get("passed") is not True or record.get("total_steps") != 2_000_000 or record.get("total_updates") != 1_995_001 or record.get("final_replay_size") != 300_000 or record.get("evaluation_points") != 200 or record.get("evaluation_episodes_per_point") != 20 or record.get("single_critic_ddpg") is not True:
            raise ValueError(f"seed {seed}: completion contract mismatch")
        if record.get("support_query_mode") != "replay_action" or record.get("support_assignment_mode") != "support" or record.get("weight_scope") != "critic_only":
            raise ValueError(f"seed {seed}: mode/scope mismatch")
        if record.get("config_parity", {}).get("passed") is not True or record.get("checkpoint_resume_audit", {}).get("passed") is not True:
            raise ValueError(f"seed {seed}: parity or resume gate failed")
        metrics = record.get("metrics", {})
        if metrics.get("all_required_fields_finite") is not True or int(metrics.get("post_warmup_update_rows_checked", 0)) <= 0:
            raise ValueError(f"seed {seed}: finite metric gate failed")
        close(metrics.get("replay_action_bin_match_min"), 1.0, 1e-12)
        close(metrics.get("replay_action_bin_match_max_abs_error"), 0.0, 1e-12)
        if float(metrics.get("normalized_weight_mean_max_abs_error", 1.0)) > 1e-6:
            raise ValueError(f"seed {seed}: normalized weight invariant failed")
        config = load_json(root / "data" / "replay" / f"seed_{seed}" / "config.json")
        if config.get("support_query_mode") != "replay_action" or config.get("support_assignment_mode") != "support" or config.get("weight_scope") != "critic_only" or config.get("total_steps") != 2_000_000 or config.get("eval_freq") != 10_000 or config.get("eval_episodes") != 20 or config.get("save_step_checkpoints") is not False:
            raise ValueError(f"seed {seed}: packaged replay config mismatch")
        diagnostic = load_json(root / "data" / "replay" / f"seed_{seed}" / "query_diagnostic_summary.json")
        if diagnostic.get("support_query_mode") != "replay_action" or diagnostic.get("support_assignment_mode") != "support" or diagnostic.get("sampled_transition_removed_from_local_pool") is not False or diagnostic.get("self_support_possible") is not True:
            raise ValueError(f"seed {seed}: packaged query definition mismatch")
        invariants = diagnostic.get("invariants", {})
        if not all(invariants.get(key) is True for key in ("all_finite", "replay_action_bin_match_all_one", "normalized_weight_mean_all_one")):
            raise ValueError(f"seed {seed}: query diagnostic invariant failed")
        replay_match = diagnostic.get("fields", {}).get("query/query_bin_matches_replay_action_rate", {})
        close(replay_match.get("min"), 1.0, 1e-12)
        close(replay_match.get("max"), 1.0, 1e-12)
    for method in ("her", "main"):
        for seed in SEEDS:
            config = load_json(root / "data" / method / f"seed_{seed}" / "config.json")
            env = load_json(root / "data" / method / f"seed_{seed}" / "env_snapshot.json")
            if config.get("record_type") != "reconstructed_metadata_not_original_run_config" or config.get("not_original_run_config") is not True or config.get("method_specific_training_fields_not_reconstructed") is not True:
                raise ValueError(f"{method} seed {seed}: comparator config is mislabelled")
            if env.get("record_type") != "same_seed_environment_proxy_not_original_snapshot" or env.get("not_original_run_env_snapshot") is not True:
                raise ValueError(f"{method} seed {seed}: environment proxy is mislabelled")

def verify_comparator_provenance(
    root: Path,
    manifest: Mapping[str, object],
    input_manifest: Mapping[str, object],
) -> None:
    packaging_commit = str(
        manifest.get("comparator_packaging_source_commit", "")
    )
    if len(packaging_commit) != 40:
        raise ValueError("invalid comparator packaging source commit")
    comparator = input_manifest.get("comparator")
    if not isinstance(comparator, dict):
        raise ValueError("missing comparator provenance")
    if (
        comparator.get("artifact_git_commit")
        != "93ae5c03633726f8e7783a157cac8516348f088f"
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
    if (
        freeze.get("schema_version") != 2
        or freeze.get("head_binding_verified") is not True
        or freeze.get("recorded_commit_was_current_head_at_write") is not True
    ):
        raise ValueError("frozen source lacks schema-v2 HEAD binding")
    rows = freeze.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty frozen source")
    expected_lines: list[str] = []
    seen_paths: set[str] = set()
    seen_repo_paths: set[str] = set()
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
            or not repo_path.endswith("/replay_action_query_2m/" + relative)
            or row.get("role") != "experiment_source"
            or row.get("head_bound") is not True
            or digest != row.get("head_sha256")
            or len(digest) != 64
            or len(oid) not in (40, 64)
            or any(char not in "0123456789abcdef" for char in (digest + oid).lower())
        ):
            raise ValueError(f"invalid HEAD-bound source metadata: {relative}")
        seen_paths.add(relative)
        seen_repo_paths.add(repo_path)
        source = root / "source_snapshot" / relative
        if (
            not source.is_file()
            or source.stat().st_size != int(row.get("size", -1))
            or sha256(source) != digest
        ):
            raise ValueError(f"frozen source mismatch: {relative}")
        expected_lines.append(f"{digest}  source_snapshot/{relative}")
    entrypoints = freeze.get("entrypoints")
    if not isinstance(entrypoints, list) or len(entrypoints) != 1:
        raise ValueError("freeze must bind exactly one root launcher")
    launcher = entrypoints[0]
    expected_repo_path = (
        "metaworld_Drawer_Open_DDPG/"
        "DDPG-HER-B3-Replay-Action-Query-2M.sh"
    )
    if not isinstance(launcher, dict):
        raise ValueError("malformed root launcher metadata")
    digest = str(launcher.get("sha256", ""))
    oid = str(launcher.get("git_blob_oid", ""))
    if (
        launcher.get("path") != expected_repo_path
        or launcher.get("repo_path") != expected_repo_path
        or launcher.get("role") != "root_launcher"
        or launcher.get("head_bound") is not True
        or launcher.get("head_sha256") != digest
        or len(digest) != 64
        or len(oid) not in (40, 64)
        or any(char not in "0123456789abcdef" for char in (digest + oid).lower())
    ):
        raise ValueError("invalid HEAD-bound root launcher metadata")
    launcher_copy = root / "source_snapshot/root_launcher" / Path(expected_repo_path).name
    if (
        not launcher_copy.is_file()
        or launcher_copy.stat().st_size != int(launcher.get("size", -1))
        or sha256(launcher_copy) != digest
    ):
        raise ValueError("frozen root launcher copy mismatch")
    expected_lines.append(
        f"{digest}  source_snapshot/root_launcher/{launcher_copy.name}"
    )
    actual_lines = (root / "frozen_source.sha256").read_text(
        encoding="utf-8"
    ).splitlines()
    if actual_lines != expected_lines:
        raise ValueError("frozen_source.sha256 mismatch")


def verify_provenance(root: Path, manifest: Mapping[str, object]) -> None:
    commit = str(manifest.get("source_commit", ""))
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit.lower()):
        raise ValueError("invalid replay source commit")
    if manifest.get("comparator_artifact_commit") != "93ae5c03633726f8e7783a157cac8516348f088f":
        raise ValueError("wrong comparator artifact commit")
    if "comparator_source_commit" in manifest:
        raise ValueError("ambiguous comparator_source_commit field is forbidden")
    if manifest.get("method_seed_coverage") != {key: list(SEEDS) for key in METHODS}:
        raise ValueError("manifest 15-run coverage mismatch")
    evaluation = manifest.get("evaluation", {})
    if evaluation.get("steps") != list(STEPS) or evaluation.get("points_per_seed") != 200 or evaluation.get("episodes_per_point") != 20 or any(evaluation.get(key) is not False for key in ("smoothing", "interpolation", "forward_fill", "extrapolation")):
        raise ValueError("manifest evaluation convention mismatch")
    if manifest.get("bootstrap", {}).get("resamples") != 3125 or manifest.get("exact_sign_flip", {}).get("assignments") != 32:
        raise ValueError("manifest paired-statistic enumeration mismatch")
    input_manifest = load_json(root / "input_manifest.json")
    completion = load_json(root / "completion_validation.json")
    freeze = load_json(root / "provenance/freeze_manifest.json")
    if input_manifest.get("source_commit") != commit or completion.get("source_commit") != commit or freeze.get("source_commit") != commit:
        raise ValueError("replay source commit disagreement")
    verify_comparator_provenance(root, manifest, input_manifest)
    inputs = input_manifest.get("inputs")
    if not isinstance(inputs, list) or int(input_manifest.get("input_file_count", -1)) != len(inputs):
        raise ValueError("input manifest count mismatch")
    roles = {str(row.get("role", "")) for row in inputs if isinstance(row, dict)}
    for method in METHODS:
        for seed in SEEDS:
            for suffix in ("metrics", "summary"):
                if f"{method}.seed_{seed}.{suffix}" not in roles:
                    raise ValueError(f"missing input provenance: {method} seed {seed} {suffix}")
    verify_frozen_snapshot(root, freeze)
    parity = load_json(root / "provenance/config_parity.json")
    if parity.get("passed") is not True or parity.get("nonallowed_differences") != [] or set(parity.get("allowed_paths", [])) != {"support_query_mode", "run_name", "output_root"}:
        raise ValueError("config parity report failed")
    seed_rows = parity.get("seeds")
    if not isinstance(seed_rows, list) or [int(row.get("seed", -1)) for row in seed_rows] != list(SEEDS):
        raise ValueError("config parity seed coverage mismatch")
    for row in seed_rows:
        if row.get("passed") is not True or row.get("nonallowed_differences") != []:
            raise ValueError("seed config parity failed")
        for difference in row.get("differences", []):
            if difference.get("allowed") is not True or difference.get("path") not in {"support_query_mode", "run_name", "output_root"}:
                raise ValueError("non-allowlisted config difference")
    source_parity = load_json(root / "provenance/source_parity.json")
    if source_parity.get("passed") is not True or source_parity.get("failures", []) != []:
        raise ValueError("source parity failed")
    resume = load_json(root / "provenance/checkpoint_resume_audit.json")
    if resume.get("passed") is not True or resume.get("seed_coverage") != list(SEEDS):
        raise ValueError("checkpoint/resume provenance failed")
    for row in resume.get("seeds", []):
        if row.get("passed") is not True or any(row.get(key) is not True for key in ("query_mode_preserved", "replay_query_is_deterministic", "learning_result_reproduced", "rng_state_restored")):
            raise ValueError("checkpoint/resume seed invariant failed")

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
        result.get("seeds") != list(SEEDS)
        or result.get("evaluation_steps") != list(STEPS)
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
        "does not remove the sampled transition",
        "self-support",
        "same intended replay-query definition used by the FPP",
        "Actor objective/update and the Bellman target are unchanged",
        "All five seeds",
        "CI includes zero",
        "reconstructed metadata",
        "same-seed Strict proxies",
    )
    if any(phrase not in readme for phrase in required_phrases):
        raise ValueError(
            "README is missing a required replay-query/provenance statement"
        )
    verify_readme_statistics(readme, aggregate, paired)

def verify(root: Path) -> dict[str, object]:
    root = root.resolve()
    manifest = load_json(root / "artifact_manifest.json")
    if manifest.get("artifact_id") != "drawer_open_ddpg_saber_replay_action_query_2m":
        raise ValueError("wrong artifact id")
    manifest_files(root, manifest)
    forbidden_files(root)
    calculated = verify_statistics(root)
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
