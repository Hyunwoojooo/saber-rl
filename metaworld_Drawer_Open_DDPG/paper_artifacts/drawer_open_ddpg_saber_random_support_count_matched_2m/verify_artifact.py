#!/usr/bin/env python3
"""Standalone verifier for the DrawerOpen DDPG Random-Support artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping

# The verifier must not mutate the artifact it is about to compare with the
# strict manifest.  Set this before importing the artifact-local analyzer so a
# plain ``python3 verify_artifact.py`` invocation cannot create __pycache__.
sys.dont_write_bytecode = True

import analyze_results as analysis


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


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: empty CSV")
    return rows


def csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ())


def close(actual: object, expected: float, tolerance: float = TOL) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"not numeric: {actual!r}") from exc
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=0.0, abs_tol=tolerance
    ):
        raise ValueError(f"numeric mismatch: {value!r} != {expected!r}")


def verify_manifest(root: Path) -> int:
    manifest = load_json(root / "artifact_manifest.json")
    rows = manifest.get("packaged_files")
    if not isinstance(rows, list):
        raise ValueError("artifact manifest missing packaged_files")
    listed: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed manifest row")
        relative = str(row.get("path", ""))
        parts = Path(relative).parts
        if (
            not relative
            or relative in listed
            or Path(relative).is_absolute()
            or ".." in parts
        ):
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
    digest = hashlib.sha256()
    for relative in sorted(listed):
        row = listed[relative]
        path = actual[relative]
        if (
            path.stat().st_size != int(row["size_bytes"])
            or sha256(path) != row["sha256"]
        ):
            raise ValueError(f"artifact hash/size mismatch: {relative}")
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode())
        digest.update(b"\n")
    if digest.hexdigest() != manifest.get(
        "package_content_sha256_excluding_manifest"
    ):
        raise ValueError("artifact content digest mismatch")
    if int(manifest.get("packaged_file_count_excluding_manifest", -1)) != len(
        actual
    ):
        raise ValueError("artifact file count mismatch")
    if manifest.get("bootstrap_resamples") != 3125:
        raise ValueError("manifest bootstrap count mismatch")
    if manifest.get("sign_flip_assignments") != 32:
        raise ValueError("manifest sign-flip count mismatch")
    if manifest.get("manifest_self_hash_policy") != (
        "artifact_manifest.json is excluded from packaged_files to avoid a "
        "recursive self-hash"
    ):
        raise ValueError("manifest self-hash exclusion policy missing")
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("manifest evaluation contract missing")
    if (
        evaluation.get("field") != analysis.PRIMARY_FIELD
        or evaluation.get("steps") != list(analysis.EVAL_STEPS)
        or evaluation.get("points_per_seed") != 200
        or any(
            evaluation.get(flag) is not False
            for flag in ("smoothing", "interpolation", "forward_fill", "extrapolation")
        )
    ):
        raise ValueError("manifest exact-grid contract mismatch")
    return len(actual)


def verify_exclusions(root: Path) -> None:
    forbidden: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if (
            path.name == "metrics.csv"
            or path.suffix.lower() in {".pt", ".pth", ".jsonl", ".log"}
            or any(
                part.lower() in {"checkpoints", "logs", "smoke", "pilot"}
                for part in relative.parts
            )
        ):
            forbidden.append(relative.as_posix())
    if forbidden:
        raise ValueError(f"forbidden payloads: {forbidden}")


def verify_statistics(root: Path) -> dict[str, object]:
    curves: dict[tuple[str, int], list[float]] = {}
    for method in analysis.METHODS:
        for seed in analysis.SEEDS:
            path = (
                root
                / "data"
                / method
                / f"seed_{seed}"
                / "metrics_selected_10k.csv"
            )
            curves[(method, seed)] = analysis.load_exact_curve(path)
    expected = analysis.analyze_curves(curves)

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
    if csv_header(per_seed_path) != expected_per_seed_header:
        raise ValueError("per-seed table schema is not aligned with strict artifacts")
    per_seed = {
        (row["method_key"], int(row["seed"])): row
        for row in csv_rows(per_seed_path)
    }
    if len(per_seed) != 15:
        raise ValueError("per-seed table must contain 15 method-seed rows")
    for row in expected["per_seed"]:
        key = (str(row["method_key"]), int(row["seed"]))
        packaged = per_seed.get(key)
        if packaged is None:
            raise ValueError(f"missing per-seed row {key}")
        if (
            packaged["method"] != row["method"]
            or packaged["run_dir"] != f"data/{key[0]}/seed_{key[1]}"
            or packaged["metrics_path"]
            != f"data/{key[0]}/seed_{key[1]}/metrics_selected_10k.csv"
        ):
            raise ValueError(f"per-seed metadata mismatch for {key}")
        for field in (
            "evaluation_points",
            "auc",
            "final_success",
            "best_success",
            "best_to_final_drop",
        ):
            close(packaged[field], float(row[field]))
        native_hit = None if packaged["hit90"] == "" else int(packaged["hit90"])
        if native_hit != row["hit90"]:
            raise ValueError(f"Hit90 mismatch for {key}")
        summary = load_json(
            root / "data" / key[0] / f"seed_{key[1]}" / "summary.json"
        )
        if summary.get("run_dir") != f"data/{key[0]}/seed_{key[1]}":
            raise ValueError(f"non-portable summary run_dir for {key}")
        summary_values = {
            "summary_auc": summary.get("eval_AUC_success"),
            "summary_auc_difference": 0.0,
            "summary_final_difference": 0.0,
            "summary_best_difference": 0.0,
        }
        for field, value in summary_values.items():
            close(packaged[field], float(value))
        close(
            summary.get("final_eval_env_success"),
            float(row["final_success"]),
        )
        close(
            summary.get("best_eval_env_success"),
            float(row["best_success"]),
        )

    aggregate = {
        row["method_key"]: row
        for row in csv_rows(root / "aggregate_results.csv")
    }
    for row in expected["aggregate"]:
        packaged = aggregate.get(str(row["method_key"]))
        if packaged is None:
            raise ValueError(f"missing aggregate row {row['method_key']}")
        for field, value in row.items():
            if field in {"method_key", "method"}:
                continue
            if value is None:
                if packaged[field] != "":
                    raise ValueError(f"expected blank aggregate field {field}")
            else:
                close(packaged[field], float(value))

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
        *[f"seed_{seed}_auc_difference" for seed in analysis.SEEDS],
    ]
    if csv_header(paired_path) != expected_paired_header:
        raise ValueError("paired table schema is not aligned with strict artifacts")
    paired = {
        row["contrast_key"]: row
        for row in csv_rows(paired_path)
    }
    for row in expected["paired_comparisons"]:
        packaged = paired.get(str(row["contrast_key"]))
        if packaged is None:
            raise ValueError(f"missing paired row {row['contrast_key']}")
        for field, value in row.items():
            if field in {"contrast_key", "contrast", "left_method", "right_method"}:
                if packaged[field] != value:
                    raise ValueError(
                        f"paired metadata mismatch for {row['contrast_key']}: {field}"
                    )
                continue
            close(packaged[field], float(value))
        if int(packaged["bootstrap_resample_count"]) != 3125:
            raise ValueError("paired bootstrap is not exhaustive 5^5")
        if int(packaged["sign_flip_assignment_count"]) != 32:
            raise ValueError("paired sign-flip test is not exhaustive 2^5")

    mean_path = root / "exact_grid_mean_curves.csv"
    expected_mean_header = ["step"]
    for method in analysis.METHODS:
        expected_mean_header.extend(
            (
                f"{method}_mean_success",
                f"{method}_sample_sd_success",
                f"{method}_method",
            )
        )
    if csv_header(mean_path) != expected_mean_header:
        raise ValueError("mean-curve schema is not aligned with strict artifacts")
    mean_rows = csv_rows(mean_path)
    if len(mean_rows) != len(expected["mean_curves"]):
        raise ValueError("mean-curve row count mismatch")
    for packaged, row in zip(mean_rows, expected["mean_curves"]):
        for field, value in row.items():
            if field.endswith("_method"):
                if packaged[field] != value:
                    raise ValueError(f"mean-curve label mismatch for {field}")
            else:
                close(packaged[field], float(value))

    analysis_json = load_json(root / "analysis_results.json")
    if (
        analysis_json.get("seeds") != list(analysis.SEEDS)
        or analysis_json.get("evaluation_field") != analysis.PRIMARY_FIELD
        or analysis_json.get("evaluation_steps") != list(analysis.EVAL_STEPS)
        or any(
            analysis_json.get(flag) is not False
            for flag in ("smoothing", "interpolation", "forward_fill", "extrapolation")
        )
    ):
        raise ValueError("analysis JSON exact-grid contract mismatch")
    if (
        analysis_json.get("aggregate") != expected["aggregate"]
        or analysis_json.get("paired_comparisons")
        != expected["paired_comparisons"]
        or analysis_json.get("mean_curves") != expected["mean_curves"]
    ):
        raise ValueError("analysis JSON results do not match regenerated results")
    return {
        "curves": curves,
        "expected": expected,
    }


def verify_completion_and_summaries(
    root: Path, calculated: Mapping[str, object]
) -> None:
    completion = load_json(root / "completion_validation.json")
    records = completion.get("records")
    if (
        completion.get("passed") is not True
        or completion.get("seed_count") != 5
        or not isinstance(records, list)
        or len(records) != 5
    ):
        raise ValueError("Random-Support completion gate is incomplete")
    expected_rows = calculated["expected"]["per_seed"]  # type: ignore[index]
    expected = {
        (str(row["method_key"]), int(row["seed"])): row for row in expected_rows
    }
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("malformed completion record")
        seed = int(record["seed"])
        if (
            record.get("passed") is not True
            or record.get("total_steps") != 2_000_000
            or record.get("total_updates") != 1_995_001
            or record.get("final_replay_size") != 300_000
            or record.get("evaluation_points") != 200
            or record.get("evaluation_first_step") != 10_000
            or record.get("evaluation_last_step") != 2_000_000
        ):
            raise ValueError(f"seed {seed}: completion contract mismatch")
        for field in (
            "summary_auc_difference",
            "summary_final_difference",
            "summary_best_difference",
        ):
            close(record[field], 0.0, 1e-12)
        per_seed_record = load_json(
            root
            / "data"
            / "random"
            / f"seed_{seed}"
            / "completion_validation.json"
        )
        if per_seed_record != record:
            raise ValueError(f"seed {seed}: duplicated completion record mismatch")
        summary = load_json(
            root / "data" / "random" / f"seed_{seed}" / "summary.json"
        )
        row = expected[("random", seed)]
        close(summary["eval_AUC_success"], float(row["auc"]))
        close(summary["final_eval_env_success"], float(row["final_success"]))
        close(summary["best_eval_env_success"], float(row["best_success"]))


def verify_count_invariant(root: Path) -> int:
    audit = load_json(root / "positive_count_invariant_audit.json")
    records = audit.get("records")
    if (
        audit.get("passed") is not True
        or audit.get("total_mismatch_count") != 0
        or not isinstance(records, list)
        or len(records) != 5
    ):
        raise ValueError("positive-count invariant audit failed or is incomplete")
    total = 0
    indexed = {
        int(row["seed"]): row for row in records if isinstance(row, dict)
    }
    if set(indexed) != set(analysis.SEEDS):
        raise ValueError("positive-count audit seed coverage mismatch")
    global_max = 0.0
    for seed in analysis.SEEDS:
        rows = csv_rows(
            root
            / "data"
            / "random"
            / f"seed_{seed}"
            / "positive_count_invariant_trace.csv"
        )
        mismatch = 0
        maximum = 0.0
        for row in rows:
            reference = float(row["reference_high_weight_rate"])
            assigned = float(row["assigned_high_weight_rate"])
            if not all(math.isfinite(value) for value in (reference, assigned)):
                raise ValueError(f"seed {seed}: non-finite invariant trace")
            difference = abs(reference - assigned)
            maximum = max(maximum, difference)
            if difference > 1e-12:
                mismatch += 1
        record = indexed[seed]
        if (
            record.get("passed") is not True
            or int(record["rows_checked"]) != len(rows)
            or int(record["mismatch_count"]) != mismatch
            or mismatch != 0
        ):
            raise ValueError(f"seed {seed}: invariant summary mismatch")
        close(record["max_abs_difference"], maximum, 1e-15)
        total += len(rows)
        global_max = max(global_max, maximum)
    if total != int(audit["total_rows_checked"]):
        raise ValueError("global invariant row-count mismatch")
    close(audit["max_abs_difference"], global_max, 1e-15)
    return total


def verify_provenance(root: Path) -> None:
    source = load_json(root / "source_provenance.json")
    if (
        source.get("random_result_commit")
        != "86912701b2ca0e26586c43748f2a984c0f301960"
        or source.get("random_original_config_available") is not False
        or source.get("random_original_environment_snapshot_available") is not False
    ):
        raise ValueError("source/config provenance contract mismatch")
    consistency = source.get("implementation_source_tree_consistency")
    if (
        not isinstance(consistency, dict)
        or consistency.get("parent_is_ancestor_of_result_commit") is not True
        or consistency.get("tree_diff_is_empty") is not True
        or "not proof" not in str(consistency.get("interpretation", ""))
    ):
        raise ValueError("source-tree consistency caveat missing")
    frozen = source.get("source_files")
    if not isinstance(frozen, list) or not frozen:
        raise ValueError("frozen source inventory missing")
    expected_lines: list[str] = []
    for row in frozen:
        if not isinstance(row, dict):
            raise ValueError("malformed frozen source row")
        relative = str(row["path"])
        path = root / "source_snapshot" / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or sha256(path) != row["sha256"]
        ):
            raise ValueError(f"frozen source mismatch: {relative}")
        expected_lines.append(f"{row['sha256']}  source_snapshot/{relative}")
    actual_lines = (root / "frozen_source.sha256").read_text(
        encoding="utf-8"
    ).splitlines()
    if actual_lines != expected_lines:
        raise ValueError("frozen_source.sha256 mismatch")

    for method in ("her", "main"):
        for seed in analysis.SEEDS:
            config = load_json(
                root / "data" / method / f"seed_{seed}" / "config.json"
            )
            env = load_json(
                root / "data" / method / f"seed_{seed}" / "env_snapshot.json"
            )
            if (
                config.get("not_original_run_config") is not True
                or env.get("not_original_run_env_snapshot") is not True
            ):
                raise ValueError(
                    f"{method} seed {seed}: reconstructed/proxy label missing"
                )
    for seed in analysis.SEEDS:
        config = load_json(
            root / "data" / "random" / f"seed_{seed}" / "config.json"
        )
        env = load_json(
            root / "data" / "random" / f"seed_{seed}" / "env_snapshot.json"
        )
        if (
            config.get("record_type")
            != "reconstructed_metadata_not_original_run_config"
            or config.get("not_original_run_config") is not True
            or config.get("support_query_mode") != "current_policy"
            or "not an originally serialized" not in str(
                config.get("support_query_mode_provenance", "")
            )
            or config.get("support_assignment_mode") != "random_count_matched"
            or config.get("weighting_mode") != "batch_norm"
            or config.get("weighting_mode_provenance")
            != "explicit argument in the frozen launcher"
            or config.get("weight_scope") != "critic_only"
            or config.get("total_steps") != 2_000_000
            or config.get("eval_freq") != 10_000
            or config.get("eval_episodes") != 20
        ):
            raise ValueError(f"random seed {seed}: reconstructed config mismatch")
        if (
            env.get("record_type")
            != "same_seed_environment_proxy_not_original_snapshot"
            or env.get("not_original_run_env_snapshot") is not True
        ):
            raise ValueError(f"random seed {seed}: environment proxy label missing")

    input_manifest = load_json(root / "input_manifest.json")
    inputs = input_manifest.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 30:
        raise ValueError("input manifest must contain 30 metrics/summary records")
    for row in inputs:
        if not isinstance(row, dict):
            raise ValueError("malformed input manifest row")
        source_path = str(row.get("source_path", ""))
        if Path(source_path).is_absolute() or "/Users/" in source_path:
            raise ValueError(f"non-portable source path: {source_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.artifact_dir.resolve()
    files = verify_manifest(root)
    verify_exclusions(root)
    calculated = verify_statistics(root)
    verify_completion_and_summaries(root, calculated)
    invariant_rows = verify_count_invariant(root)
    verify_provenance(root)
    print(
        json.dumps(
            {
                "artifact": str(root),
                "bootstrap_resamples": 3125,
                "evaluation_points_per_seed": 200,
                "files_verified": files,
                "positive_count_rows_verified": invariant_rows,
                "seed_runs_verified": 15,
                "sign_flip_assignments": 32,
                "status": "passed",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
