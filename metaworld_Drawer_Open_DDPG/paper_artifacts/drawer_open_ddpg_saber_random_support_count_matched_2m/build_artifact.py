#!/usr/bin/env python3
"""Build the compact DrawerOpen DDPG Random-Support paper artifact.

The raw Random-Support runs contain metrics and summaries but no original
config/environment snapshots.  This builder therefore packages reconstructed
configuration metadata and explicitly labelled same-seed environment proxies.
It never represents either record as an original run file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import analyze_results as analysis


ARTIFACT_ID = "drawer_open_ddpg_saber_random_support_count_matched_2m"
RANDOM_RESULT_COMMIT = "86912701b2ca0e26586c43748f2a984c0f301960"
COMPARATOR_ARTIFACT_COMMIT = "1a184f69588c0d02ece942bc106845a4516e81cb"
RANDOM_EXPERIMENT = "DDPG-HER-B3-Random-Support-2M"
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_RAW = (
    REPO_ROOT
    / "metaworld_Drawer_Open_DDPG"
    / "results_2m"
    / RANDOM_EXPERIMENT
)
DEFAULT_COMPARATOR = (
    REPO_ROOT
    / "metaworld_Drawer_Open_DDPG"
    / "paper_artifacts"
    / "drawer_open_ddpg_saber_replay_action_query_2m"
)
SOURCE_FILES = (
    "metaworld_Drawer_Open_DDPG/DDPG-HER-B3-Random-Support-2M.sh",
    "metaworld_Drawer_Open_DDPG/code/src/train_drawer_open_lite_v0.py",
    "metaworld_Drawer_Open_DDPG/code/src/metacog_lite_v0/config.py",
    "metaworld_Drawer_Open_DDPG/code/src/metacog_lite_v0/replay.py",
    "metaworld_Drawer_Open_DDPG/code/src/metacog_lite_v0/task_drawer_open.py",
    "metaworld_Drawer_Open_DDPG/code/src/metacog_lite_v0/trainer.py",
    "metaworld_Drawer_Open_DDPG/code/src/metacog_lite_v0/weights.py",
)
GENERATED_TOP_LEVEL = (
    "__pycache__",
    "data",
    "source_snapshot",
    "aggregate_results.csv",
    "analysis_results.json",
    "artifact_manifest.json",
    "completion_validation.json",
    "exact_grid_mean_curves.csv",
    "frozen_source.sha256",
    "input_manifest.json",
    "paired_comparisons.csv",
    "per_seed_results.csv",
    "positive_count_invariant_audit.json",
    "source_provenance.json",
)
AUTHORED_FILES = {
    "README.md",
    "analyze_results.py",
    "build_artifact.py",
    "verify_artifact.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def clean_json(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def portable_summary(
    summary: Mapping[str, object], method: str, seed: int
) -> dict[str, object]:
    """Preserve summary evidence while replacing host-specific run paths."""

    payload = clean_json(dict(summary))
    if not isinstance(payload, dict):
        raise TypeError("cleaned summary must remain a JSON object")
    payload["run_dir"] = f"data/{method}/seed_{seed}"
    return payload


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"{path}: refusing to write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(materialized[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def run_dir(raw_root: Path, seed: int) -> Path:
    path = (
        raw_root
        / f"drawer_open_{RANDOM_EXPERIMENT}_s{seed}_2m_cuda"
    )
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def copy_exact_curve(source: Path, destination: Path) -> list[float]:
    values = analysis.load_exact_curve(source)
    write_csv(
        destination,
        (
            {"step": step, analysis.PRIMARY_FIELD: value}
            for step, value in zip(analysis.EVAL_STEPS, values)
        ),
    )
    return values


def count_invariant_trace(
    source: Path, destination: Path, seed: int
) -> dict[str, object]:
    fields = (
        "step",
        "episode",
        "reference_high_weight_rate",
        "assigned_high_weight_rate",
    )
    selected: list[dict[str, object]] = []
    mismatch_count = 0
    max_abs_difference = 0.0
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(fields).issubset(reader.fieldnames or []):
            raise ValueError(f"{source}: missing count-invariant fields")
        for row in reader:
            reference = analysis.finite_float(row.get(fields[2]))
            assigned = analysis.finite_float(row.get(fields[3]))
            if reference is None and assigned is None:
                continue
            if reference is None or assigned is None:
                raise ValueError(f"{source}: one-sided non-finite count field")
            step = analysis.finite_float(row.get("step"))
            episode = analysis.finite_float(row.get("episode"))
            if step is None or episode is None:
                raise ValueError(f"{source}: invalid trace index")
            difference = abs(reference - assigned)
            max_abs_difference = max(max_abs_difference, difference)
            if difference > 1e-12:
                mismatch_count += 1
            selected.append(
                {
                    "step": int(step),
                    "episode": int(episode),
                    "reference_high_weight_rate": reference,
                    "assigned_high_weight_rate": assigned,
                }
            )
    if not selected:
        raise ValueError(f"{source}: no finite invariant rows")
    if mismatch_count:
        raise ValueError(
            f"{source}: Random-Support count invariant failed {mismatch_count} times"
        )
    write_csv(destination, selected)
    return {
        "seed": seed,
        "rows_checked": len(selected),
        "mismatch_count": mismatch_count,
        "max_abs_difference": max_abs_difference,
        "passed": True,
    }


def random_config(seed: int, launcher_sha: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "reconstructed_metadata_not_original_run_config",
        "not_original_run_config": True,
        "reason": (
            "The committed Random-Support result contains metrics.csv and "
            "summary.json only. Values below were reconstructed from the exact "
            "training launcher at the recorded result commit."
        ),
        "source_commit": RANDOM_RESULT_COMMIT,
        "launcher_path": SOURCE_FILES[0],
        "launcher_sha256": launcher_sha,
        "method": "b3_support_random_runtime_opt",
        "weight_scope": "critic_only",
        "support_query_mode": "current_policy",
        "support_query_mode_provenance": (
            "effective value derived from the frozen method source/default; "
            "not an originally serialized run-config field"
        ),
        "support_assignment_mode": "random_count_matched",
        "weighting_mode": "batch_norm",
        "weighting_mode_provenance": "explicit argument in the frozen launcher",
        "task_name": "drawer_open",
        "reward_type": "sparse",
        "metaworld_env_name": "drawer-open-v3",
        "metaworld_reward_function_version": "v2",
        "max_episode_steps": 200,
        "total_steps": 2_000_000,
        "start_steps": 10_000,
        "warmup_steps": 5_000,
        "updates_per_step": 1,
        "batch_size": 256,
        "buffer_capacity": 300_000,
        "her_ratio": 0.8,
        "gamma": 0.98,
        "tau": 0.005,
        "actor_lr": 0.0003,
        "critic_lr": 0.0003,
        "hidden_dim": 256,
        "noise_std": 0.2,
        "eval_freq": 10_000,
        "eval_episodes": 20,
        "lambda_support": 1.0,
        "local_search_max": 512,
        "M_B": 16,
        "M_Q_min": 4,
        "N_local_min": 4,
        "support_scan_mode": "deterministic_tail_v1",
        "seeds": {
            "env_seed": seed,
            "torch_seed": seed,
            "numpy_seed": seed,
            "action_noise_seed": seed,
            "weight_seed": 20_000 + seed,
            "eval_seed": 10_000 + seed,
        },
    }


def random_env_proxy(seed: int, comparator: Path) -> dict[str, object]:
    source = comparator / "data" / "main" / f"seed_{seed}" / "env_snapshot.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "schema_version": 1,
        "record_type": "same_seed_environment_proxy_not_original_snapshot",
        "not_original_run_env_snapshot": True,
        "reason": (
            "The Random-Support run did not preserve an environment snapshot. "
            "This nested record is the already-labelled same-seed proxy from "
            "the verified HER/Main comparator artifact."
        ),
        "proxy_artifact_commit": COMPARATOR_ARTIFACT_COMMIT,
        "proxy_source_sha256": sha256(source),
        "proxy_snapshot": read_json(source),
    }


def clean_generated(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    unexpected = {
        path.name for path in output.iterdir()
    } - AUTHORED_FILES - set(GENERATED_TOP_LEVEL)
    if unexpected:
        raise ValueError(f"unexpected files in artifact directory: {sorted(unexpected)}")
    for name in GENERATED_TOP_LEVEL:
        path = output / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def git_file(repo: Path, commit: str, relative: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{commit}:{relative}"]
    )


def package_sources(repo: Path, output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    digest_lines: list[str] = []
    for relative in SOURCE_FILES:
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(git_file(repo, RANDOM_RESULT_COMMIT, relative))
        row = {
            "path": relative,
            "size_bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "source_commit": RANDOM_RESULT_COMMIT,
        }
        rows.append(row)
        digest_lines.append(
            f"{row['sha256']}  source_snapshot/{relative}"
        )
    (output / "frozen_source.sha256").write_text(
        "\n".join(digest_lines) + "\n", encoding="utf-8"
    )
    return rows


def portable_source_path(path: Path) -> str:
    parts = path.resolve().parts
    if "metaworld_Drawer_Open_DDPG" in parts:
        index = parts.index("metaworld_Drawer_Open_DDPG")
        return Path(*parts[index:]).as_posix()
    if "replay_artifact" in parts:
        index = parts.index("replay_artifact")
        return Path("<external-verified-comparator>", *parts[index + 1 :]).as_posix()
    return f"<external>/{path.name}"


def input_record(path: Path, role: str, provenance: str) -> dict[str, object]:
    return {
        "role": role,
        "source_path": portable_source_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "provenance": provenance,
    }


def packaged_files(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]


def content_digest(rows: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["path"]).encode())
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def build(repo: Path, raw: Path, comparator: Path, output: Path) -> None:
    repo = repo.resolve()
    raw = raw.resolve()
    comparator = comparator.resolve()
    output = output.resolve()
    clean_generated(output)

    comparator_manifest = read_json(comparator / "artifact_manifest.json")
    if comparator_manifest.get("artifact_id") != (
        "drawer_open_ddpg_saber_replay_action_query_2m"
    ):
        raise ValueError("unexpected comparator artifact")

    source_rows = package_sources(repo, output)
    launcher_sha = str(source_rows[0]["sha256"])
    curves: dict[tuple[str, int], list[float]] = {}
    inputs: list[dict[str, object]] = []
    completion_rows: list[dict[str, object]] = []
    invariant_rows: list[dict[str, object]] = []

    for method in ("her", "main"):
        for seed in analysis.SEEDS:
            source_dir = comparator / "data" / method / f"seed_{seed}"
            destination = output / "data" / method / f"seed_{seed}"
            metric_source = source_dir / "metrics_selected_10k.csv"
            summary_source = source_dir / "summary.json"
            curves[(method, seed)] = copy_exact_curve(
                metric_source, destination / "metrics_selected_10k.csv"
            )
            write_json(
                destination / "summary.json",
                portable_summary(read_json(summary_source), method, seed),
            )
            shutil.copyfile(source_dir / "config.json", destination / "config.json")
            shutil.copyfile(
                source_dir / "env_snapshot.json", destination / "env_snapshot.json"
            )
            inputs.extend(
                (
                    input_record(
                        metric_source,
                        f"{method}.seed_{seed}.metrics",
                        "verified comparator compact artifact",
                    ),
                    input_record(
                        summary_source,
                        f"{method}.seed_{seed}.summary",
                        "verified comparator compact artifact",
                    ),
                )
            )

    for seed in analysis.SEEDS:
        source_dir = run_dir(raw, seed)
        destination = output / "data" / "random" / f"seed_{seed}"
        metrics_source = source_dir / "metrics.csv"
        summary_source = source_dir / "summary.json"
        curves[("random", seed)] = copy_exact_curve(
            metrics_source, destination / "metrics_selected_10k.csv"
        )
        summary = read_json(summary_source)
        write_json(
            destination / "summary.json",
            portable_summary(summary, "random", seed),
        )
        write_json(destination / "config.json", random_config(seed, launcher_sha))
        write_json(
            destination / "env_snapshot.json",
            random_env_proxy(seed, comparator),
        )
        invariant_rows.append(
            count_invariant_trace(
                metrics_source,
                destination / "positive_count_invariant_trace.csv",
                seed,
            )
        )
        run_result = analysis.run_metrics(curves[("random", seed)])
        native_auc = analysis.finite_float(summary.get("eval_AUC_success"))
        native_final = analysis.finite_float(summary.get("final_eval_env_success"))
        native_best = analysis.finite_float(summary.get("best_eval_env_success"))
        record = {
            "seed": seed,
            "passed": True,
            "total_steps": summary.get("total_steps"),
            "total_updates": summary.get("total_updates"),
            "final_replay_size": summary.get("final_replay_size"),
            "evaluation_points": run_result["evaluation_points"],
            "evaluation_first_step": analysis.EVAL_STEPS[0],
            "evaluation_last_step": analysis.EVAL_STEPS[-1],
            "summary_auc_difference": (
                None
                if native_auc is None
                else native_auc - float(run_result["auc"])
            ),
            "summary_final_difference": (
                None
                if native_final is None
                else native_final - float(run_result["final_success"])
            ),
            "summary_best_difference": (
                None
                if native_best is None
                else native_best - float(run_result["best_success"])
            ),
        }
        if (
            record["total_steps"] != 2_000_000
            or record["total_updates"] != 1_995_001
            or record["final_replay_size"] != 300_000
            or any(
                abs(float(record[key])) > 1e-12
                for key in (
                    "summary_auc_difference",
                    "summary_final_difference",
                    "summary_best_difference",
                )
            )
        ):
            raise ValueError(f"seed {seed}: completion validation failed")
        write_json(destination / "completion_validation.json", record)
        completion_rows.append(record)
        inputs.extend(
            (
                input_record(
                    metrics_source,
                    f"random.seed_{seed}.metrics",
                    f"raw result committed at {RANDOM_RESULT_COMMIT}",
                ),
                input_record(
                    summary_source,
                    f"random.seed_{seed}.summary",
                    f"raw result committed at {RANDOM_RESULT_COMMIT}",
                ),
            )
        )

    results = analysis.analyze_curves(curves)
    enriched_per_seed: list[dict[str, object]] = []
    for row in results["per_seed"]:
        method = str(row["method_key"])
        seed = int(row["seed"])
        run_path = f"data/{method}/seed_{seed}"
        summary = read_json(output / run_path / "summary.json")
        summary_auc = analysis.finite_float(summary.get("eval_AUC_success"))
        summary_final = analysis.finite_float(
            summary.get("final_eval_env_success")
        )
        summary_best = analysis.finite_float(summary.get("best_eval_env_success"))
        if summary_auc is None or summary_final is None or summary_best is None:
            raise ValueError(
                f"{run_path}/summary.json: missing trustworthy AUC/final/best fields"
            )
        enriched = {
            "method_key": method,
            "method": row["method"],
            "seed": seed,
            "run_dir": run_path,
            "metrics_path": f"{run_path}/metrics_selected_10k.csv",
            "evaluation_points": row["evaluation_points"],
            "auc": row["auc"],
            "final_success": row["final_success"],
            "best_success": row["best_success"],
            "best_to_final_drop": row["best_to_final_drop"],
            "hit90": row["hit90"],
            "summary_auc": summary_auc,
            "summary_auc_difference": summary_auc - float(row["auc"]),
            "summary_final_difference": (
                summary_final - float(row["final_success"])
            ),
            "summary_best_difference": summary_best - float(row["best_success"]),
        }
        for field in (
            "summary_auc_difference",
            "summary_final_difference",
            "summary_best_difference",
        ):
            if abs(float(enriched[field])) > 1e-12:
                raise ValueError(
                    f"{method} seed {seed}: packaged summary mismatch in {field}"
                )
        enriched_per_seed.append(enriched)
    results["per_seed"] = enriched_per_seed
    write_csv(output / "per_seed_results.csv", results["per_seed"])
    write_csv(output / "aggregate_results.csv", results["aggregate"])
    write_csv(output / "paired_comparisons.csv", results["paired_comparisons"])
    write_csv(output / "exact_grid_mean_curves.csv", results["mean_curves"])
    write_json(
        output / "analysis_results.json",
        {
            "schema_version": 1,
            "seeds": list(analysis.SEEDS),
            "evaluation_field": analysis.PRIMARY_FIELD,
            "evaluation_steps": list(analysis.EVAL_STEPS),
            "auc_convention": (
                "observed-only trapezoid divided by 2,000,000 - 10,000"
            ),
            "smoothing": False,
            "interpolation": False,
            "forward_fill": False,
            "extrapolation": False,
            "bootstrap": "ordered exhaustive paired bootstrap, 5^5=3125",
            **results,
        },
    )
    write_json(
        output / "completion_validation.json",
        {
            "schema_version": 1,
            "passed": True,
            "seed_count": len(completion_rows),
            "records": completion_rows,
        },
    )
    write_json(
        output / "positive_count_invariant_audit.json",
        {
            "schema_version": 1,
            "definition": (
                "reference_high_weight_rate equals assigned_high_weight_rate "
                "for every finite logged Random-Support row"
            ),
            "tolerance": 1e-12,
            "passed": all(row["passed"] for row in invariant_rows),
            "total_rows_checked": sum(
                int(row["rows_checked"]) for row in invariant_rows
            ),
            "total_mismatch_count": sum(
                int(row["mismatch_count"]) for row in invariant_rows
            ),
            "max_abs_difference": max(
                float(row["max_abs_difference"]) for row in invariant_rows
            ),
            "records": invariant_rows,
        },
    )
    write_json(
        output / "source_provenance.json",
        {
            "schema_version": 1,
            "random_result_commit": RANDOM_RESULT_COMMIT,
            "comparator_artifact_commit": COMPARATOR_ARTIFACT_COMMIT,
            "implementation_source_parent_commit": (
                "7c870d78b6b064d45432c84a8e9b4f1c624a2357"
            ),
            "implementation_source_tree_consistency": {
                "path": "metaworld_Drawer_Open_DDPG/code/src",
                "parent_is_ancestor_of_result_commit": True,
                "tree_diff_is_empty": True,
                "interpretation": (
                    "source-tree consistency evidence only; not proof that the "
                    "runs executed this exact commit"
                ),
            },
            "source_files": source_rows,
            "random_original_config_available": False,
            "random_original_environment_snapshot_available": False,
            "metadata_policy": (
                "config is explicitly reconstructed from frozen launcher; "
                "environment record is explicitly a same-seed proxy"
            ),
        },
    )
    write_json(
        output / "input_manifest.json",
        {
            "schema_version": 1,
            "artifact_id": ARTIFACT_ID,
            "random_result_commit": RANDOM_RESULT_COMMIT,
            "comparator_artifact_commit": COMPARATOR_ARTIFACT_COMMIT,
            "comparator_artifact_manifest_sha256": sha256(
                comparator / "artifact_manifest.json"
            ),
            "inputs": inputs,
            "transform": {
                "selected_metrics": (
                    "only exact finite observed eval/env_success_mean rows "
                    "at 10k..2M; no fallback, smoothing, or interpolation"
                ),
                "positive_count_trace": (
                    "all finite reference/assigned high-weight-rate pairs"
                ),
                "summary_run_dir": (
                    "host-specific absolute run_dir values are replaced with "
                    "package-relative data/<method>/seed_<n> paths"
                ),
                "checkpoints_logs_jsonl_copied": False,
            },
        },
    )

    files = packaged_files(output)
    write_json(
        output / "artifact_manifest.json",
        {
            "schema_version": 1,
            "artifact_id": ARTIFACT_ID,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "random_result_commit": RANDOM_RESULT_COMMIT,
            "comparator_artifact_commit": COMPARATOR_ARTIFACT_COMMIT,
            "method_seed_coverage": {
                method: list(analysis.SEEDS) for method in analysis.METHODS
            },
            "evaluation": {
                "field": analysis.PRIMARY_FIELD,
                "steps": list(analysis.EVAL_STEPS),
                "points_per_seed": len(analysis.EVAL_STEPS),
                "auc": "exact observed-only normalized trapezoid",
                "smoothing": False,
                "interpolation": False,
                "forward_fill": False,
                "extrapolation": False,
            },
            "bootstrap_resamples": 3125,
            "sign_flip_assignments": 32,
            "manifest_self_hash_policy": (
                "artifact_manifest.json is excluded from packaged_files to "
                "avoid a recursive self-hash"
            ),
            "packaged_file_count_excluding_manifest": len(files),
            "package_content_sha256_excluding_manifest": content_digest(files),
            "packaged_files": files,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--comparator-artifact", type=Path, default=DEFAULT_COMPARATOR
    )
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    build(args.repo_root, args.raw_root, args.comparator_artifact, args.output)
    print(json.dumps({"artifact": str(args.output.resolve()), "status": "built"}))


if __name__ == "__main__":
    main()
