#!/usr/bin/env python3
"""Build the DrawerClose DDPG Random-Support count-matched artifact.

The legacy run committed complete metrics and summaries, but not original
config files, environment snapshots, checkpoints, logs, or a serialized git
SHA.  This builder therefore validates only claims supported by those files,
uses explicitly labelled reconstructed/proxy metadata, and records repository
source-tree consistency without claiming cryptographic execution binding.
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
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Keep reproduction runs from mutating the packaged artifact with __pycache__.
sys.dont_write_bytecode = True

import analyze_results as analysis


ARTIFACT_ID = "drawer_close_ddpg_saber_random_support_count_matched_2m"
EXPERIMENT = "DDPG-HER-B3-Random-Support-2M"
RUN_PREFIX = "drawer_close_DDPG-HER-B3-Random-Support-2M"
RAW_RESULT_COMMIT = "ae6d93de0d1dad5bf8328fd3334addb00caec205"
LAUNCHER_CREATION_COMMIT = "bf5fe867b08ab3c374a80802dd706c2d6f88df2c"
SOURCE_TREE_OID = "78a7408bd7dd20269331e5f02143e4a195f15b90"
RANDOM_LAUNCHER_BLOB = "796b275b4f6929fc44d4e23b96597e8578466d7a"
COMPARATOR_ID = "drawer_close_ddpg_saber_context_eligible_count_matched_2m"
COMPARATOR_COMMIT = "841eb706b1dcdc12b59db23ec545099d39d3c619"
COMPARATOR_SOURCE_COMMIT = "de13e3a0a14d3dc6984d3c592452517f3e32c109"
TOOL_DIR = Path(__file__).resolve().parent
SOURCE_PATHS = (
    "metaworld_Drawer_Close_DDPG/DDPG-HER-B3-Random-Support-2M.sh",
    "metaworld_Drawer_Close_DDPG/DDPG-HER-B3-Critic-Only-2M.sh",
    "metaworld_Drawer_Close_DDPG/code/src",
)
TRACE_FIELDS = (
    "reference_high_weight_rate",
    "assigned_high_weight_rate",
    "lite/reference_high_weight_rate",
    "lite/assigned_high_weight_rate",
    "mean_w_P_reference",
    "lite/mean_w_P_reference",
    "lite/mean_w_P",
    "lite/psi_q_rank_rate",
    "support_weight_alignment_rate",
    "support_weight_overlap_rate",
)
EVAL_FIELDS = (
    "eval/return_mean",
    "eval/success_mean",
    "eval/goal_success_mean",
    "eval/env_success_mean",
    "eval/goal_success_ever_mean",
    "eval/env_success_ever_mean",
    "eval/goal_success_final_mean",
    "eval/env_success_final_mean",
    "eval/episode_length_mean",
    "eval/AUC_success",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_json(value: Any, replacements: Sequence[tuple[str, str]] = ()) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {
            str(key): clean_json(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [clean_json(item, replacements) for item in value]
    if isinstance(value, str):
        result = value
        for old, new in replacements:
            if old:
                result = result.replace(old, new)
        return result
    return value


def read_json(path: Path) -> dict[str, object]:
    require(path.is_file() and path.stat().st_size > 0, f"missing JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"{path}: expected object")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def finite(label: str, value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: nonnumeric {value!r}") from exc
    require(math.isfinite(parsed), f"{label}: non-finite")
    return parsed


def close(left: object, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(
        finite("numeric value", left),
        float(right),
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo,
        text=not binary,
        stderr=subprocess.PIPE,
    )


def logical_path(path: Path, roots: Sequence[tuple[Path, str]]) -> str:
    resolved = path.resolve()
    for root, label in roots:
        try:
            relative = resolved.relative_to(root.resolve())
        except ValueError:
            continue
        return f"{label}/{relative.as_posix()}"
    return f"<external>/{path.name}"


def input_record(
    path: Path,
    role: str,
    roots: Sequence[tuple[Path, str]],
    *,
    included: bool,
    **metadata: object,
) -> dict[str, object]:
    require(path.is_file(), f"missing input {path}")
    return {
        "role": role,
        "source_path": logical_path(path, roots),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "included_in_artifact": included,
        **metadata,
    }


def verify_upstream_manifest(root: Path) -> dict[str, object]:
    manifest = read_json(root / "artifact_manifest.json")
    rows = manifest.get("packaged_files")
    require(isinstance(rows, list), "upstream manifest malformed")
    listed = {
        str(row["path"]): row for row in rows if isinstance(row, dict)
    }
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    require(set(actual) == set(listed), "upstream artifact file set mismatch")
    for relative, path in actual.items():
        row = listed[relative]
        require(
            path.stat().st_size == int(row["size_bytes"])
            and sha256(path) == row["sha256"],
            f"upstream hash mismatch: {relative}",
        )
    return manifest


def discover_raw(root: Path) -> dict[int, Path]:
    pattern = re.compile(
        rf"^{re.escape(RUN_PREFIX)}_s([1-5])_2m_cuda$"
    )
    found: dict[int, Path] = {}
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        match = pattern.fullmatch(child.name)
        if match:
            seed = int(match.group(1))
            require(seed not in found, f"duplicate seed {seed}")
            found[seed] = child
    require(set(found) == set(analysis.SEEDS), f"seed mismatch {sorted(found)}")
    return {seed: found[seed] for seed in analysis.SEEDS}


def sanitized_summary(
    source: Path,
    replacements: Sequence[tuple[str, str]],
) -> dict[str, object]:
    return clean_json(read_json(source), replacements)


def select_and_audit_metrics(
    source: Path,
    selected_path: Path,
    trace_path: Path,
    seed: int,
) -> tuple[list[int], list[float], dict[str, object]]:
    selected: list[dict[str, str]] = []
    trace: list[dict[str, object]] = []
    eval_steps: list[int] = []
    eval_values: list[float] = []
    max_deviations = {
        "reference_minus_assigned_rate": 0.0,
        "duplicate_rate_fields": 0.0,
        "reference_raw_mean_formula": 0.0,
        "assigned_raw_mean_formula": 0.0,
        "canonical_support_rate": 0.0,
    }
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        required = {
            "step",
            "config/weight_scope",
            *TRACE_FIELDS,
            *EVAL_FIELDS,
        }
        require(
            required.issubset(fields),
            f"seed {seed}: missing fields {sorted(required - set(fields))}",
        )
        for row_index, row in enumerate(reader, start=2):
            raw_step = finite(f"seed {seed}/step", row.get("step"))
            step = int(raw_step)
            require(raw_step == step, f"seed {seed}: nonintegral step")
            if step >= 5_000:
                require(
                    row.get("config/weight_scope") == "critic_only",
                    f"seed {seed} row {row_index}: wrong scope",
                )
                values = {
                    field: finite(
                        f"seed {seed} row {row_index}/{field}", row.get(field)
                    )
                    for field in TRACE_FIELDS
                }
                ref = values["reference_high_weight_rate"]
                assigned = values["assigned_high_weight_rate"]
                ref_count = ref * 256.0
                assigned_count = assigned * 256.0
                require(
                    0.0 <= ref <= 1.0 and 0.0 <= assigned <= 1.0,
                    f"seed {seed}: rate outside [0,1]",
                )
                require(
                    close(ref_count, round(ref_count), 1e-12)
                    and close(assigned_count, round(assigned_count), 1e-12),
                    f"seed {seed}: nonintegral positive count",
                )
                deviations = {
                    "reference_minus_assigned_rate": abs(ref - assigned),
                    "duplicate_rate_fields": max(
                        abs(ref - values["lite/reference_high_weight_rate"]),
                        abs(
                            assigned
                            - values["lite/assigned_high_weight_rate"]
                        ),
                        abs(
                            values["mean_w_P_reference"]
                            - values["lite/mean_w_P_reference"]
                        ),
                    ),
                    "reference_raw_mean_formula": abs(
                        values["mean_w_P_reference"] - (1.0 + ref)
                    ),
                    "assigned_raw_mean_formula": abs(
                        values["lite/mean_w_P"] - (1.0 + assigned)
                    ),
                    "canonical_support_rate": abs(
                        values["lite/psi_q_rank_rate"] - ref
                    ),
                }
                for key, value in deviations.items():
                    max_deviations[key] = max(max_deviations[key], value)
                require(
                    max(deviations.values()) <= 1e-12,
                    f"seed {seed} row {row_index}: count invariant failed",
                )
                trace.append(
                    {
                        "row_index": row_index,
                        "step": step,
                        "reference_high_weight_rate": ref,
                        "assigned_high_weight_rate": assigned,
                        "reference_high_weight_count": int(round(ref_count)),
                        "assigned_high_weight_count": int(
                            round(assigned_count)
                        ),
                        "reference_raw_weight_mean": values[
                            "mean_w_P_reference"
                        ],
                        "assigned_raw_weight_mean": values["lite/mean_w_P"],
                        "canonical_support_positive_rate": values[
                            "lite/psi_q_rank_rate"
                        ],
                        "support_weight_alignment_rate": values[
                            "support_weight_alignment_rate"
                        ],
                        "support_weight_overlap_rate": values[
                            "support_weight_overlap_rate"
                        ],
                    }
                )
            raw_eval = row.get(analysis.PRIMARY_SUCCESS_FIELD, "")
            if raw_eval not in ("", None) and str(raw_eval).lower() != "nan":
                value = finite(
                    f"seed {seed} eval {step}", raw_eval
                )
                require(0.0 <= value <= 1.0, "success outside [0,1]")
                for field in EVAL_FIELDS:
                    finite(f"seed {seed} eval {step}/{field}", row.get(field))
                eval_steps.append(step)
                eval_values.append(value)
                selected.append(row)
    require(
        eval_steps == list(analysis.EVAL_STEPS),
        f"seed {seed}: exact grid mismatch",
    )
    require(len(trace) == 10_176, f"seed {seed}: trace row count mismatch")

    selected_path.parent.mkdir(parents=True, exist_ok=True)
    with selected_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        trace_fields = list(trace[0].keys())
        writer = csv.DictWriter(
            handle, fieldnames=trace_fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(trace)
    counts = [int(row["reference_high_weight_count"]) for row in trace]
    audit = {
        "schema_version": 1,
        "seed": seed,
        "passed": True,
        "serialized_post_warmup_rows_audited": len(trace),
        "audit_scope": (
            "every serialized post-warmup metrics row; optimizer updates "
            "between logged rows are not directly observed"
        ),
        "batch_size": 256,
        "raw_low_weight": 1.0,
        "raw_high_weight": 2.0,
        "minimum_positive_count": min(counts),
        "maximum_positive_count": max(counts),
        "max_absolute_deviations": max_deviations,
        "invariants": {
            "reference_and_assigned_positive_counts_equal": True,
            "positive_counts_integral": True,
            "reference_rate_matches_canonical_support_rate": True,
            "raw_weight_histograms_count_matched": True,
            "assignment_is_batch_wide_random": (
                "source-derived; raw metrics preserve count and overlap/"
                "alignment diagnostics but not sampled indices"
            ),
        },
    }
    return eval_steps, eval_values, audit


def normalized_auc(values: Sequence[float]) -> float:
    area = math.fsum(
        (right - left) * (left_y + right_y) / 2.0
        for left, right, left_y, right_y in zip(
            analysis.EVAL_STEPS,
            analysis.EVAL_STEPS[1:],
            values,
            values[1:],
        )
    )
    return area / (
        analysis.EVAL_STEPS[-1] - analysis.EVAL_STEPS[0]
    )


def validate_summary(
    summary: Mapping[str, object],
    steps: Sequence[int],
    values: Sequence[float],
    seed: int,
) -> dict[str, object]:
    expected_hit = next(
        (step for step, value in zip(steps, values) if value >= 0.9), None
    )
    require(summary.get("method") == "b3_support_random_runtime_opt", "method")
    require(int(summary.get("total_steps", -1)) == 2_000_000, "steps")
    require(int(summary.get("total_updates", -1)) == 1_995_001, "updates")
    require(int(summary.get("final_replay_size", -1)) == 300_000, "replay")
    require(close(summary.get("eval_AUC_success"), normalized_auc(values)), "AUC")
    require(close(summary.get("final_eval_env_success"), values[-1]), "final")
    require(close(summary.get("best_eval_env_success"), max(values)), "best")
    raw_hit = summary.get("first_hit_0.90")
    parsed_hit = None if raw_hit in (None, "", "None") else int(raw_hit)
    require(parsed_hit == expected_hit, f"seed {seed}: Hit90 mismatch")
    return {
        "auc": normalized_auc(values),
        "final_success": values[-1],
        "best_success": max(values),
        "hit90": expected_hit,
        "summary_matches_exact_grid": True,
    }


def comparator_copy(
    comparator: Path,
    stage: Path,
    input_rows: list[dict[str, object]],
    roots: Sequence[tuple[Path, str]],
) -> dict[str, object]:
    manifest = verify_upstream_manifest(comparator)
    require(
        manifest.get("artifact_id") == COMPARATOR_ID
        and manifest.get("source_commit") == COMPARATOR_SOURCE_COMMIT,
        "wrong comparator artifact",
    )
    for method in ("her", "main"):
        for seed in analysis.SEEDS:
            source = comparator / "data" / method / f"seed_{seed}"
            target = stage / "data" / method / f"seed_{seed}"
            target.mkdir(parents=True, exist_ok=True)
            for name in ("metrics_selected_10k.csv", "summary.json"):
                src = source / name
                require(src.is_file(), f"comparator missing {src}")
                if name.endswith(".json"):
                    payload = clean_json(read_json(src))
                    if isinstance(payload, dict) and "run_dir" in payload:
                        payload["run_dir"] = f"data/{method}/seed_{seed}"
                    write_json(target / name, payload)
                else:
                    (target / name).write_bytes(
                        src.read_bytes().replace(b"\r\n", b"\n")
                    )
                input_rows.append(
                    input_record(
                        src,
                        f"{method}.seed_{seed}.{name}",
                        roots,
                        included=True,
                        comparator_commit=COMPARATOR_COMMIT,
                        original_training_source_commit="unknown_not_recorded",
                    )
                )
            strict = comparator / "data" / "strict" / f"seed_{seed}"
            proxy_config = strict / "config.json"
            proxy_env = strict / "env_snapshot.json"
            write_json(
                target / "config.json",
                {
                    "schema_version": 1,
                    "record_type": "reconstructed_metadata_not_original_run_config",
                    "not_original_run_config": True,
                    "method_key": method,
                    "seed": seed,
                    "proxy_source": "same-seed strict-control config",
                    "proxy_source_sha256": sha256(proxy_config),
                    "original_training_source_commit": None,
                },
            )
            write_json(
                target / "env_snapshot.json",
                {
                    "schema_version": 1,
                    "record_type": "same_seed_environment_proxy_not_original_snapshot",
                    "not_original_run_env_snapshot": True,
                    "method_key": method,
                    "seed": seed,
                    "proxy_source_sha256": sha256(proxy_env),
                    "proxy_snapshot": clean_json(read_json(proxy_env)),
                },
            )
    upstream = comparator / "input_manifest.json"
    write_json(
        stage / "provenance" / "comparator_input_manifest.json",
        clean_json(read_json(upstream)),
    )
    return {
        "artifact_id": COMPARATOR_ID,
        "artifact_commit": COMPARATOR_COMMIT,
        "packaging_source_commit": COMPARATOR_SOURCE_COMMIT,
        "manifest_sha256": sha256(comparator / "artifact_manifest.json"),
        "arms_reused": ["her", "main"],
        "retrained": False,
        "original_training_source_commit": None,
        "metadata_policy": (
            "reused without retraining; metric row content is preserved with "
            "portable newline normalization, summary run_dir is sanitized, "
            "and config/environment records are explicitly labelled "
            "reconstructed proxies"
        ),
    }


def freeze_source(
    repo: Path,
    materialized: Path,
    stage: Path,
    input_rows: list[dict[str, object]],
    roots: Sequence[tuple[Path, str]],
) -> dict[str, object]:
    raw_tree = str(
        git(
            repo,
            "rev-parse",
            f"{RAW_RESULT_COMMIT}:metaworld_Drawer_Close_DDPG/code/src",
        )
    ).strip()
    initial_tree = str(
        git(
            repo,
            "rev-parse",
            f"{LAUNCHER_CREATION_COMMIT}:metaworld_Drawer_Close_DDPG/code/src",
        )
    ).strip()
    raw_launcher = str(
        git(
            repo,
            "rev-parse",
            f"{RAW_RESULT_COMMIT}:"
            "metaworld_Drawer_Close_DDPG/DDPG-HER-B3-Random-Support-2M.sh",
        )
    ).strip()
    initial_launcher = str(
        git(
            repo,
            "rev-parse",
            f"{LAUNCHER_CREATION_COMMIT}:"
            "metaworld_Drawer_Close_DDPG/DDPG-HER-B3-Random-Support-2M.sh",
        )
    ).strip()
    require(
        raw_tree == initial_tree == SOURCE_TREE_OID,
        "source tree consistency failed",
    )
    require(
        raw_launcher == initial_launcher == RANDOM_LAUNCHER_BLOB,
        "launcher consistency failed",
    )
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            LAUNCHER_CREATION_COMMIT,
            RAW_RESULT_COMMIT,
        ],
        cwd=repo,
        check=False,
    )
    require(ancestor.returncode == 0, "launcher commit is not ancestor")

    source_root = materialized / "metaworld_Drawer_Close_DDPG"
    files = [
        source_root / "DDPG-HER-B3-Random-Support-2M.sh",
        source_root / "DDPG-HER-B3-Critic-Only-2M.sh",
        *sorted((source_root / "code" / "src").rglob("*")),
    ]
    files = [path for path in files if path.is_file()]
    rows: list[dict[str, object]] = []
    checksum_lines: list[str] = []
    for source in files:
        relative = source.relative_to(source_root)
        repo_path = (
            Path("metaworld_Drawer_Close_DDPG") / relative
        ).as_posix()
        committed = git(
            repo,
            "show",
            f"{RAW_RESULT_COMMIT}:{repo_path}",
            binary=True,
        )
        assert isinstance(committed, bytes)
        require(source.read_bytes() == committed, f"source mismatch {repo_path}")
        destination = stage / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(committed)
        mode = str(
            git(
                repo,
                "ls-tree",
                RAW_RESULT_COMMIT,
                "--",
                repo_path,
            )
        ).split()[0]
        if mode == "100755":
            os.chmod(destination, 0o755)
        digest = sha256(destination)
        blob = str(
            git(
                repo,
                "rev-parse",
                f"{RAW_RESULT_COMMIT}:{repo_path}",
            )
        ).strip()
        rows.append(
            {
                "path": relative.as_posix(),
                "repo_path": repo_path,
                "git_blob_oid": blob,
                "mode": mode,
                "size_bytes": destination.stat().st_size,
                "sha256": digest,
            }
        )
        checksum_lines.append(f"{digest}  source_snapshot/{relative.as_posix()}")
        input_rows.append(
            input_record(
                source,
                f"source.{relative.as_posix()}",
                roots,
                included=True,
                source_consistency_commit=RAW_RESULT_COMMIT,
                execution_binding="reconstructed_not_serialized",
            )
        )
    (stage / "frozen_source.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    provenance = {
        "schema_version": 1,
        "passed": True,
        "raw_result_commit": RAW_RESULT_COMMIT,
        "launcher_creation_commit": LAUNCHER_CREATION_COMMIT,
        "launcher_creation_is_ancestor_of_result_commit": True,
        "code_src_tree_oid_at_launcher_creation": initial_tree,
        "code_src_tree_oid_at_result_commit": raw_tree,
        "random_launcher_blob_at_launcher_creation": initial_launcher,
        "random_launcher_blob_at_result_commit": raw_launcher,
        "source_tree_unchanged": True,
        "execution_source_commit_serialized_by_raw_run": False,
        "execution_binding_status": "reconstructed_not_cryptographically_bound",
        "claim_boundary": (
            "Repository history proves source-tree consistency between the "
            "launcher-creation and raw-result commits. The legacy run did not "
            "serialize a git SHA, so this is not proof of the exact checkout "
            "used by the training process."
        ),
        "file_count": len(rows),
        "files": rows,
    }
    write_json(stage / "provenance" / "source_provenance.json", provenance)
    return provenance


def reconstructed_random_config(
    seed: int,
    strict_config: Mapping[str, object],
    proxy_sha: str,
) -> dict[str, object]:
    shared_keys = (
        "task_name",
        "reward_type",
        "metaworld_env_name",
        "metaworld_reward_function_version",
        "max_episode_steps",
        "total_steps",
        "start_steps",
        "warmup_steps",
        "updates_per_step",
        "batch_size",
        "buffer_capacity",
        "her_ratio",
        "gamma",
        "tau",
        "actor_lr",
        "critic_lr",
        "hidden_dim",
        "noise_std",
        "eval_freq",
        "eval_episodes",
        "success_terminates",
        "timeout_bootstrap",
        "M_B",
        "M_Q_min",
        "N_local_min",
        "local_search_max",
        "lambda_support",
    )
    return {
        "schema_version": 1,
        "record_type": "reconstructed_metadata_not_original_run_config",
        "not_original_run_config": True,
        "seed": seed,
        "reason": "legacy raw result contains metrics.csv and summary.json only",
        "proxy_source": "same-seed strict-control config",
        "proxy_source_sha256": proxy_sha,
        "source_derived_fields": {
            "method": "b3_support_random_runtime_opt",
            "weight_scope": "critic_only",
            "weighting_mode": "batch_norm",
            "support_query_mode": "current_policy",
            "support_query_mode_provenance": "legacy source default",
            "support_assignment_mode": "batch_wide_count_matched_random",
            "random_assignment_without_replacement": True,
            "raw_low_weight": 1.0,
            "raw_high_weight": 2.0,
        },
        "reconstructed_shared_fields": {
            key: strict_config[key] for key in shared_keys if key in strict_config
        },
        "original_config_unavailable": True,
    }


def raw_copy(
    raw_root: Path,
    comparator: Path,
    stage: Path,
    input_rows: list[dict[str, object]],
    roots: Sequence[tuple[Path, str]],
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for seed, run in discover_raw(raw_root).items():
        metrics = run / "metrics.csv"
        summary_path = run / "summary.json"
        require(metrics.is_file() and summary_path.is_file(), f"seed {seed} inputs")
        target = stage / "data" / "random" / f"seed_{seed}"
        target.mkdir(parents=True, exist_ok=True)
        steps, values, audit = select_and_audit_metrics(
            metrics,
            target / "metrics_selected_10k.csv",
            target / "assignment_count_trace.csv",
            seed,
        )
        summary = sanitized_summary(summary_path, ())
        raw_summary_run_dir = summary.get("run_dir")
        require(
            isinstance(raw_summary_run_dir, str)
            and Path(raw_summary_run_dir).is_absolute()
            and Path(raw_summary_run_dir).name == run.name,
            f"seed {seed}: raw summary run_dir provenance",
        )
        gate = validate_summary(summary, steps, values, seed)
        summary["run_dir"] = f"data/random/seed_{seed}"
        write_json(target / "summary.json", summary)
        write_json(target / "positive_count_invariant.json", audit)

        strict = comparator / "data" / "strict" / f"seed_{seed}"
        proxy_config_path = strict / "config.json"
        proxy_env_path = strict / "env_snapshot.json"
        proxy_config = read_json(proxy_config_path)
        proxy_env = clean_json(read_json(proxy_env_path))
        write_json(
            target / "config.json",
            reconstructed_random_config(
                seed, proxy_config, sha256(proxy_config_path)
            ),
        )
        write_json(
            target / "env_snapshot.json",
            {
                "schema_version": 1,
                "record_type": "same_seed_environment_proxy_not_original_snapshot",
                "not_original_run_env_snapshot": True,
                "seed": seed,
                "proxy_source_sha256": sha256(proxy_env_path),
                "proxy_snapshot": proxy_env,
            },
        )
        record = {
            "seed": seed,
            "passed": True,
            "evidence_scope": "committed metrics.csv and summary.json only",
            "total_steps": 2_000_000,
            "total_updates": 1_995_001,
            "evaluation_points": 200,
            "evaluation_episodes_per_point": 20,
            "evaluation_field": analysis.PRIMARY_SUCCESS_FIELD,
            "summary": gate,
            "positive_count_invariant": audit,
            "original_config_available": False,
            "original_environment_snapshot_available": False,
            "checkpoint_available": False,
            "training_log_available": False,
            "execution_git_sha_serialized": False,
            "raw_summary_run_dir_sanitized": True,
            "raw_summary_run_dir_basename": run.name,
            "raw_summary_run_dir_sha256": hashlib.sha256(
                raw_summary_run_dir.encode("utf-8")
            ).hexdigest(),
        }
        write_json(target / "completion_validation.json", record)
        records.append(record)
        input_rows.extend(
            (
                input_record(
                    metrics,
                    f"random.seed_{seed}.metrics",
                    roots,
                    included=True,
                    compact_transform=(
                        "exact evaluation rows plus count-invariant trace"
                    ),
                    raw_result_commit=RAW_RESULT_COMMIT,
                ),
                input_record(
                    summary_path,
                    f"random.seed_{seed}.summary",
                    roots,
                    included=True,
                    raw_result_commit=RAW_RESULT_COMMIT,
                ),
            )
        )
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "passed": True,
        "seed_count": 5,
        "seeds": list(analysis.SEEDS),
        "evaluation_grid": list(analysis.EVAL_STEPS),
        "evaluation_field": analysis.PRIMARY_SUCCESS_FIELD,
        "fallback_field": None,
        "smoothing": False,
        "interpolation": False,
        "forward_fill": False,
        "extrapolation": False,
        "records": records,
        "legacy_evidence_limitations": {
            "original_config_missing": True,
            "original_environment_snapshot_missing": True,
            "checkpoint_missing": True,
            "logs_missing": True,
            "execution_git_sha_missing": True,
        },
    }


def positive_count_rollup(
    completion: Mapping[str, object],
) -> dict[str, object]:
    records = completion.get("records")
    require(isinstance(records, list), "completion records missing")
    tolerance = 1e-12
    audit_records: list[dict[str, object]] = []
    for expected_seed, record in zip(analysis.SEEDS, records):
        require(isinstance(record, dict), "malformed completion record")
        require(int(record.get("seed", -1)) == expected_seed, "audit seed order")
        audit = record.get("positive_count_invariant")
        require(isinstance(audit, dict), "missing positive-count audit")
        deviations = audit.get("max_absolute_deviations")
        require(isinstance(deviations, dict), "missing count deviations")
        difference = finite(
            f"seed {expected_seed} reference/assigned deviation",
            deviations.get("reference_minus_assigned_rate"),
        )
        rows_checked = int(audit["serialized_post_warmup_rows_audited"])
        mismatch_count = 0 if difference <= tolerance else rows_checked
        audit_records.append(
            {
                "seed": expected_seed,
                "rows_checked": rows_checked,
                "mismatch_count": mismatch_count,
                "max_abs_difference": difference,
                "passed": mismatch_count == 0,
            }
        )
    require(len(audit_records) == len(analysis.SEEDS), "count audit coverage")
    return {
        "schema_version": 1,
        "definition": (
            "reference_high_weight_rate equals assigned_high_weight_rate "
            "for every finite logged Random-Support row"
        ),
        "tolerance": tolerance,
        "records": audit_records,
        "total_rows_checked": sum(
            int(record["rows_checked"]) for record in audit_records
        ),
        "total_mismatch_count": sum(
            int(record["mismatch_count"]) for record in audit_records
        ),
        "max_abs_difference": max(
            float(record["max_abs_difference"]) for record in audit_records
        ),
        "passed": all(bool(record["passed"]) for record in audit_records),
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


def readme(stage: Path) -> str:
    aggregate = {
        row["method_key"]: row
        for row in csv.DictReader(
            (stage / "aggregate_results.csv").open(
                newline="", encoding="utf-8"
            )
        )
    }
    paired = {
        row["contrast_key"]: row
        for row in csv.DictReader(
            (stage / "paired_comparisons.csv").open(
                newline="", encoding="utf-8"
            )
        )
    }
    contrast = paired["main_minus_random"]
    table = "\n".join(
        (
            "| Method | AUC mean ± sample SD | Final mean ± sample SD |",
            "|---|---:|---:|",
            *(
                "| {label} | {auc:.6f} ± {auc_sd:.6f} | "
                "{final:.6f} ± {final_sd:.6f} |".format(
                    label=aggregate[key]["method"],
                    auc=float(aggregate[key]["auc_mean"]),
                    auc_sd=float(aggregate[key]["auc_sample_sd"]),
                    final=float(aggregate[key]["final_success_mean"]),
                    final_sd=float(
                        aggregate[key]["final_success_sample_sd"]
                    ),
                )
                for key in analysis.METHOD_ORDER
            ),
        )
    )
    return f"""# DrawerClose DDPG Random-Support Count-Matched

Compact artifact for seeds 1–5. AUC uses only `eval/env_success_mean` at the
observed 10k–2M grid (200 points), without fallback, smoothing, interpolation,
forward-fill, or extrapolation.

{table}

Main − Random AUC difference:
`{float(contrast['mean_auc_difference']):.6f}`; exhaustive paired-bootstrap
95% CI
`[{float(contrast['bootstrap_ci95_lower']):.6f},
{float(contrast['bootstrap_ci95_upper']):.6f}]`; exact two-sided sign-flip
`p={float(contrast['exact_two_sided_sign_flip_pvalue']):.6f}`.

Random-Support first computes Main SABER's support-positive count `k` on the
same minibatch, then assigns the high raw weight to exactly `k` uniformly
sampled batch positions without replacement. The packaged count traces audit
all 10,176 serialized update-era metrics rows per seed, including the first
update at step 5,000: reference and assigned
counts, raw weight histograms, and canonical support rates match exactly.
Optimizer updates between serialized rows are supported by the frozen source
definition but are not directly observed in the raw artifact.
The top-level `positive_count_invariant_audit.json` provides a machine-readable
five-seed rollup of this check.

## Provenance limitation

The legacy Random-Support result committed complete metrics and summaries but
did not preserve original config files, environment snapshots, checkpoints,
logs, or a serialized git SHA. Config/environment records in this package are
explicitly labelled reconstructed proxies. Repository history verifies that
the `code/src` tree and Random-Support launcher are identical between launcher
commit `{LAUNCHER_CREATION_COMMIT}` and raw-result commit
`{RAW_RESULT_COMMIT}`. This is source-tree consistency, **not** cryptographic
proof of the exact checkout used by the training processes.
Host-specific absolute `run_dir` strings in Random-Support summaries are
replaced with package-relative paths. Their original source-file hashes remain
recorded in `input_manifest.json`, and completion records retain only a
basename and SHA-256 digest of each original path.

HER and Main are reused, not retrained, from verified comparator artifact
commit `{COMPARATOR_COMMIT}`; their original training source commit was also
not recorded.

Verify independently from this directory:

    python3 verify_artifact.py --artifact-dir .

Full raw metrics, checkpoints, JSONL, and logs are excluded.
"""


def build(
    raw_root: Path,
    comparator: Path,
    source_materialized: Path,
    source_repo: Path,
    output: Path,
) -> None:
    for path, label in (
        (raw_root, "raw root"),
        (comparator, "comparator"),
        (source_materialized, "materialized source"),
        (source_repo, "source repository"),
    ):
        require(path.is_dir(), f"missing {label}: {path}")
    require(not output.exists(), f"output already exists: {output}")

    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{ARTIFACT_ID}.", dir=parent
    ) as temporary:
        stage = Path(temporary) / ARTIFACT_ID
        stage.mkdir()
        roots = (
            (raw_root, "raw_results"),
            (comparator, "verified_comparator"),
            (source_materialized, "materialized_source"),
        )
        input_rows: list[dict[str, object]] = []

        comparator_info = comparator_copy(
            comparator, stage, input_rows, roots
        )
        completion = raw_copy(
            raw_root, comparator, stage, input_rows, roots
        )
        write_json(stage / "completion_validation.json", completion)
        write_json(
            stage / "positive_count_invariant_audit.json",
            positive_count_rollup(completion),
        )
        source_info = freeze_source(
            source_repo,
            source_materialized,
            stage,
            input_rows,
            roots,
        )
        write_json(
            stage / "provenance" / "config_provenance.json",
            {
                "schema_version": 1,
                "passed": True,
                "original_random_run_config_available": False,
                "reconstructed_config_records_are_explicitly_labelled": True,
                "source_launcher_available": True,
                "source_method": "b3_support_random_runtime_opt",
            "source_weight_scope": "critic_only",
            "source_weighting_mode": "batch_norm",
            "source_support_query_mode": "current_policy_legacy_default",
                "source_assignment_definition": (
                    "k=sum(canonical_support_mask); sample k batch indices "
                    "uniformly without replacement"
                ),
                "config_parity_status": (
                    "source-derived shared launcher settings; not empirically "
                    "recovered from original config.json"
                ),
            },
        )

        for script in ("build_artifact.py", "analyze_results.py", "verify_artifact.py"):
            source = TOOL_DIR / script
            require(source.is_file(), f"tool missing: {source}")
            shutil.copy2(source, stage / script)

        analysis.analyze(
            (
                analysis.MethodSpec(
                    "her", "DrawerClose DDPG HER", stage / "data" / "her"
                ),
                analysis.MethodSpec(
                    "main",
                    "DrawerClose DDPG HER + SABER Main",
                    stage / "data" / "main",
                ),
                analysis.MethodSpec(
                    "random",
                    "DrawerClose DDPG Random-Support Count-Matched",
                    stage / "data" / "random",
                ),
            ),
            stage,
        )
        write_json(
            stage / "input_manifest.json",
            {
                "schema_version": 1,
                "artifact_id": ARTIFACT_ID,
                "raw_result_commit": RAW_RESULT_COMMIT,
                "comparator": comparator_info,
                "source_provenance": {
                    "execution_binding_status": source_info[
                        "execution_binding_status"
                    ],
                    "source_tree_unchanged": True,
                },
                "inputs": input_rows,
            },
        )
        (stage / "README.md").write_text(readme(stage), encoding="utf-8")
        rows = packaged_files(stage)
        write_json(
            stage / "artifact_manifest.json",
            {
                "schema_version": 1,
                "artifact_id": ARTIFACT_ID,
                "raw_result_commit": RAW_RESULT_COMMIT,
                "source_consistency_commit": RAW_RESULT_COMMIT,
                "execution_binding_status": (
                    "reconstructed_not_cryptographically_bound"
                ),
                "comparator_artifact_id": COMPARATOR_ID,
                "comparator_artifact_commit": COMPARATOR_COMMIT,
                "evaluation": {
                    "field": analysis.PRIMARY_SUCCESS_FIELD,
                    "steps": list(analysis.EVAL_STEPS),
                    "point_count_per_seed": 200,
                    "fallback_field": None,
                    "smoothing": False,
                    "interpolation": False,
                    "forward_fill": False,
                    "extrapolation": False,
                    "auc_normalization_span": 1_990_000,
                },
                "bootstrap": {
                    "scheme": "exhaustive_ordered_5_power_5",
                    "resample_count": 3_125,
                },
                "exact_sign_flip": {
                    "assignment_count": 32,
                },
                "method_seed_coverage": {
                    method: list(analysis.SEEDS)
                    for method in analysis.METHOD_ORDER
                },
                "packaged_file_count_excluding_manifest": len(rows),
                "package_content_sha256_excluding_manifest": content_digest(
                    rows
                ),
        "manifest_hash_policy": (
            "artifact_manifest.json is excluded to avoid recursive self-hash"
        ),
        "packaged_files": rows,
            },
        )
        stage.rename(output)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--comparator-artifact", type=Path, required=True)
    parser.add_argument("--source-materialized", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build(
        args.raw_root.resolve(),
        args.comparator_artifact.resolve(),
        args.source_materialized.resolve(),
        args.source_repo.resolve(),
        args.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
