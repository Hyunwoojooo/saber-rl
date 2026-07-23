#!/usr/bin/env python3
"""Build a compact DrawerClose DDPG replay-action-query paper artifact.

The builder is fail-closed and non-overwriting.  It validates the five raw 2M
runs, audits only the fields actually present in their legacy checkpoints,
reuses the already verified DrawerClose HER/Main compact cohorts, freezes the
exact execution source, runs the bundled exact-grid analyzer, and excludes raw
metrics, checkpoints, and logs from the resulting artifact.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import analyze_results as analysis


ARTIFACT_ID = "drawer_close_ddpg_saber_replay_action_query_2m"
EXPERIMENT = "DDPG-HER-B3-Replay-Action-Query-2M"
RUN_PREFIX = "drawer_close_DDPG-HER-B3-Replay-Action-Query-2M"
EXPECTED_SOURCE_COMMIT = "cb871323fed51c6c7d5f692d189ebd766f282503"
EXPECTED_SOURCE_TREE = "1eb92f2850dcd492b265d2964218c37ff5eee7ff"
COMPARATOR_ARTIFACT_ID = (
    "drawer_close_ddpg_saber_context_eligible_count_matched_2m"
)
COMPARATOR_ARTIFACT_COMMIT = "841eb706b1dcdc12b59db23ec545099d39d3c619"
COMPARATOR_PACKAGING_SOURCE_COMMIT = "de13e3a0a14d3dc6984d3c592452517f3e32c109"
SOURCE_REL = Path(
    "metaworld_Drawer_Close_DDPG/experiments/"
    "DDPG-HER-B3-Replay-Action-Query-2M"
)
MAIN_LAUNCHER_REL = Path(
    "metaworld_Drawer_Close_DDPG/DDPG-HER-B3-Critic-Only-2M.sh"
)
REPLAY_LAUNCHER_REL = SOURCE_REL / "DDPG-HER-B3-Replay-Action-Query-2M.sh"
TOOL_DIR = Path(__file__).resolve().parent

QUERY_FIELDS = (
    "query/context_count_eligible_rate",
    "query/support_positive_rate",
    "query/query_bin_present_given_context_rate",
    "query/query_bin_matches_current_policy_rate",
    "query/query_bin_matches_replay_action_rate",
    "query/high_weight_count",
    "query/raw_weight_mean",
    "query/raw_weight_std",
    "query/raw_weight_ess",
    "query/normalized_weight_mean",
    "query/normalized_weight_std",
    "query/normalized_weight_ess",
)
CRITICAL_TRAIN_FIELDS = (
    "loss/critic",
    "loss/actor",
    "loss/td_error_abs_mean",
    "loss/td_error_abs_p90",
    "value/q_mean",
    "value/target_mean",
    *QUERY_FIELDS,
)
CRITICAL_EVAL_FIELDS = (
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
REQUIRED_CONFIG: dict[str, object] = {
    "task_name": "drawer_close",
    "metaworld_env_name": "drawer-close-v3",
    "metaworld_reward_function_version": "v2",
    "reward_type": "sparse",
    "method": "b3_support_runtime_opt",
    "support_query_mode": "replay_action",
    "weight_scope": "critic_only",
    "weighting_mode": "batch_norm",
    "total_steps": 2_000_000,
    "max_episode_steps": 200,
    "start_steps": 10_000,
    "warmup_steps": 5_000,
    "updates_per_step": 1,
    "batch_size": 256,
    "buffer_capacity": 300_000,
    "her_ratio": 0.8,
    "gamma": 0.98,
    "tau": 0.005,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "hidden_dim": 256,
    "noise_std": 0.2,
    "eval_freq": 10_000,
    "eval_episodes": 20,
    "success_terminates": False,
    "timeout_bootstrap": True,
    "M_B": 16,
    "M_Q_min": 4,
    "N_local_min": 4,
    "N_gap_min": 128,
    "local_search_max": 512,
    "action_bins_per_dim": 5,
    "lambda_support": 1.0,
    "policy_support_mode": "bin",
    "support_scan_mode": "deterministic_tail_v1",
    "support_cache_enabled": True,
    "support_cache_max_entries": 200_000,
    "local_scan_cache_enabled": True,
    "local_scan_cache_max_entries": 50_000,
    "support_short_circuit": True,
    "checkpoint_every_steps": 50_000,
    "save_step_checkpoints": False,
    "device": "cuda",
}
EXPECTED_CHECKPOINT_KEYS = {
    "agent",
    "checkpoint_kind",
    "config",
    "episode_idx",
    "global_step",
    "noise_rng_state",
    "replay",
    "sample_rng_state",
    "total_updates",
    "weight_rng_state",
}
FULL_RESUME_FIELDS_NOT_SERIALIZED = (
    "python_random_state",
    "numpy_global_rng_state",
    "torch_cpu_rng_state",
    "torch_cuda_rng_state_all",
    "eval_history_env",
    "config_fingerprint",
    "checkpoint_metadata",
    "resume_audit",
)
FATAL_LOG_MARKERS = (
    "STOP after failure",
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "OutOfMemory",
    "RuntimeError",
    "AssertionError",
    "FloatingPointError",
    "Segmentation fault",
    "Killed",
    "| ERROR |",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run_git(repo: Path, *args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo,
        text=text,
        stderr=subprocess.PIPE,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    require(path.is_file() and path.stat().st_size > 0, f"missing/empty JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"{path}: expected JSON object")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def clean_json(
    value: Any,
    replacements: Sequence[tuple[str, str]] = (),
) -> Any:
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


def finite(label: str, value: object) -> float:
    parsed = analysis.finite_float(value)
    if parsed is None:
        raise ValueError(f"{label}: expected finite numeric value, got {value!r}")
    return parsed


def close(left: object, right: float, tolerance: float) -> bool:
    return math.isclose(
        finite("numeric value", left),
        float(right),
        rel_tol=0.0,
        abs_tol=tolerance,
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
    included_in_artifact: bool,
    **metadata: object,
) -> dict[str, object]:
    require(path.is_file(), f"input missing: {path}")
    return {
        "role": role,
        "source_path": logical_path(path, roots),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "included_in_artifact": included_in_artifact,
        **metadata,
    }


def verify_manifest(root: Path) -> dict[str, object]:
    manifest = read_json(root / "artifact_manifest.json")
    listed_rows = manifest.get("packaged_files")
    require(isinstance(listed_rows, list), "comparator manifest missing packaged_files")
    listed: dict[str, Mapping[str, object]] = {}
    for row in listed_rows:
        require(isinstance(row, dict), "malformed comparator manifest row")
        relative = str(row.get("path", ""))
        require(
            relative
            and relative not in listed
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            f"unsafe/duplicate comparator path {relative!r}",
        )
        listed[relative] = row
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    require(set(actual) == set(listed), "comparator artifact file-set mismatch")
    for relative, row in listed.items():
        path = actual[relative]
        require(
            path.stat().st_size == int(row["size_bytes"])
            and sha256(path) == row["sha256"],
            f"comparator artifact hash mismatch: {relative}",
        )
    return manifest


def discover_replay_runs(root: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    pattern = re.compile(
        rf"^{re.escape(RUN_PREFIX)}_s([1-5])_2m_cuda$"
    )
    for path in sorted(child for child in root.iterdir() if child.is_dir()):
        match = pattern.fullmatch(path.name)
        if match:
            seed = int(match.group(1))
            require(seed not in found, f"duplicate Replay seed {seed}")
            found[seed] = path
    require(
        set(found) == set(analysis.SEEDS),
        f"Replay seed coverage mismatch: {sorted(found)}",
    )
    return {seed: found[seed] for seed in analysis.SEEDS}


def validate_config(config: Mapping[str, object], seed: int, run_name: str) -> None:
    for key, expected in REQUIRED_CONFIG.items():
        require(key in config, f"seed {seed}: config missing {key}")
        actual = config[key]
        if isinstance(expected, float):
            require(
                close(actual, expected, 1e-12),
                f"seed {seed}: config.{key}={actual!r} != {expected!r}",
            )
        else:
            require(
                actual == expected,
                f"seed {seed}: config.{key}={actual!r} != {expected!r}",
            )
    require(config.get("run_name") == run_name, f"seed {seed}: run_name mismatch")
    require(
        "support_assignment_mode" not in config,
        "legacy DrawerClose config unexpectedly serializes support_assignment_mode",
    )
    require(
        config.get("seeds")
        == {
            "env_seed": seed,
            "torch_seed": seed,
            "numpy_seed": seed,
            "action_noise_seed": seed,
            "weight_seed": 20_000 + seed,
            "eval_seed": 10_000 + seed,
        },
        f"seed {seed}: seed mapping mismatch",
    )


def validate_env(snapshot: Mapping[str, object], seed: int) -> None:
    for key, expected_seed in (("train_env", seed), ("eval_env", 10_000 + seed)):
        env = snapshot.get(key)
        require(isinstance(env, dict), f"seed {seed}: env_snapshot.{key} missing")
        for field, expected in {
            "env_name": "drawer-close-v3",
            "max_episode_steps": 200,
            "reward_type": "sparse",
            "desired_goal_source": "_target_pos",
            "action_dim": 4,
            "goal_dim": 3,
        }.items():
            require(
                env.get(field) == expected,
                f"seed {seed}: env_snapshot.{key}.{field} mismatch",
            )
        kwargs = env.get("spec_kwargs")
        require(
            isinstance(kwargs, dict)
            and int(kwargs.get("seed", -1)) == expected_seed
            and kwargs.get("env_name") == "drawer-close-v3"
            and kwargs.get("reward_function_version") == "v2"
            and kwargs.get("terminate_on_success") is False,
            f"seed {seed}: env_snapshot.{key}.spec_kwargs mismatch",
        )
    task = snapshot.get("task_spec")
    require(
        isinstance(task, dict)
        and "DrawerClose" in str(task.get("task", ""))
        and task.get("phase_labels")
        == {"0": "approach", "1": "contact", "2": "closing", "3": "closed"},
        f"seed {seed}: DrawerClose task snapshot mismatch",
    )


def validate_and_select_metrics(
    source: Path,
    destination: Path,
    seed: int,
    warmup_steps: int,
) -> tuple[dict[str, object], list[int], list[float]]:
    values = {field: [] for field in QUERY_FIELDS}
    post_warmup_rows = 0
    selected: list[dict[str, str]] = []
    eval_steps: list[int] = []
    eval_values: list[float] = []
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        required = {
            "step",
            "config/support_query_mode",
            "config/weight_scope",
            *CRITICAL_TRAIN_FIELDS,
            *CRITICAL_EVAL_FIELDS,
        }
        missing = sorted(required - set(fields))
        require(not missing, f"seed {seed}: metrics missing columns {missing}")
        for row_index, row in enumerate(reader, start=2):
            step_value = finite(f"seed {seed} row {row_index}/step", row.get("step"))
            step = int(step_value)
            require(step_value == step, f"seed {seed}: non-integral metric step")
            if step > warmup_steps:
                post_warmup_rows += 1
                require(
                    row.get("config/support_query_mode") == "replay_action",
                    f"seed {seed} row {row_index}: wrong query mode",
                )
                require(
                    row.get("config/weight_scope") == "critic_only",
                    f"seed {seed} row {row_index}: wrong weight scope",
                )
                row_values = {
                    field: finite(
                        f"seed {seed} row {row_index}/{field}",
                        row.get(field),
                    )
                    for field in CRITICAL_TRAIN_FIELDS
                }
                require(
                    math.isclose(
                        row_values["query/query_bin_matches_replay_action_rate"],
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ),
                    f"seed {seed} row {row_index}: replay-bin match != 1",
                )
                require(
                    math.isclose(
                        row_values["query/normalized_weight_mean"],
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    ),
                    f"seed {seed} row {row_index}: normalized weight mean != 1",
                )
                require(
                    row_values["query/raw_weight_mean"] > 0.0
                    and row_values["query/normalized_weight_mean"] > 0.0
                    and 0.0
                    <= row_values["query/high_weight_count"]
                    <= REQUIRED_CONFIG["batch_size"],
                    f"seed {seed} row {row_index}: invalid weight diagnostic",
                )
                for field in (
                    "query/context_count_eligible_rate",
                    "query/support_positive_rate",
                    "query/query_bin_present_given_context_rate",
                    "query/query_bin_matches_current_policy_rate",
                    "query/query_bin_matches_replay_action_rate",
                ):
                    require(
                        0.0 <= row_values[field] <= 1.0,
                        f"seed {seed} row {row_index}: {field} outside [0,1]",
                    )
                for field in QUERY_FIELDS:
                    values[field].append(row_values[field])
            raw_eval = row.get(analysis.PRIMARY_SUCCESS_FIELD, "")
            if raw_eval not in ("", None) and str(raw_eval).lower() != "nan":
                for field in CRITICAL_EVAL_FIELDS:
                    finite(
                        f"seed {seed} eval step {step}/{field}",
                        row.get(field),
                    )
                success = finite(
                    f"seed {seed} eval step {step}/{analysis.PRIMARY_SUCCESS_FIELD}",
                    raw_eval,
                )
                require(0.0 <= success <= 1.0, "evaluation success outside [0,1]")
                eval_steps.append(step)
                eval_values.append(success)
                selected.append(row)
    require(post_warmup_rows > 0, f"seed {seed}: no post-warmup rows")
    require(
        eval_steps == list(analysis.EVAL_STEPS),
        f"seed {seed}: exact evaluation grid mismatch",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    field_summary = {
        field: {
            "count": len(items),
            "mean": math.fsum(items) / len(items),
            "min": min(items),
            "max": max(items),
        }
        for field, items in values.items()
    }
    diagnostic = {
        "schema_version": 1,
        "seed": seed,
        "post_warmup_update_rows": post_warmup_rows,
        "support_query_mode": "replay_action",
        "support_assignment_mode_serialized": None,
        "effective_support_assignment_mode": "support",
        "effective_assignment_basis": (
            "executed legacy source directly applies the canonical support "
            "membership mask; it has no randomized assignment layer"
        ),
        "sampled_transition_removed_from_local_pool": False,
        "self_support_possible": True,
        "fields": field_summary,
        "invariants": {
            "all_required_fields_finite": True,
            "replay_action_bin_match_all_one": (
                field_summary["query/query_bin_matches_replay_action_rate"]["min"]
                == 1.0
                and field_summary[
                    "query/query_bin_matches_replay_action_rate"
                ]["max"]
                == 1.0
            ),
            "normalized_weight_mean_all_one_within_1e-6": max(
                abs(item - 1.0)
                for item in values["query/normalized_weight_mean"]
            )
            <= 1e-6,
        },
    }
    return diagnostic, eval_steps, eval_values


def validate_summary(
    summary: Mapping[str, object],
    seed: int,
    steps: Sequence[int],
    values: Sequence[float],
) -> dict[str, object]:
    expected = {
        "method": "b3_support_runtime_opt",
        "support_query_mode": "replay_action",
        "total_steps": 2_000_000,
        "total_updates": 1_995_001,
        "final_replay_size": 300_000,
    }
    for key, wanted in expected.items():
        require(
            summary.get(key) == wanted,
            f"seed {seed}: summary.{key} mismatch",
        )
    require(
        "support_assignment_mode" not in summary,
        f"seed {seed}: unexpected serialized support_assignment_mode",
    )
    auc = analysis.normalized_auc(steps, values)
    final = values[-1]
    best = max(values)
    hit90 = analysis.first_hit90(steps, values)
    require(
        close(summary.get("eval_AUC_success"), auc, 1e-10)
        and close(summary.get("final_eval_env_success"), final, 1e-12)
        and close(summary.get("best_eval_env_success"), best, 1e-12),
        f"seed {seed}: summary disagrees with exact evaluation grid",
    )
    native_hit = summary.get("first_hit_0.90")
    native_hit = None if native_hit in (None, "") else int(native_hit)
    require(native_hit == hit90, f"seed {seed}: summary Hit90 mismatch")
    losses = summary.get("final_losses")
    require(isinstance(losses, dict) and losses, f"seed {seed}: final_losses missing")
    for key, value in losses.items():
        finite(f"seed {seed}: final_losses.{key}", value)
    lite = summary.get("final_lite_metrics")
    require(isinstance(lite, dict), f"seed {seed}: final_lite_metrics missing")
    for field in QUERY_FIELDS:
        finite(f"seed {seed}: final_lite_metrics.{field}", lite.get(field))
    require(
        close(
            lite.get("query/query_bin_matches_replay_action_rate"),
            1.0,
            1e-12,
        )
        and close(lite.get("query/normalized_weight_mean"), 1.0, 1e-6),
        f"seed {seed}: final query invariants failed",
    )
    return {
        "auc": auc,
        "final_success": final,
        "best_success": best,
        "hit90": hit90,
        "summary_matches_exact_grid": True,
    }


def validate_checkpoint(
    path: Path,
    config: Mapping[str, object],
    seed: int,
) -> dict[str, object]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for checkpoint audit") from exc
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict), f"seed {seed}: checkpoint is not a mapping")
    require(
        EXPECTED_CHECKPOINT_KEYS.issubset(payload),
        f"seed {seed}: checkpoint missing available-schema keys "
        f"{sorted(EXPECTED_CHECKPOINT_KEYS - set(payload))}",
    )
    require(
        int(payload.get("global_step", -1)) == 2_000_000
        and int(payload.get("total_updates", -1)) == 1_995_001,
        f"seed {seed}: checkpoint step/update mismatch",
    )
    replay = payload.get("replay")
    require(
        isinstance(replay, dict) and int(replay.get("size", -1)) == 300_000,
        f"seed {seed}: checkpoint replay size mismatch",
    )
    saved_config = payload.get("config")
    checkpoint_comparable_config = {
        key: value
        for key, value in config.items()
        if key not in {"terminal_convention", "thread_hint"}
    }
    require(
        isinstance(saved_config, dict)
        and clean_json(saved_config) == clean_json(checkpoint_comparable_config),
        f"seed {seed}: checkpoint config does not match config.json",
    )
    require(
        saved_config.get("support_query_mode") == "replay_action"
        and "support_assignment_mode" not in saved_config,
        f"seed {seed}: checkpoint query/assignment serialization mismatch",
    )
    agent = payload.get("agent")
    require(
        isinstance(agent, dict)
        and {"actor", "actor_target", "critic", "critic_target"}.issubset(agent),
        f"seed {seed}: checkpoint is not DDPG actor/single-critic",
    )
    require(
        not any(
            key in agent
            for key in ("critic2", "critic2_target", "critic_2", "critic_target_2")
        ),
        f"seed {seed}: unexpected second critic",
    )
    missing_full = [key for key in FULL_RESUME_FIELDS_NOT_SERIALIZED if key not in payload]
    require(
        set(missing_full) == set(FULL_RESUME_FIELDS_NOT_SERIALIZED),
        f"seed {seed}: legacy checkpoint schema unexpectedly changed",
    )
    return {
        "schema": "legacy_drawerclose_query_checkpoint",
        "available_field_audit_passed": True,
        "global_step": 2_000_000,
        "total_updates": 1_995_001,
        "replay_size": 300_000,
        "single_critic_ddpg": True,
        "support_query_mode_preserved": True,
        "available_semantic_config_match": True,
        "config_json_derived_fields_not_in_checkpoint": [
            "terminal_convention",
            "thread_hint",
        ],
        "support_assignment_mode_serialized": None,
        "effective_support_assignment_mode": "support",
        "serialized_rng_states_present": [
            "noise_rng_state",
            "sample_rng_state",
            "weight_rng_state",
        ],
        "resume_reproducibility_status": "not_empirically_audited",
        "resume_reproducibility_claimed": False,
        "full_resume_fields_not_serialized": list(
            FULL_RESUME_FIELDS_NOT_SERIALIZED
        ),
    }


def scan_log(path: Path, finish_marker: str) -> dict[str, object]:
    require(path.is_file() and path.stat().st_size > 0, f"missing/empty log: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    for marker in FATAL_LOG_MARKERS:
        require(marker not in text, f"fatal marker {marker!r} in {path}")
    require(finish_marker in text, f"missing finish marker in {path}")
    return {
        "passed": True,
        "finish_marker": finish_marker,
        "fatal_marker_count": 0,
    }


def launcher_arguments(text: str, function_name: str = "b3_args") -> dict[str, tuple[str, ...]]:
    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.strip() == f"{function_name}() {{"),
        None,
    )
    require(start is not None, f"launcher missing {function_name}()")
    args: dict[str, tuple[str, ...]] = {}
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped == "}":
            break
        if not stripped.startswith("--"):
            continue
        if stripped.endswith("\\"):
            stripped = stripped[:-1].rstrip()
        tokens = shlex.split(stripped)
        require(tokens and tokens[0].startswith("--"), "malformed launcher option")
        require(tokens[0] not in args, f"duplicate launcher option {tokens[0]}")
        args[tokens[0]] = tuple(tokens[1:])
    return args


def validate_launcher_parity(source_repo: Path) -> dict[str, object]:
    main_bytes = run_git(
        source_repo,
        "show",
        f"{EXPECTED_SOURCE_COMMIT}:{MAIN_LAUNCHER_REL.as_posix()}",
        text=False,
    )
    replay_bytes = run_git(
        source_repo,
        "show",
        f"{EXPECTED_SOURCE_COMMIT}:{REPLAY_LAUNCHER_REL.as_posix()}",
        text=False,
    )
    assert isinstance(main_bytes, bytes) and isinstance(replay_bytes, bytes)
    main_text = main_bytes.decode("utf-8")
    replay_text = replay_bytes.decode("utf-8")
    main_args = launcher_arguments(main_text)
    replay_args = launcher_arguments(replay_text)
    main_args["--support_query_mode"] = ("current_policy",)
    differences = {
        key: {"main": list(main_args.get(key, ())), "replay": list(replay_args.get(key, ()))}
        for key in sorted(set(main_args) | set(replay_args))
        if main_args.get(key) != replay_args.get(key)
    }
    require(
        differences
        == {
            "--support_query_mode": {
                "main": ["current_policy"],
                "replay": ["replay_action"],
            }
        },
        f"launcher scientific argument parity failed: {differences}",
    )
    invocation = re.search(
        r'run_variant\s+"[^"]+"\s+"[^"]+"\s+"([^"]+)"\s+"([^"]+)"',
        replay_text,
    )
    require(
        invocation is not None
        and invocation.group(1) == "b3_support_runtime_opt"
        and invocation.group(2) == "critic_only",
        "Replay launcher method/scope invocation mismatch",
    )
    patch = "".join(
        difflib.unified_diff(
            main_text.splitlines(keepends=True),
            replay_text.splitlines(keepends=True),
            fromfile=MAIN_LAUNCHER_REL.as_posix(),
            tofile=REPLAY_LAUNCHER_REL.as_posix(),
        )
    )
    patch = "\n".join(line.rstrip() for line in patch.splitlines()) + "\n"
    return {
        "schema_version": 1,
        "passed": True,
        "comparison_type": (
            "executed-source canonical launcher parity; historical Main run "
            "config files were not preserved"
        ),
        "historical_main_run_config_available": False,
        "scientific_argument_differences": differences,
        "allowed_scientific_difference": "support_query_mode",
        "method": "b3_support_runtime_opt",
        "weight_scope": "critic_only",
        "infrastructure_text_differences_present": True,
        "infrastructure_text_difference_description": (
            "experiment-directory repository-root discovery and run/log labels"
        ),
        "unified_diff": patch,
    }


def freeze_source(
    source_repo: Path,
    stage: Path,
    input_roots: Sequence[tuple[Path, str]],
    input_rows: list[dict[str, object]],
) -> dict[str, object]:
    head = str(run_git(source_repo, "rev-parse", "HEAD")).strip()
    require(head == EXPECTED_SOURCE_COMMIT, f"source HEAD {head} != expected commit")
    source_tree = str(
        run_git(
            source_repo,
            "rev-parse",
            f"{head}:{SOURCE_REL.as_posix()}",
        )
    ).strip()
    require(source_tree == EXPECTED_SOURCE_TREE, "execution source tree OID mismatch")
    for args in (
        ("diff", "--quiet", head, "--", SOURCE_REL.as_posix()),
        ("diff", "--cached", "--quiet", head, "--", SOURCE_REL.as_posix()),
    ):
        completed = subprocess.run(["git", *args], cwd=source_repo, check=False)
        require(completed.returncode == 0, "tracked execution source is dirty")
    listing = str(
        run_git(
            source_repo,
            "ls-tree",
            "-r",
            head,
            "--",
            SOURCE_REL.as_posix(),
        )
    ).splitlines()
    require(len(listing) == 24, f"expected 24 frozen source files, got {len(listing)}")
    rows: list[dict[str, object]] = []
    checksum_lines: list[str] = []
    prefix = SOURCE_REL.as_posix() + "/"
    for line in listing:
        metadata, repo_path = line.split("\t", 1)
        mode, object_type, oid = metadata.split()
        require(object_type == "blob" and repo_path.startswith(prefix), "bad git tree row")
        relative = repo_path[len(prefix) :]
        committed = run_git(source_repo, "show", f"{head}:{repo_path}", text=False)
        assert isinstance(committed, bytes)
        working = source_repo / repo_path
        require(working.read_bytes() == committed, f"working source mismatch: {repo_path}")
        destination = stage / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(committed)
        if mode == "100755":
            os.chmod(destination, 0o755)
        digest = sha256_bytes(committed)
        rows.append(
            {
                "path": relative,
                "repo_path": repo_path,
                "mode": mode,
                "size_bytes": len(committed),
                "sha256": digest,
                "git_blob_oid": oid,
                "head_bound": True,
            }
        )
        checksum_lines.append(f"{digest}  source_snapshot/{relative}")
        input_rows.append(
            input_record(
                working,
                f"frozen.source.{relative}",
                input_roots,
                included_in_artifact=True,
                source_commit=head,
                git_blob_oid=oid,
            )
        )
    (stage / "frozen_source.sha256").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )
    launcher = next(
        row
        for row in rows
        if row["repo_path"] == REPLAY_LAUNCHER_REL.as_posix()
    )
    return {
        "schema_version": 1,
        "source_commit": head,
        "source_tree_oid": source_tree,
        "source_root": SOURCE_REL.as_posix(),
        "file_count": len(rows),
        "tracked_source_clean": True,
        "head_binding_verified": True,
        "entrypoint": launcher,
        "files": rows,
    }


def reconstructed_config(
    method: str,
    seed: int,
    proxy: Mapping[str, object],
    proxy_sha: str,
) -> dict[str, object]:
    keys = (
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
        "seeds",
    )
    return {
        "schema_version": 1,
        "record_type": "reconstructed_metadata_not_original_run_config",
        "not_original_run_config": True,
        "method_key": method,
        "seed": seed,
        "reason": (
            "Verified comparator compact data preserves exact HER/Main metrics "
            "and summaries, but not their original configs."
        ),
        "proxy_source": "same-seed DrawerClose Strict config",
        "proxy_source_sha256": proxy_sha,
        "reconstructed_shared_fields": {
            key: proxy[key] for key in keys if key in proxy
        },
        "method_specific_training_fields_not_reconstructed": True,
    }


def proxy_environment(
    method: str,
    seed: int,
    snapshot: Mapping[str, object],
    proxy_sha: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "same_seed_environment_proxy_not_original_snapshot",
        "not_original_run_env_snapshot": True,
        "method_key": method,
        "seed": seed,
        "reason": (
            "Comparator compact data omits original HER/Main env snapshots; "
            "this is an explicitly labelled same-seed DrawerClose Strict proxy."
        ),
        "proxy_source_sha256": proxy_sha,
        "proxy_snapshot": snapshot,
    }


def copy_comparator_cohorts(
    comparator: Path,
    stage: Path,
    input_roots: Sequence[tuple[Path, str]],
    input_rows: list[dict[str, object]],
    replacements: Sequence[tuple[str, str]],
) -> dict[str, object]:
    manifest = verify_manifest(comparator)
    require(
        manifest.get("artifact_id") == COMPARATOR_ARTIFACT_ID
        and manifest.get("source_commit") == COMPARATOR_PACKAGING_SOURCE_COMMIT,
        "wrong DrawerClose comparator artifact",
    )
    upstream_path = comparator / "input_manifest.json"
    upstream = read_json(upstream_path)
    require(
        upstream.get("source_commit") == COMPARATOR_PACKAGING_SOURCE_COMMIT,
        "comparator input-manifest source mismatch",
    )
    upstream_rows = upstream.get("inputs")
    require(isinstance(upstream_rows, list), "comparator input manifest malformed")
    upstream_index = {
        str(row.get("role", "")): row
        for row in upstream_rows
        if isinstance(row, dict)
    }
    for method in ("her", "main"):
        for seed in analysis.SEEDS:
            source_dir = comparator / "data" / method / f"seed_{seed}"
            target = stage / "data" / method / f"seed_{seed}"
            target.mkdir(parents=True, exist_ok=True)
            for name, suffix in (
                ("metrics_selected_10k.csv", "metrics"),
                ("summary.json", "summary"),
            ):
                source = source_dir / name
                require(source.is_file(), f"comparator missing {source}")
                role = f"{method}.seed_{seed}.{suffix}"
                upstream_row = upstream_index.get(role)
                require(
                    isinstance(upstream_row, dict),
                    f"comparator upstream provenance missing {role}",
                )
                if name.endswith(".json"):
                    write_json(
                        target / name,
                        clean_json(read_json(source), replacements),
                    )
                else:
                    (target / name).write_bytes(
                        source.read_bytes().replace(b"\r\n", b"\n")
                    )
                input_rows.append(
                    input_record(
                        source,
                        role,
                        input_roots,
                        included_in_artifact=True,
                        comparator_artifact_commit=COMPARATOR_ARTIFACT_COMMIT,
                        packaging_artifact_source_commit=(
                            COMPARATOR_PACKAGING_SOURCE_COMMIT
                        ),
                        original_training_source_commit=None,
                        original_training_source_commit_status=(
                            "unknown_not_recorded"
                        ),
                        upstream_input_provenance=upstream_row,
                    )
                )
            strict_dir = comparator / "data" / "strict" / f"seed_{seed}"
            proxy_config_path = strict_dir / "config.json"
            proxy_env_path = strict_dir / "env_snapshot.json"
            proxy_config = read_json(proxy_config_path)
            proxy_env = clean_json(read_json(proxy_env_path), replacements)
            write_json(
                target / "config.json",
                reconstructed_config(
                    method,
                    seed,
                    proxy_config,
                    sha256(proxy_config_path),
                ),
            )
            write_json(
                target / "env_snapshot.json",
                proxy_environment(
                    method,
                    seed,
                    proxy_env,
                    sha256(proxy_env_path),
                ),
            )
            for source, suffix in (
                (proxy_config_path, "reconstructed_config_proxy"),
                (proxy_env_path, "environment_proxy"),
            ):
                input_rows.append(
                    input_record(
                        source,
                        f"{method}.seed_{seed}.{suffix}",
                        input_roots,
                        included_in_artifact=True,
                        comparator_artifact_commit=COMPARATOR_ARTIFACT_COMMIT,
                        packaging_artifact_source_commit=(
                            COMPARATOR_PACKAGING_SOURCE_COMMIT
                        ),
                        not_original_comparator_record=True,
                    )
                )
    provenance_target = stage / "provenance" / "comparator_input_manifest.json"
    write_json(provenance_target, clean_json(upstream, replacements))
    input_rows.append(
        input_record(
            upstream_path,
            "provenance.comparator_input_manifest",
            input_roots,
            included_in_artifact=True,
            comparator_artifact_commit=COMPARATOR_ARTIFACT_COMMIT,
            packaging_artifact_source_commit=COMPARATOR_PACKAGING_SOURCE_COMMIT,
        )
    )
    input_rows.append(
        input_record(
            comparator / "artifact_manifest.json",
            "provenance.comparator_artifact_manifest",
            input_roots,
            included_in_artifact=False,
            comparator_artifact_commit=COMPARATOR_ARTIFACT_COMMIT,
        )
    )
    return {
        "artifact_id": COMPARATOR_ARTIFACT_ID,
        "artifact_git_commit": COMPARATOR_ARTIFACT_COMMIT,
        "artifact_manifest_sha256": sha256(comparator / "artifact_manifest.json"),
        "packaging_artifact_source_commit": COMPARATOR_PACKAGING_SOURCE_COMMIT,
        "original_training_source_commit": None,
        "original_training_source_commit_status": "unknown_not_recorded",
        "arms_reused": ["her", "main"],
        "retrained": False,
        "upstream_input_manifest_path": (
            "provenance/comparator_input_manifest.json"
        ),
        "upstream_input_manifest_sha256": sha256(upstream_path),
        "original_comparator_config_env_available": False,
        "metadata_policy": (
            "reconstructed config metadata and explicitly labelled same-seed "
            "DrawerClose Strict environment proxy"
        ),
    }


def validate_replay_runs(
    source_repo: Path,
    replay_root: Path,
    stage: Path,
    input_roots: Sequence[tuple[Path, str]],
    input_rows: list[dict[str, object]],
    replacements: Sequence[tuple[str, str]],
) -> dict[str, object]:
    runs = discover_replay_runs(replay_root)
    experiment_root = source_repo / SOURCE_REL
    log_root = experiment_root / "logs_2m" / EXPERIMENT
    records: list[dict[str, object]] = []
    for seed, run_dir in runs.items():
        run_name = f"{RUN_PREFIX}_s{seed}_2m_cuda"
        required = (
            "config.json",
            "env_snapshot.json",
            "summary.json",
            "metrics.csv",
            "checkpoint.pt",
        )
        for name in required:
            require(
                (run_dir / name).is_file() and (run_dir / name).stat().st_size > 0,
                f"seed {seed}: missing/empty {name}",
            )
        require(
            not list(run_dir.glob("checkpoint_step_*.pt")),
            f"seed {seed}: step checkpoints must be disabled",
        )
        config = read_json(run_dir / "config.json")
        summary = read_json(run_dir / "summary.json")
        snapshot = read_json(run_dir / "env_snapshot.json")
        validate_config(config, seed, run_name)
        validate_env(snapshot, seed)
        target = stage / "data" / "replay" / f"seed_{seed}"
        diagnostic, steps, values = validate_and_select_metrics(
            run_dir / "metrics.csv",
            target / "metrics_selected_10k.csv",
            seed,
            int(config["warmup_steps"]),
        )
        summary_gate = validate_summary(summary, seed, steps, values)
        checkpoint_gate = validate_checkpoint(
            run_dir / "checkpoint.pt",
            config,
            seed,
        )
        write_json(target / "config.json", clean_json(config, replacements))
        write_json(
            target / "env_snapshot.json",
            clean_json(snapshot, replacements),
        )
        write_json(target / "summary.json", clean_json(summary, replacements))
        write_json(target / "query_diagnostic_summary.json", diagnostic)
        external_log = log_root / f"{run_name}.log"
        log_gate = scan_log(external_log, "finished run at")
        record = {
            "seed": seed,
            "run_dir": f"data/replay/seed_{seed}",
            "passed": True,
            "total_steps": 2_000_000,
            "total_updates": 1_995_001,
            "final_replay_size": 300_000,
            "evaluation_points": 200,
            "evaluation_episodes_per_point": 20,
            "single_critic_ddpg": True,
            "support_query_mode": "replay_action",
            "support_assignment_mode_serialized": None,
            "effective_support_assignment_mode": "support",
            "weight_scope": "critic_only",
            "summary": summary_gate,
            "query_diagnostics": diagnostic,
            "checkpoint_available_field_audit": checkpoint_gate,
            "environment_snapshot_validated": True,
            "external_log_audit": log_gate,
        }
        write_json(target / "completion_validation.json", record)
        records.append(record)
        for name, included in (
            ("config.json", True),
            ("env_snapshot.json", True),
            ("summary.json", True),
            ("metrics.csv", False),
            ("checkpoint.pt", False),
        ):
            input_rows.append(
                input_record(
                    run_dir / name,
                    f"replay.seed_{seed}."
                    + (
                        "completion_checkpoint_excluded"
                        if name == "checkpoint.pt"
                        else "metrics"
                        if name == "metrics.csv"
                        else name.removesuffix(".json")
                    ),
                    input_roots,
                    included_in_artifact=included and name != "metrics.csv",
                    source_commit=EXPECTED_SOURCE_COMMIT,
                    compact_transform=(
                        "exact 10k evaluation rows only"
                        if name == "metrics.csv"
                        else None
                    ),
                )
            )
        best_checkpoint = run_dir / "best_checkpoint.pt"
        if best_checkpoint.is_file():
            input_rows.append(
                input_record(
                    best_checkpoint,
                    f"replay.seed_{seed}.best_checkpoint_excluded",
                    input_roots,
                    included_in_artifact=False,
                    source_commit=EXPECTED_SOURCE_COMMIT,
                )
            )
        input_rows.append(
            input_record(
                external_log,
                f"replay.seed_{seed}.external_log_excluded",
                input_roots,
                included_in_artifact=False,
                source_commit=EXPECTED_SOURCE_COMMIT,
                validation=log_gate,
            )
        )
    orchestrator_logs = sorted((experiment_root / "orchestrator_logs").glob(f"{EXPERIMENT}_*.log"))
    require(len(orchestrator_logs) == 1, "expected one Replay orchestrator log")
    orchestrator_gate = scan_log(
        orchestrator_logs[0],
        f"{EXPERIMENT} completed successfully",
    )
    input_rows.append(
        input_record(
            orchestrator_logs[0],
            "replay.orchestrator_log_excluded",
            input_roots,
            included_in_artifact=False,
            source_commit=EXPECTED_SOURCE_COMMIT,
            validation=orchestrator_gate,
        )
    )
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "passed": True,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "seed_count": 5,
        "seeds": list(analysis.SEEDS),
        "evaluation_grid": list(analysis.EVAL_STEPS),
        "evaluation_field": analysis.PRIMARY_SUCCESS_FIELD,
        "query_contract": {
            "support_query_mode": "replay_action",
            "support_assignment_mode_serialized": None,
            "effective_support_assignment_mode": "support",
            "effective_assignment_basis": (
                "canonical membership mask is applied directly by the frozen "
                "source; no assignment-control layer exists"
            ),
            "weight_scope": "critic_only",
            "replay_action_bin_match": 1.0,
            "sampled_transition_removed_from_local_pool": False,
            "self_support_possible": True,
        },
        "resume_contract": {
            "status": "not_empirically_audited",
            "claim_made": False,
            "reason": (
                "legacy execution checkpoints contain noise/sample/weight RNG "
                "states but not the full modern resume-audit schema"
            ),
            "full_resume_fields_not_serialized": list(
                FULL_RESUME_FIELDS_NOT_SERIALIZED
            ),
        },
        "orchestrator_log_audit": orchestrator_gate,
        "records": records,
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


def readme_text(
    result: Mapping[str, object],
    source_commit: str,
) -> str:
    aggregate = {
        str(row["method_key"]): row
        for row in result["aggregate"]
        if isinstance(row, dict)
    }
    primary = next(
        row
        for row in result["paired_comparisons"]
        if isinstance(row, dict) and row["contrast_key"] == "main_minus_replay"
    )
    lines = [
        f"# {EXPERIMENT}",
        "",
        (
            "Compact, independently verifiable artifact for DrawerClose seeds "
            "1–5. Exact success AUC uses only observed 10k–2M points, with no "
            "smoothing, interpolation, forward-fill, or extrapolation."
        ),
        "",
        (
            "Primary success field: `eval/env_success_mean` only; no fallback "
            "field is used."
        ),
        "",
        (
            "Replay-action query does not remove the sampled transition from "
            "the local replay pool. Consequently the replay action can "
            "self-support when that transition is present in the bounded local "
            "scan. This matches the intended FPP replay-query definition."
        ),
        "",
        (
            "Only the SABER membership query changes. Actor objective/update "
            "and the Bellman target are unchanged. Bellman regression remains "
            "at the replay action, and weights remain critic-only."
        ),
        "",
        "| Method | AUC mean ± sample SD | Final mean ± sample SD |",
        "|---|---:|---:|",
    ]
    for key in analysis.METHOD_ORDER:
        row = aggregate[key]
        lines.append(
            f"| {row['method']} | "
            f"{float(row['auc_mean']):.6f} ± "
            f"{float(row['auc_sample_sd']):.6f} | "
            f"{float(row['final_success_mean']):.6f} ± "
            f"{float(row['final_success_sample_sd']):.6f} |"
        )
    lines += [
        "",
        (
            "Primary Main − Replay AUC difference: "
            f"{float(primary['mean_auc_difference']):.6f}; exhaustive "
            "paired-bootstrap 95% CI "
            f"[{float(primary['bootstrap_ci95_lower']):.6f}, "
            f"{float(primary['bootstrap_ci95_upper']):.6f}]; exact two-sided "
            "sign-flip "
            f"p={float(primary['exact_two_sided_sign_flip_pvalue']):.6f}."
        ),
        "",
        (
            "All five seeds and paired statistics are reported regardless of "
            "direction. If the CI includes zero, this artifact does not claim "
            "that an independent current-policy-query effect is established."
        ),
        "",
        (
            f"HER/Main were reused, not retrained, from verified artifact commit "
            f"`{COMPARATOR_ARTIFACT_COMMIT}` (packaging source "
            f"`{COMPARATOR_PACKAGING_SOURCE_COMMIT}`). Their original training "
            "source commit and original config/env snapshots were not recorded; "
            "reconstructed metadata and explicitly labelled same-seed Strict "
            "environment proxies are used."
        ),
        "",
        (
            "The legacy Replay run did not serialize "
            "`support_assignment_mode`. The artifact therefore records the "
            "serialized value as `null` and the source-derived effective mode "
            "as `support`: the canonical membership mask is directly applied "
            "without randomized assignment."
        ),
        "",
        (
            "Checkpoint available-field audits passed for all seeds. Full "
            "checkpoint/resume reproducibility is explicitly "
            "`not_empirically_audited`; this artifact makes no stronger claim."
        ),
        "",
        f"Frozen Replay execution source commit: `{source_commit}`.",
        "",
        "Verify from this directory with:",
        "",
        "    python verify_artifact.py --artifact-dir .",
        "",
        (
            "Checkpoints, best checkpoints, full raw metrics, JSONL, logs, "
            "smoke/correctness/pilot outputs, caches, and temporary files are "
            "excluded."
        ),
    ]
    return "\n".join(lines) + "\n"


def build(
    source_repo: Path,
    replay_root: Path,
    comparator: Path,
    output: Path,
) -> None:
    source_repo = source_repo.resolve()
    replay_root = replay_root.resolve()
    comparator = comparator.resolve()
    output = output.resolve()
    require((source_repo / ".git").exists(), f"not a source repository: {source_repo}")
    require(replay_root.is_dir(), f"missing Replay result root: {replay_root}")
    require(comparator.is_dir(), f"missing comparator artifact: {comparator}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{ARTIFACT_ID}.", dir=output.parent))
    input_roots = (
        (source_repo, "<execution_repo>"),
        (comparator, "<comparator_artifact>"),
    )
    replacements = (
        (source_repo.as_posix(), "."),
        (comparator.as_posix(), "<comparator_artifact>"),
        (output.parent.as_posix(), "."),
    )
    input_rows: list[dict[str, object]] = []
    try:
        freeze = freeze_source(source_repo, stage, input_roots, input_rows)
        write_json(stage / "provenance" / "freeze_manifest.json", freeze)
        launcher_parity = validate_launcher_parity(source_repo)
        write_json(
            stage / "provenance" / "config_parity.json",
            {
                **launcher_parity,
                "resolved_replay_config_contract": clean_json(REQUIRED_CONFIG),
                "resolved_replay_seed_coverage": list(analysis.SEEDS),
                "nonallowed_differences": [],
            },
        )
        (stage / "provenance" / "launcher_diff.patch").write_text(
            str(launcher_parity["unified_diff"]),
            encoding="utf-8",
        )
        comparator_info = copy_comparator_cohorts(
            comparator,
            stage,
            input_roots,
            input_rows,
            replacements,
        )
        completion = validate_replay_runs(
            source_repo,
            replay_root,
            stage,
            input_roots,
            input_rows,
            replacements,
        )
        write_json(stage / "completion_validation.json", completion)
        checkpoint_summary = {
            "schema_version": 1,
            "status": "passed_available_fields_only",
            "seed_coverage": list(analysis.SEEDS),
            "resume_reproducibility_status": "not_empirically_audited",
            "resume_reproducibility_claimed": False,
            "records": [
                {
                    "seed": record["seed"],
                    **record["checkpoint_available_field_audit"],
                }
                for record in completion["records"]
            ],
        }
        write_json(
            stage / "provenance" / "checkpoint_schema_audit.json",
            checkpoint_summary,
        )
        source_parity = {
            "schema_version": 1,
            "passed": True,
            "execution_source_commit": EXPECTED_SOURCE_COMMIT,
            "execution_source_tree_oid": EXPECTED_SOURCE_TREE,
            "tracked_source_clean": True,
            "source_file_count": freeze["file_count"],
            "current_execution_worktree_matches_commit": True,
        }
        write_json(
            stage / "provenance" / "source_parity.json",
            source_parity,
        )
        for name in ("analyze_results.py", "build_artifact.py", "verify_artifact.py"):
            source = TOOL_DIR / name
            require(source.is_file(), f"missing artifact tool: {source}")
            shutil.copyfile(source, stage / name)
        completed = subprocess.run(
            [
                sys.executable,
                str(stage / "analyze_results.py"),
                "--her-root",
                str(stage / "data" / "her"),
                "--main-root",
                str(stage / "data" / "main"),
                "--replay-root",
                str(stage / "data" / "replay"),
                "--output-dir",
                str(stage),
            ],
            cwd=stage,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        require(
            completed.returncode == 0,
            "analyze_results.py failed:\n"
            + completed.stdout
            + "\n"
            + completed.stderr,
        )
        result = read_json(stage / "analysis_results.json")
        write_json(
            stage / "input_manifest.json",
            {
                "schema_version": 1,
                "artifact_id": ARTIFACT_ID,
                "source_commit": EXPECTED_SOURCE_COMMIT,
                "source_tree_oid": EXPECTED_SOURCE_TREE,
                "comparator": comparator_info,
                "input_file_count": len(input_rows),
                "inputs": clean_json(input_rows, replacements),
                "transform": {
                    "metrics": "exact observed evaluation rows 10k..2M",
                    "json": "machine-local roots sanitized; NaN/Inf replaced by null",
                    "checkpoints_copied": False,
                    "best_checkpoints_copied": False,
                    "raw_metrics_copied": False,
                    "jsonl_copied": False,
                    "logs_copied": False,
                    "smoke_or_pilot_copied": False,
                },
            },
        )
        (stage / "README.md").write_text(
            readme_text(result, EXPECTED_SOURCE_COMMIT),
            encoding="utf-8",
        )
        records = packaged_files(stage)
        manifest = {
            "schema_version": 1,
            "artifact_id": ARTIFACT_ID,
            "experiment": EXPERIMENT,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": EXPECTED_SOURCE_COMMIT,
            "source_tree_oid": EXPECTED_SOURCE_TREE,
            "comparator_artifact_commit": COMPARATOR_ARTIFACT_COMMIT,
            "comparator_packaging_source_commit": (
                COMPARATOR_PACKAGING_SOURCE_COMMIT
            ),
            "method_seed_coverage": {
                key: list(analysis.SEEDS) for key in analysis.METHOD_ORDER
            },
            "evaluation": {
                "field": analysis.PRIMARY_SUCCESS_FIELD,
                "steps": list(analysis.EVAL_STEPS),
                "points_per_seed": 200,
                "episodes_per_point": 20,
                "auc": "observed-only normalized trapezoid",
                "smoothing": False,
                "interpolation": False,
                "forward_fill": False,
                "extrapolation": False,
            },
            "bootstrap": {
                "method": "ordered exhaustive paired bootstrap",
                "resamples": 3125,
                "confidence_level": 0.95,
            },
            "exact_sign_flip": {"two_sided": True, "assignments": 32},
            "support_assignment_mode_serialized": None,
            "effective_support_assignment_mode": "support",
            "resume_reproducibility_status": "not_empirically_audited",
            "completion_passed": True,
            "config_parity_passed": True,
            "source_parity_passed": True,
            "packaged_file_count_excluding_manifest": len(records),
            "package_content_sha256_excluding_manifest": content_digest(records),
            "packaged_files": records,
            "self_hash_note": (
                "artifact_manifest.json is excluded to avoid recursive self-hash"
            ),
        }
        write_json(stage / "artifact_manifest.json", manifest)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--comparator-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build(
        args.source_repo,
        args.replay_root,
        args.comparator_artifact,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": "built",
                "artifact": args.output.resolve().as_posix(),
                "source_commit": EXPECTED_SOURCE_COMMIT,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
