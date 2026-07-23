"""Build the DrawerOpen DDPG SABER 10-seed diagnostic paper artifact.

This builder validates every source before writing compact data.  Expected
regression values are used only as post-computation checks; they are never
written in place of computed values.
"""

from __future__ import annotations

import csv
import getpass
import gzip
import hashlib
import io
import json
import math
import re
import socket
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import rebuild_results


ARTIFACT_ID = "DRAWEROPEN-DDPG-SABER-DIAGNOSTIC-10SEED-2M"
SEEDS = tuple(range(1, 11))
EXPECTED_EVAL_STEPS = tuple(range(10_000, 2_000_001, 10_000))
EXPECTED_DIAGNOSTIC_SNAPSHOTS = 9_976

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
DRAWER_ROOT = ARTIFACT_ROOT.parents[1]
REPO_ROOT = ARTIFACT_ROOT.parents[2]

DIAGNOSTIC_RERUN_ROOT = DRAWER_ROOT / "results_2m_diag" / "DDPG-HER-B3-Critic-Only-2M-diagnostic"
EXTENSION_ROOT = DRAWER_ROOT / "results_2m" / "DDPG-HER-B3-Critic-Only-2M"
REFERENCE_ROOT = EXTENSION_ROOT

REQUIRED_DIAGNOSTIC_FILES = ("metrics.csv", "config.json", "env_snapshot.json", "summary.json")

EVAL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("step", "step"),
    ("eval/return_mean", "eval_return_mean"),
    ("eval/success_mean", "eval_success_mean"),
    ("eval/goal_success_mean", "eval_goal_success_mean"),
    ("eval/env_success_mean", "eval_env_success_mean"),
    ("eval/goal_success_ever_mean", "eval_goal_success_ever_mean"),
    ("eval/env_success_ever_mean", "eval_env_success_ever_mean"),
    ("eval/goal_success_final_mean", "eval_goal_success_final_mean"),
    ("eval/env_success_final_mean", "eval_env_success_final_mean"),
    ("eval/episode_length_mean", "eval_episode_length_mean"),
)

DIAGNOSTIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("step", "step"),
    ("episode", "episode"),
    ("train/her_relabel_ratio", "her_relabel_ratio"),
    ("loss/critic", "critic_loss"),
    ("loss/actor", "actor_loss"),
    ("loss/td_error_abs_mean", "td_error_abs_mean"),
    ("loss/td_error_abs_p90", "td_error_abs_p90"),
    ("value/q_mean", "q_mean"),
    ("value/target_mean", "target_mean"),
    ("lite/support_available_rate", "support_rate"),
    ("lite/ESS_w_normalized", "ess_normalized"),
    ("lite/psi_A_policy_rate", "psi_a_policy_rate"),
    ("lite/local_support_count_mean", "local_support_count_mean"),
    ("lite/local_support_count_p50", "local_support_count_p50"),
    ("lite/local_support_count_p90", "local_support_count_p90"),
    ("lite/mean_num_coarse_candidates", "num_coarse_candidates_mean"),
    ("lite/mean_num_scanned_candidates", "num_scanned_candidates_mean"),
    ("lite/local_scan_truncated_rate", "scan_truncated_rate"),
    ("diagnostic/td_error_support_pos_mean", "td_support_pos_mean"),
    ("diagnostic/td_error_support_neg_mean", "td_support_neg_mean"),
    ("diagnostic/td_error_enrichment_ratio", "td_enrichment_ratio"),
    ("diagnostic/residual_reduction_support_pos_mean", "residual_support_pos_mean"),
    ("diagnostic/residual_reduction_support_neg_mean", "residual_support_neg_mean"),
    ("diagnostic/residual_correction_enrichment_ratio", "residual_enrichment_ratio"),
    ("diagnostic/residual_reduction_support_pos_rate", "residual_positive_rate_pos"),
    ("diagnostic/residual_reduction_support_neg_rate", "residual_positive_rate_neg"),
    ("diagnostic/weighted_td_error_mean", "weighted_td_mean"),
    ("diagnostic/weighted_residual_reduction_mean", "weighted_residual_reduction_mean"),
    ("diagnostic/support_to_td_error_corr", "support_td_corr"),
    ("diagnostic/support_to_residual_reduction_corr", "support_residual_corr"),
    ("diagnostic/weight_to_td_error_corr", "weight_td_corr"),
    ("diagnostic/weight_to_residual_reduction_corr", "weight_residual_corr"),
    ("hand_handle_dist_mean", "hand_handle_dist_mean"),
    ("handle_goal_dist_mean", "handle_goal_dist_mean"),
    ("phase_approach_frac", "phase_approach_frac"),
    ("phase_contact_frac", "phase_contact_frac"),
    ("phase_opening_frac", "phase_opening_frac"),
    ("phase_open_frac", "phase_open_frac"),
    ("b3/support_by_phase_approach", "phase_approach_support_rate"),
    ("b3/support_by_phase_contact", "phase_contact_support_rate"),
    ("b3/support_by_phase_opening", "phase_opening_support_rate"),
    ("b3/support_by_phase_open", "phase_open_support_rate"),
    ("b3/mean_w_P_by_phase_approach", "phase_approach_mean_weight"),
    ("b3/mean_w_P_by_phase_contact", "phase_contact_mean_weight"),
    ("b3/mean_w_P_by_phase_opening", "phase_opening_mean_weight"),
    ("b3/mean_w_P_by_phase_open", "phase_open_mean_weight"),
    ("b3/fallback_by_phase_approach", "phase_approach_fallback_rate"),
    ("b3/fallback_by_phase_contact", "phase_contact_fallback_rate"),
    ("b3/fallback_by_phase_opening", "phase_opening_fallback_rate"),
    ("b3/fallback_by_phase_open", "phase_open_fallback_rate"),
    ("runtime/numba_support_scan_enabled", "numba_enabled"),
    ("runtime/numba_scanner_warmed", "numba_warmed"),
    ("runtime/numba_support_scan_fallback_rate", "numba_fallback_rate"),
    ("runtime/numba_support_queries", "numba_queries"),
    ("runtime/predicate_early_exit_rate", "predicate_early_exit_rate"),
    ("runtime/predicate_scanned_count_mean", "predicate_scanned_count_mean"),
    ("runtime/diagnostic_audit_rate", "audit_rate"),
    ("runtime/audit_len_A_Q_local_abs_diff_mean", "audit_local_count_abs_diff"),
    ("runtime/audit_len_A_Q_sup_abs_diff_mean", "audit_support_count_abs_diff"),
    ("runtime/audit_psi_A_policy_mismatch_rate", "audit_policy_support_mismatch"),
    ("runtime/support_cache_hit_rate", "support_cache_hit_rate"),
    ("runtime/local_scan_cache_hit_rate", "local_scan_cache_hit_rate"),
    ("runtime/full_context_scan_rate", "full_context_scan_rate"),
    ("runtime/exact_negative_short_circuit_rate", "exact_negative_short_circuit_rate"),
    ("time/local_context", "local_context_time_sec"),
    ("time/actor_critic_update", "actor_critic_update_time_sec"),
    ("time/numba_support_scan", "numba_scan_time_sec"),
)

CONFIG_ALLOWLIST = (
    "M_B", "M_Q_min", "N_gap_min", "N_local_min", "action_bins_per_dim", "actor_lr",
    "batch_size", "buffer_capacity", "checkpoint_every_steps", "critic_lr", "device",
    "drawer_d_goal_near", "drawer_d_opening_mid", "drawer_d_reach", "drawer_gripper_bins",
    "drawer_hand_handle_bins", "drawer_handle_goal_bins", "drawer_handle_y_bins",
    "drawer_horizon_bins", "drawer_success_threshold", "env_id", "eval_episodes", "eval_freq",
    "gamma", "her_ratio", "hidden_dim", "lambda_support", "local_scan_cache_enabled",
    "local_scan_cache_max_entries", "local_search_max", "log_every_steps", "max_episode_steps",
    "metaworld_env_name", "metaworld_reward_function_version", "method", "noise_std",
    "num_envs", "policy_delay", "policy_support_mode", "resolved_env_id", "reward_type",
    "save_step_checkpoints", "seeds", "start_steps", "success_terminates",
    "support_cache_enabled", "support_cache_max_entries", "support_scan_mode",
    "support_short_circuit", "target_noise_clip", "target_policy_noise", "task_name", "tau",
    "terminal_convention", "timeout_bootstrap", "total_steps", "updates_per_step",
    "warmup_steps", "weight_scope", "weighting_mode",
)

ENV_ALLOWLIST = (
    "action_dim", "action_high", "action_low", "desired_goal_source", "distance_threshold",
    "env_id", "env_name", "goal_dim", "max_episode_steps", "obs_dim", "reward_type", "state_dim",
)
ENV_SPEC_ALLOWLIST = ("env_name", "reward_function_version", "seed", "terminate_on_success")
TASK_SPEC_ALLOWLIST = (
    "coarse_index_key", "d_goal_near", "d_opening_mid", "d_reach", "epsilon_g",
    "full_context_key", "gripper_bins", "hand_handle_bins", "handle_goal_bins", "handle_y_bins",
    "horizon_bins", "observation_layout", "phase_definition", "phase_labels", "task",
)

SUMMARY_ALLOWLIST = (
    "best_eval_env_success", "best_eval_goal_success", "best_eval_return",
    "best_eval_selection_success", "best_eval_step", "best_eval_success", "episodes",
    "eval_AUC_success", "final_eval_env_success", "final_eval_env_success_ever",
    "final_eval_env_success_final", "final_eval_goal_success", "final_eval_goal_success_ever",
    "final_eval_goal_success_final", "final_eval_return", "final_eval_success", "final_replay_size",
    "final_steps_per_sec", "first_hit_0.25", "first_hit_0.50", "first_hit_0.75",
    "first_hit_0.90", "method", "runtime_total_sec", "selection_metric", "total_steps",
    "total_updates",
)
FINAL_LOSS_ALLOWLIST = (
    "actor", "critic", "q_abs_max", "q_abs_mean", "q_mean", "q_std", "target_mean",
    "target_std", "td_error_abs_mean", "td_error_abs_p90",
)
FINAL_LITE_ALLOWLIST = (
    "diagnostic/support_positive_rate", "diagnostic/td_error_support_pos_mean",
    "diagnostic/td_error_support_neg_mean", "diagnostic/residual_reduction_support_pos_mean",
    "diagnostic/residual_reduction_support_neg_mean", "diagnostic/support_to_td_error_corr",
    "diagnostic/support_to_residual_reduction_corr", "lite/ESS_w_normalized",
    "lite/local_scan_truncated_rate", "lite/support_available_rate",
    "runtime/numba_support_scan_enabled", "runtime/numba_scanner_warmed",
    "runtime/numba_support_scan_fallback_rate", "runtime/audit_psi_A_policy_mismatch_rate",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any, precision: int = 17) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if value == 0.0:
            return "0"
        return format(value, f".{precision}g")
    return str(value)


def _finite(value: Any) -> bool:
    if value is None or value == "":
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _number(value: Any, integer: bool = False) -> int | float | None:
    if not _finite(value):
        return None
    number = float(value)
    return int(number) if integer else number


def _json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _canonical(row.get(field)) for field in fieldnames})


def _write_deterministic_gzip_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _canonical(row.get(field)) for field in fieldnames})
    encoded = buffer.getvalue().encode("utf-8")
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=raw_handle, mtime=0) as gzip_handle:
            gzip_handle.write(encoded)


def _seed_from_name(name: str) -> int | None:
    match = re.search(r"_s(\d+)_", name)
    return int(match.group(1)) if match else None


def _discover_runs(root: Path) -> dict[int, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"source cohort directory is missing: {root}")
    discovered: dict[int, Path] = {}
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        seed = _seed_from_name(child.name)
        if seed is None:
            continue
        if seed in discovered:
            raise ValueError(f"multiple source run directories resolve to seed {seed}: {discovered[seed]} and {child}")
        discovered[seed] = child
    return discovered


def _validate_source_presence() -> tuple[dict[int, Path], dict[int, Path]]:
    rerun = _discover_runs(DIAGNOSTIC_RERUN_ROOT)
    extension_and_reference = _discover_runs(EXTENSION_ROOT)
    missing: list[str] = []
    diagnostic: dict[int, Path] = {}
    reference: dict[int, Path] = {}
    for seed in range(1, 6):
        if seed not in rerun:
            missing.append(str(DIAGNOSTIC_RERUN_ROOT / f"<seed-{seed}-diagnostic-run>"))
        else:
            diagnostic[seed] = rerun[seed]
        if seed not in extension_and_reference:
            missing.append(str(EXTENSION_ROOT / f"<seed-{seed}-non-diagnostic-reference-run>"))
        else:
            reference[seed] = extension_and_reference[seed]
    for seed in range(6, 11):
        if seed not in extension_and_reference:
            missing.append(str(EXTENSION_ROOT / f"<seed-{seed}-diagnostic-run>"))
        else:
            diagnostic[seed] = extension_and_reference[seed]
    for seed, run_dir in sorted(diagnostic.items()):
        for filename in REQUIRED_DIAGNOSTIC_FILES:
            if not (run_dir / filename).is_file():
                missing.append(str(run_dir / filename))
    for seed, run_dir in sorted(reference.items()):
        if not (run_dir / "metrics.csv").is_file():
            missing.append(str(run_dir / "metrics.csv"))
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"required source inputs are missing; artifact was not built:\n{joined}")
    if tuple(sorted(diagnostic)) != SEEDS:
        raise ValueError(f"diagnostic seed IDs must be exactly 1-10, got {tuple(sorted(diagnostic))}")
    return diagnostic, reference


def _source_entry(path: Path, seed: int, cohort: str, role: str) -> dict[str, Any]:
    return {
        "cohort": cohort,
        "role": role,
        "seed": seed,
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _collect_source_manifest(diagnostic: Mapping[int, Path], reference: Mapping[int, Path]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for seed, run_dir in sorted(diagnostic.items()):
        cohort = "diagnostic_rerun_seeds_1_5" if seed <= 5 else "diagnostic_extension_seeds_6_10"
        for filename in REQUIRED_DIAGNOSTIC_FILES:
            role = "raw_metrics" if filename == "metrics.csv" else filename.removesuffix(".json")
            entries.append(_source_entry(run_dir / filename, seed, cohort, role))
    for seed, run_dir in sorted(reference.items()):
        entries.append(_source_entry(run_dir / "metrics.csv", seed, "original_non_diagnostic_seeds_1_5", "performance_reference_raw_metrics"))
    entries.sort(key=lambda entry: (entry["cohort"], entry["seed"], entry["role"], entry["path"]))
    return {
        "artifact_id": ARTIFACT_ID,
        "schema_version": 1,
        "source_input_count": len(entries),
        "diagnostic_raw_metrics_count": 10,
        "performance_reference_raw_metrics_count": 5,
        "source_inputs": entries,
    }


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def _project(row: Mapping[str, str], mapping: Sequence[tuple[str, str]]) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for raw_name, compact_name in mapping:
        projected[compact_name] = _number(row.get(raw_name), integer=compact_name in {"step", "episode"})
    return projected


def _equal_projected(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[bool, list[str]]:
    differing = [key for key in left if left.get(key) != right.get(key)]
    return not differing, differing


def _deduplicate(
    grouped: Mapping[int, list[tuple[int, dict[str, Any]]]],
    seed: int,
    row_kind: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    duplicate_step_count = 0
    duplicate_row_count = 0
    for step in sorted(grouped):
        candidates = sorted(grouped[step], key=lambda item: item[0])
        if len(candidates) > 1:
            duplicate_step_count += 1
            duplicate_row_count += len(candidates) - 1
            reference = candidates[0][1]
            for source_row, candidate in candidates[1:]:
                equal, differing = _equal_projected(reference, candidate)
                if not equal:
                    raise ValueError(
                        f"seed {seed} {row_kind} duplicate step {step} differs in analysis fields {differing} "
                        f"at source row {source_row}"
                    )
        selected.append(candidates[-1][1])
    return selected, {
        "artifact_id": ARTIFACT_ID,
        "seed": seed,
        "row_kind": row_kind,
        "candidate_row_count": sum(len(rows) for rows in grouped.values()),
        "unique_step_count": len(selected),
        "duplicate_step_count": duplicate_step_count,
        "removed_duplicate_row_count": duplicate_row_count,
        "duplicate_analysis_values_identical": True,
        "selection_rule": "choose_last_raw_row_after_exact_analysis_field_equality_check",
    }


def _validate_eval(seed: int, rows: Sequence[Mapping[str, Any]], source: Path) -> None:
    steps = tuple(int(row["step"]) for row in rows)
    if steps != EXPECTED_EVAL_STEPS:
        missing = sorted(set(EXPECTED_EVAL_STEPS) - set(steps))
        extra = sorted(set(steps) - set(EXPECTED_EVAL_STEPS))
        raise ValueError(f"seed {seed} exact 10k evaluation grid mismatch in {source}; missing={missing}, extra={extra}")
    for row in rows:
        success = row["eval_success_mean"]
        if not isinstance(success, float) or not 0.0 <= success <= 1.0:
            raise ValueError(f"seed {seed} evaluation success is outside [0,1] at step {row['step']}: {success}")


def _validate_diagnostic(seed: int, rows: Sequence[Mapping[str, Any]], source: Path) -> None:
    if len(rows) != EXPECTED_DIAGNOSTIC_SNAPSHOTS:
        raise ValueError(
            f"seed {seed} diagnostic snapshot count mismatch in {source}: "
            f"expected {EXPECTED_DIAGNOSTIC_SNAPSHOTS}, got {len(rows)}"
        )
    unit_interval_fields = (
        "support_rate", "ess_normalized", "psi_a_policy_rate", "scan_truncated_rate",
        "residual_positive_rate_pos", "residual_positive_rate_neg", "phase_approach_frac",
        "phase_contact_frac", "phase_opening_frac", "phase_open_frac",
        "phase_approach_support_rate", "phase_contact_support_rate", "phase_opening_support_rate",
        "phase_open_support_rate", "numba_enabled", "numba_warmed", "numba_fallback_rate",
        "audit_policy_support_mismatch",
    )
    correlation_fields = ("support_td_corr", "support_residual_corr", "weight_td_corr", "weight_residual_corr")
    for row in rows:
        step = int(row["step"])
        phase_sum = sum(float(row[field]) for field in (
            "phase_approach_frac", "phase_contact_frac", "phase_opening_frac", "phase_open_frac"
        ))
        if abs(phase_sum - 1.0) > 1e-6:
            raise ValueError(f"seed {seed} phase fractions sum to {phase_sum} at diagnostic step {step}")
        for field in unit_interval_fields:
            value = row.get(field)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"seed {seed} {field} outside [0,1] at step {step}: {value}")
        for field in correlation_fields:
            value = row.get(field)
            if value is not None and not -1.0 <= float(value) <= 1.0:
                raise ValueError(f"seed {seed} {field} outside [-1,1] at step {step}: {value}")


def _extract_metrics(
    path: Path,
    seed: int,
    require_diagnostics: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    header = _read_header(path)
    required = {raw for raw, _compact in EVAL_COLUMNS}
    if require_diagnostics:
        required.update(raw for raw, _compact in DIAGNOSTIC_COLUMNS)
        required.add("diagnostic/support_positive_rate")
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"source metrics schema missing required columns in {path}: {missing}")

    eval_grouped: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    diagnostic_grouped: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for source_row, row in enumerate(csv.DictReader(handle), start=2):
            step = _number(row.get("step"), integer=True)
            if step is None:
                continue
            if _finite(row.get("eval/success_mean")):
                projected = _project(row, EVAL_COLUMNS)
                eval_grouped.setdefault(int(step), []).append((source_row, projected))
            if require_diagnostics and _finite(row.get("diagnostic/support_positive_rate")):
                projected = _project(row, DIAGNOSTIC_COLUMNS)
                diagnostic_grouped.setdefault(int(step), []).append((source_row, projected))

    eval_rows, eval_audit = _deduplicate(eval_grouped, seed, "evaluation")
    _validate_eval(seed, eval_rows, path)
    audits = [eval_audit]
    diagnostic_rows: list[dict[str, Any]] = []
    if require_diagnostics:
        diagnostic_rows, diagnostic_audit = _deduplicate(diagnostic_grouped, seed, "diagnostic")
        _validate_diagnostic(seed, diagnostic_rows, path)
        audits.append(diagnostic_audit)
    return eval_rows, diagnostic_rows, audits


def _cohort(seed: int) -> str:
    return "diagnostic_rerun_seeds_1_5" if seed <= 5 else "diagnostic_extension_seeds_6_10"


def _attach_identity(rows: Sequence[Mapping[str, Any]], seed: int, cohort: str) -> list[dict[str, Any]]:
    return [{"artifact_id": ARTIFACT_ID, "cohort": cohort, "seed": seed, **dict(row)} for row in rows]


def _pick(source: Mapping[str, Any], allowlist: Sequence[str]) -> dict[str, Any]:
    return {key: _json_safe(source[key]) for key in allowlist if key in source}


def _sanitize_config(path: Path, seed: int, cohort: str) -> dict[str, Any]:
    source = _json_load(path)
    seeds = source.get("seeds")
    if not isinstance(seeds, dict):
        raise ValueError(f"seed mapping missing from config: {path}")
    expected = {
        "env_seed": seed,
        "torch_seed": seed,
        "numpy_seed": seed,
        "action_noise_seed": seed,
        "weight_seed": 20_000 + seed,
        "eval_seed": 10_000 + seed,
    }
    for key, value in expected.items():
        if int(seeds.get(key, -1)) != value:
            raise ValueError(f"config seed identity mismatch for seed {seed}: {key}={seeds.get(key)} expected {value}")
    if int(source.get("total_steps", -1)) != 2_000_000:
        raise ValueError(f"config total_steps mismatch for seed {seed}: {source.get('total_steps')}")
    if source.get("method") != "b3_support_runtime_opt" or source.get("weight_scope") != "critic_only":
        raise ValueError(f"unexpected method/weight scope for seed {seed}: {source.get('method')}/{source.get('weight_scope')}")
    return {"artifact_id": ARTIFACT_ID, "cohort": cohort, "seed": seed, **_pick(source, CONFIG_ALLOWLIST)}


def _sanitize_env_block(block: Mapping[str, Any]) -> dict[str, Any]:
    output = _pick(block, ENV_ALLOWLIST)
    spec = block.get("spec_kwargs")
    if isinstance(spec, dict):
        output["spec_kwargs"] = _pick(spec, ENV_SPEC_ALLOWLIST)
    return output


def _sanitize_env(path: Path, seed: int, cohort: str) -> dict[str, Any]:
    source = _json_load(path)
    output: dict[str, Any] = {"artifact_id": ARTIFACT_ID, "cohort": cohort, "seed": seed}
    for key in ("train_env", "eval_env"):
        block = source.get(key)
        if not isinstance(block, dict):
            raise ValueError(f"{key} missing from environment snapshot: {path}")
        output[key] = _sanitize_env_block(block)
    task_spec = source.get("task_spec")
    if not isinstance(task_spec, dict):
        raise ValueError(f"task_spec missing from environment snapshot: {path}")
    output["task_spec"] = _pick(task_spec, TASK_SPEC_ALLOWLIST)
    return output


def _sanitize_summary(path: Path, seed: int, cohort: str) -> dict[str, Any]:
    source = _json_load(path)
    output: dict[str, Any] = {"artifact_id": ARTIFACT_ID, "cohort": cohort, "seed": seed, **_pick(source, SUMMARY_ALLOWLIST)}
    losses = source.get("final_losses")
    if isinstance(losses, dict):
        output["final_losses"] = _pick(losses, FINAL_LOSS_ALLOWLIST)
    lite = source.get("final_lite_metrics")
    if isinstance(lite, dict):
        output["final_lite_metrics"] = _pick(lite, FINAL_LITE_ALLOWLIST)
    return output


def _result_hashes() -> dict[str, str]:
    return {
        path.name: _sha256(path)
        for path in sorted((ARTIFACT_ROOT / "results").glob("*.csv"))
    }


def _assert_close(name: str, actual: float, expected: float, tolerance: float) -> dict[str, Any]:
    passed = math.isfinite(actual) and abs(actual - expected) <= tolerance
    return {
        "name": name,
        "actual": actual,
        "expected_approx": expected,
        "absolute_tolerance": tolerance,
        "passed": passed,
    }


def _regression_checks(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    performance = analysis["performance_all_10"]
    support = analysis["support_activation_all_10"]
    td = analysis["td_residual_all_10_summary"]["pre_update_absolute_td_error"]
    residual = analysis["td_residual_all_10_summary"]["same_minibatch_residual_reduction"]
    successful_td = analysis["td_residual_successful_9_summary"]["pre_update_absolute_td_error"]
    successful_residual = analysis["td_residual_successful_9_summary"]["same_minibatch_residual_reduction"]
    phase = analysis["phase_all_10"]
    seed7 = analysis["seed7_final_400k"]
    checks = [
        _assert_close("diagnostic_all10_final_success_mean", float(performance["final_success_mean"]), 0.890, 0.001),
        _assert_close("diagnostic_all10_auc_success_mean", float(performance["auc_success_mean"]), 0.653, 0.001),
        _assert_close("diagnostic_all10_support_positive_rate", float(support["support_positive_rate_mean"]), 0.4293, 0.0002),
        _assert_close("diagnostic_all10_td_support_positive_mean", float(td["support_positive_mean"]), 0.2913, 0.0002),
        _assert_close("diagnostic_all10_td_support_negative_mean", float(td["support_negative_mean"]), 0.2490, 0.0002),
        _assert_close("diagnostic_all10_td_ratio", float(td["positive_to_negative_ratio"]), 1.170, 0.003),
        _assert_close("diagnostic_all10_residual_support_positive_mean", float(residual["support_positive_mean"]), 0.00754, 0.00006),
        _assert_close("diagnostic_all10_residual_support_negative_mean", float(residual["support_negative_mean"]), 0.00359, 0.00006),
        _assert_close("diagnostic_all10_residual_ratio", float(residual["positive_to_negative_ratio"]), 2.102, 0.04),
        _assert_close("diagnostic_all10_phase_approach_support_rate", float(phase["approach"]["support_positive_rate_within_phase"]), 0.7367, 0.0015),
        _assert_close("diagnostic_all10_phase_contact_support_rate", float(phase["contact"]["support_positive_rate_within_phase"]), 0.6559, 0.0015),
        _assert_close("diagnostic_all10_phase_opening_support_rate", float(phase["opening"]["support_positive_rate_within_phase"]), 0.6410, 0.0015),
        _assert_close("diagnostic_all10_phase_open_support_rate", float(phase["open"]["support_positive_rate_within_phase"]), 0.3255, 0.0015),
        _assert_close("successful9_td_support_positive_mean", float(successful_td["support_positive_mean"]), 0.3204, 0.0002),
        _assert_close("successful9_td_support_negative_mean", float(successful_td["support_negative_mean"]), 0.2739, 0.0002),
        _assert_close("successful9_residual_support_positive_mean", float(successful_residual["support_positive_mean"]), 0.00843, 0.00006),
        _assert_close("successful9_residual_support_negative_mean", float(successful_residual["support_negative_mean"]), 0.00403, 0.00006),
        _assert_close("seed7_final400k_support_mean", float(seed7["seed7_support_rate_mean"]), 0.993, 0.0015),
        _assert_close("seed7_final400k_support_ge_095_rate", float(seed7["seed7_support_ge_0_95_snapshot_rate"]), 0.9915, 0.0015),
    ]
    for name, actual, expected in (
        ("diagnostic_all10_support_n_seeds", int(support["n_seeds"]), 10),
        ("diagnostic_all10_td_n_seeds", int(td["n_seeds"]), 10),
        ("diagnostic_all10_td_positive_seed_count", int(td["positive_seed_count"]), 10),
        ("diagnostic_all10_residual_n_seeds", int(residual["n_seeds"]), 10),
        ("diagnostic_all10_residual_positive_seed_count", int(residual["positive_seed_count"]), 10),
    ):
        checks.append({"name": name, "actual": actual, "expected": expected, "passed": actual == expected})
    for phase_name in rebuild_results.PHASES:
        expected_support_n = 9 if phase_name == "opening" else 10
        for metric_name, expected in (
            ("batch_fraction_n_seeds", 10),
            ("support_rate_n_seeds", expected_support_n),
        ):
            actual = int(phase[phase_name][metric_name])
            checks.append({
                "name": f"diagnostic_all10_{phase_name}_{metric_name}",
                "actual": actual,
                "expected": expected,
                "passed": actual == expected,
            })
    checks.append({
        "name": "diagnostic_all10_hit90_seed_count",
        "actual": int(performance["hit90_seed_count"]),
        "expected": 9,
        "passed": int(performance["hit90_seed_count"]) == 9,
    })
    return checks


def _format_float(value: Any, digits: int = 4) -> str:
    number = float(value)
    return f"{number:.{digits}f}"


def _write_readme_and_report(
    analysis: Mapping[str, Any],
    duplicate_audits: Sequence[Mapping[str, Any]],
) -> None:
    performance = analysis["performance_all_10"]
    td = analysis["td_residual_all_10_summary"]["pre_update_absolute_td_error"]
    residual = analysis["td_residual_all_10_summary"]["same_minibatch_residual_reduction"]
    successful_td = analysis["td_residual_successful_9_summary"]["pre_update_absolute_td_error"]
    successful_residual = analysis["td_residual_successful_9_summary"]["same_minibatch_residual_reduction"]
    phase = analysis["phase_all_10"]
    seed7 = analysis["seed7_final_400k"]
    duplicate_removed = sum(int(row["removed_duplicate_row_count"]) for row in duplicate_audits)

    readme = f"""# DrawerOpen DDPG SABER 10-seed diagnostic artifact

Artifact ID: `{ARTIFACT_ID}`

이 패키지는 sparse MetaWorld DrawerOpen-v3에서 DDPG+HER+SABER critic-only diagnostic run 10개를 compact하고 재생성 가능한 형태로 제공한다.

## Cohort

- Seeds 1–5: diagnostic logging 재실행 cohort
- Seeds 6–10: 추가 diagnostic seed-extension cohort
- 기존 seeds 1–5 비-diagnostic run: 성능 재현 reference 전용

Seeds 1–5와 6–10은 같은 한 번의 matched 10-seed launch가 아니므로 `unmatched extension`이다. 기존 비-diagnostic seeds 1–5는 diagnostic 통계에 포함하지 않는다.

## 구조

- `data/seeds/seed_XX/eval_10k.csv`: seed당 10k–2M exact 평가점 200개
- `data/seeds/seed_XX/diagnostic_snapshots.csv.gz`: seed당 step-deduplicated snapshot 9,976개
- `data/seeds/seed_XX/{{config,env_snapshot,summary}}.json`: allowlist 익명화 metadata
- `data/reproduction_reference/`: 기존 seeds 1–5 비-diagnostic 평가 reference
- `results/diagnostic_all_10_support_activation_summary.csv`: all-10 support-positive rate 집계
- `results/diagnostic_all_10_td_residual*.csv`: outcome과 무관하게 seed 1–10을 포함한 primary diagnostic 집계
- `results/successful_9seed_td_residual*.csv`: Hit90 도달 seed만 포함한 outcome-stratified 보조 집계
- `results/`: compact data에서 재생성되는 분석 CSV
- `source_input_manifest.json`: 원본 입력의 상대경로, byte size, SHA256
- `artifact_manifest.json`: artifact payload의 byte size와 SHA256

## 재생성

Artifact root에서 다음을 실행한다.

```bash
python code/rebuild_results.py
```

재생성기는 raw 결과 디렉터리를 읽지 않고 `data/`만 사용한다. Python 3와 SciPy가 필요하다.

## 중복 제거

동일 step의 분석 대상 필드가 수치적으로 완전히 같은지 먼저 확인한 뒤 raw 파일에서 가장 마지막에 나타난 행을 선택한다. 분석 필드가 하나라도 다르면 builder는 중단한다. 이번 패키지에서 제거된 중복행은 {duplicate_removed:,}개다.

## Diagnostic 집계 원칙

Primary diagnostic aggregate는 성능 결과와 무관하게 instrumented seed 1–10을 모두 포함한다. Hit90에 도달한 9개 seed만의 결과는 outcome-stratified secondary analysis로 별도 보존한다. Phase별 support rate는 해당 phase 관측이 있는 seed만 평균하며 각 결과 행에 유효 seed 수를 기록한다.

## 필수 해석 제한

1. Residual reduction은 critic·actor·target update 후 **같은 weighted minibatch**에서 측정한 in-sample 값이다. Held-out 일반화나 인과적 성능 향상을 증명하지 않는다.
2. Phase는 원래 환경 goal이 아니라 `desired_goal_train` 기준이며 HER relabeled goal을 포함한다.
3. Support lookup은 `local_search_max=512` 제한을 사용한다. 512/full-scan 동등성은 이 artifact로 증명되지 않는다.
4. Primary TD/residual 및 phase 집계는 all-10이지만, Hit90 전후와 exploratory correlation은 정의상 Hit90 도달 9개 seed만 사용한다. Correlation은 multiple-comparison uncorrected다.
5. Checkpoint, raw `metrics.csv`, JSONL, run log는 payload에서 제외했다.
"""
    (ARTIFACT_ROOT / "README.md").write_text(readme, encoding="utf-8")

    report = f"""# DrawerOpen DDPG SABER 10-seed diagnostic 보고서

Artifact ID: `{ARTIFACT_ID}`

## 1. 데이터와 처리

Diagnostic cohort의 seed ID는 정확히 1–10이다. 모든 seed는 10,000부터 2,000,000까지 10k 간격의 평가점 200개와 고유 diagnostic snapshot 9,976개를 갖는다. 동일 step 중복은 분석 대상 값의 동일성을 검증한 뒤 마지막 raw 행을 선택했다. Snapshot은 자기상관이 있으므로 통계적 독립 표본으로 취급하지 않고 seed 내부 평균을 만든 뒤 seed를 독립 단위로 사용했다.

Seeds 1–5는 diagnostic logging 재실행이고 seeds 6–10은 별도의 extension이므로 하나의 matched launch가 아니다. 기존 seeds 1–5 비-diagnostic 결과는 aggregate 재현 reference로만 사용했다.

## 2. 성능

- Final success 평균: `{_format_float(performance['final_success_mean'], 3)}`
- Success AUC 평균: `{_format_float(performance['auc_success_mean'], 4)}`
- Hit90: `{int(performance['hit90_seed_count'])}/10` seeds
- Hit90 도달 seed만의 평균: `{float(performance['hit90_mean_successful_only']):.0f}` steps

## 3. 전체 10-seed primary diagnostic

전체 seed의 평균 support-positive rate는 `{_format_float(analysis['support_mean_all_10'], 4)}`다.

Support-positive의 업데이트 전 절대 TD error는 `{_format_float(td['support_positive_mean'], 4)}`, support-negative는 `{_format_float(td['support_negative_mean'], 4)}`로 비율은 `{_format_float(td['positive_to_negative_ratio'], 3)}x`다. 차이는 `{int(td['positive_seed_count'])}/10` seed에서 양수다.

같은 weighted minibatch에서 residual reduction은 support-positive `{_format_float(residual['support_positive_mean'], 5)}`, support-negative `{_format_float(residual['support_negative_mean'], 5)}`로 비율은 `{_format_float(residual['positive_to_negative_ratio'], 2)}x`다. 차이는 `{int(residual['positive_seed_count'])}/10` seed에서 양수다. 이 값은 same-minibatch local optimization response이며 독립 batch 또는 unseen transition의 개선을 뜻하지 않는다.

## 4. Outcome-stratified successful 9-seed

Hit90에 도달한 9개 seed만 집계하면 업데이트 전 절대 TD error는 support-positive `{_format_float(successful_td['support_positive_mean'], 4)}`, support-negative `{_format_float(successful_td['support_negative_mean'], 4)}`이고, residual reduction은 각각 `{_format_float(successful_residual['support_positive_mean'], 5)}`와 `{_format_float(successful_residual['support_negative_mean'], 5)}`다. 이 결과는 outcome에 조건화된 secondary analysis이며 primary all-10 aggregate를 대체하지 않는다.

## 5. Training-goal phase allocation: all 10 seeds

| Phase | Batch fraction | Support-positive within phase | Support-rate n |
| --- | ---: | ---: | ---: |
| Approach | {100 * phase['approach']['batch_fraction']:.2f}% | {100 * phase['approach']['support_positive_rate_within_phase']:.2f}% | {int(phase['approach']['support_rate_n_seeds'])} |
| Contact | {100 * phase['contact']['batch_fraction']:.2f}% | {100 * phase['contact']['support_positive_rate_within_phase']:.2f}% | {int(phase['contact']['support_rate_n_seeds'])} |
| Opening | {100 * phase['opening']['batch_fraction']:.2f}% | {100 * phase['opening']['support_positive_rate_within_phase']:.2f}% | {int(phase['opening']['support_rate_n_seeds'])} |
| Open | {100 * phase['open']['batch_fraction']:.2f}% | {100 * phase['open']['support_positive_rate_within_phase']:.2f}% | {int(phase['open']['support_rate_n_seeds'])} |

Phase는 observation과 `desired_goal_train`으로 정의되므로 HER relabeled goal을 포함한다. 따라서 `open`을 원래 환경 성공으로 해석해서는 안 된다.

Seed 7에는 opening-phase 관측이 없어 opening support rate만 9개 seed로 계산했다. 값을 대체하거나 0으로 imputing하지 않았으며, batch fraction은 zero-observation seed를 포함해 10개 seed 모두로 평균했다.

## 6. Hit90 전후와 seed 7

Hit90 전후 결과는 시간적 기술 통계이며 인과 방향을 분리하지 않는다. Seed 7은 400–800k 구간부터 성공 seed와 분기하며, 마지막 400k support 평균은 `{_format_float(seed7['seed7_support_rate_mean'], 4)}`, support≥0.95 snapshot 비율은 `{100 * seed7['seed7_support_ge_0_95_snapshot_rate']:.2f}%`다. 이는 초기 접촉 탐색 실패 뒤 support가 비구분적으로 포화된 단일 failure case다.

## 7. Scanner와 통계 한계

원본 run은 Numba scanner를 요청했지만 diagnostic cohort에서 scanner enabled/warmed 값은 0이었다. Python fallback audit의 policy-support mismatch는 0이었으나, 이는 같은 512-candidate 제한 아래 구현 일치만 뜻한다. Local scan truncation이 높으므로 512/1024/full-scan label agreement가 별도로 필요하다.

Exploratory correlation은 성공 seed 9개만 사용하고 multiple-comparison correction을 하지 않았다. p-value는 가설 생성용이며 확증적 결과가 아니다.

## 8. Payload 제외

Checkpoint, raw `metrics.csv`, `metrics.jsonl`, run log 및 기타 JSONL은 포함하지 않았다. 원본 파일은 `source_input_manifest.json`의 SHA256으로만 식별한다.
"""
    (ARTIFACT_ROOT / "report.md").write_text(report, encoding="utf-8")


def _text_payload(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""


def _anonymity_scan(paths: Sequence[Path]) -> list[dict[str, str]]:
    username = getpass.getuser()
    hostname = socket.gethostname()
    literal_tokens = [
        "/" + "home" + "/",
        "/" + "Users" + "/",
        "git" + "@",
        "git" + "hub.com",
    ]
    if username:
        literal_tokens.append(username)
    if hostname:
        literal_tokens.append(hostname)
    email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    findings: list[dict[str, str]] = []
    for path in paths:
        text = _text_payload(path)
        for token in literal_tokens:
            if token and token in text:
                findings.append({"path": path.relative_to(ARTIFACT_ROOT).as_posix(), "match": token})
        if email.search(text):
            findings.append({"path": path.relative_to(ARTIFACT_ROOT).as_posix(), "match": "email_pattern"})
    return findings


def _payload_paths(include_manifest: bool = False) -> list[Path]:
    excluded_names = {"artifact_manifest.json"} if not include_manifest else set()
    return sorted(
        path
        for path in ARTIFACT_ROOT.rglob("*")
        if path.is_file()
        and path.name not in excluded_names
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )


def _write_artifact_manifest() -> dict[str, Any]:
    payloads = []
    for path in _payload_paths(include_manifest=False):
        payloads.append({
            "path": path.relative_to(ARTIFACT_ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    manifest = {
        "artifact_id": ARTIFACT_ID,
        "schema_version": 1,
        "payload_count": len(payloads),
        "total_payload_bytes_excluding_manifest": sum(item["size_bytes"] for item in payloads),
        "max_payload_bytes": max(item["size_bytes"] for item in payloads),
        "github_file_limit_bytes": 100_000_000,
        "excluded_payload_types": ["checkpoint", "raw_metrics.csv", "JSONL", "run_log"],
        "payloads": payloads,
    }
    _write_json(ARTIFACT_ROOT / "artifact_manifest.json", manifest)
    return manifest


def _verify_manifest(manifest: Mapping[str, Any]) -> None:
    for item in manifest["payloads"]:
        path = ARTIFACT_ROOT / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"manifest payload missing: {item['path']}")
        if path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"manifest size mismatch: {item['path']}")
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"manifest SHA256 mismatch: {item['path']}")


def main() -> int:
    diagnostic_runs, reference_runs = _validate_source_presence()
    source_manifest = _collect_source_manifest(diagnostic_runs, reference_runs)
    source_hashes_before = {entry["path"]: entry["sha256"] for entry in source_manifest["source_inputs"]}

    data_seed_root = ARTIFACT_ROOT / "data" / "seeds"
    reference_root = ARTIFACT_ROOT / "data" / "reproduction_reference"
    duplicate_audits: list[dict[str, Any]] = []

    eval_fields = ("artifact_id", "cohort", "seed", *(compact for _raw, compact in EVAL_COLUMNS))
    diagnostic_fields = ("artifact_id", "cohort", "seed", *(compact for _raw, compact in DIAGNOSTIC_COLUMNS))

    for seed in SEEDS:
        cohort = _cohort(seed)
        run_dir = diagnostic_runs[seed]
        eval_rows, diagnostic_rows, audits = _extract_metrics(run_dir / "metrics.csv", seed, require_diagnostics=True)
        duplicate_audits.extend(audits)
        seed_root = data_seed_root / f"seed_{seed:02d}"
        _write_csv(seed_root / "eval_10k.csv", eval_fields, _attach_identity(eval_rows, seed, cohort))
        _write_deterministic_gzip_csv(
            seed_root / "diagnostic_snapshots.csv.gz",
            diagnostic_fields,
            _attach_identity(diagnostic_rows, seed, cohort),
        )
        _write_json(seed_root / "config.json", _sanitize_config(run_dir / "config.json", seed, cohort))
        _write_json(seed_root / "env_snapshot.json", _sanitize_env(run_dir / "env_snapshot.json", seed, cohort))
        _write_json(seed_root / "summary.json", _sanitize_summary(run_dir / "summary.json", seed, cohort))

    for seed in range(1, 6):
        eval_rows, _diagnostic_rows, audits = _extract_metrics(
            reference_runs[seed] / "metrics.csv", seed, require_diagnostics=False
        )
        duplicate_audits.extend(audits)
        rows = _attach_identity(eval_rows, seed, "original_non_diagnostic_reference")
        _write_csv(reference_root / f"seed_{seed:02d}_eval_10k.csv", eval_fields, rows)

    _write_json(ARTIFACT_ROOT / "source_input_manifest.json", source_manifest)
    duplicate_fields = (
        "artifact_id", "seed", "row_kind", "candidate_row_count", "unique_step_count",
        "duplicate_step_count", "removed_duplicate_row_count", "duplicate_analysis_values_identical",
        "selection_rule",
    )
    _write_csv(ARTIFACT_ROOT / "validation" / "duplicate_step_audit.csv", duplicate_fields, duplicate_audits)

    first_analysis = rebuild_results.build_results(ARTIFACT_ROOT)
    first_hashes = _result_hashes()
    second_analysis = rebuild_results.build_results(ARTIFACT_ROOT)
    second_hashes = _result_hashes()
    if first_hashes != second_hashes:
        raise ValueError("results rebuild is not byte-identical across two consecutive executions")

    regression_checks = _regression_checks(second_analysis)
    failed_regressions = [check for check in regression_checks if not check["passed"]]
    if failed_regressions:
        raise ValueError(
            "computed regression values differ from expected cohort checks; investigate source cohort, deduplication, "
            f"finite filtering, aggregation, and success definition: {failed_regressions}"
        )

    source_hashes_after = {
        entry["path"]: _sha256(REPO_ROOT / entry["path"]) for entry in source_manifest["source_inputs"]
    }
    if source_hashes_before != source_hashes_after:
        raise ValueError("one or more source input SHA256 values changed during artifact construction")

    _write_readme_and_report(second_analysis, duplicate_audits)
    pre_validation_paths = _payload_paths(include_manifest=False)
    anonymity_findings = _anonymity_scan(pre_validation_paths)
    if anonymity_findings:
        raise ValueError(f"anonymity scan failed before manifest creation: {anonymity_findings}")

    snapshot_counts = {
        str(seed): int(second_analysis["diagnostic_seed_stats"][seed]["snapshot_count"]) for seed in SEEDS
    }
    validation_summary = {
        "artifact_id": ARTIFACT_ID,
        "seed_ids": list(SEEDS),
        "seed_ids_exact_1_to_10": True,
        "evaluation_grid": {"start": 10_000, "stop": 2_000_000, "stride": 10_000, "points_per_seed": 200},
        "evaluation_grid_valid_all_seeds": True,
        "diagnostic_snapshot_counts": snapshot_counts,
        "diagnostic_snapshot_count_valid_all_seeds": all(
            count == EXPECTED_DIAGNOSTIC_SNAPSHOTS for count in snapshot_counts.values()
        ),
        "primary_diagnostic_cohort": "diagnostic_all_10",
        "primary_diagnostic_n_seeds": int(
            second_analysis["td_residual_all_10_summary"]["pre_update_absolute_td_error"]["n_seeds"]
        ),
        "primary_diagnostic_includes_failed_seed": any(
            not bool(second_analysis["diagnostic_seed_stats"][seed]["successful"]) for seed in SEEDS
        ),
        "outcome_stratified_secondary_cohort": "diagnostic_successful_9",
        "outcome_stratified_secondary_n_seeds": len(second_analysis["successful_seeds"]),
        "phase_support_rate_seed_counts": {
            phase: int(second_analysis["phase_all_10"][phase]["support_rate_n_seeds"])
            for phase in rebuild_results.PHASES
        },
        "phase_missing_observation_rule": (
            "exclude_undefined_within_phase_support_rate_without_imputation_and_report_n"
        ),
        "phase_fraction_sum_tolerance": 1e-6,
        "phase_fraction_and_required_range_checks_passed": True,
        "duplicate_analysis_values_identical": all(
            bool(row["duplicate_analysis_values_identical"]) for row in duplicate_audits
        ),
        "duplicate_selection_rule": "choose_last_raw_row_after_exact_analysis_field_equality_check",
        "results_rebuild_runs": 2,
        "results_byte_identical": True,
        "source_input_count": source_manifest["source_input_count"],
        "source_sha256_unchanged": True,
        "regression_checks": regression_checks,
        "regression_checks_passed": True,
        "anonymity_scan_passed": True,
        "anonymity_finding_count": 0,
        "inference_unit": "seed",
        "exploratory_correlation_n": 9,
        "exploratory_multiple_comparison_corrected": False,
    }
    _write_json(ARTIFACT_ROOT / "validation" / "validation_summary.json", validation_summary)

    manifest = _write_artifact_manifest()
    if int(manifest["max_payload_bytes"]) >= int(manifest["github_file_limit_bytes"]):
        raise ValueError(
            f"payload exceeds GitHub 100MB limit: {manifest['max_payload_bytes']} bytes"
        )
    _verify_manifest(manifest)
    final_anonymity_findings = _anonymity_scan(_payload_paths(include_manifest=True))
    if final_anonymity_findings:
        raise ValueError(f"final anonymity scan failed: {final_anonymity_findings}")

    manifest_path = ARTIFACT_ROOT / "artifact_manifest.json"
    artifact_size = sum(path.stat().st_size for path in _payload_paths(include_manifest=True))
    print(json.dumps({
        "artifact_id": ARTIFACT_ID,
        "artifact_size_bytes": artifact_size,
        "payload_count": manifest["payload_count"],
        "source_input_count": source_manifest["source_input_count"],
        "artifact_manifest_external_sha256": _sha256(manifest_path),
        "result_csv_count": len(rebuild_results.RESULT_FILENAMES),
        "status": "ready_to_commit",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
