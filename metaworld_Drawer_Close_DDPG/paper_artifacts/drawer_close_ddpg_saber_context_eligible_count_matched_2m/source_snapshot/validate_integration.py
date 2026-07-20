#!/usr/bin/env python3
"""Correctness/smoke gate; never interprets short-run performance."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

import run as experiment_run


STRICT_FIELDS = (
    "strict/main_positive_count",
    "strict/context_eligible_count",
    "strict/control_positive_count",
    "strict/main_positive_rate",
    "strict/context_eligible_rate",
    "strict/control_positive_rate",
    "strict/main_control_overlap_count",
    "strict/main_control_overlap_rate",
    "strict/control_selected_outside_eligible",
    "strict/raw_weight_mean",
    "strict/normalized_weight_mean",
    "strict/normalized_weight_min",
    "strict/normalized_weight_max",
    "runtime/support_lookup_sec_per_update",
    "runtime/strict_assignment_sec_per_update",
    "runtime/numba_support_scan_enabled",
    "runtime/numba_scanner_warmed",
    "runtime/numba_support_scan_fallback_rate",
)
NUMBA_FIELDS = (
    "runtime/numba_support_scan_enabled",
    "runtime/numba_scanner_warmed",
    "runtime/numba_support_scan_fallback_rate",
)


def _number(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    if raw == "":
        raise RuntimeError(f"missing metric value {key!r}")
    value = float(raw)
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite metric {key!r}: {raw!r}")
    return value


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_run(run_dir: Path, mode: experiment_run.Mode, seed: int) -> dict[str, Any]:
    required_files = (
        "config.json",
        "env_snapshot.json",
        "metrics.csv",
        "summary.json",
        "checkpoint.pt",
    )
    missing = [name for name in required_files if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"{run_dir}: missing required files {missing}")

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    expected_config = {
        "method": "b3_support_runtime_opt",
        "weight_scope": "critic_only",
        "support_query_mode": "current_policy",
        "support_assignment_mode": "context_eligible_count_matched",
        "weighting_mode": "batch_norm",
        "total_steps": mode.steps,
        "eval_freq": mode.eval_freq,
        "eval_episodes": mode.eval_episodes,
        "batch_size": 256,
        "buffer_capacity": 300_000,
        "local_search_max": 512,
        "M_B": 16,
        "M_Q_min": 4,
        "N_local_min": 4,
        "action_bins_per_dim": 5,
    }
    config_failures = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in expected_config.items()
        if config.get(key) != expected
    }
    seeds = config.get("seeds", {})
    if seeds.get("assignment_seed") != 30_000 + seed:
        config_failures["seeds.assignment_seed"] = {
            "expected": 30_000 + seed,
            "actual": seeds.get("assignment_seed"),
        }
    if config_failures:
        raise RuntimeError(f"{run_dir}: config failures {config_failures}")

    if int(summary.get("total_steps", -1)) != mode.steps:
        raise RuntimeError(f"{run_dir}: incomplete summary total_steps")
    expected_updates = mode.steps - mode.warmup_steps + 1
    if int(summary.get("total_updates", -1)) != expected_updates:
        raise RuntimeError(
            f"{run_dir}: total_updates={summary.get('total_updates')} expected={expected_updates}"
        )
    expected_replay = min(mode.steps, 300_000)
    if int(summary.get("final_replay_size", -1)) != expected_replay:
        raise RuntimeError(f"{run_dir}: final replay size mismatch")

    rows = _load_csv(run_dir / "metrics.csv")
    if not rows:
        raise RuntimeError(f"{run_dir}: empty metrics.csv")
    missing_headers = [field for field in STRICT_FIELDS if field not in rows[0]]
    if missing_headers:
        raise RuntimeError(f"{run_dir}: missing strict metric columns {missing_headers}")
    checked_updates = 0
    numba_rows_checked = 0
    for row in rows:
        if row.get("strict/main_positive_count", "") == "":
            continue
        values = {
            field: _number(row, field)
            for field in STRICT_FIELDS
            if field not in NUMBA_FIELDS
        }
        if values["strict/main_positive_count"] != values["strict/control_positive_count"]:
            raise RuntimeError(f"{run_dir}: exact-k mismatch in metrics row")
        if values["strict/main_positive_count"] > values["strict/context_eligible_count"]:
            raise RuntimeError(f"{run_dir}: Main positive outside E")
        if values["strict/control_selected_outside_eligible"] != 0.0:
            raise RuntimeError(f"{run_dir}: control selected outside E")
        if not math.isclose(
            values["strict/normalized_weight_mean"], 1.0, rel_tol=0.0, abs_tol=2e-6
        ):
            raise RuntimeError(f"{run_dir}: normalized weight mean is not one")
        if values["strict/normalized_weight_min"] <= 0.0:
            raise RuntimeError(f"{run_dir}: non-positive normalized weight")
        if values["strict/normalized_weight_max"] <= 0.0:
            raise RuntimeError(f"{run_dir}: non-positive normalized weight max")
        if values["runtime/support_lookup_sec_per_update"] < 0.0:
            raise RuntimeError(f"{run_dir}: negative support timing")
        if values["runtime/strict_assignment_sec_per_update"] < 0.0:
            raise RuntimeError(f"{run_dir}: negative assignment timing")
        step = int(float(row["step"]))
        if step >= mode.warmup_steps:
            numba = {field: _number(row, field) for field in NUMBA_FIELDS}
            if numba["runtime/numba_support_scan_enabled"] != 1.0:
                raise RuntimeError(f"{run_dir}: Numba support scanner is not enabled")
            if numba["runtime/numba_scanner_warmed"] != 1.0:
                raise RuntimeError(f"{run_dir}: Numba support scanner is not warmed")
            if numba["runtime/numba_support_scan_fallback_rate"] != 0.0:
                raise RuntimeError(f"{run_dir}: Numba support scanner used fallback")
            numba_rows_checked += 1
        checked_updates += 1
    if checked_updates == 0:
        raise RuntimeError(f"{run_dir}: no strict metric rows were checked")
    if numba_rows_checked == 0:
        raise RuntimeError(f"{run_dir}: no post-warmup Numba metric rows were checked")

    eval_rows = [row for row in rows if row.get("eval/env_success_mean", "") != ""]
    expected_grid = list(range(mode.eval_freq, mode.steps + 1, mode.eval_freq))
    actual_grid = [int(float(row["step"])) for row in eval_rows]
    if actual_grid != expected_grid:
        raise RuntimeError(
            f"{run_dir}: evaluation grid mismatch actual={actual_grid} expected={expected_grid}"
        )
    for row in eval_rows:
        value = _number(row, "eval/env_success_mean")
        if value < 0.0 or value > 1.0:
            raise RuntimeError(f"{run_dir}: invalid evaluation success {value}")

    diagnostics_candidates = (
        run_dir / "strict_diagnostics_1000.csv",
        run_dir / "assignment_diagnostics_1000.csv",
    )
    diagnostics_path = next((path for path in diagnostics_candidates if path.is_file()), None)
    if diagnostics_path is None:
        raise RuntimeError(f"{run_dir}: strict diagnostic trace is missing")
    diagnostic_rows = _load_csv(diagnostics_path)
    if not diagnostic_rows:
        raise RuntimeError(f"{run_dir}: strict diagnostic trace is empty")
    for row in diagnostic_rows:
        for key in (
            "count_equality_violation_sum",
            "control_selected_outside_eligible_sum",
            "fallback_sampling_used_sum",
            "replacement_sampling_used_sum",
            "invariant_failure_count",
        ):
            if key in row and _number(row, key) != 0.0:
                raise RuntimeError(f"{run_dir}: diagnostic failure {key}={row[key]}")

    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    checkpoint_required = (
        "assignment_rng_state",
        "assignment_diagnostics_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state_all",
        "numpy_global_rng_state",
        "resume_audit",
    )
    missing_checkpoint = [key for key in checkpoint_required if key not in checkpoint]
    if missing_checkpoint:
        raise RuntimeError(f"{run_dir}: checkpoint audit fields missing {missing_checkpoint}")
    resume_audit = checkpoint["resume_audit"]
    if not bool(resume_audit.get("episode_boundary", False)):
        raise RuntimeError(f"{run_dir}: final checkpoint is not at an episode boundary")
    if int(checkpoint.get("global_step", -1)) != mode.steps:
        raise RuntimeError(f"{run_dir}: checkpoint global step mismatch")

    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "steps": mode.steps,
        "updates": expected_updates,
        "eval_points": len(eval_rows),
        "strict_metric_rows_checked": checked_updates,
        "numba_metric_rows_checked": numba_rows_checked,
        "strict_diagnostic_windows": len(diagnostic_rows),
        "passed": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=sorted(experiment_run.MODES))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mode = experiment_run.MODES[args.mode]
    seeds = tuple(args.seeds) if args.seeds else mode.seeds
    reports = [
        validate_run(mode.output_root / experiment_run.run_name(mode, seed), mode, seed)
        for seed in seeds
    ]
    report = {"mode": args.mode, "seeds": reports, "passed": True}
    output = args.output or experiment_run.EXPERIMENT_DIR / "reports" / f"{args.mode}_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
