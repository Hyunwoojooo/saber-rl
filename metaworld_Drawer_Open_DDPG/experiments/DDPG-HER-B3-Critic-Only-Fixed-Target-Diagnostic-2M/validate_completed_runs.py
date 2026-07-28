#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


WARMUP_STEPS = 5_000
HORIZON = 200
BATCH_SIZE = 256


def number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def validate_run(run_dir: Path, expected_steps: int) -> None:
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.csv"
    if not summary_path.is_file() or not metrics_path.is_file():
        raise AssertionError(f"missing summary or metrics in {run_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required_summary = {
        "total_steps": expected_steps,
        "fixed_target_residual_diagnostic": True,
        "live_target_residual_diagnostic": True,
        "top10_exact_k": True,
        "diagnostic_snapshot_only": True,
        "resume_supported": False,
    }
    for key, expected in required_summary.items():
        if summary.get(key) != expected:
            raise AssertionError(
                f"{run_dir}: summary[{key!r}]={summary.get(key)!r}, expected {expected!r}"
            )

    final = summary.get("final_lite_metrics", {})
    exact_runtime = {
        "runtime/numba_support_scan_enabled": 1.0,
        "runtime/numba_scanner_warmed": 1.0,
        "runtime/numba_support_scan_fallback_rate": 0.0,
        "runtime/audit_psi_A_policy_mismatch_rate": 0.0,
        "diagnostic/fixed_target_from_pre_update": 1.0,
        "diagnostic/same_minibatch_one_update": 1.0,
        "diagnostic/snapshot_only_collection": 1.0,
    }
    for key, expected in exact_runtime.items():
        observed = number(final.get(key))
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(
                f"{run_dir}: final_lite_metrics[{key!r}]={observed}, expected {expected}"
            )

    observed_steps: set[int] = set()
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not math.isclose(
                number(row.get("diagnostic/fixed_target_from_pre_update")),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                continue
            step = int(float(row["step"]))
            observed_steps.add(step)
            for key in (
                "diagnostic/same_minibatch_one_update",
                "diagnostic/snapshot_only_collection",
            ):
                if not math.isclose(number(row.get(key)), 1.0, rel_tol=0.0, abs_tol=1e-12):
                    raise AssertionError(f"{run_dir}: {key} is not 1 at step {step}")
            if not math.isclose(
                number(row.get("diagnostic/td_error_top10_count")),
                float(math.ceil(0.1 * BATCH_SIZE)),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise AssertionError(f"{run_dir}: incorrect exact top-10% count at step {step}")

    first_step = math.ceil(WARMUP_STEPS / HORIZON) * HORIZON
    expected_snapshot_steps = set(range(first_step, expected_steps + 1, HORIZON))
    if observed_steps != expected_snapshot_steps:
        missing = sorted(expected_snapshot_steps - observed_steps)[:10]
        extra = sorted(observed_steps - expected_snapshot_steps)[:10]
        raise AssertionError(
            f"{run_dir}: diagnostic grid mismatch; "
            f"observed={len(observed_steps)} expected={len(expected_snapshot_steps)} "
            f"missing={missing} extra={extra}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()
    if len(args.run_dirs) != 5:
        raise SystemExit(f"expected exactly five run directories, got {len(args.run_dirs)}")
    for raw in args.run_dirs:
        validate_run(Path(raw), args.steps)
    print(f"validated {len(args.run_dirs)} fixed-target diagnostic runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
