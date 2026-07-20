#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SLUG = "DDPG-HER-B3-Context-Count-Only-2M"
REQUIRED_QUERY_FIELDS = (
    "query/context_count_eligible_rate",
    "query/support_positive_rate",
    "query/high_weight_count",
    "query/raw_weight_mean",
    "query/raw_weight_std",
    "query/raw_weight_ess",
    "query/normalized_weight_mean",
    "query/normalized_weight_std",
    "query/normalized_weight_ess",
    "query/all_eligible_update",
    "query/all_ineligible_update",
    "query/mixed_weight_update",
    "query/action_membership_lookup_count",
    "query/invariant_violation_count",
)


def finite(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise AssertionError(f"non-finite {key}: {row[key]!r}")
    return value


def validate_run(run_dir: Path, expected_steps: int, expected_eval_freq: int, expected_eval_episodes: int) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    if config["support_query_mode"] != "context_count_only":
        raise AssertionError("config does not record context_count_only")
    if summary.get("support_query_mode") != "context_count_only":
        raise AssertionError("summary does not record context_count_only")
    if summary.get("control_description") != "Context eligibility only (count-unmatched descriptive control)":
        raise AssertionError("summary descriptive label mismatch")
    if int(summary["total_steps"]) != expected_steps:
        raise AssertionError("total step mismatch")
    if int(summary["total_updates"]) <= 0:
        raise AssertionError("no learning updates were recorded")
    if expected_steps == 2_000_000:
        expected_config = {"max_episode_steps": 200, "start_steps": 10_000, "warmup_steps": 5_000, "updates_per_step": 1, "batch_size": 256, "buffer_capacity": 300_000, "her_ratio": 0.8, "gamma": 0.98, "tau": 0.005, "actor_lr": 3e-4, "critic_lr": 3e-4, "hidden_dim": 256, "noise_std": 0.2, "M_B": 16, "M_Q_min": 4, "N_local_min": 4, "local_search_max": 512, "action_bins_per_dim": 5, "lambda_support": 1.0, "weighting_mode": "batch_norm", "weight_scope": "critic_only", "support_scan_mode": "deterministic_tail_v1"}
        mismatched = {key: (config.get(key), value) for key, value in expected_config.items() if config.get(key) != value}
        if mismatched:
            raise AssertionError(f"full-run config mismatch: {mismatched}")
        if int(summary["final_replay_size"]) != 300_000:
            raise AssertionError("final replay size is not 300000")

    with (run_dir / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    train_rows = [row for row in rows if row.get("query/context_count_eligible_rate", "") not in {"", "nan", "NaN"}]
    if not train_rows:
        raise AssertionError("no context-only training diagnostics")
    for row in train_rows:
        for key in REQUIRED_QUERY_FIELDS:
            if key not in row or row[key] == "":
                raise AssertionError(f"missing diagnostic {key}")
        eligible = finite(row, "query/context_count_eligible_rate")
        support = finite(row, "query/support_positive_rate")
        if abs(eligible - support) > 1e-7:
            raise AssertionError("support mask differs from context eligibility")
        if finite(row, "query/action_membership_lookup_count") != 0.0:
            raise AssertionError("action membership lookup was used")
        if finite(row, "query/invariant_violation_count") != 0.0:
            raise AssertionError("context-only invariant violation")
        if abs(finite(row, "query/normalized_weight_mean") - 1.0) > 2e-6:
            raise AssertionError("normalized weight mean is not one")
        if abs(
            finite(row, "query/all_eligible_update")
            + finite(row, "query/all_ineligible_update")
            + finite(row, "query/mixed_weight_update")
            - 1.0
        ) > 1e-7:
            raise AssertionError("update-type indicators are not one-hot")

    eval_rows = [row for row in rows if row.get("eval/env_success_mean", "") not in {"", "nan", "NaN"}]
    observed_steps = [int(float(row["step"])) for row in eval_rows]
    expected_grid = list(range(expected_eval_freq, expected_steps + 1, expected_eval_freq))
    if observed_steps != expected_grid:
        raise AssertionError(f"evaluation grid mismatch: {observed_steps[:5]} ...")
    if int(config["eval_episodes"]) != expected_eval_episodes:
        raise AssertionError("evaluation episode count mismatch")
    return {
        "run_dir": str(run_dir),
        "steps": expected_steps,
        "updates": int(summary["total_updates"]),
        "train_rows_checked": len(train_rows),
        "eval_points": len(eval_rows),
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--eval-freq", type=int, required=True)
    parser.add_argument("--eval-episodes", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    cohort = args.root / SLUG
    results = []
    for seed in args.seeds:
        candidates = sorted(cohort.glob(f"drawer_open_{SLUG}_s{seed}_*_cuda"))
        if len(candidates) != 1:
            raise AssertionError(f"seed {seed}: expected one run, found {len(candidates)}")
        results.append(validate_run(candidates[0], args.steps, args.eval_freq, args.eval_episodes))
    print(json.dumps({"passed": True, "runs": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
