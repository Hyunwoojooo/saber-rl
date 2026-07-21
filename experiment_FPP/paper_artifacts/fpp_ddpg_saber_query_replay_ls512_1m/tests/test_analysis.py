#!/usr/bin/env python3
"""Synthetic fail-closed tests for exact replay-query analysis."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

from query_ablation import (
    BOOTSTRAP_SEED,
    EXPECTED_CONFIG,
    GRID,
    MAIN_KEY,
    REPLAY_KEY,
    SEEDS,
    TOTAL_STEPS,
    TOTAL_UPDATES,
    AnalysisError,
    analyze_query_ablation,
    discover_run,
    exact_sign_flip_pvalue,
    normalized_trapezoid_auc,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def env_snapshot() -> dict[str, Any]:
    environment = {
        "env_id": "FetchPickAndPlace-v4",
        "obs_dim": 25,
        "goal_dim": 3,
        "state_dim": 28,
        "action_dim": 4,
        "action_low": [-1.0, -1.0, -1.0, -1.0],
        "action_high": [1.0, 1.0, 1.0, 1.0],
        "max_episode_steps": 50,
        "distance_threshold": 0.05,
        "spec_kwargs": {"reward_type": "sparse"},
    }
    return {
        "train_env": dict(environment),
        "eval_env": dict(environment),
        "task_spec": {"task": "FetchPickAndPlace", "epsilon_g": 0.05},
    }


def resolved_config(seed: int, method_key: str, *, total_steps: int = TOTAL_STEPS) -> dict[str, Any]:
    config = dict(EXPECTED_CONFIG)
    config.update(
        {
            "total_steps": total_steps,
            "terminal_convention": {
                "success_terminates": False,
                "timeout_bootstrap": True,
            },
            "smoke": False,
            "run_name": f"synthetic_{method_key}_s{seed}_{total_steps}_ls512",
            "output_root": f"/synthetic/{method_key}",
            "seeds": {
                "env_seed": seed,
                "torch_seed": seed,
                "numpy_seed": seed,
                "action_noise_seed": seed,
                "eval_seed": 10_000 + seed,
                "weight_seed": 20_000 + seed,
            },
        }
    )
    if method_key == REPLAY_KEY:
        config["support_query_mode"] = "replay_action"
    return config


def curve_for(seed: int, method_key: str) -> list[float]:
    penalty = 0.04 if method_key == REPLAY_KEY else 0.0
    return [
        max(0.0, min(1.0, 0.01 * index + 0.002 * seed - penalty))
        for index in range(1, len(GRID) + 1)
    ]


def write_run(
    root: Path,
    seed: int,
    method_key: str,
    *,
    total_steps: int = TOTAL_STEPS,
    curve: Sequence[float] | None = None,
) -> Path:
    run_dir = root / f"run_{method_key}_s{seed}_{total_steps}"
    config = resolved_config(seed, method_key, total_steps=total_steps)
    values = list(curve if curve is not None else curve_for(seed, method_key))
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "env_snapshot.json", env_snapshot())
    if total_steps == TOTAL_STEPS:
        auc = normalized_trapezoid_auc(values)
        hit = next((step for step, value in zip(GRID, values) if value >= 0.90), None)
        summary = {
            "method": "b3_support_runtime_opt",
            "total_steps": TOTAL_STEPS,
            "total_updates": TOTAL_UPDATES,
            "eval_AUC_success": auc,
            "final_eval_success": values[-1],
            "best_eval_success": max(values),
            "first_hit_0.90": None if hit is None else str(hit),
            "final_lite_metrics": {"legacy/optional_nan": float("nan")},
        }
        write_json(run_dir / "summary.json", summary)
        fields = [
            "step",
            "eval/env_success_mean",
            "eval/success_mean",
            "config/support_query_mode",
            "query/context_count_eligible_rate",
            "query/support_positive_rate",
            "query/query_bin_present_given_context_rate",
            "query/query_bin_matches_current_policy_rate",
            "query/query_bin_matches_replay_action_rate",
            "lite/psi_q_rank_rate",
        ]
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for index, (step, value) in enumerate(zip(GRID, values)):
                row = {
                    "step": step,
                    "eval/env_success_mean": "" if index == 0 else value,
                    "eval/success_mean": value,
                    "config/support_query_mode": (
                        "replay_action" if method_key == REPLAY_KEY else ""
                    ),
                    "lite/psi_q_rank_rate": 0.20 + seed * 0.01,
                }
                if method_key == REPLAY_KEY:
                    row.update(
                        {
                            "query/context_count_eligible_rate": 0.4,
                            "query/support_positive_rate": 0.3,
                            "query/query_bin_present_given_context_rate": 0.75,
                            "query/query_bin_matches_current_policy_rate": 0.2,
                            "query/query_bin_matches_replay_action_rate": 1.0,
                        }
                    )
                writer.writerow(row)
    return run_dir


def make_fixture(base: Path) -> tuple[Path, Path]:
    main_root, replay_root = base / "main", base / "replay"
    for seed in SEEDS:
        write_run(main_root, seed, MAIN_KEY)
        write_run(replay_root, seed, REPLAY_KEY)
    # These historical extras must not pollute exact resolved-config discovery.
    write_run(main_root, 1, MAIN_KEY, total_steps=500_000)
    write_run(main_root, 6, MAIN_KEY)
    return main_root, replay_root


class QueryAnalysisTests(unittest.TestCase):
    def test_exact_analysis_discovery_statistics_and_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            main_root, replay_root = make_fixture(Path(temporary))
            bundle = analyze_query_ablation(main_root, replay_root)
            self.assertEqual(bundle.seeds, SEEDS)
            self.assertEqual(len(bundle.per_seed_rows), 10)
            self.assertEqual(len(bundle.mean_curve_rows), 2 * len(GRID))
            inference = bundle.inference_rows[0]
            self.assertEqual(inference["bootstrap_seed"], BOOTSTRAP_SEED)
            self.assertEqual(inference["bootstrap_resamples"], 10_000)
            self.assertEqual(inference["exact_sign_flip_assignments"], 32)
            self.assertEqual(inference["positive_count"], 5)
            self.assertGreater(float(inference["mean_delta"]), 0.0)
            for run in bundle.runs_by_method[REPLAY_KEY]:
                match = run.diagnostics["query/query_bin_matches_replay_action_rate"]
                self.assertEqual(match.finite_count, len(GRID))
                self.assertEqual(match.mean, 1.0)
            # First row uses the required env-success fallback only once per run.
            for row in bundle.per_seed_rows:
                self.assertEqual(row["success_fallback_checkpoint_count"], 1)
                self.assertEqual(row["success_primary_checkpoint_count"], 99)
            self.assertTrue(discover_run(main_root, MAIN_KEY, 1).name.endswith("1000000"))

    def test_non_query_config_difference_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            main_root, replay_root = make_fixture(Path(temporary))
            replay_run = discover_run(replay_root, REPLAY_KEY, 3)
            config_path = replay_run / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["gamma"] = 0.97
            write_json(config_path, config)
            with self.assertRaisesRegex(AnalysisError, "config.gamma|config parity"):
                analyze_query_ablation(main_root, replay_root)

    def test_replay_match_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            main_root, replay_root = make_fixture(Path(temporary))
            replay_run = discover_run(replay_root, REPLAY_KEY, 2)
            metrics = replay_run / "metrics.csv"
            with metrics.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[20]["query/query_bin_matches_replay_action_rate"] = "0.99"
            with metrics.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(AnalysisError, "match rate"):
                analyze_query_ablation(main_root, replay_root)

    def test_missing_and_duplicate_grid_fail(self) -> None:
        for operation, pattern in (("missing", "missing exact"), ("duplicate", "duplicate")):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temporary:
                main_root, replay_root = make_fixture(Path(temporary))
                run = discover_run(replay_root, REPLAY_KEY, 1)
                metrics = run / "metrics.csv"
                with metrics.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                if operation == "missing":
                    rows.pop(10)
                else:
                    rows.append(dict(rows[10]))
                with metrics.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaisesRegex(AnalysisError, pattern):
                    analyze_query_ablation(main_root, replay_root)

    def test_exact_sign_flip_enumerates_all_assignments(self) -> None:
        self.assertEqual(exact_sign_flip_pvalue([1.0] * 5), 2.0 / 32.0)


if __name__ == "__main__":
    unittest.main()
