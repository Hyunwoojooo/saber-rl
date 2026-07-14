#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def load_module():
    path = HERE / "analyze_results.py"
    spec = importlib.util.spec_from_file_location("analyze_query_results", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_metrics(path: Path, rows: list[list[object]], extra_fields: tuple[str, ...] = ()) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["step", "train/loss", "eval/env_success_mean", "eval/success_mean", *extra_fields]
        )
        writer.writerows(rows)


def test_known_auc_and_duplicate_train_eval_rows() -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics.csv"
        write_metrics(
            path,
            [
                [10, 1.0, "", ""],
                [10, "", 0.0, 0.9],
                [20, 0.5, "", ""],
                [20, "", 0.5, 0.1],
                [20, "", 0.5, 0.2],
                [30, "", 1.0, 0.0],
            ],
        )
        before = path.read_bytes()
        steps, values = module.load_eval_curve(path, (10, 20, 30))
        assert path.read_bytes() == before
    np.testing.assert_array_equal(steps, np.asarray([10.0, 20.0, 30.0]))
    np.testing.assert_array_equal(values, np.asarray([0.0, 0.5, 1.0]))
    assert module.normalized_auc(steps, values) == 0.5
    assert module.first_hit90(steps, values) == 30


def test_missing_checkpoint_raises() -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics.csv"
        write_metrics(path, [[10, "", 0.0, ""], [30, "", 1.0, ""]])
        try:
            module.load_eval_curve(path, (10, 20, 30))
        except ValueError as exc:
            assert "missing exact evaluation checkpoints [20]" in str(exc)
        else:
            raise AssertionError("missing evaluation checkpoint must fail")


def test_mechanism_exact_snapshots() -> None:
    module = load_module()
    field = "query/support_positive_rate"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics.csv"
        write_metrics(
            path,
            [
                [10, "", 0.1, "", 0.2],
                [10, "", 0.1, "", 0.2],
                [20, "", 0.3, "", 0.6],
            ],
            (field,),
        )
        result = module.load_mechanism_snapshots(path, (10, 20))
    assert result[field] == 0.4
    assert result[f"{field}__checkpoint_count"] == 2
    assert result["query/context_count_eligible_rate"] is None


def fixed_config(mode: str) -> dict[str, object]:
    return {
        "run_name": f"synthetic_{mode}",
        "output_root": f"runs/{mode}",
        "support_query_mode": mode,
        "total_steps": 1_000_000,
        "batch_size": 256,
        "her_ratio": 0.8,
        "updates_per_step": 1,
        "eval_freq": 10_000,
        "eval_episodes": 20,
        "local_search_max": 512,
        "M_B": 16,
        "M_Q_min": 4,
        "N_local_min": 4,
        "action_bins_per_dim": 5,
        "lambda_support": 1.0,
        "weight_scope": "critic_only",
        "weighting_mode": "batch_norm",
        "support_scan_mode": "deterministic_tail_v1",
        "gamma": 0.98,
    }


def synthetic_rows(module):
    rows = []
    methods = {
        "her": ("TD3+HER", 0.1, None),
        "saber": ("Main SABER", 0.3, "current_policy"),
        "replay": ("Replay-query", 0.2, "replay_action"),
        "shuffled": ("Shuffled-policy-query", 0.25, "shuffled_policy"),
        "context_only": ("Context-count-only", 0.15, "context_count_only"),
    }
    for key, (label, offset, mode) in methods.items():
        for seed in module.SEEDS:
            config = {} if mode is None else fixed_config(mode)
            rows.append(
                {
                    "method_key": key,
                    "method": label,
                    "seed": seed,
                    "recomputed_auc": offset + seed * 0.01,
                    "native_auc": offset + seed * 0.01,
                    "final_success": offset + seed * 0.02,
                    "hit90": 100_000 + seed * 10_000 if seed != 5 else None,
                    "support_query_mode": mode,
                    "_config": config,
                }
            )
    return rows


def test_exact_seed_pairing() -> None:
    module = load_module()
    rows = synthetic_rows(module)
    paired = module.paired_rows(rows)
    first = [row for row in paired if row["comparison"] == "Main SABER - Replay-query"]
    assert [row["seed"] for row in first] == [1, 2, 3, 4, 5]
    np.testing.assert_allclose([row["auc_difference"] for row in first], 0.1)
    broken = [row for row in rows if not (row["method_key"] == "replay" and row["seed"] == 3)]
    try:
        module.paired_rows(broken)
    except ValueError as exc:
        assert "paired seed coverage" in str(exc)
    else:
        raise AssertionError("inexact seed pairing must fail")


def test_condition_config_validation() -> None:
    module = load_module()
    rows = synthetic_rows(module)
    assert module.validate_condition_configs(rows) is None

    broken_mode = [dict(row) for row in rows]
    target = next(
        row for row in broken_mode if row["method_key"] == "shuffled" and row["seed"] == 3
    )
    target["support_query_mode"] = "replay_action"
    try:
        module.validate_condition_configs(broken_mode)
    except ValueError as exc:
        assert "expected support_query_mode" in str(exc)
    else:
        raise AssertionError("wrong query mode must fail")

    broken_config = [dict(row) for row in rows]
    target = next(
        row for row in broken_config if row["method_key"] == "context_only" and row["seed"] == 2
    )
    target["_config"] = dict(target["_config"])
    target["_config"]["gamma"] = 0.95
    try:
        module.validate_condition_configs(broken_config)
    except ValueError as exc:
        assert "non-query config differs" in str(exc)
    else:
        raise AssertionError("non-query config difference must fail")


def test_paired_inference() -> None:
    module = load_module()
    differences = [0.01, 0.02, -0.01, 0.03, 0.00]
    first = module.paired_percentile_bootstrap(differences, resamples=1_000, rng_seed=27_027)
    second = module.paired_percentile_bootstrap(differences, resamples=1_000, rng_seed=27_027)
    assert first == second
    assert module.exact_sign_flip_pvalue([1, 1, 1, 1, 1]) == 0.0625

    paired = module.paired_rows(synthetic_rows(module))
    leave_one_out = module.leave_one_seed_out_rows(paired)
    assert len(leave_one_out) == len(module.COMPARISONS) * len(module.SEEDS)
    first_comparison = [
        row for row in leave_one_out if row["comparison"] == "Main SABER - Replay-query"
    ]
    assert [row["omitted_seed"] for row in first_comparison] == [1, 2, 3, 4, 5]
    np.testing.assert_allclose([row["mean_auc_difference"] for row in first_comparison], 0.1)


def main() -> int:
    test_known_auc_and_duplicate_train_eval_rows()
    test_missing_checkpoint_raises()
    test_mechanism_exact_snapshots()
    test_exact_seed_pairing()
    test_condition_config_validation()
    test_paired_inference()
    print("test_analyze_results.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
