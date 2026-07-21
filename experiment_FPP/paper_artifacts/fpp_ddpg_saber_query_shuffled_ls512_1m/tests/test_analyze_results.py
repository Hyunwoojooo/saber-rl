#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def load_analysis():
    path = HERE / "analyze_results.py"
    spec = importlib.util.spec_from_file_location("ddpg_shuffled_analysis_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def canonical_config(seed: int, mode: str | None, method_key: str) -> dict[str, object]:
    config: dict[str, object] = {
        "run_name": f"synthetic_{method_key}_s{seed}",
        "output_root": f"runs/{method_key}",
        "total_steps": 1_000_000,
        "method": "b3_support_runtime_opt",
        "weight_scope": "critic_only",
        "weighting_mode": "batch_norm",
        "batch_size": 256,
        "buffer_capacity": 300_000,
        "her_ratio": 0.8,
        "updates_per_step": 1,
        "eval_freq": 10_000,
        "eval_episodes": 20,
        "gamma": 0.98,
        "seeds": {
            "env_seed": seed,
            "torch_seed": seed,
            "numpy_seed": seed,
            "action_noise_seed": seed,
            "eval_seed": 10_000 + seed,
            "weight_seed": 20_000 + seed,
        },
    }
    if mode is not None:
        config["support_query_mode"] = mode
    return config


def curve(method_key: str, seed: int) -> np.ndarray:
    offset = 0.10 if method_key == "main" else 0.0
    return np.clip(np.linspace(0.0, 0.9, 100) + offset + seed * 0.001, 0.0, 1.0)


def write_run(root: Path, method_key: str, seed: int, *, total_steps: int = 1_000_000) -> Path:
    analysis = load_analysis()
    mode = None if method_key == "main" else "shuffled_policy"
    run = root / f"synthetic_{method_key}_s{seed}_{total_steps}"
    run.mkdir(parents=True)
    config = canonical_config(seed, mode, method_key)
    config["total_steps"] = total_steps
    values = curve(method_key, seed)
    steps = np.asarray(analysis.EVAL_STEPS, dtype=np.float64)
    auc = analysis.normalized_auc(steps, values)
    summary = {
        "total_steps": total_steps,
        "total_updates": 995_001 if total_steps == 1_000_000 else total_steps - 4_999,
        "final_replay_size": 300_000 if total_steps == 1_000_000 else total_steps,
        "eval_AUC_success": auc,
        "final_eval_env_success": float(values[-1]),
    }
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run / "env_snapshot.json").write_text(json.dumps({"synthetic": True}) + "\n", encoding="utf-8")
    fields = [
        "step",
        "eval/env_success_mean",
        "eval/success_mean",
        "loss/critic",
        "loss/actor",
        "loss/td_error_abs_mean",
        "value/q_mean",
        "value/target_mean",
        "train/replay_size",
        "lite/psi_q_rank_rate",
        "lite/mean_w_P",
        *analysis.DIAGNOSTIC_FIELDS,
    ]
    with (run / "metrics_selected_10k.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for step, success in zip(analysis.EVAL_STEPS, values):
            row: dict[str, object] = {
                "step": step,
                "eval/env_success_mean": float(success),
                "eval/success_mean": 1.0 - float(success),
                "loss/critic": 0.5,
                "loss/actor": -0.2,
                "loss/td_error_abs_mean": 0.1,
                "value/q_mean": 2.0,
                "value/target_mean": 2.1,
                "train/replay_size": min(step, 300_000),
                "lite/psi_q_rank_rate": 0.2,
                "lite/mean_w_P": 1.2,
            }
            if method_key == "shuffled":
                row.update(
                    {
                        "query/context_count_eligible_rate": 0.8,
                        "query/support_positive_rate": 0.2,
                        "query/query_bin_present_given_context_rate": 0.25,
                        "query/query_bin_matches_current_policy_rate": 0.1,
                        "query/query_bin_matches_replay_action_rate": 0.05,
                        "query/high_weight_count": 51,
                        "query/raw_weight_mean": 1.2,
                        "query/normalized_weight_mean": 1.0,
                    }
                )
            writer.writerow(row)
    return run


def build_synthetic_roots(tmp: Path) -> tuple[Path, Path]:
    main_root = tmp / "main"
    shuffled_root = tmp / "shuffled"
    for seed in range(1, 6):
        write_run(main_root, "main", seed)
        write_run(shuffled_root, "shuffled", seed)
    write_run(main_root, "main", 1, total_steps=500_000)
    write_run(main_root, "main", 6)
    return main_root, shuffled_root


def test_primary_field_only_and_exact_grid() -> None:
    analysis = load_analysis()
    with tempfile.TemporaryDirectory() as tmp:
        metrics = Path(tmp) / "metrics.csv"
        with metrics.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("step", "eval/env_success_mean", "eval/success_mean")
            )
            writer.writeheader()
            writer.writerow({"step": 10, "eval/env_success_mean": 0.0, "eval/success_mean": 1.0})
            writer.writerow({"step": 20, "eval/env_success_mean": 0.5, "eval/success_mean": 0.0})
            writer.writerow({"step": 30, "eval/env_success_mean": 1.0, "eval/success_mean": 0.0})
        steps, values = analysis.load_eval_curve(metrics, (10, 20, 30))
        np.testing.assert_array_equal(values, (0.0, 0.5, 1.0))
        assert analysis.normalized_auc(steps, values) == 0.5

        fallback_only = Path(tmp) / "fallback.csv"
        with fallback_only.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("step", "eval/env_success_mean", "eval/success_mean"))
            writer.writeheader()
            writer.writerow({"step": 10, "eval/env_success_mean": "", "eval/success_mean": 1.0})
        try:
            analysis.load_eval_curve(fallback_only, (10,))
        except ValueError as exc:
            assert "missing=[10]" in str(exc)
        else:
            raise AssertionError("fallback success field must not be accepted")


def test_discovery_filters_partial_and_extra_seed() -> None:
    analysis = load_analysis()
    with tempfile.TemporaryDirectory() as tmp:
        main_root, _ = build_synthetic_roots(Path(tmp))
        found = analysis.discover_runs(main_root, "main")
        assert list(found) == [1, 2, 3, 4, 5]
        assert all("1000000" in path.name for path in found.values())


def test_full_analysis_and_paired_inference() -> None:
    analysis = load_analysis()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        main_root, shuffled_root = build_synthetic_roots(root)
        payload = analysis.run_analysis(main_root, shuffled_root, root / "results")
        assert payload["validation_passed"] is True
        inference = payload["paired_inference"]
        assert inference["positive_seed_count"] == 5
        assert inference["zero_seed_count"] == 0
        assert inference["negative_seed_count"] == 0
        assert inference["bootstrap_resamples"] == 10_000
        assert inference["bootstrap_rng_seed"] == 27_027
        assert inference["sign_flip_assignment_count"] == 32
        assert inference["exact_two_sided_sign_flip_pvalue"] == 0.0625
        assert math.isclose(float(inference["mean_auc_difference"]), 0.1, abs_tol=2e-4)
        assert len(inference["leave_one_seed_out_means"]) == 5
        validation = json.loads((root / "results" / "validation.json").read_text(encoding="utf-8"))
        assert validation["run_count"] == 10
        assert validation["config_parity"]["passed"] is True


def test_bootstrap_reproducibility_and_config_fail_fast() -> None:
    analysis = load_analysis()
    differences = [0.1, 0.2, -0.1, 0.0, 0.05]
    first = analysis.paired_percentile_bootstrap(differences)
    second = analysis.paired_percentile_bootstrap(differences)
    assert first == second
    assert analysis.exact_sign_flip_pvalue([1, 1, 1, 1, 1]) == 0.0625

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        main_root, shuffled_root = build_synthetic_roots(root)
        config_path = shuffled_root / "synthetic_shuffled_s3_1000000" / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["gamma"] = 0.95
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        try:
            analysis.run_analysis(main_root, shuffled_root, root / "bad-results")
        except ValueError as exc:
            assert "config parity failed" in str(exc)
        else:
            raise AssertionError("unexpected config difference must fail")


def main() -> int:
    test_primary_field_only_and_exact_grid()
    test_discovery_filters_partial_and_extra_seed()
    test_full_analysis_and_paired_inference()
    test_bootstrap_reproducibility_and_config_fail_fast()
    print("test_analyze_results.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
