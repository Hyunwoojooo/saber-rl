import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
from protocol import ROOT, make_config, expected_reference_config
from records import (accepted_attempt, list_attempts, local_output, read_json, write_json,
                     require_production_gate)


def synthetic_attempt(base, seed=1, reference=False):
    """Only tabular fixtures, never a trained model or a usable checkpoint."""
    arm = "support" if reference else "pal"
    d = Path(base) / arm / f"seed_{seed:02d}" / "attempt_01"
    cfg = (expected_reference_config(seed, d) if reference else make_config("pal", seed, d).as_dict())
    write_json(d / "status.json", {"status": "complete", "profile": "production", "arm": arm,
               "seed": seed, "checkpoint_continuation": False, "source_manifest_sha256": "test"})
    write_json(d / "run/config.json", cfg)
    write_json(d / "requested_config.json", cfg)
    write_json(d / "environment.json", {key: "synthetic-test-only"
                                        for key in analyze.ENVIRONMENT_FIELDS})
    summary = {"total_steps": 2_000_000, "total_updates": 1_995_001,
               "final_eval_env_success": 0.5, "runtime_total_sec": 300,
               "legacy_live_target_diagnostic_enabled": False,
               "method": "pal", "pal_experiment": True, "critic_objective": "pal",
               "pal_alpha": 0.4, "pal_min_priority": 1.0,
               "support_computation_enabled": False, "replay_support_index_slots": 0,
               "assignment_experiment": False}
    write_json(d / "run/summary.json", summary)
    (d / "run/checkpoint.pt").write_bytes(b"synthetic-test-only")
    fields = ["step", "eval/env_success_mean", "runtime/total_sec", "loss/critic", "loss/actor",
              "loss/td_error_abs_mean", "value/q_mean", "value/target_mean", "pal/normalizer",
              "pal/base_loss", "pal/large_error_fraction"]
    with (d / "run/metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(1, 201):
            writer.writerow({"step": i*10000, "eval/env_success_mean": 0.5, "runtime/total_sec": i,
                "loss/critic": 0.5, "loss/actor": 1, "loss/td_error_abs_mean": 0.5,
                "value/q_mean": -1, "value/target_mean": -0.5, "pal/normalizer": 1,
                "pal/base_loss": 0.5, "pal/large_error_fraction": 0})
    return d


class AnalysisTests(unittest.TestCase):
    def test_auc_uses_full_prespecified_span(self):
        x = np.arange(10000, 2000001, 10000)
        y = (x-x[0])/(x[-1]-x[0])
        metrics = analyze.curve_metrics(x, y, 1_500_000)
        self.assertAlmostEqual(metrics["auc"], 0.5)
        self.assertEqual(metrics["final_success"], 1)
        with self.assertRaises(ValueError):
            analyze.curve_metrics(x[x != 1_500_000], y[x != 1_500_000], 1_500_000)

    def test_joint_seed_resampling_and_missing_seed_rejection(self):
        values = np.column_stack([np.arange(10)/10, np.arange(10)/20])
        result = analyze.paired_bootstrap(values, resamples=200, seed=123)
        rows = np.random.default_rng(123).integers(0, 10, size=(200, 10))
        expected = np.quantile((values[:, 0]-values[:, 1])[rows].mean(axis=1), [0.025, 0.975])
        np.testing.assert_array_equal(result["ci95"], expected)
        self.assertEqual(result["positive_pairs"], 9)
        self.assertEqual(result["tied_pairs"], 1)
        for broken in (values[:-1], np.full((10, 2), np.nan)):
            with self.assertRaises(ValueError):
                analyze.paired_bootstrap(broken)

    def test_full_budget_and_loss_scope_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = synthetic_attempt(tmp)
            self.assertEqual(analyze.validate_run(d)["auc"], 0.5)
            summary = read_json(d / "run/summary.json")
            summary["replay_support_index_slots"] = 1
            write_json(d / "run/summary.json", summary)
            with self.assertRaises(ValueError):
                analyze.validate_run(d)

    def test_eval_checkpoint_missing_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = synthetic_attempt(tmp)
            p = d / "run/metrics.csv"
            lines = p.read_text().splitlines()
            p.write_text("\n".join(lines[:7] + lines[8:]) + "\n")
            with self.assertRaisesRegex(ValueError, "Evaluation grid"):
                analyze.validate_run(d)

    def test_nonfinite_training_loss_is_not_imputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = synthetic_attempt(tmp)
            p = d / "run/metrics.csv"
            with p.open(newline="") as f:
                reader = csv.DictReader(f)
                fields, rows = reader.fieldnames, list(reader)
            rows[3]["loss/critic"] = "nan"
            with p.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                analyze.validate_run(d)

    def test_all_ten_statuses_retained_when_runs_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_attempt(tmp, 1)
            rows, accepted, problems = analyze.collect_cohort(tmp, "pal", "test")
            self.assertEqual(len(rows), 10)
            self.assertEqual(set(accepted), {1})
            self.assertEqual(len(problems), 9)

    def test_attempt_discovery_and_replacement_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = synthetic_attempt(tmp)
            (d.parent / "attempt_01.log").write_text("log")
            self.assertEqual(list_attempts(d.parent), [d])
            write_json(d.parent / "attempt_02/status.json", {"status": "complete", "retry_reason": "retry"})
            with self.assertRaises(ValueError):
                accepted_attempt(d.parent)

    def test_reference_config_rejects_changed_reward_and_seed(self):
        expected = expected_reference_config(1, "/tmp/unused")
        for key, value in (("gamma", 0.99), ("metaworld_reward_function_version", "v3"),
                           ("her_ratio", 0.7), ("device", "cpu")):
            changed = {**expected, key: value}
            with self.assertRaises(ValueError):
                analyze.compare_config(changed, expected)

    def test_environment_requires_same_host_and_stack(self):
        a = {key: "same" for key in analyze.ENVIRONMENT_FIELDS}
        analyze.audit_environment(a, a)
        for key in analyze.ENVIRONMENT_FIELDS:
            with self.assertRaises(ValueError):
                analyze.audit_environment(a, {**a, key: "different"})

    def test_output_cannot_target_existing_assignment_experiment(self):
        with self.assertRaises(ValueError):
            local_output(ROOT.parent / "2026-09-25_draweropen_ddpg_assignment/outputs")
        self.assertEqual(local_output(ROOT / "outputs/new"), ROOT / "outputs/new")

    def test_cpu_gate_does_not_authorize_cuda(self):
        gate = {"passed": True, "tests_passed": True, "source_manifest_sha256": "test", "device": "cpu"}
        with patch("records.read_json", return_value=gate):
            require_production_gate("test", "cpu")
            for source, device in (("test", "cuda"), ("changed", "cpu")):
                with self.assertRaises(ValueError):
                    require_production_gate(source, device)

    def test_cuda_production_needs_all_final_checks(self):
        from final_checks import CHECKS
        gate = {"passed": True, "tests_passed": True, "source_manifest_sha256": "test",
                "device": "cuda", "environment": {"host": "synthetic"}}
        final = {"passed": True, "source_manifest_sha256": "test",
                 "environment": gate["environment"], "checks": CHECKS}
        with patch("records.read_json", side_effect=[gate, final]):
            require_production_gate("test", "cuda")
        for change in ({"passed": False}, {"checks": CHECKS[:-1]},
                       {"source_manifest_sha256": "stale"}, {"environment": {}}):
            with patch("records.read_json", side_effect=[gate, {**final, **change}]), self.assertRaises(ValueError):
                require_production_gate("test", "cuda")

    def test_complete_analysis_cli_and_mismatched_environment(self):
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pal"
            reference = Path(tmp) / "reference"
            write_json(root / "production/plan.json", {"source_manifest_sha256": "test",
                       "arms": ["pal"], "seeds": list(range(1, 11))})
            write_json(reference / "outputs/production/plan.json", {"source_manifest_sha256": "test",
                       "arms": ["support", "random", "td_topk"], "seeds": list(range(1, 11))})
            for seed in range(1, 11):
                synthetic_attempt(root / "production", seed)
                synthetic_attempt(reference / "outputs/production", seed, reference=True)
            with patch.object(analyze, "verify", return_value="test"), \
                 patch.object(analyze, "local_output", return_value=root), \
                 patch.object(analyze, "audit_reference_source", return_value=reference), \
                 patch.object(analyze, "REFERENCE_MANIFEST_SHA256", "test"), \
                 patch.object(sys, "argv", ["analyze.py", "--output-root", str(root),
                                            "--saber-root", str(reference)]), redirect_stdout(io.StringIO()):
                analyze.main()
                report = read_json(root / "analysis/report.json")
                self.assertEqual(report["inference"]["mean"], 0)
                self.assertEqual(report["inference"]["ci95"], [0, 0])
                self.assertFalse(report["runtime_comparison_valid"])
                path = root / "production/pal/seed_01/attempt_01/environment.json"
                write_json(path, {**read_json(path), "host": "different"})
                with self.assertRaises(SystemExit):
                    analyze.main()
                self.assertIsNone(read_json(root / "analysis/report.json")["inference"])


if __name__ == "__main__":
    unittest.main()
