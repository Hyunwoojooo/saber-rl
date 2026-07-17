#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


FOLDER = Path(__file__).resolve().parent
REPO_ROOT = FOLDER.parents[1]
BASELINE_CONFIG = REPO_ROOT / "experiment_FPP" / "td3_her_baseline_ls512" / "config.json"


def load_runner():
    spec = importlib.util.spec_from_file_location("td3_her_per_full_run", FOLDER / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def option(command: list[str], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


class RunConfigParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads((FOLDER / "config.json").read_text())
        self.baseline = json.loads(BASELINE_CONFIG.read_text())

    def test_matched_td3_and_environment_settings(self) -> None:
        keys = [
            "task_name",
            "reward_type",
            "total_steps",
            "seeds",
            "batch_size",
            "her_ratio",
            "updates_per_step",
            "eval_freq",
            "eval_episodes",
            "local_search_max",
            "td3_policy_delay",
            "td3_target_policy_noise",
            "td3_target_noise_clip",
        ]
        for key in keys:
            self.assertEqual(self.config[key], self.baseline[key], key)
        self.assertEqual(self.config["trainer_method"], "td3_her_per_full")
        self.assertEqual(self.config["weight_scope"], "critic_only")
        self.assertEqual(self.config["weighting_mode"], "batch_norm")
        self.assertFalse(self.config["support_lookup"])
        self.assertEqual(self.config["per_alpha"], 0.6)
        self.assertEqual(self.config["per_beta_start"], 0.4)
        self.assertEqual(self.config["per_beta_end"], 1.0)
        self.assertEqual(self.config["per_beta_steps"], 1_000_000)
        self.assertEqual(self.config["eps_PER"], 1e-6)

    def test_canonical_command_routes_to_full_per(self) -> None:
        runner = load_runner()
        command = runner.build_command(
            seed=3,
            steps=12345,
            device="cpu",
            output_root=None,
            smoke=False,
            validate_only=False,
            resume=False,
            resume_path=None,
        )
        self.assertEqual(option(command, "--method"), "td3_her_per_full")
        self.assertEqual(option(command, "--total_steps"), "12345")
        self.assertEqual(option(command, "--device"), "cpu")
        self.assertEqual(option(command, "--per_alpha"), "0.6")
        self.assertEqual(option(command, "--per_beta_start"), "0.4")
        self.assertEqual(option(command, "--per_beta_end"), "1.0")
        self.assertEqual(option(command, "--per_beta_steps"), "1000000")
        self.assertEqual(option(command, "--td3_policy_delay"), "2")
        self.assertEqual(option(command, "--td3_target_policy_noise"), "0.2")
        self.assertEqual(option(command, "--td3_target_noise_clip"), "0.5")
        self.assertIn("experiment_FPP/runs/td3_her_per_full_ls512", option(command, "--output_root"))
        self.assertIn("td3_her_per_full_ls512_s3_12345_ls512", option(command, "--run_name"))
        self.assertIn("--no-support_cache_enabled", command)
        self.assertIn("--no-local_scan_cache_enabled", command)

        resumed = runner.build_command(
            seed=3,
            steps=20000,
            device="cpu",
            output_root="/tmp/per-full-resume",
            smoke=False,
            validate_only=False,
            resume=True,
            resume_path="/tmp/per-full-resume/checkpoint.pt",
        )
        self.assertIn("--resume", resumed)
        self.assertEqual(option(resumed, "--resume_path"), "/tmp/per-full-resume/checkpoint.pt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
