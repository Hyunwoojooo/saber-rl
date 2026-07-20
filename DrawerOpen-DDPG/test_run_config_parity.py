#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


FOLDER = Path(__file__).resolve().parent
REPO_ROOT = FOLDER.parent


def load_runner():
    spec = importlib.util.spec_from_file_location("drawer_per_full_run", FOLDER / "run.py")
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

    def test_frozen_paper_protocol(self) -> None:
        expected = {
            "task_name": "drawer_open",
            "backbone": "DDPG",
            "trainer_method": "ddpg_her_per_full",
            "total_steps": 2_000_000,
            "seeds": [1, 2, 3, 4, 5],
            "max_episode_steps": 200,
            "batch_size": 256,
            "buffer_capacity": 300_000,
            "her_ratio": 0.8,
            "updates_per_step": 1,
            "eval_freq": 10_000,
            "eval_episodes": 20,
            "per_alpha": 0.6,
            "per_beta_start": 0.4,
            "per_beta_end": 1.0,
            "per_beta_steps": 2_000_000,
            "eps_PER": 1e-6,
            "weight_scope": "critic_only",
            "weighting_mode": "batch_norm",
            "support_lookup": False,
        }
        for key, value in expected.items():
            self.assertEqual(self.config[key], value, key)
        self.assertFalse(self.config["support_cache_enabled"])
        self.assertFalse(self.config["local_scan_cache_enabled"])
        self.assertFalse(self.config["save_step_checkpoints"])
        self.assertFalse(self.config["success_terminates"])
        self.assertTrue(self.config["timeout_bootstrap"])

    def test_fpp_replay_and_drawer_backbone_source_parity(self) -> None:
        local_source = FOLDER / "code" / "src" / "metacog_lite_v0"
        fpp_replay = REPO_ROOT / self.config["fpp_full_per_replay_reference"]
        self.assertEqual((local_source / "replay.py").read_bytes(), fpp_replay.read_bytes())

        backbone_source = REPO_ROOT / self.config["matched_backbone_reference"] / "metacog_lite_v0"
        for filename in ("agent.py", "networks.py", "envs.py"):
            self.assertEqual(
                (local_source / filename).read_bytes(),
                (backbone_source / filename).read_bytes(),
                filename,
            )

    def test_canonical_command_routes_every_full_per_setting(self) -> None:
        runner = load_runner()
        command = runner.build_command(
            seed=3,
            steps=12_345,
            device="cpu",
            output_root=None,
            smoke=False,
            validate_only=False,
            resume=False,
            resume_path=None,
        )
        self.assertEqual(option(command, "--method"), "ddpg_her_per_full")
        self.assertEqual(option(command, "--task_name"), "drawer_open")
        self.assertEqual(option(command, "--total_steps"), "12345")
        self.assertEqual(option(command, "--device"), "cpu")
        self.assertEqual(option(command, "--per_alpha"), "0.6")
        self.assertEqual(option(command, "--per_beta_start"), "0.4")
        self.assertEqual(option(command, "--per_beta_end"), "1.0")
        self.assertEqual(option(command, "--per_beta_steps"), "2000000")
        self.assertEqual(option(command, "--weight_scope"), "critic_only")
        self.assertIn("ddpg_her_per_full_s3_12345", option(command, "--run_name"))
        self.assertIn("--no-support_cache_enabled", command)
        self.assertIn("--no-local_scan_cache_enabled", command)
        self.assertIn("--no-numba_support_scan", command)
        self.assertIn("--no_save_step_checkpoints", command)

        smoke = runner.build_command(1, None, "cpu", None, True, False)
        self.assertTrue(option(smoke, "--run_name").endswith("_smoke"))
        self.assertIn("--smoke", smoke)

    def test_resume_is_explicit(self) -> None:
        runner = load_runner()
        command = runner.build_command(
            seed=2,
            steps=2_000_000,
            device="cuda",
            output_root="/tmp/full-per-resume",
            smoke=False,
            validate_only=False,
            resume=True,
            resume_path="/tmp/full-per-resume/checkpoint.pt",
        )
        self.assertIn("--resume", command)
        self.assertEqual(option(command, "--resume_path"), "/tmp/full-per-resume/checkpoint.pt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
