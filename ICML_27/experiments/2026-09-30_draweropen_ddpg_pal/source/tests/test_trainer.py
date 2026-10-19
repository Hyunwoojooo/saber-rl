"""A tiny mocked environment checks PAL wiring; real smoke remains separate."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol import ROOT, make_config
from metacog_lite_v0 import trainer
from metacog_lite_v0.envs import GoalObservation


class TinyEnv:
    obs_dim, goal_dim, action_dim, state_dim = 3, 1, 2, 4
    max_episode_steps = 2
    action_low, action_high = np.full(2, -1, np.float32), np.full(2, 1, np.float32)

    def __init__(self):
        self.t = 0
        self.reset_seeds = []

    def observation(self):
        return GoalObservation(np.array([self.t, 0, 0], np.float32),
                               np.array([self.t], np.float32), np.array([9], np.float32), 0.0)

    def reset(self, seed=None):
        self.t = 0
        self.reset_seeds.append(seed)
        return self.observation()

    def step(self, action):
        self.t += 1
        return self.observation(), -1.0, False, self.t == 2, {"env_success": 0, "goal_success": 0}

    def compute_reward(self, achieved, desired, info=None):
        return -(np.abs(achieved-desired).reshape(-1) > 0.05).astype(np.float32)

    def success_from_goal(self, achieved, desired):
        return (np.abs(achieved-desired).reshape(-1) <= 0.05).astype(np.float32)

    def snapshot(self):
        return {"mock": True}

    def close(self):
        pass


class TrainerTests(unittest.TestCase):
    def test_no_support_query_and_same_eval_seed_formula(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("PAL executed support-only code")

        env, evaluation = TinyEnv(), TinyEnv()
        task = SimpleNamespace(phase_names=("phase",), metadata=lambda: {},
                               coarse_hash=forbidden, action_bin_batch=forbidden)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config("pal", 1, tmp, "smoke", "cpu")
            cfg.total_steps = 4
            cfg.start_steps = 2
            cfg.warmup_steps = 2
            cfg.batch_size = 2
            cfg.buffer_capacity = 8
            cfg.hidden_dim = 8
            cfg.eval_freq = cfg.checkpoint_every_steps = 2
            checkpoints = []
            with patch.object(trainer, "make_env", side_effect=[env, evaluation]), \
                 patch.object(trainer, "make_task_spec", return_value=task), \
                 patch.object(trainer, "LiteWeightComputer", side_effect=forbidden), \
                 patch.object(trainer, "save_checkpoint", side_effect=lambda p, v: checkpoints.append(v)):
                directory = trainer.run_training(cfg, ROOT)
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(summary["total_updates"], 3)
            self.assertFalse(summary["support_computation_enabled"])
            self.assertEqual(summary["replay_support_index_slots"], 0)
            self.assertFalse(summary["legacy_live_target_diagnostic_enabled"])
            self.assertFalse((directory / "assignment_windows.csv").exists())
            self.assertTrue(checkpoints)
            self.assertTrue(all(p["critic_objective"] == "pal" for p in checkpoints))
            self.assertTrue(all(not p["checkpoint_continuation_permitted"] for p in checkpoints))
            self.assertEqual(evaluation.reset_seeds, [10021, 10022, 10041, 10042])
            with (directory / "metrics.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(all(float(r["pal/normalizer"]) >= 1 for r in rows))
            self.assertTrue(all(np.isnan(float(r["lite/support_available_rate"])) for r in rows))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
