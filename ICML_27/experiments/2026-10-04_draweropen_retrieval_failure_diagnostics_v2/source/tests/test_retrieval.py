"""Deferred until PAL ends: synthetic NumPy/Torch/copied-scanner checks."""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream/code/src"))
import numpy as np
from retrieval_core import choose_candidates, scan_reference, summarize_method
from metacog_lite_v0.replay import HERReplayBuffer, ReplayActionIndex, SampleBatch
from metacog_lite_v0.task_drawer_open import MetaWorldDrawerOpenTaskSpec


def example(n=800, witness_at_end=False):
    task = MetaWorldDrawerOpenTaskSpec(np.full(4, -1, np.float32), np.ones(4, np.float32), .05)
    replay = HERReplayBuffer(capacity=n, obs_dim=39, goal_dim=3, action_dim=4)
    obs = np.zeros(39, np.float32)
    obs[5] = .1
    goal = obs[4:7].copy()
    for i in range(n):
        witness = i == (n - 1 if witness_at_end else 0)
        action = np.full(4, .8 if witness else -.8, np.float32)
        replay.add_transition(obs, goal, goal, action, -1., obs, goal, False, False, 200,
                              episode_id=i // 200, episode_step=i % 200,
                              coarse_key=task.coarse_hash(obs, 200),
                              action_bin=int(task.action_bin_batch(action[None], 5)[0]))
    cfg = SimpleNamespace(N_local_min=4, M_Q_min=4, M_B=16)
    return task, replay, obs, goal, cfg


class RetrievalTests(unittest.TestCase):
    def test_tail_not_sorted_or_recency_rebuilt(self):
        candidates = np.array([8, 1, 6, 3, 2])
        np.testing.assert_array_equal(choose_candidates(candidates, "tail", 2), [3, 2])

    def test_random_without_replacement_and_subset(self):
        selected = choose_candidates(np.arange(1000), "random", 512, np.random.default_rng(7))
        self.assertEqual(len(selected), 512)
        self.assertEqual(len(np.unique(selected)), 512)
        self.assertTrue(np.all((selected >= 0) & (selected < 1000)))

    def test_small_or_empty_buckets(self):
        for n in (0, 2, 512):
            for mode in ("tail", "random", "full"):
                np.testing.assert_array_equal(choose_candidates(np.arange(n), mode, 512), np.arange(n))

    def test_random_repeat_reproducible(self):
        a = choose_candidates(np.arange(900), "random", 512, np.random.default_rng(8))
        b = choose_candidates(np.arange(900), "random", 512, np.random.default_rng(8))
        np.testing.assert_array_equal(a, b)

    def test_index_restore_preserves_valid_order(self):
        index = ReplayActionIndex(6)
        for slot in (4, 1, 3, 1, 0):
            index.update(slot, 10, slot)
        restored = ReplayActionIndex(6)
        restored.load_state_dict(copy.deepcopy(index.state_dict()), 6)
        np.testing.assert_array_equal(restored.candidates(10, 6), [4, 1, 3, 0])

    def test_support_outside_tail_is_detected(self):
        task, replay, obs, goal, cfg = example()
        key = task.full_context_hash(obs, goal, 200)
        policy_bin = int(task.action_bin_batch(np.full((1, 4), .8, np.float32), 5)[0])
        tail = scan_reference(task, replay, np.arange(800)[-512:], goal, key, policy_bin, cfg)
        full = scan_reference(task, replay, np.arange(800), goal, key, policy_bin, cfg)
        self.assertTrue(tail["eligible"])
        self.assertFalse(tail["support"])
        self.assertTrue(full["support"])

    def test_population_threshold_still_required(self):
        task, replay, obs, goal, cfg = example(n=2)
        key = task.full_context_hash(obs, goal, 200)
        policy_bin = int(task.action_bin_batch(np.full((1, 4), .8, np.float32), 5)[0])
        result = scan_reference(task, replay, np.arange(2), goal, key, policy_bin, cfg)
        self.assertTrue(result["occupancy"])
        self.assertFalse(result["support"])

    def test_her_query_goal_not_stored_goal(self):
        task, replay, obs, goal, cfg = example(n=8)
        replay.desired_goal[:] = 100
        key = task.full_context_hash(obs, goal, 200)
        policy_bin = int(task.action_bin_batch(np.full((1, 4), .8, np.float32), 5)[0])
        result = scan_reference(task, replay, np.arange(8), goal, key, policy_bin, cfg)
        self.assertEqual(result["local_count"], 8)
        self.assertTrue(result["support"])

    def test_full_context_implies_coarse_key(self):
        task, _, _, _, _ = example(n=2)
        rng = np.random.default_rng(73)
        obs = rng.normal(size=(3000, 39)).astype(np.float32)
        h = rng.integers(1, 201, size=3000)
        goals = np.repeat(np.zeros((1, 3), np.float32), 3000, axis=0)
        full = task.full_context_hash_batch(obs, goals, h)
        coarse = task.coarse_hash_batch(obs, h)
        for key in np.unique(full):
            self.assertEqual(len(np.unique(coarse[full == key])), 1)

    def test_zero_full_positive_has_na_miss_rate(self):
        row = {"query": 0, "support": False, "eligible": False, "occupancy": False,
               "reference_lookup_seconds": .001, "coarse_count": 0, "scanned_count": 0,
               "local_count": 0, "truncated": False}
        result = summarize_method([row], {0: row}, {0: row})
        self.assertIsNone(result["miss_rate_given_full_positive"])
        self.assertEqual(result["support_disagreement_vs_full"], 0)

    def test_impossible_false_positive_blocks(self):
        row = {"query": 0, "support": True, "eligible": True}
        full = {0: {"support": False, "eligible": True}}
        with self.assertRaises(ValueError):
            summarize_method([row], full, full)

    def test_optimized_frozen_tail_masks_and_weights(self):
        import torch
        from metacog_lite_v0.config import LiteConfig
        from metacog_lite_v0.weights import GapStats, LiteWeightComputer
        torch.set_num_threads(1)

        class Policy:
            def policy_actions(self, state):
                return torch.full((len(state), 4), .8, dtype=torch.float32)

        for n, at_end in ((2, False), (32, False), (800, False), (800, True)):
            with self.subTest(n=n, witness_at_end=at_end):
                task, replay, obs, goal, _ = example(n=n, witness_at_end=at_end)
                cfg = LiteConfig(task_name="drawer_open", method="b3_support_runtime_opt",
                                 assignment_experiment=True, assignment_rule="support",
                                 N_local_min=4, M_Q_min=4, M_B=16, action_bins_per_dim=5,
                                 local_search_max=512, support_scan_mode="deterministic_tail_v1",
                                 policy_support_mode="bin", support_query_mode="current_policy",
                                 support_assignment_mode="support", weight_scope="critic_only",
                                 weighting_mode="batch_norm", lambda_support=1.0)
                gap = GapStats(cfg.gap_window, cfg.N_gap_min, cfg.gap_q_low, cfg.gap_q_high, cfg.eps_G)
                computer = LiteWeightComputer(cfg, task, gap)
                batch = SampleBatch(indices=np.array([0]), observation=obs[None], achieved_goal=goal[None],
                                    desired_goal_train=goal[None], next_observation=obs[None],
                                    next_achieved_goal=goal[None], action_exec=np.zeros((1, 4), np.float32),
                                    reward_ext=np.array([-1.]), done=np.array([False]), timeout=np.array([False]),
                                    h=np.array([200]), her_relabel_ratio=1.0)
                output = computer.compute(batch, replay, Policy(), np.random.default_rng(1), torch.device("cpu"))
                key = task.full_context_hash(obs, goal, 200)
                policy_bin = int(task.action_bin_batch(np.full((1, 4), .8, np.float32), 5)[0])
                candidates = replay.action_index.candidates(task.coarse_hash(obs, 200), replay.size)
                result = scan_reference(task, replay, choose_candidates(candidates, "tail", 512),
                                        goal, key, policy_bin, cfg)
                self.assertEqual(bool(output.support_mask[0]), result["support"])
                self.assertEqual(bool(output.context_eligible_mask[0]), result["eligible"])
                np.testing.assert_allclose(output.omega_critic.mean(), 1.)
                np.testing.assert_array_equal(output.omega_actor, [1.])


if __name__ == "__main__":
    unittest.main()
