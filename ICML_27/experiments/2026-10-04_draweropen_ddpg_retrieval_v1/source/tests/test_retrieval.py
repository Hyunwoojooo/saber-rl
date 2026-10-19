import copy
import csv
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol import ROOT, make_config, evaluation_seed
from analyze import joint_bootstrap, evaluation_curve, curve_metrics
from metacog_lite_v0.replay import HERReplayBuffer, ReplayActionIndex
from metacog_lite_v0.task_drawer_open import MetaWorldDrawerOpenTaskSpec
from metacog_lite_v0.weights import LiteWeightComputer, GapStats
from check_tail_parity import identical
import legacy_checks


def world(capacity=1024, writes=1400):
    task = MetaWorldDrawerOpenTaskSpec(np.full(4, -1, np.float32), np.ones(4, np.float32), epsilon_g=.05)
    replay = HERReplayBuffer(capacity, 7, 3, 4)
    for i in range(writes):
        obs = np.array([0, 0, 0, .2, .25, .1, 0], np.float32)
        if i % 17 == 0:
            obs[4] = .8
        action = np.full(4, 0 if i in (400, 600, 800) else .9, np.float32)
        goal = np.array([.6, .1, 0], np.float32)
        replay.add_transition(obs, obs[4:7], goal, action, -1, obs, obs[4:7], False,
                              i % 200 == 199, 20, i // 200, i % 200,
                              task.coarse_hash(obs, 20), int(task.action_bin_batch(action[None], 5)[0]))
    batch = replay.sample(64, .8, np.random.default_rng(18))
    actor = SimpleNamespace(policy_actions=lambda state: torch.zeros((len(state), 4)))
    return task, replay, batch, actor


def computer(task, mode="random512", seed=1):
    cfg = make_config(mode, seed, ROOT / "outputs/test-unused", device="cpu")
    return LiteWeightComputer(cfg, task, GapStats(50000, 128, .1, .9, 1e-6))


def compute(c, replay, batch, actor, rng=None):
    return c.compute(batch, replay, actor, rng or np.random.default_rng(20001),
                     torch.device("cpu"), assignment_rng=np.random.default_rng(30001))


class RetrievalTests(unittest.TestCase):
    def test_protocol_only_changes_scan_mode(self):
        a = make_config("tail", 4, Path("/tmp/test")).as_dict()
        b = make_config("random512", 4, Path("/tmp/test")).as_dict()
        self.assertEqual([k for k in a if a[k] != b[k]], ["support_scan_mode"])
        self.assertEqual(a["assignment_rule"], "support")
        for arm in ("tail", "random512"):
            cfg = make_config(arm, 4, Path("/tmp/test"))
            cfg.resume = True
            with self.assertRaises(ValueError):
                cfg.normalize()
        self.assertEqual(evaluation_seed(4, 10000, 3), 110007)

    def test_random_retrieval_rejects_other_assignment_rules(self):
        cfg = make_config("random512", 1, Path("/tmp/test"))
        cfg.assignment_rule = "random"
        with self.assertRaises(ValueError):
            cfg.normalize()

    def test_live_unique_uniform_without_replacement(self):
        idx = ReplayActionIndex(50)
        for i in range(50):
            idx.update(i, 2, i % 5)
            idx.update(i, 2, i % 5)
        idx.clear_slot(3)
        idx.update(7, 9, 1)
        valid = set(idx.candidates(2, 50))
        self.assertEqual(len(valid), 48)
        counts = np.zeros(50)
        rng = np.random.default_rng(77)
        for _ in range(5000):
            sample, n, truncated = idx.sample_candidates(2, 50, 8, rng)
            self.assertEqual((len(sample), len(set(sample)), n, truncated), (8, 8, 48, True))
            self.assertTrue(set(sample) <= valid)
            counts[sample] += 1
        np.testing.assert_allclose(counts[list(valid)], 5000 * 8 / 48, rtol=.15)

    def test_empty_and_small_buckets_do_not_advance_rng(self):
        idx = ReplayActionIndex(12)
        rng = np.random.default_rng(11)
        initial = copy.deepcopy(rng.bit_generator.state)
        sample, n, trunc = idx.sample_candidates(1, 12, 512, rng)
        self.assertEqual((len(sample), n, trunc), (0, 0, False))
        for i in range(12):
            idx.update(i, 1, 0)
        sample, n, trunc = idx.sample_candidates(1, 12, 512, rng)
        np.testing.assert_array_equal(sample, np.arange(12))
        self.assertEqual(rng.bit_generator.state, initial)

    def test_small_buckets_all_modes_identical(self):
        task, replay, batch, actor = world(64, 100)
        a, b = computer(task, "tail"), computer(task)
        wa, wb = compute(a, replay, batch, actor), compute(b, replay, batch, actor)
        for field in ("support_mask", "context_eligible_mask", "omega_critic", "omega_actor"):
            np.testing.assert_array_equal(getattr(wa, field), getattr(wb, field))

    def test_tail_matches_frozen_masks_weights_after_wrap(self):
        task, replay, batch, actor = world()
        legacy_checks.load_upstream()
        old_module = importlib.import_module("assignment_upstream.weights")
        new = computer(task, "tail")
        old = old_module.LiteWeightComputer(copy.deepcopy(new.cfg), task,
                                           old_module.GapStats(50000, 128, .1, .9, 1e-6))
        for _ in range(3):
            a, b = compute(old, replay, batch, actor), compute(new, replay, batch, actor)
            for field in ("support_mask", "context_eligible_mask", "omega_critic", "omega_actor",
                          "num_coarse_candidates", "num_scanned_candidates"):
                np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        self.assertTrue(new._numba_support_scan_enabled)

    def test_shared_subset_within_update_redraw_next_update(self):
        task, replay, batch, actor = world()
        c = computer(task)
        compute(c, replay, batch, actor)
        first = {k: v.indices.copy() for k, v in c._update_scans.items()}
        rng_state = copy.deepcopy(c._retrieval_rng.bit_generator.state)
        for key, selected in first.items():
            scanned, hit = c._runtime_opt_local_scan(key, replay)
            self.assertTrue(hit)
            np.testing.assert_array_equal(selected, scanned.indices)
        self.assertEqual(rng_state, c._retrieval_rng.bit_generator.state)
        compute(c, replay, batch, actor)
        self.assertNotEqual(rng_state, c._retrieval_rng.bit_generator.state)
        self.assertTrue(any(not np.array_equal(v, c._update_scans[k].indices) for k, v in first.items()))
        self.assertEqual(len(c._local_scan_cache), 0)

    def test_optimized_masks_equal_exact_selected_subset_reference(self):
        task, replay, batch, actor = world()
        c = computer(task)
        wb = compute(c, replay, batch, actor)
        coarse = task.coarse_hash_batch(batch.observation, batch.h)
        keys = task.full_context_hash_batch(batch.observation, batch.desired_goal_train, batch.h)
        policy_bin = int(task.action_bin_batch(np.zeros((1, 4), np.float32), 5)[0])
        for i, key in enumerate(coarse):
            scan = c._update_scans[int(key)]
            selected = scan.indices
            self.assertEqual(len(selected), min(512, len(replay.action_index.candidates(int(key), replay.size))))
            goals = np.repeat(batch.desired_goal_train[i:i+1], len(selected), axis=0)
            full = task.full_context_hash_batch(replay.observation[selected], goals, replay.h[selected])
            matching = selected[full == keys[i]]
            eligible = len(matching) >= 4
            support = eligible and bool(np.any(replay.action_bin[matching] == policy_bin))
            self.assertEqual(bool(wb.context_eligible_mask[i]), eligible)
            self.assertEqual(bool(wb.support_mask[i]), support)
        raw = 1 + wb.support_mask.astype(np.float32)
        np.testing.assert_allclose(wb.omega_critic, raw / raw.mean(), atol=0, rtol=0)
        np.testing.assert_array_equal(wb.omega_actor, np.ones(len(batch.h)))

    def test_no_sampling_her_exploration_global_rng_side_effects(self):
        task, replay, batch, actor = world()
        c = computer(task)
        caller = np.random.default_rng(20001)
        state = copy.deepcopy(caller.bit_generator.state)
        np_state, torch_state = np.random.get_state(), torch.get_rng_state().clone()
        compute(c, replay, batch, actor, caller)
        identical(np_state, np.random.get_state())
        identical(torch_state, torch.get_rng_state())
        self.assertEqual(state, caller.bit_generator.state)

    def test_cache_toggle_does_not_change_random_subsets_or_masks(self):
        task, replay, batch, actor = world()
        a, b = computer(task), computer(task)
        b.cfg.support_cache_enabled = False
        b.cfg.local_scan_cache_enabled = False
        for _ in range(3):
            x, y = compute(a, replay, batch, actor), compute(b, replay, batch, actor)
            np.testing.assert_array_equal(x.support_mask, y.support_mask)
            identical(a.retrieval_state_dict(), b.retrieval_state_dict())

    def test_retrieval_rng_checkpoint_roundtrip(self):
        task, replay, batch, actor = world()
        a, b = computer(task), computer(task)
        compute(a, replay, batch, actor)
        b.load_retrieval_state_dict(copy.deepcopy(a.retrieval_state_dict()))
        wa, wb = compute(a, replay, batch, actor), compute(b, replay, batch, actor)
        np.testing.assert_array_equal(wa.support_mask, wb.support_mask)
        for key in a._update_scans:
            np.testing.assert_array_equal(a._update_scans[key].indices, b._update_scans[key].indices)
        identical(a.retrieval_state_dict(), b.retrieval_state_dict())
        with self.assertRaises(ValueError):
            computer(task, seed=2).load_retrieval_state_dict(a.retrieval_state_dict())

    def test_frozen_learning_modules_byte_identical(self):
        for name in ("replay.py", "agent.py", "task_drawer_open.py", "envs.py", "networks.py",
                     "numba_support_scan_drawer_open.py", "assignment_comparison.py"):
            self.assertEqual((ROOT / "code/src/metacog_lite_v0" / name).read_bytes(),
                             (ROOT / "upstream/code/src/metacog_lite_v0" / name).read_bytes())


class AnalysisTests(unittest.TestCase):
    def test_bootstrap_direction_seed_unit_and_quantiles(self):
        a = np.arange(20, dtype=float).reshape(10, 2) / 20
        result = joint_bootstrap(a)["random512_minus_tail"]
        self.assertAlmostEqual(result["mean"], .05)
        self.assertEqual(result["positive_pairs"], 10)
        np.testing.assert_allclose(result["ci95"], [.05, .05])
        a[:, 1] = np.linspace(0, 1, 10)[::-1]
        rows = np.random.default_rng(20261004).integers(0, 10, size=(10000, 10))
        ref = np.quantile((a[:, 1]-a[:, 0])[rows].mean(axis=1), [.025, .975])
        np.testing.assert_array_equal(joint_bootstrap(a)["random512_minus_tail"]["ci95"], ref)

    def test_incomplete_and_nonfinite_ci_rejected(self):
        for values in (np.zeros((9, 2)), np.zeros((10, 3)), np.full((10, 2), np.nan)):
            with self.assertRaises(ValueError):
                joint_bootstrap(values)

    def test_auc_exact_span_and_missing_checkpoint(self):
        grid = np.arange(10000, 2000001, 10000)
        self.assertEqual(curve_metrics(grid, np.ones(200), 1500000)["auc"], 1)
        self.assertEqual(curve_metrics(grid, np.zeros(200), 1500000)["hit90"], None)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.csv"
            with path.open("w") as f:
                f.write("step,eval/env_success_mean\n10000,0\n30000,0\n")
            with self.assertRaises(ValueError):
                evaluation_curve(path, grid)


def load_tests(loader, tests, pattern):
    for cls in (legacy_checks.AssignmentTests, legacy_checks.UpdateTests):
        tests.addTests(loader.loadTestsFromTestCase(cls))
    for name in ("test_strict_context_eligible_count_matched", "test_strict_ddpg_loss_scope",
                 "test_strict_drawer_open_scanner", "test_strict_config_resume_contract"):
        module = importlib.import_module(name)
        for key in sorted(vars(module)):
            if key.startswith("test_") and callable(getattr(module, key)):
                tests.addTest(unittest.FunctionTestCase(getattr(module, key)))
    return tests


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
