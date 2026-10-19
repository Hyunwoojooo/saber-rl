import copy
import importlib
import json
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from test_contract import ROOT, reference_package, agent_kwargs, update_inputs, assert_nested, fixture
from protocol import SPEC, make_config, make_gpu_validation_config
from metacog_lite_v0.agent import SACAgent
from metacog_lite_v0.replay import HERReplayBuffer
from metacog_lite_v0.hgr import HGRSampler
from metacog_lite_v0.baselines import apply_baseline_weights, per_beta_at_step
from metacog_lite_v0.weights import LiteWeightComputer, GapStats
from metacog_lite_v0.trainer import make_agent


def fill(replay, sampler=None, start=0, count=8, length=4):
    for i in range(start, start + count):
        slot = replay.ptr
        ep, step = divmod(i, length)
        replay.add_transition(observation=np.array([i, -i], np.float32),
            achieved_goal=np.array([i], np.float32), desired_goal=np.array([-100 - ep], np.float32),
            action_exec=np.array([.1 * (i % 10)], np.float32), reward_ext=-1.,
            next_observation=np.array([i + 1, -i - 1], np.float32),
            next_achieved_goal=np.array([i + 1], np.float32), done=step == length - 1,
            timeout=step == length - 1, h=length - step, episode_id=ep, episode_step=step,
            coarse_key=ep, action_bin=i % 5)
        if sampler is not None:
            sampler.on_replay_add(slot, ep, step, replay, episode_done=step == length - 1)


def small_replay(capacity=16):
    return HERReplayBuffer(capacity=capacity, obs_dim=2, goal_dim=1, action_dim=1)


def small_hgr(capacity=16, cls=HGRSampler):
    cfg = make_config("drawer_open", "hgr", 1, "cpu")
    cfg.buffer_capacity = capacity
    return cls(cfg)


def compare_batches(test, a, b, keys):
    for key in keys:
        x, y = getattr(a, key), getattr(b, key)
        if isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y, err_msg=key)
        else:
            test.assertEqual(x, y, key)


class BaselineContract(unittest.TestCase):
    def test_twenty_configs_only_declared_differences(self):
        count = 0
        for task in SPEC["tasks"]:
            for seed in SPEC["seeds"]:
                her = make_config(task, "her", seed, "cpu").as_dict()
                for method in SPEC["methods"]:
                    cfg = make_config(task, method, seed, "cpu")
                    data = cfg.as_dict()
                    self.assertEqual({k for k in data if data[k] != her[k]}, {"method", "weight_scope", "run_name"})
                    cfg.normalize()
                    self.assertEqual(data, cfg.as_dict())
                    self.assertEqual(cfg.per_beta_steps, 2000000)
                    self.assertEqual(cfg.hgr_beta_anneal_updates, 1995001)
                    self.assertEqual(cfg.weight_scope, "critic_only")
                    self.assertEqual(cfg.lambda_support, 0)
                    count += 1
        self.assertEqual(count, 20)

    def test_validation_keeps_annealing_schedule_and_training_scale(self):
        for method in SPEC["methods"]:
            cfg = make_gpu_validation_config("drawer_close", method, 1, "stability", ROOT / "validation/test")
            self.assertEqual((cfg.total_steps, cfg.batch_size, cfg.buffer_capacity, cfg.warmup_steps),
                             (100000, 256, 300000, 5000))
            self.assertEqual((cfg.per_beta_steps, cfg.hgr_beta_anneal_updates), (2000000, 1995001))

    def test_agents_and_no_support_lookup(self):
        _, task, replay, batch = fixture("drawer_open")
        for method, reduction in (("hgr", "min_abs"), ("per", "mean_abs")):
            cfg = make_config("drawer_open", method, 1, "cpu")
            agent = make_agent(cfg, 15, 4, -np.ones(4, np.float32), np.ones(4, np.float32), torch.device("cpu"))
            self.assertIsInstance(agent, SACAgent)
            self.assertEqual(agent.priority_reduction, reduction)
            wc = LiteWeightComputer(cfg, task, GapStats(1000, 128, .6, .95, 1e-6))
            wb = wc.compute(batch, replay, None, None, torch.device("cpu"))
            np.testing.assert_array_equal(wb.omega_critic, np.ones(3))
            np.testing.assert_array_equal(wb.psi_q_rank, np.zeros(3))

    def test_priority_uses_same_preupdate_target_and_does_not_change_rng_or_update(self):
        for mode in ("min_abs", "mean_abs"):
            a, b = SACAgent(**agent_kwargs()), SACAgent(**agent_kwargs())
            b.load_state_dict(copy.deepcopy(a.state_dict()))
            a.priority_reduction = mode
            data = update_inputs()
            data["omega_critic"] = torch.arange(1, 9, dtype=torch.float32) / 4.5
            torch.manual_seed(802)
            with torch.no_grad():
                action, logp = a.actor.sample(data["next_state"], a.action_scale_t, a.action_bias_t)
                target = data["reward"] + a.gamma * data["nonterminal"] * (
                    torch.minimum(a.critic1_target(data["next_state"], action),
                                  a.critic2_target(data["next_state"], action)) - a.alpha * logp)
                d1 = (a.critic1(data["state"], data["action_exec"]) - target).abs()
                d2 = (a.critic2(data["state"], data["action_exec"]) - target).abs()
                expected = torch.minimum(d1, d2) if mode == "min_abs" else .5 * (d1 + d2)
            torch.manual_seed(802)
            ma = a.update(**data)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(802)
            mb = b.update(**data)
            np.testing.assert_array_equal(a.last_priority_td_abs, expected.numpy())
            torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
            assert_nested(self, vars(ma), vars(mb))
            assert_nested(self, a.state_dict(), b.state_dict())


class PERContract(unittest.TestCase):
    def test_sampling_and_uniform_rng_parity_with_frozen_sources(self):
        for label in ("per", "current"):
            cls = importlib.import_module(reference_package(label) + ".replay").HERReplayBuffer
            a, b = small_replay(), cls(capacity=16, obs_dim=2, goal_dim=1, action_dim=1)
            fill(a, count=24)
            fill(b, count=24)
            keys = ["indices", "observation", "desired_goal_train", "action_exec", "her_relabel_ratio", "done", "timeout"]
            for seed in range(10):
                x, y = np.random.default_rng(seed), np.random.default_rng(seed)
                compare_batches(self, a.sample(8, .8, x), b.sample(8, .8, y), keys)
                self.assertEqual(x.bit_generator.state, y.bit_generator.state)
            if label == "per":
                a.priority[:] = b.priority[:] = np.arange(1, 17, dtype=np.float32)
                for seed in range(10):
                    x, y = np.random.default_rng(seed), np.random.default_rng(seed)
                    compare_batches(self, a.sample_prioritized(8, .8, x, .6, .4, 1e-6),
                                    b.sample_prioritized(8, .8, y, .6, .4, 1e-6),
                                    keys + ["sampling_prob", "is_weight", "sampling_priority"])
                    self.assertEqual(x.bit_generator.state, y.bit_generator.state)

    def test_probabilities_weights_and_future_convention(self):
        replay = small_replay()
        fill(replay)
        replay.priority[:8] = np.arange(1, 9, dtype=np.float32)
        prob = replay.priority[:8].astype(np.float64) ** .6
        prob /= prob.sum()
        counts = np.zeros(8, dtype=np.int64)
        for seed in range(200):
            batch = replay.sample_prioritized(8, 1., np.random.default_rng(seed), .6, .4, 1e-6)
            counts += np.bincount(batch.indices, minlength=8)
            np.testing.assert_allclose(batch.sampling_prob, prob[batch.indices], rtol=1e-6)
            expected = (8 * prob[batch.indices]) ** -.4
            expected /= expected.max()
            np.testing.assert_allclose(batch.is_weight, expected, rtol=1e-6)
            for pos, slot in enumerate(batch.indices):
                if replay.episode_step[slot] == 3:
                    np.testing.assert_array_equal(batch.desired_goal_train[pos], replay.desired_goal[slot])
                else:
                    future = int(batch.desired_goal_train[pos, 0]) - 1
                    self.assertEqual(replay.episode_id[future], replay.episode_id[slot])
                    self.assertGreater(replay.episode_step[future], replay.episode_step[slot])
        np.testing.assert_allclose(counts / counts.sum(), prob, atol=.04)

    def test_mean_normalization_and_beta(self):
        cfg = make_config("drawer_close", "per", 1, "cpu")
        for step, expected in ((0, .4), (1000000, .7), (2000000, 1), (3000000, 1)):
            self.assertAlmostEqual(per_beta_at_step(cfg, step), expected)
        wb = SimpleNamespace(extra_metrics={})
        batch = SimpleNamespace(indices=np.arange(3), is_weight=np.array([.2, .5, 1], np.float32),
                                per_beta=.4, sampling_prob=np.array([.1, .2, .3]))
        apply_baseline_weights(cfg, wb, batch, None)
        np.testing.assert_allclose(wb.omega_critic, batch.is_weight / batch.is_weight.mean())
        np.testing.assert_array_equal(wb.omega_actor, [1, 1, 1])
        batch.is_weight[0] = float("nan")
        with self.assertRaises(FloatingPointError):
            apply_baseline_weights(cfg, wb, batch, None)

    def test_duplicate_priority_max_checkpoint_and_corruption(self):
        replay = small_replay()
        fill(replay)
        replay.update_priorities(np.array([1, 1, 2]), np.array([.2, 3, 1]), 1e-6)
        self.assertAlmostEqual(float(replay.priority[1]), 3.000001, places=5)
        saved = copy.deepcopy(replay.state_dict())
        restored = small_replay()
        restored.load_state_dict(saved)
        for key in ("priority", "action_exec", "episode_id"):
            np.testing.assert_array_equal(getattr(replay, key), getattr(restored, key))
        a = replay.sample_prioritized(8, .8, np.random.default_rng(9), .6, .4, 1e-6)
        b = restored.sample_prioritized(8, .8, np.random.default_rng(9), .6, .4, 1e-6)
        compare_batches(self, a, b, ["indices", "desired_goal_train", "is_weight"])
        with self.assertRaises(FloatingPointError):
            replay.update_priorities([0], [float("nan")], 1e-6)
        with self.assertRaises(IndexError):
            replay.update_priorities([-1], [1], 1e-6)
        replay.priority[0] = float("nan")
        with self.assertRaises(FloatingPointError):
            replay.sample_prioritized(8, .8, np.random.default_rng(9), .6, .4, 1e-6)


class HGRContract(unittest.TestCase):
    def test_sampling_update_and_rng_parity_with_fpp(self):
        original = importlib.import_module(reference_package("hgr") + ".hgr").HGRSampler
        a, b = small_hgr(), small_hgr(cls=original)
        ra, rb = small_replay(), small_replay()
        fill(ra, a)
        fill(rb, b)
        keys = ["indices", "desired_goal_train", "action_exec", "her_relabel", "hgr_episode_id",
                "hgr_pair_index", "hgr_goal_slot", "hgr_episode_probability", "hgr_pair_probability", "hgr_is_weight"]
        for seed in range(20):
            x, y = np.random.default_rng(seed), np.random.default_rng(seed)
            ba, bb = a.sample_batch(ra, 8, .8, x), b.sample_batch(rb, 8, .8, y)
            compare_batches(self, ba, bb, keys)
            self.assertEqual(x.bit_generator.state, y.bit_generator.state)
            td = np.linspace(.1, 2, 8, dtype=np.float32)
            self.assertEqual(a.update_priorities(ba, td), b.update_priorities(bb, td))
            for ep in a._tables:
                np.testing.assert_array_equal(a._tables[ep].priorities, b._tables[ep].priorities)

    def test_triangular_pairs_native_immediate_goal_and_weights(self):
        replay, hgr = small_replay(), small_hgr()
        fill(replay, hgr)
        self.assertEqual(hgr.pair_table_stats()[:2], (2, 20))
        saw_immediate = False
        for seed in range(30):
            b = hgr.sample_batch(replay, 8, 1., np.random.default_rng(seed))
            for pos, slot in enumerate(b.indices):
                goal = b.hgr_goal_slot[pos]
                self.assertEqual(replay.episode_id[slot], replay.episode_id[goal])
                self.assertGreaterEqual(replay.episode_step[goal], replay.episode_step[slot])
                saw_immediate |= goal == slot
                np.testing.assert_array_equal(b.desired_goal_train[pos], replay.next_achieved_goal[goal])
        self.assertTrue(saw_immediate)
        b.hgr_is_weight = np.linspace(.1, 1, 8, dtype=np.float32)
        wb = SimpleNamespace()
        apply_baseline_weights(hgr.cfg, wb, b, hgr)
        np.testing.assert_array_equal(wb.omega_critic, b.hgr_is_weight)
        self.assertLess(wb.omega_critic.mean(), 1)
        np.testing.assert_array_equal(wb.omega_actor, np.ones(8))

    def test_episode_and_pair_probability_and_is_formula(self):
        replay, hgr = small_replay(), small_hgr()
        fill(replay, hgr)
        for ep, table in hgr._tables.items():
            table.priorities[:] = np.arange(1, 11, dtype=np.float32) * (ep + 1)
            table.priority_sum = float(table.priorities.sum(dtype=np.float64))
        ids, probabilities, means = hgr.episode_distribution()
        expected = means.astype(np.float64) ** .6
        expected /= expected.sum()
        np.testing.assert_allclose(probabilities, expected, rtol=1e-6)
        b = hgr.sample_batch(replay, 8, .8, np.random.default_rng(55))
        mask = b.her_relabel > .5
        raw = (len(ids) * b.hgr_episode_probability[mask].astype(np.float64)) ** -.4
        raw *= (10 * b.hgr_pair_probability[mask].astype(np.float64)) ** -.4
        np.testing.assert_allclose(b.hgr_is_weight[mask], raw / raw.max(), rtol=1e-6)
        np.testing.assert_array_equal(b.hgr_is_weight[~mask], np.ones((~mask).sum()))
        hgr.critic_updates = hgr.beta_anneal_updates
        self.assertEqual(hgr.current_betas(), (1., 1.))

    def test_ring_overwrite_and_checkpoint_roundtrip(self):
        replay, hgr = small_replay(8), small_hgr(8)
        fill(replay, hgr)
        fill(replay, hgr, start=8, count=1)
        self.assertEqual(set(hgr._tables), {1})
        fill(replay, hgr, start=9, count=3)
        self.assertEqual(set(hgr._tables), {1, 2})
        b = hgr.sample_batch(replay, 8, 1., np.random.default_rng(3))
        hgr.update_priorities(b, np.linspace(.3, 2, 8, dtype=np.float32))
        restored = small_hgr(8)
        restored.load_state_dict(copy.deepcopy(hgr.state_dict()), replay)
        self.assertEqual(restored.critic_updates, hgr.critic_updates)
        a = hgr.sample_batch(replay, 8, .8, np.random.default_rng(7))
        b = restored.sample_batch(replay, 8, .8, np.random.default_rng(7))
        compare_batches(self, a, b, ["indices", "desired_goal_train", "hgr_pair_index", "hgr_is_weight"])
        with self.assertRaises(FloatingPointError):
            hgr.update_priorities(a, np.full(8, np.nan))
        bad = copy.deepcopy(hgr.state_dict())
        bad["tables"][1]["priorities"][0] = np.nan
        with self.assertRaises(FloatingPointError):
            restored.load_state_dict(bad, replay)


if __name__ == "__main__":
    unittest.main()
