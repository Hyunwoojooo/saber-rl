import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protocol import make_config, validate_parity
from manifest import verify_upstream
from analyze import auc, read_curve, paired_intervals

import numpy as np
import torch
from metacog_lite_v0.agent import SACAgent
from metacog_lite_v0.trainer import make_agent, make_env, make_task_spec, evaluate_agent
from metacog_lite_v0.drawer_adapter import nonterminal_mask
from metacog_lite_v0.replay import HERReplayBuffer, SampleBatch
from metacog_lite_v0.weights import GapStats, LiteWeightComputer
from metacog_lite_v0.io_utils import make_run_dir, to_jsonable


def reference_package(label):
    name = f"reference_{label}"
    if name not in sys.modules:
        folder = ROOT / f"upstream/{label}/code/src/metacog_lite_v0"
        spec = importlib.util.spec_from_file_location(name, folder / "__init__.py", submodule_search_locations=[str(folder)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return name


def agent_kwargs():
    return dict(state_dim=15, action_dim=4, hidden_dim=32, actor_lr=3e-4, critic_lr=3e-4,
                gamma=0.98, tau=0.005, action_low=-np.ones(4, np.float32), action_high=np.ones(4, np.float32),
                device=torch.device("cpu"), l2_action_penalty=0.0,
                alpha=0.2, auto_alpha=True, alpha_lr=3e-4, target_entropy=-4.0)


def update_inputs():
    torch.manual_seed(71)
    return dict(state=torch.randn(8, 15), action_exec=torch.tanh(torch.randn(8, 4)),
                reward=-torch.ones(8), next_state=torch.randn(8, 15), nonterminal=torch.ones(8),
                omega_critic=torch.ones(8), omega_actor=torch.ones(8))


def assert_nested(test, a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            assert_nested(test, a[key], b[key])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_nested(test, x, y)
    else:
        test.assertEqual(a, b)


class SACContract(unittest.TestCase):
    def test_20_configs_and_normalization_idempotent(self):
        self.assertEqual(len(validate_parity()), 10)
        for task in ("drawer_open", "drawer_close"):
            for method in ("her", "saber"):
                cfg = make_config(task, method, 5, "cpu")
                old = copy.deepcopy(cfg.as_dict())
                cfg.normalize()
                self.assertEqual(old, cfg.as_dict())
                self.assertEqual((cfg.total_steps, cfg.max_episode_steps), (2000000, 200))
                self.assertTrue(cfg.sac_auto_alpha)
                self.assertEqual(cfg.sac_target_entropy, -4)
        for seed in (0, 6, 10, 11):
            with self.assertRaises(ValueError):
                make_config("drawer_close", "her", seed)

    def test_upstream_unchanged(self):
        self.assertEqual(verify_upstream(check_origins=True), 39)

    def test_fpp_core_parity_three_updates_and_optimizer(self):
        original = importlib.import_module(reference_package("fpp") + ".agent").SACAgent
        torch.manual_seed(11)
        current = SACAgent(**agent_kwargs())
        reference = original(**agent_kwargs())
        reference.load_state_dict(copy.deepcopy(current.state_dict()))
        batch = update_inputs()
        for i, mask in enumerate((torch.ones(8), torch.tensor([1., 2.] * 4) / 1.5, torch.ones(8))):
            batch["omega_critic"] = mask
            torch.manual_seed(200 + i)
            a = current.update(**batch)
            torch.manual_seed(200 + i)
            b = reference.update(**batch)
            assert_nested(self, vars(a), vars(b))
            assert_nested(self, current.state_dict(), reference.state_dict())
        self.assertNotEqual(float(current.alpha.detach()), 0.2)

    def test_twin_loss_replay_action_and_target_formula(self):
        agent = SACAgent(**agent_kwargs())
        batch = update_inputs()
        batch["omega_critic"] = torch.tensor([1., 2.] * 4) / 1.5
        batch["nonterminal"][0] = 0
        torch.manual_seed(119)
        with torch.no_grad():
            action, logp = agent.actor.sample(batch["next_state"], agent.action_scale_t, agent.action_bias_t)
            target = batch["reward"] + .98 * batch["nonterminal"] * (
                torch.minimum(agent.critic1_target(batch["next_state"], action),
                              agent.critic2_target(batch["next_state"], action)) - agent.alpha * logp)
            expected = sum((batch["omega_critic"] * (q(batch["state"], batch["action_exec"]) - target).square()).mean()
                           for q in (agent.critic1, agent.critic2))
        actions = []
        handle = agent.critic1.register_forward_pre_hook(lambda _module, args: actions.append(args[1].detach().clone()))
        torch.manual_seed(119)
        metrics = agent.update(**batch)
        handle.remove()
        self.assertAlmostEqual(metrics.critic_loss, expected.item(), places=6)
        torch.testing.assert_close(actions[0], batch["action_exec"], atol=0, rtol=0)
        self.assertEqual(target[0], batch["reward"][0])

    def test_actor_weights_ignored_and_query_deterministic(self):
        a, b = SACAgent(**agent_kwargs()), SACAgent(**agent_kwargs())
        b.load_state_dict(copy.deepcopy(a.state_dict()))
        batch = update_inputs()
        state = torch.get_rng_state().clone()
        query = a.policy_actions(batch["state"])
        mean, _ = a.actor(batch["state"])
        torch.testing.assert_close(query, torch.tanh(mean))
        torch.testing.assert_close(state, torch.get_rng_state())
        torch.manual_seed(7)
        x = a.update(**batch)
        batch["omega_actor"] = torch.arange(8, dtype=torch.float32) * 100
        torch.manual_seed(7)
        y = b.update(**batch)
        assert_nested(self, vars(x), vars(y))
        assert_nested(self, a.state_dict(), b.state_dict())

    def test_nonfinite_and_broadcast_rejected_before_update(self):
        agent = SACAgent(**agent_kwargs())
        batch = update_inputs()
        before = copy.deepcopy(agent.state_dict())
        batch["reward"][0] = float("nan")
        with self.assertRaises(FloatingPointError):
            agent.update(**batch)
        assert_nested(self, before, agent.state_dict())
        batch = update_inputs()
        batch["omega_critic"] = torch.ones((8, 1))
        with self.assertRaises(ValueError):
            agent.update(**batch)

    def test_no_overwrite_or_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run_dir(root, "run", False)
            with self.assertRaises(FileExistsError):
                make_run_dir(root, "run", False)
            with self.assertRaises(ValueError):
                make_run_dir(root, "run", True)

    def test_nonterminal_contract(self):
        cfg = make_config("drawer_close", "her", 1, "cpu")
        batch = SimpleNamespace(done=np.array([0, 1, 1]), timeout=np.array([0, 0, 1]))
        np.testing.assert_array_equal(nonterminal_mask(cfg, batch, np.ones(3)), [1, 0, 1])
        cfg.timeout_bootstrap = False
        np.testing.assert_array_equal(nonterminal_mask(cfg, batch, np.zeros(3)), [1, 0, 0])

    def test_auc_complete_grid_and_pairing(self):
        self.assertAlmostEqual(auc([(10000, 0), (2000000, 1)]), .5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows([["step", "eval/env_success_mean"], [10000, .2], [30000, .8]])
            with self.assertRaises(ValueError):
                read_curve(path, [10000, 20000, 30000])
            self.assertEqual(len(read_curve(path, [10000, 30000])), 2)
        ci = paired_intervals([.1] * 5)
        np.testing.assert_allclose(ci["ci95"], [.1, .1])
        self.assertEqual(ci["n_pairs"], 5)
        for n in (4, 6, 10):
            with self.assertRaises(ValueError):
                paired_intervals([.1] * n)


def fixture(task_name):
    cfg = make_config(task_name, "saber", 1, "cpu")
    env = SimpleNamespace(action_low=-np.ones(4, np.float32), action_high=np.ones(4, np.float32), distance_threshold=.05)
    task = make_task_spec(cfg, env)
    replay = HERReplayBuffer(capacity=64, obs_dim=12, goal_dim=3, action_dim=4)
    goal = np.array([.1, .2, .3], np.float32)
    for i in range(35):
        group = 0 if i < 16 else 1 if i < 32 else 2
        obs = np.zeros(12, np.float32)
        obs[:3] = [.08, .18, .30]
        obs[4:7] = [.09, .19, .30]
        obs[3] = [-.7, .2, .7][group]
        action = np.full(4, .8 if group == 1 else 0, np.float32)
        replay.add_transition(observation=obs, achieved_goal=obs[4:7], desired_goal=goal + 1,
            action_exec=action, reward_ext=-1, next_observation=obs, next_achieved_goal=goal,
            done=False, timeout=False, h=20, episode_id=group, episode_step=i,
            coarse_key=task.coarse_hash(obs, 20), action_bin=int(task.action_bin_batch(action[None], 5)[0]))
    ids = np.array([0, 16, 32])
    batch = SampleBatch(indices=ids, observation=replay.observation[ids].copy(),
        achieved_goal=replay.achieved_goal[ids].copy(), desired_goal_train=np.tile(goal, (3, 1)),
        next_observation=replay.next_observation[ids].copy(), next_achieved_goal=replay.next_achieved_goal[ids].copy(),
        action_exec=replay.action_exec[ids].copy(), reward_ext=replay.reward_ext[ids].copy(),
        done=replay.done[ids].copy(), timeout=replay.timeout[ids].copy(), h=replay.h[ids].copy(), her_relabel_ratio=1.)
    return cfg, task, replay, batch


class ZeroQuery:
    def policy_actions(self, state):
        self.state = state.detach().clone()
        return torch.zeros((len(state), 4))


class SupportContract(unittest.TestCase):
    def test_both_tasks_masks_normalization_cache_numba_and_reference(self):
        original = importlib.import_module(reference_package("drawer") + ".weights")
        for task_name in ("drawer_open", "drawer_close"):
            cfg, task, replay, batch = fixture(task_name)
            before_actions = batch.action_exec.copy()
            query = ZeroQuery()
            gap = GapStats(1000, 128, .6, .95, 1e-6)
            for numba in (True, False):
                for cache in (True, False):
                    cfg.numba_support_scan = numba
                    cfg.support_cache_enabled = cfg.local_scan_cache_enabled = cache
                    wc = LiteWeightComputer(cfg, task, gap)
                    if numba:
                        self.assertTrue(wc._numba_scanner_warmed)
                    old_cfg = copy.deepcopy(cfg)
                    old_cfg.method = "b3_support_runtime_opt"
                    old_wc = original.LiteWeightComputer(old_cfg, task, gap)
                    rng = np.random.default_rng(101)
                    rng_before = copy.deepcopy(rng.bit_generator.state)
                    torch_before = torch.get_rng_state().clone()
                    for _ in range(2):
                        wb = wc.compute(batch, replay, query, rng, torch.device("cpu"))
                        old = old_wc.compute(batch, replay, query, np.random.default_rng(101), torch.device("cpu"))
                        np.testing.assert_array_equal(wb.psi_q_rank, [1, 0, 0])
                        np.testing.assert_array_equal(wb.w_p_raw, [2, 1, 1])
                        np.testing.assert_allclose(wb.omega_critic, [1.5, .75, .75])
                        np.testing.assert_array_equal(wb.omega_actor, [1, 1, 1])
                        np.testing.assert_array_equal(wb.omega_critic, old.omega_critic)
                        np.testing.assert_array_equal(wb.psi_q_rank, old.psi_q_rank)
                    self.assertEqual(rng_before, rng.bit_generator.state)
                    torch.testing.assert_close(torch_before, torch.get_rng_state())
                    np.testing.assert_array_equal(query.state.numpy()[:, -3:], batch.desired_goal_train)
                    np.testing.assert_array_equal(before_actions, batch.action_exec)

    def test_baseline_never_queries_support(self):
        cfg, task, replay, batch = fixture("drawer_open")
        cfg = make_config("drawer_open", "her", 1, "cpu")
        wc = LiteWeightComputer(cfg, task, GapStats(1000, 128, .6, .95, 1e-6))
        wb = wc.compute(batch, replay, None, None, torch.device("cpu"))
        np.testing.assert_array_equal(wb.omega_critic, np.ones(3))
        self.assertFalse(wc._numba_support_scan_enabled)

    def test_uniform_when_all_masks_agree(self):
        cfg, task, replay, batch = fixture("drawer_close")
        wc = LiteWeightComputer(cfg, task, GapStats(1000, 128, .6, .95, 1e-6))
        for source in (0, 1):
            data = copy.deepcopy(batch)
            for name, value in vars(data).items():
                if isinstance(value, np.ndarray):
                    setattr(data, name, np.repeat(value[source:source+1], 3, axis=0))
            wb = wc.compute(data, replay, ZeroQuery(), np.random.default_rng(0), torch.device("cpu"))
            np.testing.assert_array_equal(wb.omega_critic, np.ones(3))

    def test_strict_future_and_replay_actions_preserved(self):
        _, _, replay, _ = fixture("drawer_open")
        for i in range(replay.size):
            replay.next_achieved_goal[i] = [i, i, i]
        rng = np.random.default_rng(13)
        for slot in range(replay.size):
            goal = replay._sample_future_goal(slot, rng)
            if slot in (15, 31, 34):
                self.assertIsNone(goal)
            else:
                future = int(goal[0])
                self.assertGreater(replay.episode_step[future], replay.episode_step[slot])
                self.assertEqual(replay.episode_id[future], replay.episode_id[slot])
        batch = replay.sample(32, 1., rng)
        np.testing.assert_array_equal(batch.action_exec, replay.action_exec[batch.indices])

    def test_cache_invalidation_after_ring_overwrite(self):
        cfg, task, replay, batch = fixture("drawer_close")
        wc = LiteWeightComputer(cfg, task, GapStats(1000, 128, .6, .95, 1e-6))
        rng = np.random.default_rng(1)
        first = wc.compute(batch, replay, ZeroQuery(), rng, torch.device("cpu"))
        self.assertEqual(first.psi_q_rank[0], 1)
        obs = batch.observation[0]
        action = np.ones(4, np.float32) * .8
        for i in range(64):
            replay.add_transition(observation=obs, achieved_goal=obs[4:7], desired_goal=batch.desired_goal_train[0],
                action_exec=action, reward_ext=-1, next_observation=obs, next_achieved_goal=obs[4:7],
                done=False, timeout=False, h=20, episode_id=9, episode_step=i,
                coarse_key=task.coarse_hash(obs, 20), action_bin=int(task.action_bin_batch(action[None], 5)[0]))
        second = wc.compute(batch, replay, ZeroQuery(), rng, torch.device("cpu"))
        np.testing.assert_array_equal(second.psi_q_rank, np.zeros(3))

    def test_tail_512_and_goal_conditioning_against_direct_oracle(self):
        for task_name in ("drawer_open", "drawer_close"):
            cfg, task, _, small = fixture(task_name)
            replay = HERReplayBuffer(capacity=1024, obs_dim=12, goal_dim=3, action_dim=4)
            obs = small.observation[0]
            goal = small.desired_goal_train[0]
            for i in range(1200):
                action = np.full(4, 0 if i < 600 else .8, np.float32)
                replay.add_transition(observation=obs, achieved_goal=obs[4:7], desired_goal=goal,
                    action_exec=action, reward_ext=-1, next_observation=obs, next_achieved_goal=goal,
                    done=False, timeout=False, h=20, episode_id=i // 200, episode_step=i % 200,
                    coarse_key=task.coarse_hash(obs, 20), action_bin=int(task.action_bin_batch(action[None], 5)[0]))
            batch = replay.sample(128, .8, np.random.default_rng(43))
            batch.desired_goal_train[::3] += .5
            scanned, n, truncated = replay.indexed_reference_indices_deterministic(
                task.coarse_hash(obs, 20), 512, "deterministic_tail_v1")
            self.assertEqual((len(scanned), n, truncated), (512, 1024, True))
            expected = []
            for obs_i, goal_i, h_i in zip(batch.observation, batch.desired_goal_train, batch.h):
                key = task.full_context_hash(obs_i, goal_i, int(h_i))
                matches = scanned[task.full_context_hash_batch(replay.observation[scanned], goal_i[None], replay.h[scanned]) == key]
                zero_bin = task.action_bin_batch(np.zeros((1, 4), np.float32), 5)[0]
                expected.append(float(len(matches) >= 4 and np.any(replay.action_bin[matches] == zero_bin)))
            for early_exit in (True, False):
                cfg.support_predicate_early_exit = early_exit
                wc = LiteWeightComputer(cfg, task, GapStats(1000, 128, .6, .95, 1e-6))
                wb = wc.compute(batch, replay, ZeroQuery(), np.random.default_rng(0), torch.device("cpu"))
                np.testing.assert_array_equal(wb.psi_q_rank, expected)
                self.assertLessEqual(float(np.max(wb.num_scanned_candidates)), 512)

    def test_support_path_changes_critic_without_changing_alpha_rule(self):
        cfg, task, replay, batch = fixture("drawer_open")
        wc = LiteWeightComputer(cfg, task, GapStats(1000, 128, .6, .95, 1e-6))
        a, b = SACAgent(**agent_kwargs()), SACAgent(**agent_kwargs())
        with torch.no_grad():
            a.actor.mean.weight.zero_()
            a.actor.mean.bias.zero_()
        b.load_state_dict(copy.deepcopy(a.state_dict()))
        wb = wc.compute(batch, replay, a, np.random.default_rng(0), torch.device("cpu"))
        np.testing.assert_array_equal(wb.psi_q_rank, [1, 0, 0])
        data = dict(state=torch.from_numpy(np.concatenate([batch.observation, batch.desired_goal_train], axis=1)),
                    action_exec=torch.from_numpy(batch.action_exec), reward=-torch.ones(3),
                    next_state=torch.from_numpy(np.concatenate([batch.next_observation, batch.desired_goal_train], axis=1)),
                    nonterminal=torch.ones(3), omega_critic=torch.ones(3), omega_actor=torch.ones(3))
        torch.manual_seed(89)
        her = b.update(**data)
        data["omega_critic"] = torch.from_numpy(wb.omega_critic)
        torch.manual_seed(89)
        saber = a.update(**data)
        self.assertNotEqual(her.critic_loss, saber.critic_loss)
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(a.critic1.parameters(), b.critic1.parameters())))
        self.assertEqual(her.sac_alpha_loss, saber.sac_alpha_loss)
        self.assertEqual(her.sac_alpha, saber.sac_alpha)


class EnvironmentContract(unittest.TestCase):
    def test_real_drawer_reset_reward_horizon_and_factory(self):
        for task_name in ("drawer_open", "drawer_close"):
            cfg = make_config(task_name, "her", 1, "cpu")
            env = make_env(cfg, 1, False)
            try:
                a = env.reset(seed=913)
                for _ in range(5):
                    env.step(np.ones(4, np.float32) * .7)
                b = env.reset(seed=913)
                np.testing.assert_array_equal(a.observation, b.observation)
                np.testing.assert_array_equal(a.desired_goal, b.desired_goal)
                self.assertNotEqual(env.desired_goal_source, "synthetic_fallback")
                self.assertEqual((env.goal_dim, env.action_dim), (3, 4))
                json.dumps(to_jsonable(env.snapshot()))
                agent = make_agent(cfg, env.state_dim, 4, env.action_low, env.action_high, torch.device("cpu"))
                self.assertIsInstance(agent, SACAgent)
                self.assertTrue(agent.auto_alpha)
                for step in range(200):
                    obs, reward, term, trunc, _ = env.step(np.zeros(4, np.float32))
                    self.assertEqual(reward, env.compute_reward(obs.achieved_goal[None], obs.desired_goal[None])[0])
                    self.assertIn(reward, (-1., 0.))
                    self.assertEqual(bool(term or trunc), step == 199)
                self.assertTrue(trunc)
            finally:
                env.close()

    def test_eval_seed_pairing(self):
        class TinyEnv:
            max_episode_steps = 1
            def __init__(self):
                self.seeds = []
            def reset(self, seed):
                self.seeds.append(seed)
                return SimpleNamespace(flat_state=lambda: np.zeros(3), observation=np.zeros(3), desired_goal=np.zeros(3))
            def step(self, action):
                obs = SimpleNamespace(flat_state=lambda: np.zeros(3), observation=np.zeros(3), desired_goal=np.zeros(3))
                return obs, -1., False, True, {"goal_success": 0., "env_success": 0.}
        actor = SimpleNamespace(act=lambda *a, **k: np.zeros(4))
        for step in (10000, 2000000):
            first, second = TinyEnv(), TinyEnv()
            for env in (first, second):
                evaluate_agent(actor, env, 10001 + step * 10, 3)
            self.assertEqual(first.seeds, second.seeds)
            self.assertEqual(first.seeds, [10001 + step * 10 + i for i in range(3)])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
