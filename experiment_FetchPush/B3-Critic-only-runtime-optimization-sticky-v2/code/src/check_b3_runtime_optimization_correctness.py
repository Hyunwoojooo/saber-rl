#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

from metacog_lite_v0.config import LiteConfig, SeedConfig
from metacog_lite_v0.replay import HERReplayBuffer, ReplayActionIndex, SampleBatch
from metacog_lite_v0.weights import GapStats, LiteWeightComputer


class SyntheticTask:
    action_low = np.asarray([-1.0, -1.0], dtype=np.float32)
    action_high = np.asarray([1.0, 1.0], dtype=np.float32)

    def coarse_hash_batch(self, obs: np.ndarray, h: np.ndarray) -> np.ndarray:
        del h
        return obs[:, 0].astype(np.int64)

    def coarse_hash(self, obs: np.ndarray, h: int) -> int:
        del h
        return int(obs[0])

    def full_context_hash_batch(self, obs: np.ndarray, goal: np.ndarray, h: np.ndarray) -> np.ndarray:
        coarse = obs[:, 0].astype(np.int64)
        phase = (obs[:, 1] > 0.5).astype(np.int64)
        goal_bin = (goal[:, 0] > 0.5).astype(np.int64)
        h_bin = (h.astype(np.int64) // 10).astype(np.int64)
        return coarse * 1000 + phase * 100 + goal_bin * 10 + h_bin

    def full_context_hash(self, obs: np.ndarray, goal: np.ndarray, h: int) -> int:
        return int(self.full_context_hash_batch(obs[None, :], goal[None, :], np.asarray([h]))[0])

    def action_bin_batch(self, action: np.ndarray, bins_per_dim: int) -> np.ndarray:
        del bins_per_dim
        x_pos = (action[:, 0] >= 0.0).astype(np.int64)
        y_pos = (action[:, 1] >= 0.0).astype(np.int64)
        return x_pos + 2 * y_pos


class SyntheticAgent:
    def policy_actions(self, state_t: torch.Tensor) -> torch.Tensor:
        state = state_t.detach().cpu().numpy()
        action = np.zeros((state.shape[0], 2), dtype=np.float32)
        phase_high = state[:, 1] > 0.5
        action[phase_high] = np.asarray([0.75, -0.75], dtype=np.float32)
        action[np.logical_not(phase_high)] = np.asarray([-0.75, 0.75], dtype=np.float32)
        return torch.as_tensor(action, dtype=torch.float32, device=state_t.device)


def make_config(*, optimized: bool, scan_mode: str) -> LiteConfig:
    cfg = LiteConfig(
        task_name="fetch_push",
        method="b3_support_only",
        reward_type="sparse",
        batch_size=12,
        M_B=8,
        M_Q_min=4,
        N_local_min=4,
        local_search_max=8,
        action_bins_per_dim=5,
        support_lookup_optimization=optimized,
        support_cache_enabled=optimized,
        support_short_circuit=optimized,
        support_batch_query_coalescing=False,
        support_deterministic_scan=True,
        support_scan_mode=scan_mode,
        support_scan_seed=12345,
        support_local_scan_cache_enabled=optimized,
        weighting_mode="batch_norm",
        weight_scope="critic_only",
        seeds=SeedConfig(weight_seed=123),
    )
    cfg.normalize()
    return cfg


def make_replay(task: SyntheticTask) -> HERReplayBuffer:
    replay = HERReplayBuffer(capacity=128, obs_dim=2, goal_dim=1, action_dim=2)
    actions = [
        np.asarray([0.75, -0.75], dtype=np.float32),
        np.asarray([-0.75, 0.75], dtype=np.float32),
        np.asarray([0.75, 0.75], dtype=np.float32),
    ]
    slot = 0
    for coarse in (0, 1):
        for phase in (0.0, 1.0):
            for h in (10, 20):
                for rep in range(6):
                    obs = np.asarray([float(coarse), phase], dtype=np.float32)
                    goal = np.asarray([float(rep % 2)], dtype=np.float32)
                    action = actions[(rep + int(phase)) % len(actions)]
                    replay.add_transition(
                        observation=obs,
                        achieved_goal=goal,
                        desired_goal=goal,
                        action_exec=action,
                        reward_ext=0.0,
                        next_observation=obs,
                        next_achieved_goal=goal,
                        done=False,
                        timeout=False,
                        h=h,
                        episode_id=slot,
                        episode_step=0,
                        coarse_key=task.coarse_hash(obs, h),
                        action_bin=int(task.action_bin_batch(action[None, :], 5)[0]),
                    )
                    slot += 1
    return replay


def make_batch() -> SampleBatch:
    unique_queries = [
        ([0.0, 0.0], [0.0], 10),
        ([0.0, 1.0], [0.0], 10),
        ([1.0, 0.0], [1.0], 10),
        ([1.0, 1.0], [1.0], 10),
        ([0.0, 0.0], [1.0], 20),
        ([1.0, 1.0], [0.0], 20),
    ]
    rows = unique_queries + unique_queries
    obs_rows = [row[0] for row in rows]
    goals = [row[1] for row in rows]
    hs = [row[2] for row in rows]
    obs = np.asarray(obs_rows, dtype=np.float32)
    goal = np.asarray(goals, dtype=np.float32)
    h = np.asarray(hs, dtype=np.int16)
    zeros_g = np.zeros_like(goal, dtype=np.float32)
    return SampleBatch(
        indices=np.arange(obs.shape[0], dtype=np.int64),
        observation=obs.copy(),
        achieved_goal=zeros_g.copy(),
        desired_goal_train=goal.copy(),
        next_observation=obs.copy(),
        next_achieved_goal=zeros_g.copy(),
        action_exec=np.zeros((obs.shape[0], 2), dtype=np.float32),
        reward_ext=np.zeros(obs.shape[0], dtype=np.float32),
        done=np.zeros(obs.shape[0], dtype=np.bool_),
        timeout=np.zeros(obs.shape[0], dtype=np.bool_),
        h=h.copy(),
        her_relabel_ratio=1.0,
    )


def compute(config: LiteConfig, task: SyntheticTask, replay: HERReplayBuffer, batch: SampleBatch, seed: int):
    gap_stats = GapStats(maxlen=100, n_min=4, q_low=0.6, q_high=0.95, eps_g=1e-6)
    computer = LiteWeightComputer(config, task, gap_stats)
    return computer.compute(
        batch=batch,
        replay=replay,
        agent=SyntheticAgent(),  # type: ignore[arg-type]
        rng=np.random.default_rng(seed),
        device=torch.device("cpu"),
    ), computer


def assert_close(name: str, left: np.ndarray, right: np.ndarray) -> None:
    if not np.allclose(left, right, rtol=0.0, atol=1e-7):
        raise AssertionError(f"{name} mismatch\nleft={left}\nright={right}")


def assert_equal(name: str, left: object, right: object) -> None:
    if left != right:
        raise AssertionError(f"{name} mismatch: {left!r} != {right!r}")


def run_replay_action_index_tests() -> dict[str, Any]:
    index = ReplayActionIndex(capacity=32)

    index.update(slot=3, key=10, action_bin=2)
    index.update(slot=3, key=10, action_bin=2)
    state = index.state_dict()
    assert_equal("same slot/key live count", state["live_count_by_key"][10], 1)
    assert_equal("same slot/key candidate count", int(index.candidates(10, 32).size), 1)
    assert_equal("same slot/key action bin", bool(index.has_action_bin(10, 2)), True)

    index.update(slot=3, key=10, action_bin=4)
    assert_equal("old action bin removed", bool(index.has_action_bin(10, 2)), False)
    assert_equal("new action bin present", bool(index.has_action_bin(10, 4)), True)

    index.update(slot=3, key=11, action_bin=1)
    assert_equal("old key removed", int(index.candidates(10, 32).size), 0)
    assert_equal("new key candidate count", int(index.candidates(11, 32).size), 1)
    assert_equal("new key action bin", bool(index.has_action_bin(11, 1)), True)

    index.clear_slot(3)
    assert_equal("clear slot candidate count", int(index.candidates(11, 32).size), 0)
    assert_equal("clear slot action bin", bool(index.has_action_bin(11, 1)), False)

    for slot in range(20):
        index.update(slot=slot, key=7, action_bin=slot % 4)
    bins = sorted(int(v) for v in index.action_bins(7).tolist())
    assert_equal("metadata action bins", bins, [0, 1, 2, 3])
    for bin_id in range(4):
        assert_equal(f"has_action_bin {bin_id}", bool(index.has_action_bin(7, bin_id)), True)
    assert_equal("missing action bin", bool(index.has_action_bin(7, 9)), False)

    first, n_first, trunc_first = index.sample_candidates_direct(
        key=7,
        max_scan=8,
        scan_mode="sticky_random_v1",
        scan_seed=2026,
    )
    second, n_second, trunc_second = index.sample_candidates_direct(
        key=7,
        max_scan=8,
        scan_mode="sticky_random_v1",
        scan_seed=2026,
    )
    if not np.array_equal(first, second):
        raise AssertionError(f"sticky_random_v1 not reproducible\nfirst={first}\nsecond={second}")
    assert_equal("sticky n_candidates", n_first, 20)
    assert_equal("sticky n repeat", n_second, 20)
    assert_equal("sticky truncated", trunc_first, True)
    assert_equal("sticky truncated repeat", trunc_second, True)
    assert_equal("sticky scan size", int(first.size), 8)

    index.update(slot=20, key=7, action_bin=2)
    third, _, _ = index.sample_candidates_direct(
        key=7,
        max_scan=8,
        scan_mode="sticky_random_v1",
        scan_seed=2026,
    )
    if np.array_equal(first, third):
        raise AssertionError("sticky_random_v1 subset did not change after bucket version changed")

    return {
        "same_slot_same_key_no_duplicate": True,
        "slot_key_change_updates_counts": True,
        "action_bin_metadata_matches_bruteforce": True,
        "clear_slot_updates_metadata": True,
        "sticky_random_v1_reproducible": True,
        "sample_size_matches_budget": True,
    }


def main() -> int:
    task = SyntheticTask()
    replay = make_replay(task)
    batch = make_batch()
    replay_index_report = run_replay_action_index_tests()

    mode_reports: dict[str, Any] = {}
    for scan_mode in ("deterministic_back_v1", "deterministic_even_v1", "sticky_random_v1"):
        naive_cfg = make_config(optimized=False, scan_mode=scan_mode)
        opt_cfg = make_config(optimized=True, scan_mode=scan_mode)

        naive, _ = compute(naive_cfg, task, replay, batch, seed=999)
        opt_first, opt_computer = compute(opt_cfg, task, replay, batch, seed=999)
        opt_second = opt_computer.compute(
            batch=batch,
            replay=replay,
            agent=SyntheticAgent(),  # type: ignore[arg-type]
            rng=np.random.default_rng(999),
            device=torch.device("cpu"),
        )

        for label, opt in (("optimized_first", opt_first), ("optimized_second_cached", opt_second)):
            assert_close(f"{scan_mode}: {label}: psi_q_rank", naive.psi_q_rank, opt.psi_q_rank)
            assert_close(f"{scan_mode}: {label}: psi_A_policy", naive.psi_a_policy, opt.psi_a_policy)
            assert_close(f"{scan_mode}: {label}: q_candidate_fallback", naive.q_candidate_fallback, opt.q_candidate_fallback)
            assert_close(f"{scan_mode}: {label}: w_P", naive.w_p_raw, opt.w_p_raw)
            assert_close(f"{scan_mode}: {label}: Omega_critic", naive.omega_critic, opt.omega_critic)
            assert_close(f"{scan_mode}: {label}: Omega_actor", naive.omega_actor, opt.omega_actor)
            assert_close(f"{scan_mode}: {label}: len_A_Q_replay_local", naive.len_a_q_local, opt.len_a_q_local)
            assert_close(f"{scan_mode}: {label}: len_A_Q_sup", naive.len_a_q_sup, opt.len_a_q_sup)
            assert_close(f"{scan_mode}: {label}: num_scanned_candidates", naive.num_scanned_candidates, opt.num_scanned_candidates)
            assert_close(f"{scan_mode}: {label}: local_scan_truncated", naive.local_scan_truncated, opt.local_scan_truncated)

        mode_reports[scan_mode] = {
            "naive_support_positive_rate": float(np.mean(naive.psi_q_rank)),
            "optimized_support_positive_rate": float(np.mean(opt_second.psi_q_rank)),
            "optimized_runtime_metrics": opt_second.extra_metrics,
            "truncated_path_exercised": bool(np.mean(naive.local_scan_truncated) > 0.0),
            "local_scan_cache_hit_observed": float(
                opt_second.extra_metrics.get("runtime/local_scan_cache_hit_rate", 0.0)
            ) > 0.0,
            "support_cache_hit_observed": float(
                opt_second.extra_metrics.get("runtime/support_cache_hit_rate", 0.0)
            ) > 0.0,
        }

    report: dict[str, Any] = {
        "status": "passed",
        "replay_action_index_v2_unit_tests": True,
        "replay_action_index_v2": replay_index_report,
        "semantic_equivalence": True,
        "diagnostic_equivalence": True,
        "scan_modes_checked": sorted(mode_reports.keys()),
        "truncated_path_exercised": all(
            bool(mode_report["truncated_path_exercised"])
            for mode_report in mode_reports.values()
        ),
        "deterministic_local_scan_cache": True,
        "sticky_random_v1_reproducible": True,
        "support_lookup_optimization": {
            "versioned_support_cache": True,
            "batch_query_coalescing": False,
            "exact_negative_short_circuit": True,
            "deterministic_truncated_scan": True,
            "replay_action_index_v2": True,
            "coarse_action_bin_metadata": True,
            "goal_aware_cache_key": True,
        },
        "modes": mode_reports,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
