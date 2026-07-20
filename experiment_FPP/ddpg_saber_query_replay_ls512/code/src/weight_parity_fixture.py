from __future__ import annotations

import json
import os

import numpy as np
import torch

from metacog_lite_v0.config import LiteConfig
from metacog_lite_v0.replay import HERReplayBuffer, SampleBatch
from metacog_lite_v0.task_fetch_pick_and_place import FetchPickAndPlaceTaskSpec
from metacog_lite_v0.weights import GapStats, LiteWeightComputer


ARRAY_FIELDS = (
    "omega_critic",
    "omega_actor",
    "w_p_raw",
    "g_sup",
    "kappa",
    "psi_q_rank",
    "q_candidate_fallback",
    "psi_a_policy",
    "len_a_q_sup",
    "len_a_q_local",
    "num_coarse_candidates",
    "num_scanned_candidates",
    "local_scan_truncated",
    "q_imp_minus_q_policy",
    "unrestricted_gap",
    "phase",
)


class VectorAgent:
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = torch.as_tensor(actions, dtype=torch.float32)

    def policy_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.actions.to(state.device)


def _obs() -> np.ndarray:
    obs = np.zeros(25, dtype=np.float32)
    obs[3:6] = np.asarray([0.04, 0.0, 0.44], dtype=np.float32)
    obs[9:11] = np.asarray([0.03, 0.03], dtype=np.float32)
    return obs


def _fixture():
    task = FetchPickAndPlaceTaskSpec(
        action_low=np.full(4, -1.0, dtype=np.float32),
        action_high=np.full(4, 1.0, dtype=np.float32),
        epsilon_g=0.05,
    )
    cfg = LiteConfig(
        task_name="fetch_pick_and_place",
        method="b3_support_runtime_opt",
        batch_size=4,
        M_B=16,
        M_Q_min=4,
        N_local_min=4,
        local_search_max=512,
        action_bins_per_dim=5,
        lambda_support=1.0,
        policy_support_mode="bin",
        support_scan_mode="deterministic_tail_v1",
        support_cache_enabled=True,
        local_scan_cache_enabled=True,
        weighting_mode="batch_norm",
        weight_scope="critic_only",
    )
    cfg.normalize()
    cfg.numba_support_scan = os.environ.get("PARITY_NUMBA", "0") == "1"
    cfg.support_predicate_early_exit = cfg.numba_support_scan
    cfg.support_diagnostic_audit_interval = 0

    action_a = np.asarray([0.6, 0.0, 0.0, 0.0], dtype=np.float32)
    action_b = np.asarray([-0.6, 0.0, 0.0, 0.0], dtype=np.float32)
    obs = _obs()
    goal = np.asarray([0.04, 0.0, 0.70], dtype=np.float32)
    replay = HERReplayBuffer(capacity=16, obs_dim=25, goal_dim=3, action_dim=4)
    action_bin = int(task.action_bin_batch(action_a[None, :], 5)[0])
    for row in range(4):
        replay.add_transition(
            observation=obs,
            achieved_goal=goal,
            desired_goal=goal,
            action_exec=action_a,
            reward_ext=0.0,
            next_observation=obs,
            next_achieved_goal=goal,
            done=False,
            timeout=False,
            h=20,
            episode_id=row,
            episode_step=0,
            coarse_key=task.coarse_hash(obs, 20),
            action_bin=action_bin,
        )
    observations = np.repeat(obs[None, :], 4, axis=0)
    goals = np.repeat(goal[None, :], 4, axis=0)
    batch = SampleBatch(
        indices=np.arange(4, dtype=np.int64),
        observation=observations,
        achieved_goal=goals.copy(),
        desired_goal_train=goals.copy(),
        next_observation=observations.copy(),
        next_achieved_goal=goals.copy(),
        action_exec=np.stack([action_b, action_a, action_b, action_a]),
        reward_ext=np.zeros(4, dtype=np.float32),
        done=np.zeros(4, dtype=np.bool_),
        timeout=np.zeros(4, dtype=np.bool_),
        h=np.full(4, 20, dtype=np.int16),
        her_relabel_ratio=1.0,
    )
    agent = VectorAgent(np.stack([action_a, action_a, action_b, action_b]))
    computer = LiteWeightComputer(cfg, task, GapStats(100, 10, 0.6, 0.95, 1e-6))
    return computer, batch, replay, agent


def _array_payload(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "bytes_hex": contiguous.tobytes().hex(),
    }


def emit() -> None:
    computer, batch, replay, agent = _fixture()
    rng = np.random.default_rng(20001)
    passes = []
    for _ in range(2):
        wb = computer.compute(batch, replay, agent, rng, torch.device("cpu"))
        passes.append({name: _array_payload(getattr(wb, name)) for name in ARRAY_FIELDS})
    print(json.dumps({"passes": passes, "rng_state": rng.bit_generator.state}, sort_keys=True))


if __name__ == "__main__":
    emit()
