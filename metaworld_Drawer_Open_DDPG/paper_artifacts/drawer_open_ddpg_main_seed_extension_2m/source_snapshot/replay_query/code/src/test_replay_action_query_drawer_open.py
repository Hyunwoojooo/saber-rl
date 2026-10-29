#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from metacog_lite_v0.config import LiteConfig
from metacog_lite_v0.numba_support_scan_drawer_open import NUMBA_AVAILABLE
from metacog_lite_v0.replay import HERReplayBuffer, SampleBatch
from metacog_lite_v0.task_drawer_open import MetaWorldDrawerOpenTaskSpec
from metacog_lite_v0.trainer import (
    CHECKPOINT_SCHEMA_VERSION,
    capture_global_rng_state,
    checkpoint_config_fingerprint,
    validate_resume_payload,
)
from metacog_lite_v0.weights import GapStats, LiteWeightComputer
from train_drawer_open_lite_v0 import config_from_args, parse_args


PARITY_ARRAY_FIELDS = (
    "omega_critic",
    "omega_actor",
    "w_p_raw",
    "g_sup",
    "kappa",
    "psi_q_rank",
    "q_candidate_fallback",
    "psi_a_policy",
    "q_imp_minus_q_policy",
    "unrestricted_gap",
    "phase",
)


class FixedPolicyAgent:
    def __init__(self, action: np.ndarray) -> None:
        self.action = np.asarray(action, dtype=np.float32)

    def policy_actions(self, state: torch.Tensor) -> torch.Tensor:
        if self.action.ndim == 2:
            if self.action.shape[0] != state.shape[0]:
                raise ValueError("per-row policy action batch size mismatch")
            action = self.action
        else:
            action = np.repeat(self.action[None, :], state.shape[0], axis=0)
        return torch.as_tensor(action, dtype=torch.float32, device=state.device)


class SyntheticReplay:
    def __init__(
        self,
        task: MetaWorldDrawerOpenTaskSpec,
        observation: np.ndarray,
        horizon: np.ndarray,
        actions: np.ndarray,
    ) -> None:
        self.observation = observation.astype(np.float32, copy=True)
        self.h = horizon.astype(np.int64, copy=True)
        self.action_exec = actions.astype(np.float32, copy=True)
        self.action_bin = task.action_bin_batch(self.action_exec, 5)
        self.size = int(actions.shape[0])
        self.action_dim = int(actions.shape[1])
        self.version = 1
        self.last_scanned = np.zeros(0, dtype=np.int64)

    def indexed_reference_indices_deterministic(
        self,
        coarse_key: int,
        max_n: int,
        mode: str,
    ) -> tuple[np.ndarray, int, bool]:
        del coarse_key
        assert mode == "deterministic_tail_v1"
        start = max(0, self.size - int(max_n))
        self.last_scanned = np.arange(start, self.size, dtype=np.int64)
        return self.last_scanned.copy(), self.size, self.size > int(max_n)

    def action_index_version(self, coarse_key: int) -> int:
        del coarse_key
        return self.version

    def action_index_bins(self, coarse_key: int) -> np.ndarray:
        del coarse_key
        return np.unique(self.action_bin).astype(np.int64)

    def action_index_has_action_bin(self, coarse_key: int, action_bin: int) -> bool:
        del coarse_key
        return bool(np.any(self.action_bin == int(action_bin)))


def fixture() -> tuple[
    MetaWorldDrawerOpenTaskSpec,
    SyntheticReplay,
    SampleBatch,
    np.ndarray,
]:
    task = MetaWorldDrawerOpenTaskSpec(
        action_low=np.full(4, -1.0, dtype=np.float32),
        action_high=np.full(4, 1.0, dtype=np.float32),
        epsilon_g=0.05,
    )
    observation = np.zeros((4, 7), dtype=np.float32)
    observation[:, 3] = 0.2
    observation[:, 4:7] = np.asarray([0.25, 0.1, 0.0], dtype=np.float32)
    goal = np.repeat(
        np.asarray([[0.6, 0.1, 0.0]], dtype=np.float32), 4, axis=0
    )
    horizon = np.full(4, 20, dtype=np.int64)
    replay_actions = np.zeros((4, 4), dtype=np.float32)
    # Only the sampled transition at replay slot zero occupies this action bin.
    replay_actions[0] = -0.9
    replay = SyntheticReplay(task, observation, horizon, replay_actions)
    batch = SampleBatch(
        indices=np.asarray([0], dtype=np.int64),
        observation=observation[:1].copy(),
        achieved_goal=np.zeros((1, 3), dtype=np.float32),
        desired_goal_train=goal[:1].copy(),
        next_observation=observation[:1].copy(),
        next_achieved_goal=np.zeros((1, 3), dtype=np.float32),
        action_exec=replay_actions[:1].copy(),
        reward_ext=np.zeros(1, dtype=np.float32),
        done=np.zeros(1, dtype=np.float32),
        timeout=np.zeros(1, dtype=np.float32),
        h=horizon[:1].copy(),
        her_relabel_ratio=0.8,
    )
    policy_action = np.full(4, 0.9, dtype=np.float32)
    return task, replay, batch, policy_action


def weight_config(
    mode: str,
    *,
    cache: bool,
    numba: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        method="b3_support_runtime_opt",
        support_query_mode=mode,
        support_assignment_mode="support",
        M_B=16,
        M_Q_min=4,
        N_local_min=4,
        local_search_max=512,
        action_bins_per_dim=5,
        policy_support_mode="bin",
        policy_action_radius=0.35,
        support_scan_mode="deterministic_tail_v1",
        support_cache_enabled=cache,
        support_cache_max_entries=64,
        local_scan_cache_enabled=cache,
        local_scan_cache_max_entries=64,
        support_short_circuit=True,
        numba_support_scan=numba,
        support_predicate_early_exit=True,
        support_diagnostic_audit_interval=1,
        lambda_support=1.0,
        weighting_mode="batch_norm",
        weight_scope="critic_only",
        eps=1e-8,
    )


def compute(mode: str, *, cache: bool, numba: bool = False):
    task, replay, batch, policy_action = fixture()
    computer = LiteWeightComputer(
        weight_config(mode, cache=cache, numba=numba),
        task,
        GapStats(32, 128, 0.60, 0.95, 1e-6),
    )
    rng = np.random.default_rng(20_001)
    before = copy.deepcopy(rng.bit_generator.state)
    torch_before = torch.random.get_rng_state().clone()
    result = computer.compute(
        batch=batch,
        replay=replay,  # type: ignore[arg-type]
        agent=FixedPolicyAgent(policy_action),  # type: ignore[arg-type]
        rng=rng,
        device=torch.device("cpu"),
    )
    assert before == rng.bit_generator.state
    assert torch.equal(torch_before, torch.random.get_rng_state())
    return computer, replay, batch, policy_action, result


def test_query_modes_split_support_and_replay_action_is_exact() -> None:
    current = compute("current_policy", cache=False)[-1]
    computer, replay, batch, policy_action, replay_query = compute(
        "replay_action", cache=False
    )
    np.testing.assert_array_equal(current.psi_a_query, np.asarray([0.0]))
    np.testing.assert_array_equal(replay_query.psi_a_query, np.asarray([1.0]))
    np.testing.assert_array_equal(current.psi_q_rank, np.asarray([0.0]))
    np.testing.assert_array_equal(replay_query.psi_q_rank, np.asarray([1.0]))
    assert replay_query.extra_metrics[
        "query/query_bin_matches_replay_action_rate"
    ] == 1.0
    assert replay_query.extra_metrics["query/support_positive_rate"] == 1.0
    assert replay_query.extra_metrics["query/high_weight_count"] == 1.0
    assert replay_query.extra_metrics["query/raw_weight_mean"] == 2.0
    assert replay_query.extra_metrics["query/normalized_weight_mean"] == 1.0
    np.testing.assert_array_equal(replay_query.omega_actor, np.ones(1))
    np.testing.assert_allclose(np.mean(replay_query.omega_critic), 1.0)

    selected, metrics = computer._support_query_actions(
        batch, policy_action[None, :]
    )
    assert selected is batch.action_exec
    np.testing.assert_array_equal(selected, batch.action_exec)
    assert metrics["query/query_bin_matches_replay_action_rate"] == 1.0
    # Slot zero is intentionally retained, enabling the defined self-support.
    assert 0 in replay.last_scanned


def test_cache_numba_and_current_policy_parity() -> None:
    current_off = compute("current_policy", cache=False)[-1]
    cfg_default = weight_config("current_policy", cache=False, numba=False)
    delattr(cfg_default, "support_query_mode")
    task, replay, batch, policy_action = fixture()
    default_computer = LiteWeightComputer(
        cfg_default, task, GapStats(32, 128, 0.60, 0.95, 1e-6)
    )
    default = default_computer.compute(
        batch,
        replay,  # type: ignore[arg-type]
        FixedPolicyAgent(policy_action),  # type: ignore[arg-type]
        np.random.default_rng(20_001),
        torch.device("cpu"),
    )
    for field in (
        "psi_a_query",
        "psi_q_rank",
        "w_p_raw",
        "omega_critic",
        "omega_actor",
    ):
        np.testing.assert_array_equal(getattr(default, field), getattr(current_off, field))

    replay_off = compute("replay_action", cache=False)[-1]
    replay_on = compute("replay_action", cache=True)[-1]
    for field in ("psi_a_query", "psi_q_rank", "w_p_raw", "omega_critic"):
        np.testing.assert_array_equal(getattr(replay_off, field), getattr(replay_on, field))

    if NUMBA_AVAILABLE:
        replay_numba = compute("replay_action", cache=False, numba=True)[-1]
        for field in ("psi_a_query", "psi_q_rank", "w_p_raw", "omega_critic"):
            np.testing.assert_array_equal(
                getattr(replay_off, field), getattr(replay_numba, field)
            )


def test_cache_key_mode_and_bin_and_invalid_mode() -> None:
    task, _replay, batch, policy_action = fixture()
    cfg = weight_config("current_policy", cache=True, numba=False)
    computer = LiteWeightComputer(cfg, task, GapStats(8, 128, 0.1, 0.9, 1e-6))
    kwargs = dict(
        coarse_key=1,
        full_key=2,
        goal=batch.desired_goal_train[0],
        h=20,
        bucket_version=3,
    )
    current = computer._support_cache_key(**kwargs, policy_bin=4)
    cfg.support_query_mode = "replay_action"
    replay_same_bin = computer._support_cache_key(**kwargs, policy_bin=4)
    replay_other_bin = computer._support_cache_key(**kwargs, policy_bin=5)
    assert current != replay_same_bin
    assert replay_same_bin != replay_other_bin
    assert ("query_bin", 4) == replay_same_bin[-2:]

    cfg.support_query_mode = "invalid"
    try:
        computer._support_query_actions(batch, policy_action[None, :])
    except ValueError as exc:
        assert "support_query_mode" in str(exc)
    else:
        raise AssertionError("invalid support_query_mode was accepted")


def _parity_array_payload(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "bytes_hex": contiguous.tobytes().hex(),
    }


def emit_current_policy_parity() -> None:
    """Emit a source-root-independent legacy parity payload as JSON."""
    task, _synthetic, batch_one, policy_action = fixture()
    cfg = LiteConfig(
        task_name="drawer_open",
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
        support_assignment_mode="support",
        support_query_mode="current_policy",
    )
    cfg.normalize()
    cfg.numba_support_scan = os.environ.get("PARITY_NUMBA", "0") == "1"
    cfg.support_predicate_early_exit = cfg.numba_support_scan
    cfg.support_diagnostic_audit_interval = 0

    observation = np.repeat(batch_one.observation, 4, axis=0)
    goals = np.repeat(batch_one.desired_goal_train, 4, axis=0)
    action_supported = np.full(4, -0.9, dtype=np.float32)
    action_unsupported = np.full(4, 0.9, dtype=np.float32)
    replay = HERReplayBuffer(capacity=16, obs_dim=7, goal_dim=3, action_dim=4)
    action_bin = int(task.action_bin_batch(action_supported[None, :], 5)[0])
    for row in range(4):
        replay.add_transition(
            observation=observation[row],
            achieved_goal=goals[row],
            desired_goal=goals[row],
            action_exec=action_supported,
            reward_ext=0.0,
            next_observation=observation[row],
            next_achieved_goal=goals[row],
            done=False,
            timeout=False,
            h=20,
            episode_id=row,
            episode_step=0,
            coarse_key=task.coarse_hash(observation[row], 20),
            action_bin=action_bin,
        )
    batch = SampleBatch(
        indices=np.arange(4, dtype=np.int64),
        observation=observation,
        achieved_goal=goals.copy(),
        desired_goal_train=goals.copy(),
        next_observation=observation.copy(),
        next_achieved_goal=goals.copy(),
        action_exec=np.stack(
            [
                action_unsupported,
                action_supported,
                action_unsupported,
                action_supported,
            ]
        ),
        reward_ext=np.zeros(4, dtype=np.float32),
        done=np.zeros(4, dtype=np.bool_),
        timeout=np.zeros(4, dtype=np.bool_),
        h=np.full(4, 20, dtype=np.int16),
        her_relabel_ratio=1.0,
    )
    policy_actions = np.stack(
        [
            action_supported,
            action_supported,
            action_unsupported,
            policy_action,
        ]
    )
    computer = LiteWeightComputer(
        cfg, task, GapStats(100, 10, 0.6, 0.95, 1e-6)
    )
    rng = np.random.default_rng(20_001)
    passes = []
    for _ in range(2):
        wb = computer.compute(
            batch,
            replay,
            FixedPolicyAgent(policy_actions),  # type: ignore[arg-type]
            rng,
            torch.device("cpu"),
        )
        passes.append(
            {
                name: _parity_array_payload(getattr(wb, name))
                for name in PARITY_ARRAY_FIELDS
            }
        )
    print(
        json.dumps(
            {"passes": passes, "rng_state": rng.bit_generator.state},
            sort_keys=True,
        )
    )


def _run_parity_subprocess(source_root: Path, fixture_root: Path) -> dict[str, object]:
    env = os.environ.copy()
    paths = [str(source_root), str(fixture_root)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from test_replay_action_query_drawer_open import "
                "emit_current_policy_parity; emit_current_policy_parity()"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(result.stdout)


def test_current_policy_frozen_source_subprocess_parity() -> None:
    fixture_root = Path(__file__).resolve().parent
    drawer_root = fixture_root.parents[2]
    baseline_root = (
        drawer_root
        / "paper_artifacts"
        / "drawer_open_ddpg_saber_context_eligible_count_matched_2m"
        / "source_snapshot"
        / "code"
        / "src"
    )
    if not baseline_root.is_dir():
        raise FileNotFoundError(f"frozen baseline source missing: {baseline_root}")
    legacy = _run_parity_subprocess(baseline_root, fixture_root)
    current = _run_parity_subprocess(fixture_root, fixture_root)
    if current != legacy:
        raise AssertionError(
            "current_policy weight arrays or weight RNG state differ from "
            "the verified frozen DrawerOpen source"
        )


def replay_config() -> LiteConfig:
    config = LiteConfig(
        task_name="drawer_open",
        method="b3_support_runtime_opt",
        weight_scope="critic_only",
        weighting_mode="batch_norm",
        support_query_mode="replay_action",
        support_assignment_mode="support",
    )
    config.normalize()
    return config


def replay_resume_payload(config: LiteConfig) -> dict[str, object]:
    fingerprint = checkpoint_config_fingerprint(config)
    global_step = 200
    total_updates = 3
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_metadata": {
            "global_step": global_step,
            "total_updates": total_updates,
            "at_episode_boundary": True,
            "config_fingerprint": fingerprint,
        },
        "config": config.as_dict(),
        "config_fingerprint": fingerprint,
        "global_step": global_step,
        "episode_idx": 1,
        "total_updates": total_updates,
        "episode_state": {
            "at_episode_boundary": True,
            "episode_step": 0,
        },
        "agent": {},
        "replay": {},
        "gap_stats": {},
        "last_eval_metrics": {},
        "eval_history_env": [],
        "last_loss": {},
        "last_lite": {},
        "last_her_ratio": 0.0,
        "noise_rng_state": np.random.default_rng(1).bit_generator.state,
        "sample_rng_state": np.random.default_rng(2).bit_generator.state,
        "weight_rng_state": np.random.default_rng(3).bit_generator.state,
        "resume_audit": {
            "schema_version": 1,
            "strict_control": False,
            "replay_action_query": True,
            "episode_boundary": True,
            "global_step": global_step,
            "total_updates": total_updates,
            "rng_state_fields": [
                "python_random_state",
                "numpy_global_rng_state",
                "torch_cpu_rng_state",
                "torch_cuda_rng_state_all",
                "noise_rng_state",
                "sample_rng_state",
                "weight_rng_state",
            ],
            "config_fingerprint": fingerprint,
        },
        **capture_global_rng_state(),
    }


def test_cli_config_and_resume_contract() -> None:
    args = parse_args(["--support_query_mode", "replay_action"])
    parsed = config_from_args(args)
    parsed.normalize()
    assert parsed.support_query_mode == "replay_action"
    assert parsed.support_assignment_mode == "support"

    config = replay_config()
    payload = replay_resume_payload(config)
    result = validate_resume_payload(payload, config, torch.device("cpu"))
    assert result["replay_action_query"] is True
    assert result["config_fingerprint_matched"] is True
    assert "assignment_rng_state" not in payload
    assert "assignment_diagnostics_state" not in payload

    changed = replace(config, support_query_mode="current_policy")
    try:
        validate_resume_payload(payload, changed, torch.device("cpu"))
    except RuntimeError as exc:
        assert "training config differs" in str(exc)
    else:
        raise AssertionError("changed query mode was accepted on resume")

    missing_rng = copy.deepcopy(payload)
    missing_rng.pop("weight_rng_state")
    try:
        validate_resume_payload(missing_rng, config, torch.device("cpu"))
    except RuntimeError as exc:
        assert "missing required state" in str(exc)
    else:
        raise AssertionError("replay resume accepted a missing weight RNG")

    mid_episode = copy.deepcopy(payload)
    mid_episode["episode_state"] = {
        "at_episode_boundary": False,
        "episode_step": 7,
    }
    try:
        validate_resume_payload(mid_episode, config, torch.device("cpu"))
    except RuntimeError as exc:
        assert "episode-boundary" in str(exc)
    else:
        raise AssertionError("replay resume accepted a mid-episode checkpoint")

    invalid = replace(config, support_query_mode="not_a_mode")
    try:
        invalid.normalize()
    except ValueError as exc:
        assert "support_query_mode" in str(exc)
    else:
        raise AssertionError("invalid config query mode was accepted")


def main() -> int:
    test_query_modes_split_support_and_replay_action_is_exact()
    test_cache_numba_and_current_policy_parity()
    test_cache_key_mode_and_bin_and_invalid_mode()
    test_current_policy_frozen_source_subprocess_parity()
    test_cli_config_and_resume_contract()
    print("test_replay_action_query_drawer_open.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
