#!/usr/bin/env python3
"""Actual DrawerOpen replay-query/DDPG split checkpoint-resume audit."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
CODE_SRC = HERE / "code" / "src"
if str(CODE_SRC) not in sys.path:
    sys.path.insert(0, str(CODE_SRC))

from metacog_lite_v0.agent import DDPGAgent
from metacog_lite_v0.config import LiteConfig
from metacog_lite_v0.numba_support_scan_drawer_open import NUMBA_AVAILABLE
from metacog_lite_v0.replay import HERReplayBuffer
from metacog_lite_v0.task_drawer_open import MetaWorldDrawerOpenTaskSpec
from metacog_lite_v0.trainer import (
    CHECKPOINT_SCHEMA_VERSION,
    GLOBAL_RNG_STATE_FIELDS,
    capture_global_rng_state,
    checkpoint_config_fingerprint,
    load_checkpoint,
    restore_global_rng_state,
    save_checkpoint,
    validate_resume_payload,
)
from metacog_lite_v0.weights import GapStats, LiteWeightComputer


SEEDS = (1, 2, 3, 4, 5)
UPDATES = 8
SPLIT_UPDATE = 4
BATCH_SIZE = 8
OBS_DIM = 7
GOAL_DIM = 3
ACTION_DIM = 4
STATE_DIM = OBS_DIM + GOAL_DIM
QUERY_MODE = "replay_action"
ASSIGNMENT_MODE = "support"
WEIGHT_SCOPE = "critic_only"

WEIGHT_ARRAY_FIELDS = (
    "omega_critic",
    "omega_actor",
    "w_p_raw",
    "psi_q_rank",
    "psi_a_policy",
    "len_a_q_sup",
    "len_a_q_local",
    "num_coarse_candidates",
    "num_scanned_candidates",
    "local_scan_truncated",
)
QUERY_METRIC_FIELDS = (
    "query/query_bin_matches_current_policy_rate",
    "query/query_bin_matches_replay_action_rate",
    "query/context_count_eligible_rate",
    "query/support_positive_rate",
    "query/query_bin_present_given_context_rate",
    "query/high_weight_count",
    "query/raw_weight_mean",
    "query/normalized_weight_mean",
)


def equal_state(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            equal_state(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            equal_state(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def make_config() -> LiteConfig:
    config = LiteConfig(
        task_name="drawer_open",
        method="b3_support_runtime_opt",
        batch_size=BATCH_SIZE,
        M_B=16,
        M_Q_min=4,
        N_local_min=4,
        local_search_max=512,
        action_bins_per_dim=5,
        policy_support_mode="bin",
        support_query_mode=QUERY_MODE,
        support_assignment_mode=ASSIGNMENT_MODE,
        support_scan_mode="deterministic_tail_v1",
        support_cache_enabled=True,
        support_cache_max_entries=200_000,
        local_scan_cache_enabled=True,
        local_scan_cache_max_entries=50_000,
        support_short_circuit=True,
        numba_support_scan=True,
        support_predicate_early_exit=True,
        support_diagnostic_audit_interval=0,
        lambda_support=1.0,
        weighting_mode="batch_norm",
        weight_scope=WEIGHT_SCOPE,
    )
    config.normalize()
    return config


def make_task() -> MetaWorldDrawerOpenTaskSpec:
    return MetaWorldDrawerOpenTaskSpec(
        action_low=np.full(ACTION_DIM, -1.0, dtype=np.float32),
        action_high=np.full(ACTION_DIM, 1.0, dtype=np.float32),
        epsilon_g=0.05,
    )


def make_agent(seed: int) -> DDPGAgent:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return DDPGAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=16,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.98,
        tau=0.005,
        action_low=np.full(ACTION_DIM, -1.0, dtype=np.float32),
        action_high=np.full(ACTION_DIM, 1.0, dtype=np.float32),
        device=torch.device("cpu"),
        l2_action_penalty=0.0,
    )


def make_streams(seed: int) -> dict[str, np.random.Generator]:
    return {
        "sample": np.random.default_rng(seed + 17),
        "noise": np.random.default_rng(seed),
        "weight": np.random.default_rng(20_000 + seed),
    }


def generator_states(
    streams: dict[str, np.random.Generator],
) -> dict[str, object]:
    return {
        name: copy.deepcopy(rng.bit_generator.state)
        for name, rng in streams.items()
    }


def restore_generator_states(
    streams: dict[str, np.random.Generator],
    states: dict[str, object],
) -> None:
    for name, state in states.items():
        streams[name].bit_generator.state = state


def _observation(group: int) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[3] = 0.2
    if group == 0:
        obs[4:7] = np.asarray([0.25, 0.10, 0.0], dtype=np.float32)
    elif group == 1:
        obs[4:7] = np.asarray([0.10, 0.00, 0.0], dtype=np.float32)
    else:
        obs[4:7] = np.asarray([0.50, 0.25, 0.0], dtype=np.float32)
    return obs


def make_replay(task: MetaWorldDrawerOpenTaskSpec, seed: int) -> HERReplayBuffer:
    replay = HERReplayBuffer(
        capacity=64,
        obs_dim=OBS_DIM,
        goal_dim=GOAL_DIM,
        action_dim=ACTION_DIM,
    )
    goal = np.asarray([0.60, 0.10, 0.0], dtype=np.float32)
    actions = np.asarray(
        [
            [-0.9, -0.9, -0.9, -0.9],
            [-0.3, 0.2, -0.3, 0.2],
            [0.3, -0.2, 0.3, -0.2],
            [0.9, 0.9, 0.9, 0.9],
        ],
        dtype=np.float32,
    )
    # 28 samples share an eligible context; groups of 3 and 1 are ineligible.
    episode_specs = [(0, 4)] * 7 + [(1, 3), (2, 1)]
    slot = 0
    for episode_id, (group, length) in enumerate(episode_specs):
        obs = _observation(group)
        for episode_step in range(length):
            action = actions[(slot + seed) % actions.shape[0]]
            final = episode_step == length - 1
            replay.add_transition(
                observation=obs,
                achieved_goal=goal,
                desired_goal=goal,
                action_exec=action,
                reward_ext=-1.0,
                next_observation=obs,
                next_achieved_goal=goal,
                done=final,
                timeout=final,
                h=20,
                episode_id=episode_id,
                episode_step=episode_step,
                coarse_key=task.coarse_hash(obs, 20),
                action_bin=int(task.action_bin_batch(action[None, :], 5)[0]),
            )
            slot += 1
    if replay.size != 32:
        raise AssertionError(f"unexpected replay size: {replay.size}")
    return replay


def make_gap_stats() -> GapStats:
    return GapStats(100, 128, 0.60, 0.95, 1e-6)


def make_weight_computer(
    config: LiteConfig,
    task: MetaWorldDrawerOpenTaskSpec,
    gap_stats: GapStats,
) -> LiteWeightComputer:
    return LiteWeightComputer(config, task, gap_stats)


def _semantic_support_cache_key(key: tuple[object, ...]) -> str:
    """Normalize the replay-index epoch while preserving query semantics."""
    normalized = list(key)
    if len(normalized) > 4:
        normalized[4] = "<bucket_version>"
    return repr(tuple(normalized))


def _semantic_local_cache_key(key: tuple[object, ...]) -> str:
    """Normalize the replay-index epoch while preserving lookup semantics."""
    normalized = list(key)
    if len(normalized) > 1:
        normalized[1] = "<bucket_version>"
    return repr(tuple(normalized))


def cache_snapshot(computer: LiteWeightComputer) -> dict[str, object]:
    return {
        "support_cache_size": len(computer._support_cache),
        "local_scan_cache_size": len(computer._local_scan_cache),
        "support_cache_keys": tuple(repr(key) for key in computer._support_cache),
        "local_scan_cache_keys": tuple(
            repr(key) for key in computer._local_scan_cache
        ),
        "semantic_support_cache_keys": tuple(
            _semantic_support_cache_key(key)
            for key in computer._support_cache
        ),
        "semantic_local_scan_cache_keys": tuple(
            _semantic_local_cache_key(key)
            for key in computer._local_scan_cache
        ),
        "runtime_context_calls": int(computer._runtime_context_calls),
        "numba_support_scan_enabled": bool(
            computer._numba_support_scan_enabled
        ),
    }


def run_actual_update(
    agent: DDPGAgent,
    computer: LiteWeightComputer,
    streams: dict[str, np.random.Generator],
    replay: HERReplayBuffer,
) -> dict[str, object]:
    batch = replay.sample(BATCH_SIZE, 0.8, streams["sample"])
    state_np = np.concatenate(
        [batch.observation, batch.desired_goal_train], axis=1
    ).astype(np.float32)
    next_state_np = np.concatenate(
        [batch.next_observation, batch.desired_goal_train], axis=1
    ).astype(np.float32)
    state = torch.as_tensor(state_np, dtype=torch.float32)
    next_state = torch.as_tensor(next_state_np, dtype=torch.float32)
    action = torch.as_tensor(batch.action_exec, dtype=torch.float32)
    reward = torch.as_tensor(batch.reward_ext, dtype=torch.float32)
    env_terminal = batch.done.astype(np.float32) * (
        1.0 - batch.timeout.astype(np.float32)
    )
    nonterminal = torch.as_tensor(1.0 - env_terminal, dtype=torch.float32)

    weight_rng_before = copy.deepcopy(streams["weight"].bit_generator.state)
    wb = computer.compute(
        batch=batch,
        replay=replay,
        agent=agent,
        rng=streams["weight"],
        device=torch.device("cpu"),
    )
    weight_rng_unchanged = equal_state(
        weight_rng_before, streams["weight"].bit_generator.state
    )
    with torch.no_grad():
        policy_action = (
            agent.policy_actions(state).detach().cpu().numpy().astype(np.float32)
        )
    query_action, direct_query_metrics = computer._support_query_actions(
        batch, policy_action
    )
    query_action_exact = np.array_equal(query_action, batch.action_exec)
    query_match_exact = (
        wb.extra_metrics["query/query_bin_matches_replay_action_rate"] == 1.0
        and direct_query_metrics[
            "query/query_bin_matches_replay_action_rate"
        ]
        == 1.0
    )
    if not np.array_equal(wb.omega_actor, np.ones(BATCH_SIZE, dtype=np.float32)):
        raise AssertionError("replay query weights leaked into actor loss")

    # Exercise and checkpoint every non-query RNG family. Replay-query itself
    # must leave the weight RNG untouched.
    streams["noise"].normal(size=ACTION_DIM)
    random.random()
    np.random.random()
    torch.rand(1)
    update = agent.update(
        state=state,
        action_exec=action,
        reward=reward,
        next_state=next_state,
        nonterminal=nonterminal,
        omega_critic=torch.as_tensor(wb.omega_critic, dtype=torch.float32),
        omega_actor=torch.as_tensor(wb.omega_actor, dtype=torch.float32),
    )
    return {
        "batch_indices": batch.indices.copy(),
        "desired_goal_train": batch.desired_goal_train.copy(),
        "query_action": query_action.copy(),
        "query_action_exact": query_action_exact,
        "query_match_exact": query_match_exact,
        "weight_rng_unchanged": weight_rng_unchanged,
        "weight_arrays": {
            name: np.asarray(getattr(wb, name)).copy()
            for name in WEIGHT_ARRAY_FIELDS
        },
        "query_metrics": {
            name: float(wb.extra_metrics[name])
            for name in QUERY_METRIC_FIELDS
        },
        "critic_loss": update.critic_loss,
        "actor_loss": update.actor_loss,
        "td_error_abs_values": update.td_error_abs_values.copy(),
    }


def run_actual_updates(
    agent: DDPGAgent,
    computer: LiteWeightComputer,
    streams: dict[str, np.random.Generator],
    replay: HERReplayBuffer,
    count: int,
) -> list[dict[str, object]]:
    return [
        run_actual_update(agent, computer, streams, replay)
        for _ in range(count)
    ]


def make_checkpoint_payload(
    config: LiteConfig,
    agent: DDPGAgent,
    replay: HERReplayBuffer,
    gap_stats: GapStats,
    streams: dict[str, np.random.Generator],
    last_trace: dict[str, object],
) -> dict[str, object]:
    fingerprint = checkpoint_config_fingerprint(config)
    rng_fields = list(GLOBAL_RNG_STATE_FIELDS) + [
        "noise_rng_state",
        "sample_rng_state",
        "weight_rng_state",
    ]
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_metadata": {
            "global_step": SPLIT_UPDATE,
            "total_updates": SPLIT_UPDATE,
            "at_episode_boundary": True,
            "config_fingerprint": fingerprint,
        },
        "config": config.as_dict(),
        "config_fingerprint": fingerprint,
        "global_step": SPLIT_UPDATE,
        "episode_idx": 1,
        "total_updates": SPLIT_UPDATE,
        "episode_state": {
            "at_episode_boundary": True,
            "episode_step": 0,
        },
        "agent": agent.state_dict(),
        "replay": replay.state_dict(),
        "gap_stats": gap_stats.state_dict(),
        "noise_rng_state": copy.deepcopy(streams["noise"].bit_generator.state),
        "sample_rng_state": copy.deepcopy(streams["sample"].bit_generator.state),
        "weight_rng_state": copy.deepcopy(streams["weight"].bit_generator.state),
        "last_eval_metrics": {},
        "eval_history_env": [],
        "last_loss": {
            "critic": float(last_trace["critic_loss"]),
            "actor": float(last_trace["actor_loss"]),
        },
        "last_lite": dict(last_trace["query_metrics"]),
        "last_her_ratio": 0.8,
        "resume_audit": {
            "schema_version": 1,
            "strict_control": False,
            "replay_action_query": True,
            "episode_boundary": True,
            "global_step": SPLIT_UPDATE,
            "total_updates": SPLIT_UPDATE,
            "rng_state_fields": rng_fields,
            "config_fingerprint": fingerprint,
        },
        **capture_global_rng_state(),
    }


def cache_keys_encode_mode_and_bin(snapshot: dict[str, object]) -> bool:
    keys = tuple(snapshot["support_cache_keys"])
    return bool(keys) and all(
        "replay_action" in key and "query_bin" in key for key in keys
    )


def audit_seed(seed: int, temp_root: Path) -> dict[str, object]:
    config = make_config()
    task = make_task()

    continuous_replay = make_replay(task, seed)
    continuous_agent = make_agent(seed)
    continuous_streams = make_streams(seed)
    continuous_gap = make_gap_stats()
    continuous_computer = make_weight_computer(
        config, task, continuous_gap
    )
    continuous_first = run_actual_updates(
        continuous_agent,
        continuous_computer,
        continuous_streams,
        continuous_replay,
        SPLIT_UPDATE,
    )
    continuous_cache_at_split = cache_snapshot(continuous_computer)
    continuous_second = run_actual_updates(
        continuous_agent,
        continuous_computer,
        continuous_streams,
        continuous_replay,
        UPDATES - SPLIT_UPDATE,
    )
    continuous_trace = continuous_first + continuous_second
    continuous_learning = copy.deepcopy(continuous_agent.state_dict())
    continuous_stream_state = generator_states(continuous_streams)
    continuous_global = capture_global_rng_state()
    continuous_cache_final = cache_snapshot(continuous_computer)

    split_replay = make_replay(task, seed)
    split_agent = make_agent(seed)
    split_streams = make_streams(seed)
    split_gap = make_gap_stats()
    split_computer = make_weight_computer(config, task, split_gap)
    first_trace = run_actual_updates(
        split_agent,
        split_computer,
        split_streams,
        split_replay,
        SPLIT_UPDATE,
    )
    split_cache_at_checkpoint = cache_snapshot(split_computer)
    payload = make_checkpoint_payload(
        config,
        split_agent,
        split_replay,
        split_gap,
        split_streams,
        first_trace[-1],
    )
    if "assignment_rng_state" in payload or "assignment_diagnostics_state" in payload:
        raise AssertionError("support-mode checkpoint contains strict assignment state")
    checkpoint_path = temp_root / f"seed_{seed}.pt"
    save_checkpoint(checkpoint_path, payload)

    resumed_config = make_config()
    resumed_task = make_task()
    resumed_agent = make_agent(seed + 1_000)
    resumed_replay = HERReplayBuffer(64, OBS_DIM, GOAL_DIM, ACTION_DIM)
    resumed_streams = make_streams(seed + 1_000)
    resumed_gap = make_gap_stats()
    resumed_computer = make_weight_computer(
        resumed_config, resumed_task, resumed_gap
    )
    cold_cache_after_restart = cache_snapshot(resumed_computer)
    loaded = load_checkpoint(checkpoint_path)
    resume_validation = validate_resume_payload(
        loaded, resumed_config, torch.device("cpu")
    )
    resumed_agent.load_state_dict(loaded["agent"])
    resumed_replay.load_state_dict(loaded["replay"])
    resumed_gap.load_state_dict(loaded["gap_stats"])
    restore_generator_states(
        resumed_streams,
        {
            "noise": loaded["noise_rng_state"],
            "sample": loaded["sample_rng_state"],
            "weight": loaded["weight_rng_state"],
        },
    )
    restore_global_rng_state(
        loaded, strict=True, device=torch.device("cpu")
    )
    second_trace = run_actual_updates(
        resumed_agent,
        resumed_computer,
        resumed_streams,
        resumed_replay,
        UPDATES - SPLIT_UPDATE,
    )
    resumed_trace = first_trace + second_trace
    resumed_cache_final = cache_snapshot(resumed_computer)

    trace_exact = equal_state(continuous_trace, resumed_trace)
    weight_arrays_exact = all(
        equal_state(
            left["weight_arrays"], right["weight_arrays"]
        )
        for left, right in zip(continuous_trace, resumed_trace)
    )
    query_metrics_exact = all(
        equal_state(left["query_metrics"], right["query_metrics"])
        for left, right in zip(continuous_trace, resumed_trace)
    )
    query_action_exact = all(
        bool(record["query_action_exact"])
        and bool(record["query_match_exact"])
        for record in resumed_trace
    )
    weight_rng_isolated = all(
        bool(record["weight_rng_unchanged"])
        for record in resumed_trace
    )
    model_exact = equal_state(
        continuous_learning, resumed_agent.state_dict()
    )
    losses_exact = all(
        left["critic_loss"] == right["critic_loss"]
        and left["actor_loss"] == right["actor_loss"]
        and equal_state(
            left["td_error_abs_values"], right["td_error_abs_values"]
        )
        for left, right in zip(continuous_trace, resumed_trace)
    )
    streams_exact = equal_state(
        continuous_stream_state, generator_states(resumed_streams)
    )
    global_exact = equal_state(
        continuous_global, capture_global_rng_state()
    )
    mode_preserved = (
        loaded["config"]["support_query_mode"] == QUERY_MODE
        and loaded["config"]["support_assignment_mode"] == ASSIGNMENT_MODE
        and resumed_config.support_query_mode == QUERY_MODE
        and resume_validation["replay_action_query"] is True
        and resume_validation["config_fingerprint_matched"] is True
    )
    cache_split_exact = equal_state(
        continuous_cache_at_split, split_cache_at_checkpoint
    )
    cold_cache_verified = (
        cold_cache_after_restart["support_cache_size"] == 0
        and cold_cache_after_restart["local_scan_cache_size"] == 0
        and cold_cache_after_restart["runtime_context_calls"] == 0
    )
    cache_rebuilt = (
        resumed_cache_final["support_cache_size"] > 0
        and resumed_cache_final["local_scan_cache_size"] > 0
    )
    resumed_support_keys = set(
        resumed_cache_final["semantic_support_cache_keys"]
    )
    continuous_support_keys = set(
        continuous_cache_final["semantic_support_cache_keys"]
    )
    resumed_local_keys = set(
        resumed_cache_final["semantic_local_scan_cache_keys"]
    )
    continuous_local_keys = set(
        continuous_cache_final["semantic_local_scan_cache_keys"]
    )
    bucket_version_invalidation_observed = (
        set(resumed_cache_final["support_cache_keys"])
        != set(continuous_cache_final["support_cache_keys"])
        and resumed_support_keys.issubset(continuous_support_keys)
    )
    cache_keys_valid = (
        resumed_support_keys.issubset(continuous_support_keys)
        and resumed_local_keys.issubset(continuous_local_keys)
        and cache_keys_encode_mode_and_bin(split_cache_at_checkpoint)
        and cache_keys_encode_mode_and_bin(resumed_cache_final)
    )
    numba_active = all(
        bool(snapshot["numba_support_scan_enabled"])
        for snapshot in (
            continuous_cache_at_split,
            split_cache_at_checkpoint,
            cold_cache_after_restart,
            resumed_cache_final,
        )
    )

    row: dict[str, object] = {
        "seed": seed,
        "query_mode_preserved": mode_preserved,
        "replay_query_is_deterministic": (
            trace_exact and query_action_exact and weight_rng_isolated
        ),
        "learning_result_reproduced": model_exact and losses_exact,
        "rng_state_restored": streams_exact and global_exact,
        "actual_lite_weight_computer_used": True,
        "actual_her_replay_used": True,
        "numba_support_scan_active": numba_active,
        "weight_arrays_exact": weight_arrays_exact,
        "query_metrics_exact": query_metrics_exact,
        "query_action_exact": query_action_exact,
        "query_does_not_consume_weight_rng": weight_rng_isolated,
        "model_optimizer_state_exact": model_exact,
        "agent_loss_trace_exact": losses_exact,
        "dedicated_rng_streams_exact": streams_exact,
        "global_rng_streams_exact": global_exact,
        "checkpoint_resume_validator_passed": True,
        "checkpoint_excludes_strict_assignment_state": True,
        "cache_state_at_split_exact": cache_split_exact,
        "cache_cold_restart_verified": cold_cache_verified,
        "cache_rebuilt_after_resume": cache_rebuilt,
        "cache_keys_mode_and_query_bin_valid": cache_keys_valid,
        "cache_bucket_version_invalidation_observed": (
            bucket_version_invalidation_observed
        ),
        "cache_cold_resume_semantic_exact": (
            weight_arrays_exact and query_metrics_exact and losses_exact
        ),
        "cache_state": {
            "continuous_at_split": continuous_cache_at_split,
            "split_at_checkpoint": split_cache_at_checkpoint,
            "cold_after_restart": cold_cache_after_restart,
            "continuous_final": continuous_cache_final,
            "resumed_final": resumed_cache_final,
        },
        "updates": UPDATES,
        "split_update": SPLIT_UPDATE,
    }
    required = (
        "query_mode_preserved",
        "replay_query_is_deterministic",
        "learning_result_reproduced",
        "rng_state_restored",
        "numba_support_scan_active",
        "weight_arrays_exact",
        "query_metrics_exact",
        "query_action_exact",
        "model_optimizer_state_exact",
        "agent_loss_trace_exact",
        "cache_state_at_split_exact",
        "cache_cold_restart_verified",
        "cache_rebuilt_after_resume",
        "cache_keys_mode_and_query_bin_valid",
        "cache_bucket_version_invalidation_observed",
        "cache_cold_resume_semantic_exact",
    )
    row["passed"] = all(row[key] is True for key in required)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not NUMBA_AVAILABLE:
        raise RuntimeError(
            "actual replay-query checkpoint audit requires Numba"
        )
    with tempfile.TemporaryDirectory(
        prefix="drawer_replay_query_resume_audit_"
    ) as temp:
        rows = [audit_seed(seed, Path(temp)) for seed in SEEDS]
    payload = {
        "schema_version": 2,
        "audit": (
            "actual_drawer_open_lite_weight_computer_numba_"
            "ddpg_split_resume"
        ),
        "seeds": rows,
        "seed_coverage": list(SEEDS),
        "passed": all(row["passed"] for row in rows),
        "support_query_mode": QUERY_MODE,
        "support_assignment_mode": ASSIGNMENT_MODE,
        "weight_scope": WEIGHT_SCOPE,
        "actual_lite_weight_computer": True,
        "actual_her_replay": True,
        "numba_required": True,
        "cache_policy": (
            "derived support/local-scan caches are intentionally cold after "
            "resume; semantic weights, query metrics, losses, and learning "
            "state must remain exact"
        ),
        "device": "cpu",
        "performance_used_for_gate_decision": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
