from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from .replay import HERReplayBuffer, SampleBatch


@dataclass
class HGRBatchInfo:
    metrics: Dict[str, float]
    relabel_mask: np.ndarray
    selected_future_slots: np.ndarray


class HGRSampler:
    """Two-step Hindsight Goal Ranking sampler for DDPG+HER.

    HGR changes only HER sampling: it samples high-TD-error episodes more often,
    then ranks future achieved-goal candidates by their current TD error. Loss
    weights remain all ones in the trainer.
    """

    def __init__(
        self,
        capacity: int,
        priority_eps: float = 1e-6,
        priority_init: float = 1.0,
        priority_clip_max: float = 50.0,
        alpha_episode: float = 1.0,
        alpha_goal: float = 1.0,
        uniform_mix: float = 0.0,
        max_future_candidates: int = 50,
        episode_priority_ema: float = 0.0,
    ) -> None:
        self.capacity = int(capacity)
        self.priority_eps = float(priority_eps)
        self.priority_init = float(priority_init)
        self.priority_clip_max = float(priority_clip_max)
        self.alpha_episode = float(alpha_episode)
        self.alpha_goal = float(alpha_goal)
        self.uniform_mix = float(np.clip(uniform_mix, 0.0, 1.0))
        self.max_future_candidates = int(max_future_candidates)
        self.episode_priority_ema = float(np.clip(episode_priority_ema, 0.0, 1.0))

        self.slot_priority = np.full(self.capacity, self.priority_init, dtype=np.float32)
        self.slot_episode_id = np.full(self.capacity, -1, dtype=np.int64)
        self.slot_episode_step = np.full(self.capacity, -1, dtype=np.int32)
        self._episode_priority_sum: Dict[int, float] = {}
        self._episode_priority_count: Dict[int, int] = {}
        self._last_update_clip_rate = 0.0
        self._last_update_mean_priority = float(self.priority_init)

    @classmethod
    def from_config(cls, config: Any, capacity: int) -> "HGRSampler":
        return cls(
            capacity=capacity,
            priority_eps=float(config.hgr_priority_eps),
            priority_init=float(config.hgr_priority_init),
            priority_clip_max=float(config.hgr_priority_clip_max),
            alpha_episode=float(config.hgr_alpha_episode),
            alpha_goal=float(config.hgr_alpha_goal),
            uniform_mix=float(config.hgr_uniform_mix),
            max_future_candidates=int(config.hgr_max_future_candidates),
            episode_priority_ema=float(config.hgr_episode_priority_ema),
        )

    def on_replay_add(self, slot: int, episode_id: int, episode_step: int) -> None:
        slot_i = int(slot)
        if slot_i < 0 or slot_i >= self.capacity:
            raise IndexError(f"HGR slot out of range: {slot_i}")
        old_ep = int(self.slot_episode_id[slot_i])
        if old_ep >= 0:
            self._remove_episode_priority(old_ep, float(self.slot_priority[slot_i]))

        priority = float(np.clip(self.priority_init, self.priority_eps, self.priority_clip_max))
        self.slot_priority[slot_i] = priority
        self.slot_episode_id[slot_i] = int(episode_id)
        self.slot_episode_step[slot_i] = int(episode_step)
        self._add_episode_priority(int(episode_id), priority)

    def rebuild_from_replay(self, replay: HERReplayBuffer) -> None:
        self.slot_priority.fill(float(np.clip(self.priority_init, self.priority_eps, self.priority_clip_max)))
        self.slot_episode_id.fill(-1)
        self.slot_episode_step.fill(-1)
        self._episode_priority_sum.clear()
        self._episode_priority_count.clear()
        for slot in range(int(replay.size)):
            ep = int(replay.episode_id[slot])
            if ep < 0:
                continue
            self.slot_episode_id[slot] = ep
            self.slot_episode_step[slot] = int(replay.episode_step[slot])
            self._add_episode_priority(ep, float(self.slot_priority[slot]))

    @torch.no_grad()
    def sample(
        self,
        replay: HERReplayBuffer,
        batch_size: int,
        her_ratio: float,
        rng: np.random.Generator,
        agent: Any,
        env: Any,
        device: torch.device,
        success_terminates: bool,
        timeout_bootstrap: bool,
    ) -> tuple[SampleBatch, HGRBatchInfo]:
        if replay.size < batch_size:
            raise ValueError(f"Not enough replay samples: {replay.size} < {batch_size}")

        t0 = time.perf_counter()
        b = int(batch_size)
        indices = np.empty(b, dtype=np.int64)
        desired_goal_train = np.empty((b, replay.goal_dim), dtype=np.float32)
        relabel_mask = np.zeros(b, dtype=np.bool_)
        selected_future_slots = np.full(b, -1, dtype=np.int64)

        episode_dist = self._eligible_episode_distribution(replay)
        relabel_requested = rng.random(b) < float(her_ratio)
        relabeled = 0
        hgr_fallbacks = 0
        goal_entropies: list[float] = []
        selected_ranks: list[float] = []
        candidate_td_means: list[float] = []
        candidate_td_maxes: list[float] = []

        for out_i in range(b):
            if relabel_requested[out_i] and episode_dist is not None:
                sampled = self._sample_ranked_future_goal(
                    replay=replay,
                    episode_dist=episode_dist,
                    rng=rng,
                    agent=agent,
                    env=env,
                    device=device,
                    success_terminates=success_terminates,
                    timeout_bootstrap=timeout_bootstrap,
                )
                if sampled is not None:
                    slot, future_slot, goal, metrics = sampled
                    indices[out_i] = int(slot)
                    desired_goal_train[out_i] = goal
                    selected_future_slots[out_i] = int(future_slot)
                    relabel_mask[out_i] = True
                    relabeled += 1
                    goal_entropies.append(float(metrics["goal_priority_entropy"]))
                    selected_ranks.append(float(metrics["goal_selected_rank"]))
                    candidate_td_means.append(float(metrics["candidate_td_error_mean"]))
                    candidate_td_maxes.append(float(metrics["candidate_td_error_max"]))
                    continue
                hgr_fallbacks += 1

            slot = int(rng.integers(0, replay.size, endpoint=False))
            indices[out_i] = slot
            desired_goal_train[out_i] = replay.desired_goal[slot]

        sample_sec = float(time.perf_counter() - t0)
        episode_priorities = (
            episode_dist["episode_priorities"]
            if episode_dist is not None
            else np.zeros((0,), dtype=np.float32)
        )
        metrics = {
            "hgr/episode_priority_mean": _mean(episode_priorities, default=float(self.priority_init)),
            "hgr/episode_priority_std": _std(episode_priorities),
            "hgr/episode_priority_max": _max(episode_priorities, default=float(self.priority_init)),
            "hgr/eligible_episode_count": float(episode_priorities.size),
            "hgr/goal_priority_entropy_mean": _mean_list(goal_entropies),
            "hgr/goal_selected_rank_mean": _mean_list(selected_ranks),
            "hgr/candidate_td_error_mean": _mean_list(candidate_td_means),
            "hgr/candidate_td_error_max": _max_list(candidate_td_maxes),
            "hgr/relabel_fraction": float(relabeled / max(1, b)),
            "hgr/fallback_fraction": float(hgr_fallbacks / max(1, int(np.count_nonzero(relabel_requested)))),
            "hgr/priority_clip_rate": float(self._last_update_clip_rate),
            "hgr/update_priority_mean": float(self._last_update_mean_priority),
            "runtime/hgr_sampling_sec_per_update": sample_sec,
        }

        batch = SampleBatch(
            indices=indices,
            observation=replay.observation[indices].copy(),
            achieved_goal=replay.achieved_goal[indices].copy(),
            desired_goal_train=desired_goal_train,
            next_observation=replay.next_observation[indices].copy(),
            next_achieved_goal=replay.next_achieved_goal[indices].copy(),
            action_exec=replay.action_exec[indices].copy(),
            reward_ext=replay.reward_ext[indices].copy(),
            done=replay.done[indices].copy(),
            timeout=replay.timeout[indices].copy(),
            h=replay.h[indices].copy(),
            her_relabel_ratio=float(relabeled / max(1, b)),
        )
        return batch, HGRBatchInfo(metrics=metrics, relabel_mask=relabel_mask, selected_future_slots=selected_future_slots)

    def update_priorities(self, indices: np.ndarray, td_abs_values: np.ndarray, replay: HERReplayBuffer) -> Dict[str, float]:
        clipped = 0
        updated = 0
        new_values: list[float] = []
        for slot_raw, td_raw in zip(np.asarray(indices).reshape(-1), np.asarray(td_abs_values).reshape(-1)):
            slot = int(slot_raw)
            if slot < 0 or slot >= self.capacity:
                continue
            td_value = float(abs(float(td_raw)))
            if not np.isfinite(td_value):
                continue
            if td_value > self.priority_clip_max:
                clipped += 1
            new_priority = float(np.clip(td_value, self.priority_eps, self.priority_clip_max))
            old_priority = float(self.slot_priority[slot])
            if self.episode_priority_ema > 0.0:
                new_priority = (
                    self.episode_priority_ema * old_priority
                    + (1.0 - self.episode_priority_ema) * new_priority
                )
                new_priority = float(np.clip(new_priority, self.priority_eps, self.priority_clip_max))
            ep = int(self.slot_episode_id[slot])
            if ep < 0 or slot >= replay.size or int(replay.episode_id[slot]) != ep:
                continue
            self.slot_priority[slot] = new_priority
            self._episode_priority_sum[ep] = float(self._episode_priority_sum.get(ep, 0.0) + new_priority - old_priority)
            updated += 1
            new_values.append(new_priority)

        self._last_update_clip_rate = float(clipped / max(1, int(np.asarray(td_abs_values).size)))
        self._last_update_mean_priority = _mean_list(new_values, default=self._last_update_mean_priority)
        return self.last_update_metrics()

    def last_update_metrics(self) -> Dict[str, float]:
        return {
            "hgr/priority_clip_rate": float(self._last_update_clip_rate),
            "hgr/update_priority_mean": float(self._last_update_mean_priority),
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "capacity": self.capacity,
            "priority_eps": self.priority_eps,
            "priority_init": self.priority_init,
            "priority_clip_max": self.priority_clip_max,
            "alpha_episode": self.alpha_episode,
            "alpha_goal": self.alpha_goal,
            "uniform_mix": self.uniform_mix,
            "max_future_candidates": self.max_future_candidates,
            "episode_priority_ema": self.episode_priority_ema,
            "slot_priority": self.slot_priority,
            "slot_episode_id": self.slot_episode_id,
            "slot_episode_step": self.slot_episode_step,
            "episode_priority_sum": self._episode_priority_sum,
            "episode_priority_count": self._episode_priority_count,
            "last_update_clip_rate": self._last_update_clip_rate,
            "last_update_mean_priority": self._last_update_mean_priority,
        }

    def load_state_dict(self, payload: Dict[str, object], replay: HERReplayBuffer) -> None:
        self.slot_priority[:] = np.asarray(payload.get("slot_priority", self.slot_priority), dtype=np.float32)[: self.capacity]
        self.slot_episode_id[:] = np.asarray(payload.get("slot_episode_id", self.slot_episode_id), dtype=np.int64)[: self.capacity]
        self.slot_episode_step[:] = np.asarray(payload.get("slot_episode_step", self.slot_episode_step), dtype=np.int32)[: self.capacity]
        self._episode_priority_sum = {
            int(k): float(v) for k, v in payload.get("episode_priority_sum", {}).items()
        }
        self._episode_priority_count = {
            int(k): int(v) for k, v in payload.get("episode_priority_count", {}).items()
        }
        self._last_update_clip_rate = float(payload.get("last_update_clip_rate", 0.0))
        self._last_update_mean_priority = float(payload.get("last_update_mean_priority", self.priority_init))
        self._drop_stale_slots(replay)

    def _eligible_episode_distribution(self, replay: HERReplayBuffer) -> Optional[Dict[str, Any]]:
        episodes: list[int] = []
        priorities: list[float] = []
        slots_by_episode: Dict[int, np.ndarray] = {}
        for ep, count in self._episode_priority_count.items():
            if int(count) < 2:
                continue
            slots = self._valid_episode_slots(replay, int(ep))
            if slots.size < 2:
                continue
            ep_priority = self._episode_priority(int(ep))
            if not np.isfinite(ep_priority):
                ep_priority = float(self.priority_init)
            episodes.append(int(ep))
            priorities.append(float(np.clip(ep_priority, self.priority_eps, self.priority_clip_max)))
            slots_by_episode[int(ep)] = slots

        if not episodes:
            return None
        priority_arr = np.asarray(priorities, dtype=np.float64)
        weights = np.power(priority_arr + self.priority_eps, self.alpha_episode)
        if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0.0:
            weights = np.ones_like(priority_arr, dtype=np.float64)
        probs = weights / float(weights.sum())
        if self.uniform_mix > 0.0:
            probs = (1.0 - self.uniform_mix) * probs + self.uniform_mix / float(probs.size)
            probs = probs / float(probs.sum())
        return {
            "episodes": np.asarray(episodes, dtype=np.int64),
            "probs": probs.astype(np.float64),
            "episode_priorities": priority_arr.astype(np.float32),
            "slots_by_episode": slots_by_episode,
        }

    @torch.no_grad()
    def _sample_ranked_future_goal(
        self,
        replay: HERReplayBuffer,
        episode_dist: Dict[str, Any],
        rng: np.random.Generator,
        agent: Any,
        env: Any,
        device: torch.device,
        success_terminates: bool,
        timeout_bootstrap: bool,
    ) -> Optional[tuple[int, int, np.ndarray, Dict[str, float]]]:
        episodes = episode_dist["episodes"]
        probs = episode_dist["probs"]
        slots_by_episode = episode_dist["slots_by_episode"]
        ep = int(episodes[int(rng.choice(episodes.size, p=probs))])
        slots = np.asarray(slots_by_episode[ep], dtype=np.int64)
        if slots.size < 2:
            return None
        base_slot = int(slots[int(rng.integers(0, slots.size - 1, endpoint=False))])
        step = int(replay.episode_step[base_slot])
        future_slots = slots[replay.episode_step[slots] > step]
        if future_slots.size == 0:
            return None
        if self.max_future_candidates > 0 and future_slots.size > self.max_future_candidates:
            positions = np.linspace(0, future_slots.size - 1, num=self.max_future_candidates, dtype=np.int64)
            future_slots = future_slots[positions].astype(np.int64)

        td_abs = self._candidate_td_errors(
            replay=replay,
            base_slot=base_slot,
            future_slots=future_slots,
            agent=agent,
            env=env,
            device=device,
            success_terminates=success_terminates,
            timeout_bootstrap=timeout_bootstrap,
        )
        if td_abs.size == 0:
            return None
        clipped_td = np.clip(td_abs.astype(np.float64), self.priority_eps, self.priority_clip_max)
        goal_weights = np.power(clipped_td + self.priority_eps, self.alpha_goal)
        if not np.all(np.isfinite(goal_weights)) or float(goal_weights.sum()) <= 0.0:
            goal_weights = np.ones_like(clipped_td, dtype=np.float64)
        goal_probs = goal_weights / float(goal_weights.sum())
        selected_pos = int(rng.choice(goal_probs.size, p=goal_probs))
        selected_future_slot = int(future_slots[selected_pos])
        selected_goal = replay.next_achieved_goal[selected_future_slot].copy()
        selected_td = float(td_abs[selected_pos])
        rank = 1 + int(np.count_nonzero(td_abs > selected_td))
        entropy = float(-(goal_probs * np.log(goal_probs + self.priority_eps)).sum())
        metrics = {
            "goal_priority_entropy": entropy,
            "goal_selected_rank": float(rank),
            "candidate_td_error_mean": float(np.mean(td_abs)),
            "candidate_td_error_max": float(np.max(td_abs)),
        }
        return base_slot, selected_future_slot, selected_goal.astype(np.float32), metrics

    @torch.no_grad()
    def _candidate_td_errors(
        self,
        replay: HERReplayBuffer,
        base_slot: int,
        future_slots: np.ndarray,
        agent: Any,
        env: Any,
        device: torch.device,
        success_terminates: bool,
        timeout_bootstrap: bool,
    ) -> np.ndarray:
        candidate_slots = np.asarray(future_slots, dtype=np.int64)
        n = int(candidate_slots.size)
        if n <= 0:
            return np.zeros((0,), dtype=np.float32)
        goals = replay.next_achieved_goal[candidate_slots].astype(np.float32, copy=True)
        obs = np.repeat(replay.observation[int(base_slot)][None, :], n, axis=0)
        next_obs = np.repeat(replay.next_observation[int(base_slot)][None, :], n, axis=0)
        action = np.repeat(replay.action_exec[int(base_slot)][None, :], n, axis=0)
        next_achieved = np.repeat(replay.next_achieved_goal[int(base_slot)][None, :], n, axis=0)
        reward = np.asarray(env.compute_reward(next_achieved, goals, {}), dtype=np.float32).reshape(-1)
        if hasattr(env, "success_from_goal"):
            success = np.asarray(env.success_from_goal(next_achieved, goals), dtype=np.float32).reshape(-1)
        else:
            success = np.zeros(n, dtype=np.float32)
        env_terminal = float(bool(replay.done[int(base_slot)])) * (1.0 - float(bool(replay.timeout[int(base_slot)])))
        if not timeout_bootstrap:
            env_terminal = max(env_terminal, float(bool(replay.timeout[int(base_slot)])))
        terminal = np.full(n, env_terminal, dtype=np.float32)
        if success_terminates:
            terminal = np.maximum(terminal, success.astype(np.float32))
        nonterminal = (1.0 - terminal).astype(np.float32)

        state_np = np.concatenate([obs, goals], axis=1).astype(np.float32)
        next_state_np = np.concatenate([next_obs, goals], axis=1).astype(np.float32)
        td_abs = agent.td_error_abs(
            torch.as_tensor(state_np, dtype=torch.float32, device=device),
            torch.as_tensor(action, dtype=torch.float32, device=device),
            torch.as_tensor(reward, dtype=torch.float32, device=device),
            torch.as_tensor(next_state_np, dtype=torch.float32, device=device),
            torch.as_tensor(nonterminal, dtype=torch.float32, device=device),
        )
        return td_abs.detach().cpu().numpy().astype(np.float32).reshape(-1)

    def _valid_episode_slots(self, replay: HERReplayBuffer, episode_id: int) -> np.ndarray:
        slots = replay._episode_to_slots.get(int(episode_id), [])
        valid = [
            int(slot)
            for slot in slots
            if 0 <= int(slot) < int(replay.size) and int(replay.episode_id[int(slot)]) == int(episode_id)
        ]
        if not valid:
            return np.zeros((0,), dtype=np.int64)
        valid.sort(key=lambda slot: int(replay.episode_step[slot]))
        return np.asarray(valid, dtype=np.int64)

    def _episode_priority(self, episode_id: int) -> float:
        count = int(self._episode_priority_count.get(int(episode_id), 0))
        if count <= 0:
            return float(self.priority_init)
        return float(self._episode_priority_sum.get(int(episode_id), 0.0) / max(1, count))

    def _add_episode_priority(self, episode_id: int, priority: float) -> None:
        ep = int(episode_id)
        self._episode_priority_sum[ep] = float(self._episode_priority_sum.get(ep, 0.0) + float(priority))
        self._episode_priority_count[ep] = int(self._episode_priority_count.get(ep, 0) + 1)

    def _remove_episode_priority(self, episode_id: int, priority: float) -> None:
        ep = int(episode_id)
        count = int(self._episode_priority_count.get(ep, 0))
        if count <= 1:
            self._episode_priority_sum.pop(ep, None)
            self._episode_priority_count.pop(ep, None)
            return
        self._episode_priority_sum[ep] = float(self._episode_priority_sum.get(ep, 0.0) - float(priority))
        self._episode_priority_count[ep] = count - 1

    def _drop_stale_slots(self, replay: HERReplayBuffer) -> None:
        self._episode_priority_sum.clear()
        self._episode_priority_count.clear()
        for slot in range(self.capacity):
            ep = int(self.slot_episode_id[slot])
            is_stale = ep < 0 or slot >= replay.size or int(replay.episode_id[slot]) != ep
            if is_stale:
                self.slot_episode_id[slot] = -1
                self.slot_episode_step[slot] = -1
                continue
            self._add_episode_priority(ep, float(self.slot_priority[slot]))


def _mean(values: np.ndarray, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=np.float32)
    return float(np.mean(arr)) if arr.size else float(default)


def _std(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    return float(np.std(arr)) if arr.size else 0.0


def _max(values: np.ndarray, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=np.float32)
    return float(np.max(arr)) if arr.size else float(default)


def _mean_list(values: list[float], default: float = 0.0) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float32))) if values else float(default)


def _max_list(values: list[float], default: float = 0.0) -> float:
    return float(np.max(np.asarray(values, dtype=np.float32))) if values else float(default)
