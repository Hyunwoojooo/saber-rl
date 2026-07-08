from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from .agent import DDPGAgent
from .config import LiteConfig
from .replay import HERReplayBuffer, SampleBatch


@dataclass
class HGRChoice:
    slot: int
    desired_goal: np.ndarray
    future_slot: int
    entropy: float
    selected_rank: float
    td_mean: float
    td_max: float


class HGRSampler:
    """Two-step Hindsight Goal Ranking sampler for DDPG+HER.

    Step 1 samples an episode using mean transition TD-error priority.
    Step 2 samples a future achieved goal in that episode using candidate TD-error.
    The sampler changes only replay/HER goal selection; loss weights remain all one.
    """

    def __init__(self, config: LiteConfig, env: Any, device: torch.device) -> None:
        self.cfg = config
        self.env = env
        self.device = device
        self.latest_metrics: Dict[str, float] = self._empty_metrics()

    def _empty_metrics(self) -> Dict[str, float]:
        return {
            "hgr/episode_priority_mean": float("nan"),
            "hgr/episode_priority_std": float("nan"),
            "hgr/episode_priority_max": float("nan"),
            "hgr/goal_priority_entropy_mean": float("nan"),
            "hgr/goal_selected_rank_mean": float("nan"),
            "hgr/candidate_td_error_mean": float("nan"),
            "hgr/candidate_td_error_max": float("nan"),
            "hgr/relabel_fraction": 0.0,
            "hgr/priority_clip_rate": 0.0,
            "hgr/fallback_fraction": 0.0,
            "runtime/hgr_sampling_sec_per_update": 0.0,
        }

    def sample_batch(
        self,
        replay: HERReplayBuffer,
        agent: DDPGAgent,
        batch_size: int,
        her_ratio: float,
        rng: np.random.Generator,
    ) -> SampleBatch:
        if replay.size < int(batch_size):
            raise ValueError(f"Not enough replay samples: {replay.size} < {batch_size}")
        t0 = time.perf_counter()
        b = int(batch_size)
        indices = np.zeros(b, dtype=np.int64)
        desired_goal_train = np.zeros((b, replay.goal_dim), dtype=np.float32)
        her_relabel = np.zeros(b, dtype=np.float32)
        future_slot = np.full(b, -1, dtype=np.int64)

        episode_ids, episode_probs, slots_by_episode = self._episode_distribution(replay)
        query_batch_indices: list[int] = []
        query_slots: list[int] = []
        query_futures: list[np.ndarray] = []
        fallback_count = 0

        for i in range(b):
            if rng.random() >= float(her_ratio) or episode_ids.size == 0:
                slot = int(rng.integers(0, replay.size, endpoint=False))
                indices[i] = slot
                desired_goal_train[i] = replay.desired_goal[slot]
                continue
            ep_pos = int(rng.choice(np.arange(episode_ids.size), p=episode_probs))
            ep = int(episode_ids[ep_pos])
            episode_slots = slots_by_episode.get(ep, [])
            if len(episode_slots) < 2:
                fallback_count += 1
                slot = int(rng.integers(0, replay.size, endpoint=False))
                indices[i] = slot
                desired_goal_train[i] = replay.desired_goal[slot]
                continue
            slot_pos = int(rng.integers(0, len(episode_slots) - 1, endpoint=False))
            slot = int(episode_slots[slot_pos])
            futures = self._future_slots_from_episode(episode_slots, slot_pos)
            if futures.size == 0:
                fallback_count += 1
                indices[i] = slot
                desired_goal_train[i] = replay.desired_goal[slot]
                continue
            indices[i] = slot
            query_batch_indices.append(i)
            query_slots.append(slot)
            query_futures.append(futures)

        if query_slots:
            td_by_query = self._candidate_td_errors_batch(replay, agent, query_slots, query_futures)
        else:
            td_by_query = []

        entropies: list[float] = []
        selected_ranks: list[float] = []
        td_means: list[float] = []
        td_maxes: list[float] = []
        for batch_i, slot, futures, td_abs in zip(query_batch_indices, query_slots, query_futures, td_by_query):
            choice = self._choose_future_goal(replay, slot, futures, td_abs, rng)
            if choice is None:
                fallback_count += 1
                desired_goal_train[batch_i] = replay.desired_goal[slot]
                continue
            desired_goal_train[batch_i] = choice.desired_goal
            her_relabel[batch_i] = 1.0
            future_slot[batch_i] = choice.future_slot
            entropies.append(choice.entropy)
            selected_ranks.append(choice.selected_rank)
            td_means.append(choice.td_mean)
            td_maxes.append(choice.td_max)

        self.latest_metrics = replay.hgr_priority_stats(
            eps=float(self.cfg.hgr_priority_eps),
            clip_max=float(self.cfg.hgr_priority_clip_max),
        )
        self.latest_metrics.update(
            {
                "hgr/goal_priority_entropy_mean": float(np.mean(entropies)) if entropies else float("nan"),
                "hgr/goal_selected_rank_mean": float(np.mean(selected_ranks)) if selected_ranks else float("nan"),
                "hgr/candidate_td_error_mean": float(np.mean(td_means)) if td_means else float("nan"),
                "hgr/candidate_td_error_max": float(np.max(td_maxes)) if td_maxes else float("nan"),
                "hgr/relabel_fraction": float(np.mean(her_relabel)) if b else 0.0,
                "hgr/fallback_fraction": float(fallback_count / max(1, b)),
                "runtime/hgr_sampling_sec_per_update": float(time.perf_counter() - t0),
            }
        )
        return SampleBatch(
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
            her_relabel_ratio=float(np.mean(her_relabel)) if b else 0.0,
            her_relabel=her_relabel,
            hgr_future_slot=future_slot,
        )

    def _episode_distribution(
        self,
        replay: HERReplayBuffer,
    ) -> tuple[np.ndarray, np.ndarray, Dict[int, list[int]]]:
        eps = float(self.cfg.hgr_priority_eps)
        clip_max = float(self.cfg.hgr_priority_clip_max)
        episodes: list[int] = []
        priorities: list[float] = []
        slots_by_episode: Dict[int, list[int]] = {}
        for ep in list(replay._episode_to_slots.keys()):
            slots = replay.episode_slots(int(ep))
            if len(slots) < 2:
                continue
            episodes.append(int(ep))
            slots_by_episode[int(ep)] = slots
            priorities.append(replay.episode_priority(int(ep), eps=eps, clip_max=clip_max))
        if not episodes:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float64),
                slots_by_episode,
            )
        priority_arr = np.asarray(priorities, dtype=np.float64)
        priority_arr = np.nan_to_num(priority_arr, nan=1.0, posinf=clip_max, neginf=eps)
        weights = np.power(np.maximum(priority_arr, eps), float(self.cfg.hgr_alpha_episode))
        if not np.isfinite(weights).all() or float(np.sum(weights)) <= 0.0:
            probs = np.full(len(episodes), 1.0 / float(len(episodes)), dtype=np.float64)
        else:
            probs = weights.astype(np.float64)
            probs = probs / float(np.sum(probs))
            probs[-1] += 1.0 - float(np.sum(probs))
        return np.asarray(episodes, dtype=np.int64), probs, slots_by_episode

    def _future_slots_from_episode(self, episode_slots: list[int], slot_pos: int) -> np.ndarray:
        future = np.asarray(episode_slots[int(slot_pos) + 1 :], dtype=np.int64)
        if future.size == 0:
            return np.zeros((0,), dtype=np.int64)
        max_i = int(self.cfg.hgr_max_future_candidates)
        if max_i > 0 and future.size > max_i:
            positions = np.linspace(0, future.size - 1, num=max_i, dtype=np.int64)
            future = future[positions]
        return future.astype(np.int64, copy=False)

    @torch.no_grad()
    def _candidate_td_errors_batch(
        self,
        replay: HERReplayBuffer,
        agent: DDPGAgent,
        slots: list[int],
        futures_by_query: list[np.ndarray],
    ) -> list[np.ndarray]:
        counts = [int(np.asarray(futures).size) for futures in futures_by_query]
        total = int(sum(counts))
        if total <= 0:
            return [np.zeros((0,), dtype=np.float32) for _ in slots]
        slot_rep = np.empty(total, dtype=np.int64)
        future_cat = np.empty(total, dtype=np.int64)
        cursor = 0
        for slot, futures in zip(slots, futures_by_query):
            futures_arr = np.asarray(futures, dtype=np.int64).reshape(-1)
            n = int(futures_arr.size)
            slot_rep[cursor : cursor + n] = int(slot)
            future_cat[cursor : cursor + n] = futures_arr
            cursor += n
        goals = replay.next_achieved_goal[future_cat].astype(np.float32)
        next_ag = replay.next_achieved_goal[slot_rep].astype(np.float32)
        reward = self.env.compute_reward(next_ag, goals, {}).astype(np.float32)
        success = self.env.success_from_goal(next_ag, goals).astype(np.float32)
        env_terminal = replay.done[slot_rep].astype(np.float32) * (1.0 - replay.timeout[slot_rep].astype(np.float32))
        if not bool(self.cfg.timeout_bootstrap):
            env_terminal = np.maximum(env_terminal, replay.timeout[slot_rep].astype(np.float32))
        terminal = env_terminal.copy()
        if bool(self.cfg.success_terminates):
            terminal = np.maximum(terminal, success)
        nonterminal = (1.0 - terminal).astype(np.float32)
        state = np.concatenate([replay.observation[slot_rep], goals], axis=1).astype(np.float32)
        next_state = np.concatenate([replay.next_observation[slot_rep], goals], axis=1).astype(np.float32)
        action_exec = replay.action_exec[slot_rep].astype(np.float32)
        td_tensor = self._td_error_abs_tensor(
            agent=agent,
            state=state,
            action_exec=action_exec,
            reward=reward,
            next_state=next_state,
            nonterminal=nonterminal,
        )
        td_flat = td_tensor.reshape(-1).astype(np.float32)
        out: list[np.ndarray] = []
        cursor = 0
        for count in counts:
            out.append(td_flat[cursor : cursor + int(count)].copy())
            cursor += int(count)
        return out

    @torch.no_grad()
    def _td_error_abs_tensor(
        self,
        agent: Optional[DDPGAgent],
        state: np.ndarray,
        action_exec: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        nonterminal: np.ndarray,
    ) -> np.ndarray:
        # Used by fake-batch tests and by the vectorized sampler.
        td_abs = agent.td_error_abs(
            torch.as_tensor(state, dtype=torch.float32, device=self.device),
            torch.as_tensor(action_exec, dtype=torch.float32, device=self.device),
            torch.as_tensor(reward, dtype=torch.float32, device=self.device),
            torch.as_tensor(next_state, dtype=torch.float32, device=self.device),
            torch.as_tensor(nonterminal, dtype=torch.float32, device=self.device),
        )
        return td_abs.detach().cpu().numpy().astype(np.float32)

    def _choose_future_goal(
        self,
        replay: HERReplayBuffer,
        slot: int,
        future: np.ndarray,
        td_abs: np.ndarray,
        rng: np.random.Generator,
    ) -> Optional[HGRChoice]:
        future_arr = np.asarray(future, dtype=np.int64).reshape(-1)
        td_arr = np.asarray(td_abs, dtype=np.float32).reshape(-1)
        if future_arr.size == 0:
            return None
        if td_arr.size != future_arr.size or not np.any(np.isfinite(td_arr)):
            probs = np.full(future_arr.size, 1.0 / float(future_arr.size), dtype=np.float64)
            td_arr = np.ones(future_arr.size, dtype=np.float32)
        else:
            clean = np.nan_to_num(td_arr, nan=0.0, posinf=float(self.cfg.hgr_priority_clip_max), neginf=0.0)
            weights = np.power(np.maximum(clean, 0.0) + float(self.cfg.hgr_priority_eps), float(self.cfg.hgr_alpha_goal))
            if not np.isfinite(weights).all() or float(np.sum(weights)) <= 0.0:
                probs = np.full(future_arr.size, 1.0 / float(future_arr.size), dtype=np.float64)
            else:
                probs = weights.astype(np.float64)
                probs = probs / float(np.sum(probs))
                probs[-1] += 1.0 - float(np.sum(probs))
        chosen_pos = int(rng.choice(np.arange(future_arr.size), p=probs))
        chosen_future = int(future_arr[chosen_pos])
        order = np.argsort(-td_arr, kind="mergesort")
        rank_positions = np.empty(order.size, dtype=np.int64)
        rank_positions[order] = np.arange(1, order.size + 1, dtype=np.int64)
        entropy = float(-(probs * np.log(probs + float(self.cfg.hgr_priority_eps))).sum())
        return HGRChoice(
            slot=int(slot),
            desired_goal=replay.next_achieved_goal[chosen_future].copy(),
            future_slot=chosen_future,
            entropy=entropy,
            selected_rank=float(rank_positions[chosen_pos]),
            td_mean=float(np.mean(td_arr)) if td_arr.size else float("nan"),
            td_max=float(np.max(td_arr)) if td_arr.size else float("nan"),
        )

    @torch.no_grad()
    def candidate_td_errors(
        self,
        replay: HERReplayBuffer,
        agent: DDPGAgent,
        slot: int,
        future_slots: np.ndarray,
    ) -> np.ndarray:
        return self._candidate_td_errors_batch(replay, agent, [int(slot)], [np.asarray(future_slots, dtype=np.int64)])[0]

    def update_priorities(self, replay: HERReplayBuffer, indices: np.ndarray, td_abs: np.ndarray) -> Dict[str, float]:
        replay.update_hgr_priorities(
            indices=indices,
            td_abs=td_abs,
            eps=float(self.cfg.hgr_priority_eps),
            clip_max=float(self.cfg.hgr_priority_clip_max),
        )
        return replay.hgr_priority_stats(
            eps=float(self.cfg.hgr_priority_eps),
            clip_max=float(self.cfg.hgr_priority_clip_max),
        )
