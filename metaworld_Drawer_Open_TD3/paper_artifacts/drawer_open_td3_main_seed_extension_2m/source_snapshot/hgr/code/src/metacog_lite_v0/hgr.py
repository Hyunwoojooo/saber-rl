from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .config import LiteConfig
from .replay import HERReplayBuffer, SampleBatch


HGR_STATE_SCHEMA = "hgr_episode_pair_priority_v1"


@dataclass
class EpisodePairTable:
    episode_id: int
    slots: np.ndarray
    priorities: np.ndarray
    priority_sum: float

    @property
    def pair_count(self) -> int:
        return int(self.priorities.size)

    @property
    def mean_priority(self) -> float:
        return float(self.priority_sum / max(1, self.pair_count))


class HGRSampler:
    """Original two-step HGR over completed-episode transition-goal pairs.

    Each completed episode owns a triangular priority table for every valid
    pair (j, i), where transition j is relabeled with achieved goal g_i and
    j < i. Pair priorities are initialized to the global maximum, sampled in
    two stages, and updated only for pairs used by the critic update.
    """

    def __init__(self, config: LiteConfig, env: Any = None, device: Any = None) -> None:
        del env, device
        self.cfg = config
        self.capacity = int(config.buffer_capacity)
        self.priority_eps = float(config.hgr_priority_eps)
        self.priority_init = float(config.hgr_priority_init)
        self.priority_clip_max = float(config.hgr_priority_clip_max)
        self.alpha_episode = float(config.hgr_alpha_episode)
        self.alpha_goal = float(config.hgr_alpha_goal)
        self.beta_episode_initial = float(config.hgr_beta_episode_initial)
        self.beta_goal_initial = float(config.hgr_beta_goal_initial)
        self.beta_episode_final = float(config.hgr_beta_episode_final)
        self.beta_goal_final = float(config.hgr_beta_goal_final)
        self.uniform_mix = float(config.hgr_uniform_mix)
        first_update_step = max(int(config.warmup_steps), int(config.batch_size))
        default_anneal_updates = max(
            1,
            (int(config.total_steps) - first_update_step + 1) * int(config.updates_per_step),
        )
        configured_anneal = int(config.hgr_beta_anneal_updates)
        self.beta_anneal_updates = configured_anneal if configured_anneal > 0 else default_anneal_updates

        self.max_priority = max(self.priority_eps, self.priority_init)
        self.critic_updates = 0
        self.slot_episode_id = np.full(self.capacity, -1, dtype=np.int64)
        self.slot_episode_step = np.full(self.capacity, -1, dtype=np.int16)
        self._episode_slots: Dict[int, list[int]] = {}
        self._tables: Dict[int, EpisodePairTable] = {}
        self._pair_position_cache: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._last_update_clip_rate = 0.0
        self._last_safety_replacement_rate = 0.0
        self._last_update_mean_priority = float(self.max_priority)
        self.latest_metrics: Dict[str, float] = self._empty_metrics()

    def _empty_metrics(self) -> Dict[str, float]:
        beta_episode, beta_goal = self.current_betas()
        return {
            "hgr/episode_priority_mean": float(self.max_priority),
            "hgr/episode_priority_std": 0.0,
            "hgr/episode_priority_max": float(self.max_priority),
            "hgr/eligible_episode_count": 0.0,
            "hgr/pair_table_count": 0.0,
            "hgr/pair_priority_count": 0.0,
            "hgr/pair_priority_memory_mb": 0.0,
            "hgr/goal_priority_entropy_mean": float("nan"),
            "hgr/goal_selected_rank_mean": float("nan"),
            "hgr/candidate_td_error_mean": float("nan"),
            "hgr/candidate_td_error_max": float("nan"),
            "hgr/relabel_fraction": 0.0,
            "hgr/fallback_fraction": 0.0,
            "hgr/episode_probability_mean": float("nan"),
            "hgr/pair_probability_mean": float("nan"),
            "hgr/is_weight_raw_mean": 1.0,
            "hgr/is_weight_raw_max": 1.0,
            "hgr/is_weight_mean": 1.0,
            "hgr/is_weight_max": 1.0,
            "hgr/is_weight_min": 1.0,
            "hgr/actor_weight_mean": 1.0,
            "hgr/beta_episode": beta_episode,
            "hgr/beta_goal": beta_goal,
            "hgr/beta_progress": self.beta_progress(),
            "hgr/beta_episode_next": beta_episode,
            "hgr/beta_goal_next": beta_goal,
            "hgr/beta_progress_next": self.beta_progress(),
            "hgr/max_priority": float(self.max_priority),
            "hgr/priority_clip_enabled": float(self.priority_clip_max > 0.0),
            "hgr/priority_clip_rate": 0.0,
            "hgr/priority_safety_replacement_rate": 0.0,
            "hgr/update_priority_mean": float(self.max_priority),
            "runtime/hgr_sampling_sec_per_update": 0.0,
        }

    def beta_progress(self) -> float:
        return float(np.clip(self.critic_updates / max(1, self.beta_anneal_updates), 0.0, 1.0))

    def current_betas(self) -> tuple[float, float]:
        progress = self.beta_progress()
        beta_episode = self.beta_episode_initial + progress * (
            self.beta_episode_final - self.beta_episode_initial
        )
        beta_goal = self.beta_goal_initial + progress * (self.beta_goal_final - self.beta_goal_initial)
        return float(beta_episode), float(beta_goal)

    def on_replay_add(
        self,
        slot: int,
        episode_id: int,
        episode_step: int,
        replay: HERReplayBuffer,
        *,
        episode_done: bool,
    ) -> None:
        slot_i = int(slot)
        if slot_i < 0 or slot_i >= self.capacity:
            raise IndexError(f"HGR slot out of range: {slot_i}")
        old_episode = int(self.slot_episode_id[slot_i])
        if old_episode >= 0 and old_episode != int(episode_id):
            self._remove_episode(old_episode)

        ep = int(episode_id)
        self.slot_episode_id[slot_i] = ep
        self.slot_episode_step[slot_i] = int(episode_step)
        slots = self._episode_slots.setdefault(ep, [])
        if slot_i not in slots:
            slots.append(slot_i)
            slots.sort(key=lambda value: int(self.slot_episode_step[int(value)]))
        if episode_done:
            self._finalize_episode(ep, replay)

    def _finalize_episode(self, episode_id: int, replay: HERReplayBuffer) -> bool:
        ep = int(episode_id)
        slots = self._valid_sorted_slots(ep, replay)
        if not slots:
            return False
        steps = [int(replay.episode_step[slot]) for slot in slots]
        if steps != list(range(len(slots))) or not bool(replay.done[slots[-1]]):
            return False
        pair_count = len(slots) * (len(slots) + 1) // 2
        if pair_count <= 0:
            return False
        initial = float(self.max_priority)
        priorities = np.full(pair_count, initial, dtype=np.float32)
        self._tables[ep] = EpisodePairTable(
            episode_id=ep,
            slots=np.asarray(slots, dtype=np.int64),
            priorities=priorities,
            priority_sum=float(initial * pair_count),
        )
        self._episode_slots[ep] = slots
        return True

    def _remove_episode(self, episode_id: int) -> None:
        ep = int(episode_id)
        self._tables.pop(ep, None)
        slots = self._episode_slots.pop(ep, [])
        for slot in slots:
            if 0 <= int(slot) < self.capacity and int(self.slot_episode_id[int(slot)]) == ep:
                self.slot_episode_id[int(slot)] = -1
                self.slot_episode_step[int(slot)] = -1

    def _valid_sorted_slots(self, episode_id: int, replay: HERReplayBuffer) -> list[int]:
        ep = int(episode_id)
        slots = [
            int(slot)
            for slot in self._episode_slots.get(ep, [])
            if 0 <= int(slot) < int(replay.size) and int(replay.episode_id[int(slot)]) == ep
        ]
        slots.sort(key=lambda value: int(replay.episode_step[int(value)]))
        return slots

    def _pair_positions(self, episode_length: int) -> tuple[np.ndarray, np.ndarray]:
        length = int(episode_length)
        cached = self._pair_position_cache.get(length)
        if cached is not None:
            return cached
        transition_positions = np.concatenate(
            [np.full(length - j, j, dtype=np.int16) for j in range(length)]
        )
        # next_achieved_goal at slot k is g_{k+1}; k >= j therefore encodes j < i.
        goal_positions = np.concatenate(
            [np.arange(j, length, dtype=np.int16) for j in range(length)]
        )
        self._pair_position_cache[length] = (transition_positions, goal_positions)
        return transition_positions, goal_positions

    def _priority_probabilities(self, priorities: np.ndarray, alpha: float) -> np.ndarray:
        values = np.asarray(priorities, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return np.zeros((0,), dtype=np.float64)
        if float(alpha) == 0.0:
            probs = np.full(values.size, 1.0 / float(values.size), dtype=np.float64)
        else:
            masses = np.power(np.maximum(values, self.priority_eps), float(alpha))
            total = float(np.sum(masses))
            if not np.all(np.isfinite(masses)) or not np.isfinite(total) or total <= 0.0:
                probs = np.full(values.size, 1.0 / float(values.size), dtype=np.float64)
            else:
                probs = masses / total
        if self.uniform_mix > 0.0:
            probs = (1.0 - self.uniform_mix) * probs + self.uniform_mix / float(values.size)
        probs = probs / float(np.sum(probs))
        probs[-1] += 1.0 - float(np.sum(probs))
        return probs

    def episode_distribution(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        episode_ids = np.asarray(sorted(self._tables), dtype=np.int64)
        if episode_ids.size == 0:
            return episode_ids, np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float32)
        means = np.asarray(
            [self._tables[int(ep)].mean_priority for ep in episode_ids],
            dtype=np.float32,
        )
        probs = self._priority_probabilities(means, self.alpha_episode)
        return episode_ids, probs, means

    def pair_distribution(self, episode_id: int) -> np.ndarray:
        table = self._tables.get(int(episode_id))
        if table is None:
            return np.zeros((0,), dtype=np.float64)
        return self._priority_probabilities(table.priorities, self.alpha_goal)

    def sample_batch(
        self,
        replay: HERReplayBuffer,
        batch_size: int,
        her_ratio: float,
        rng: np.random.Generator,
    ) -> SampleBatch:
        if replay.size < int(batch_size):
            raise ValueError(f"Not enough replay samples: {replay.size} < {batch_size}")
        started = time.perf_counter()
        b = int(batch_size)
        indices = rng.integers(0, replay.size, size=b, endpoint=False, dtype=np.int64)
        desired_goal_train = replay.desired_goal[indices].copy()
        her_relabel = np.zeros(b, dtype=np.float32)
        hgr_episode_id = np.full(b, -1, dtype=np.int64)
        hgr_pair_index = np.full(b, -1, dtype=np.int32)
        hgr_goal_slot = np.full(b, -1, dtype=np.int64)
        episode_probability = np.ones(b, dtype=np.float64)
        pair_probability = np.ones(b, dtype=np.float64)
        is_weight_raw = np.ones(b, dtype=np.float64)
        is_weight = np.ones(b, dtype=np.float32)

        relabel_requested = rng.random(b) < float(her_ratio)
        relabel_positions = np.flatnonzero(relabel_requested)
        episode_ids, episode_probs, episode_priorities = self.episode_distribution()
        beta_episode, beta_goal = self.current_betas()
        entropies: list[float] = []
        ranks: list[float] = []
        sampled_priorities: list[float] = []
        fallback_count = 0

        if episode_ids.size > 0 and relabel_positions.size > 0:
            chosen_episode_positions = rng.choice(
                episode_ids.size,
                size=relabel_positions.size,
                replace=True,
                p=episode_probs,
            )
            for batch_position_raw, episode_position_raw in zip(
                relabel_positions, chosen_episode_positions
            ):
                batch_position = int(batch_position_raw)
                episode_position = int(episode_position_raw)
                episode_id = int(episode_ids[episode_position])
                table = self._tables.get(episode_id)
                if table is None or table.pair_count <= 0:
                    fallback_count += 1
                    continue
                pair_probs = self.pair_distribution(episode_id)
                pair_index = int(rng.choice(table.pair_count, p=pair_probs))
                transition_positions, goal_positions = self._pair_positions(table.slots.size)
                transition_slot = int(table.slots[int(transition_positions[pair_index])])
                goal_slot = int(table.slots[int(goal_positions[pair_index])])
                if (
                    int(replay.episode_id[transition_slot]) != episode_id
                    or int(replay.episode_id[goal_slot]) != episode_id
                ):
                    fallback_count += 1
                    continue

                indices[batch_position] = transition_slot
                desired_goal_train[batch_position] = replay.next_achieved_goal[goal_slot]
                her_relabel[batch_position] = 1.0
                hgr_episode_id[batch_position] = episode_id
                hgr_pair_index[batch_position] = pair_index
                hgr_goal_slot[batch_position] = goal_slot
                episode_probability[batch_position] = float(episode_probs[episode_position])
                pair_probability[batch_position] = float(pair_probs[pair_index])
                log_weight = (
                    -beta_episode
                    * math.log(max(self.priority_eps, episode_ids.size * episode_probs[episode_position]))
                    -beta_goal
                    * math.log(max(self.priority_eps, table.pair_count * pair_probs[pair_index]))
                )
                is_weight_raw[batch_position] = math.exp(min(80.0, log_weight))
                entropy = float(-np.sum(pair_probs * np.log(pair_probs + self.priority_eps)))
                entropies.append(entropy)
                selected_priority = float(table.priorities[pair_index])
                sampled_priorities.append(selected_priority)
                ranks.append(float(1 + np.count_nonzero(table.priorities > selected_priority)))
        else:
            fallback_count = int(relabel_positions.size)

        actual_relabel = her_relabel > 0.5
        if np.any(actual_relabel):
            max_weight = float(np.max(is_weight_raw[actual_relabel]))
            is_weight[actual_relabel] = (
                is_weight_raw[actual_relabel] / max(self.priority_eps, max_weight)
            ).astype(np.float32)

        table_count, pair_count, memory_mb = self.pair_table_stats()
        normalized_relabel_weights = is_weight[actual_relabel]
        raw_relabel_weights = is_weight_raw[actual_relabel]
        self.latest_metrics = {
            "hgr/episode_priority_mean": _mean(episode_priorities, self.max_priority),
            "hgr/episode_priority_std": _std(episode_priorities),
            "hgr/episode_priority_max": _max(episode_priorities, self.max_priority),
            "hgr/eligible_episode_count": float(episode_ids.size),
            "hgr/pair_table_count": float(table_count),
            "hgr/pair_priority_count": float(pair_count),
            "hgr/pair_priority_memory_mb": float(memory_mb),
            "hgr/goal_priority_entropy_mean": _mean_list(entropies, float("nan")),
            "hgr/goal_selected_rank_mean": _mean_list(ranks, float("nan")),
            "hgr/candidate_td_error_mean": _mean_list(sampled_priorities, float("nan")),
            "hgr/candidate_td_error_max": _max_list(sampled_priorities, float("nan")),
            "hgr/relabel_fraction": float(np.mean(her_relabel)) if b else 0.0,
            "hgr/fallback_fraction": float(fallback_count / max(1, relabel_positions.size)),
            "hgr/episode_probability_mean": _mean(episode_probability[actual_relabel], float("nan")),
            "hgr/pair_probability_mean": _mean(pair_probability[actual_relabel], float("nan")),
            "hgr/is_weight_raw_mean": _mean(raw_relabel_weights, 1.0),
            "hgr/is_weight_raw_max": _max(raw_relabel_weights, 1.0),
            "hgr/is_weight_mean": _mean(normalized_relabel_weights, 1.0),
            "hgr/is_weight_max": _max(normalized_relabel_weights, 1.0),
            "hgr/is_weight_min": _min(normalized_relabel_weights, 1.0),
            "hgr/actor_weight_mean": 1.0,
            "hgr/beta_episode": float(beta_episode),
            "hgr/beta_goal": float(beta_goal),
            "hgr/beta_progress": self.beta_progress(),
            "hgr/max_priority": float(self.max_priority),
            "hgr/priority_clip_enabled": float(self.priority_clip_max > 0.0),
            "hgr/priority_clip_rate": float(self._last_update_clip_rate),
            "hgr/priority_safety_replacement_rate": float(self._last_safety_replacement_rate),
            "hgr/update_priority_mean": float(self._last_update_mean_priority),
            "runtime/hgr_sampling_sec_per_update": float(time.perf_counter() - started),
        }
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
            hgr_episode_id=hgr_episode_id,
            hgr_pair_index=hgr_pair_index,
            hgr_goal_slot=hgr_goal_slot,
            hgr_episode_probability=episode_probability.astype(np.float32),
            hgr_pair_probability=pair_probability.astype(np.float32),
            hgr_is_weight=is_weight,
        )

    def loss_weights(self, batch: SampleBatch) -> tuple[np.ndarray, np.ndarray]:
        batch_size = int(batch.indices.size)
        critic = (
            np.asarray(batch.hgr_is_weight, dtype=np.float32).reshape(-1)
            if batch.hgr_is_weight is not None
            else np.ones(batch_size, dtype=np.float32)
        )
        if critic.size != batch_size or not np.all(np.isfinite(critic)) or np.any(critic <= 0.0):
            raise ValueError("Invalid HGR critic IS weights")
        critic = np.minimum(critic, 1.0).astype(np.float32)
        actor = np.ones(batch_size, dtype=np.float32)
        return critic, actor

    def update_priorities(self, batch: SampleBatch, td_abs_before: np.ndarray) -> Dict[str, float]:
        td_values = np.asarray(td_abs_before, dtype=np.float32).reshape(-1)
        relabel = np.asarray(batch.her_relabel, dtype=np.float32).reshape(-1) > 0.5
        episode_ids = np.asarray(batch.hgr_episode_id, dtype=np.int64).reshape(-1)
        pair_indices = np.asarray(batch.hgr_pair_index, dtype=np.int64).reshape(-1)
        n = min(td_values.size, relabel.size, episode_ids.size, pair_indices.size)
        updated_priorities: list[float] = []
        clipped = 0
        replaced = 0
        for position in range(n):
            if not bool(relabel[position]):
                continue
            table = self._tables.get(int(episode_ids[position]))
            pair_index = int(pair_indices[position])
            if table is None or pair_index < 0 or pair_index >= table.pair_count:
                continue
            td_value = abs(float(td_values[position]))
            if not np.isfinite(td_value):
                td_value = float(self.max_priority)
                replaced += 1
            new_priority = max(self.priority_eps, td_value + self.priority_eps)
            if self.priority_clip_max > 0.0 and new_priority > self.priority_clip_max:
                new_priority = self.priority_clip_max
                clipped += 1
            old_priority = float(table.priorities[pair_index])
            table.priorities[pair_index] = np.float32(new_priority)
            table.priority_sum += float(new_priority - old_priority)
            self.max_priority = max(float(self.max_priority), float(new_priority))
            updated_priorities.append(float(new_priority))

        self.critic_updates += 1
        attempted = max(1, int(np.count_nonzero(relabel[:n])))
        self._last_update_clip_rate = float(clipped / attempted)
        self._last_safety_replacement_rate = float(replaced / attempted)
        if updated_priorities:
            self._last_update_mean_priority = _mean_list(
                updated_priorities, self._last_update_mean_priority
            )
        return self.last_update_metrics()

    def last_update_metrics(self) -> Dict[str, float]:
        beta_episode, beta_goal = self.current_betas()
        return {
            "hgr/beta_episode_next": beta_episode,
            "hgr/beta_goal_next": beta_goal,
            "hgr/beta_progress_next": self.beta_progress(),
            "hgr/max_priority": float(self.max_priority),
            "hgr/priority_clip_enabled": float(self.priority_clip_max > 0.0),
            "hgr/priority_clip_rate": float(self._last_update_clip_rate),
            "hgr/priority_safety_replacement_rate": float(self._last_safety_replacement_rate),
            "hgr/update_priority_mean": float(self._last_update_mean_priority),
        }

    def pair_table_stats(self) -> tuple[int, int, float]:
        pair_count = int(sum(table.pair_count for table in self._tables.values()))
        memory_bytes = int(sum(table.priorities.nbytes for table in self._tables.values()))
        return len(self._tables), pair_count, memory_bytes / (1024.0 * 1024.0)

    def state_dict(self) -> Dict[str, object]:
        beta_episode, beta_goal = self.current_betas()
        return {
            "schema": HGR_STATE_SCHEMA,
            "max_priority": float(self.max_priority),
            "critic_updates": int(self.critic_updates),
            "beta_anneal_updates": int(self.beta_anneal_updates),
            "beta_progress": self.beta_progress(),
            "beta_episode": beta_episode,
            "beta_goal": beta_goal,
            "tables": {
                int(ep): {
                    "slots": table.slots,
                    "priorities": table.priorities,
                    "priority_sum": float(table.priority_sum),
                }
                for ep, table in self._tables.items()
            },
            "last_update_clip_rate": float(self._last_update_clip_rate),
            "last_safety_replacement_rate": float(self._last_safety_replacement_rate),
            "last_update_mean_priority": float(self._last_update_mean_priority),
        }

    def load_state_dict(self, payload: Dict[str, object], replay: HERReplayBuffer) -> None:
        self._rebuild_slot_index(replay)
        if payload.get("schema") != HGR_STATE_SCHEMA:
            raise ValueError(
                "Incompatible HGR checkpoint schema; do not resume a legacy "
                "transition-priority run as the original pair-priority experiment"
            )
        self._tables.clear()
        self.max_priority = max(self.priority_eps, float(payload.get("max_priority", self.priority_init)))
        self.critic_updates = max(0, int(payload.get("critic_updates", 0)))
        self.beta_anneal_updates = max(1, int(payload.get("beta_anneal_updates", self.beta_anneal_updates)))
        for episode_raw, table_payload_raw in payload.get("tables", {}).items():
            ep = int(episode_raw)
            table_payload = dict(table_payload_raw)
            slots = np.asarray(table_payload.get("slots", []), dtype=np.int64).reshape(-1)
            priorities = np.asarray(table_payload.get("priorities", []), dtype=np.float32).reshape(-1)
            expected_pairs = slots.size * (slots.size + 1) // 2
            valid_slots = (
                slots.size > 0
                and np.all(slots >= 0)
                and np.all(slots < int(replay.size))
                and np.all(replay.episode_id[slots] == ep)
            )
            if not valid_slots or priorities.size != expected_pairs:
                continue
            clean = np.nan_to_num(
                priorities,
                nan=self.max_priority,
                posinf=self.max_priority,
                neginf=self.priority_eps,
            )
            clean = np.maximum(clean, self.priority_eps).astype(np.float32)
            if self.priority_clip_max > 0.0:
                clean = np.minimum(clean, self.priority_clip_max).astype(np.float32)
            self._tables[ep] = EpisodePairTable(
                episode_id=ep,
                slots=slots.copy(),
                priorities=clean,
                priority_sum=float(np.sum(clean, dtype=np.float64)),
            )
            self.max_priority = max(self.max_priority, float(np.max(clean)))
        self._last_update_clip_rate = float(payload.get("last_update_clip_rate", 0.0))
        self._last_safety_replacement_rate = float(payload.get("last_safety_replacement_rate", 0.0))
        self._last_update_mean_priority = float(
            payload.get("last_update_mean_priority", self.max_priority)
        )
        self.latest_metrics = self._empty_metrics()

    def rebuild_from_replay(self, replay: HERReplayBuffer) -> None:
        self._rebuild_slot_index(replay)
        self._tables.clear()
        self.max_priority = max(self.priority_eps, self.priority_init)
        for episode_id in sorted(self._episode_slots):
            self._finalize_episode(int(episode_id), replay)
        self.latest_metrics = self._empty_metrics()

    def _rebuild_slot_index(self, replay: HERReplayBuffer) -> None:
        self.slot_episode_id.fill(-1)
        self.slot_episode_step.fill(-1)
        self._episode_slots.clear()
        for slot in range(int(replay.size)):
            ep = int(replay.episode_id[slot])
            if ep < 0:
                continue
            self.slot_episode_id[slot] = ep
            self.slot_episode_step[slot] = int(replay.episode_step[slot])
            self._episode_slots.setdefault(ep, []).append(int(slot))
        for slots in self._episode_slots.values():
            slots.sort(key=lambda value: int(self.slot_episode_step[int(value)]))


def _mean(values: np.ndarray, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.mean(arr)) if arr.size else float(default)


def _std(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.std(arr)) if arr.size else 0.0


def _max(values: np.ndarray, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.max(arr)) if arr.size else float(default)


def _min(values: np.ndarray, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.min(arr)) if arr.size else float(default)


def _mean_list(values: list[float], default: float = 0.0) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else float(default)


def _max_list(values: list[float], default: float = 0.0) -> float:
    return float(np.max(np.asarray(values, dtype=np.float64))) if values else float(default)
