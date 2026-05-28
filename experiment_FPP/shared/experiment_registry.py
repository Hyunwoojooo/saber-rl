from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentSpec:
    folder: str
    run_slug: str
    trainer_method: str
    weight_scope: str
    local_search_max: int
    support_lookup: bool
    reference: str | None
    purpose: str
    total_steps: int = 1_000_000
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    reward_type: str = "sparse"
    her_ratio: float = 0.8
    batch_size: int = 256
    updates_per_step: int = 1
    eval_freq: int = 10_000
    eval_episodes: int = 20
    M_B: int = 16
    M_Q_min: int = 4
    N_local_min: int = 4
    N_gap_min: int = 128
    lambda_support: float = 1.0
    lambda_TD: float = 1.0
    U_TD_max: float = 1.0
    per_alpha: float = 0.6
    eps_PER: float = 1e-6
    U_PER_max: float = 2.0


EXPERIMENTS: dict[str, ExperimentSpec] = {
    'B0': ExperimentSpec(
        folder='B0',
        run_slug='b0',
        trainer_method='b0_ddpg_her',
        weight_scope='none',
        local_search_max=1024,
        support_lookup=False,
        lambda_support=0.0,
        reference=None,
        purpose='Clean DDPG+HER baseline with Omega=1.',
    ),
    'B3-original': ExperimentSpec(
        folder='B3-original',
        run_slug='b3_original',
        trainer_method='b3_support_only',
        weight_scope='actor_critic',
        local_search_max=1024,
        support_lookup=True,
        reference='B0',
        purpose='Required actor-critic B3 support-only reference.',
    ),
    'B3-critic-only': ExperimentSpec(
        folder='B3-critic-only',
        run_slug='b3_critic_only',
        trainer_method='b3_support_only',
        weight_scope='critic_only',
        local_search_max=1024,
        support_lookup=True,
        reference='B0,B3-original',
        purpose='SABER main candidate: support reliability weighting on critic loss only.',
    ),
    'B3-critic-only-random': ExperimentSpec(
        folder='B3-critic-only-random',
        run_slug='b3_critic_only_random',
        trainer_method='b3_support_random',
        weight_scope='critic_only',
        local_search_max=1024,
        support_lookup=True,
        reference='B3-critic-only',
        purpose='Positive-count control: random K high-weight critic samples with B3 diagnostics retained.',
    ),
    'B3-critic-only-shuffled': ExperimentSpec(
        folder='B3-critic-only-shuffled',
        run_slug='b3_critic_only_shuffled',
        trainer_method='b3_support_shuffled',
        weight_scope='critic_only',
        local_search_max=1024,
        support_lookup=True,
        reference='B3-critic-only',
        purpose='Weight-distribution control: shuffle B3 raw weights within batch for critic loss.',
    ),
    'B3-critic-only-runtime-optimization': ExperimentSpec(
        folder='B3-critic-only-runtime-optimization',
        run_slug='b3_critic_only_runtime_opt',
        trainer_method='b3_support_runtime_opt',
        weight_scope='critic_only',
        local_search_max=1024,
        support_lookup=True,
        reference='B3-critic-only',
        purpose='Exact-semantics runtime-optimization path for critic-only B3 support weighting.',
    ),
    'B3-critic-only-fast-512': ExperimentSpec(
        folder='B3-critic-only-fast-512',
        run_slug='b3_critic_only_fast_512',
        trainer_method='b3_support_only',
        weight_scope='critic_only',
        local_search_max=512,
        support_lookup=True,
        reference='B3-critic-only',
        purpose='FetchPickAndPlace local_search_max sensitivity run at 512.',
    ),
    'B3-critic-only-fast-2048': ExperimentSpec(
        folder='B3-critic-only-fast-2048',
        run_slug='b3_critic_only_fast_2048',
        trainer_method='b3_support_only',
        weight_scope='critic_only',
        local_search_max=2048,
        support_lookup=True,
        reference='B3-critic-only',
        purpose='FetchPickAndPlace local_search_max sensitivity run at 2048.',
    ),
    'TD-error': ExperimentSpec(
        folder='TD-error',
        run_slug='td_error',
        trainer_method='td_error',
        weight_scope='critic_only',
        local_search_max=1024,
        support_lookup=False,
        reference='B0,B3-critic-only',
        purpose='Critic-only TD-error loss-weighting baseline.',
        lambda_TD=1.0,
        U_TD_max=1.0,
    ),
    'PER-style': ExperimentSpec(
        folder='PER-style',
        run_slug='per_style',
        trainer_method='per_style_loss_weighting',
        weight_scope='critic_only',
        local_search_max=1024,
        support_lookup=False,
        reference='B0,B3-critic-only',
        purpose='PER-style TD-error priority-derived critic loss weighting with uniform replay sampling.',
        per_alpha=0.6,
        eps_PER=1e-06,
        U_PER_max=2.0,
    ),
}


def repo_root_from_shared() -> Path:
    return Path(__file__).resolve().parents[3]


def experiment_dir(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / name


def load_folder_config(name: str) -> dict[str, Any]:
    path = experiment_dir(name) / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment config: {path}")
    import json

    return json.loads(path.read_text())


def run_name(spec: ExperimentSpec, seed: int, steps: int | None = None) -> str:
    n_steps = int(spec.total_steps if steps is None else steps)
    return f"fetch_pick_and_place_{spec.run_slug}_s{int(seed)}_{n_steps}_ls{int(spec.local_search_max)}"
