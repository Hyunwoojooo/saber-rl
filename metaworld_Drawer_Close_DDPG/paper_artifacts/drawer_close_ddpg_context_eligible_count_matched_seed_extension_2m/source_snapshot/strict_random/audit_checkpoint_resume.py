#!/usr/bin/env python3
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
from metacog_lite_v0.trainer import (
    capture_global_rng_state,
    load_checkpoint,
    restore_global_rng_state,
    save_checkpoint,
)
from metacog_lite_v0.weights import (
    context_eligible_count_matched_assignment,
)


SEEDS = (1, 2, 3, 4, 5)
UPDATES = 8
SPLIT_UPDATE = 4
BATCH_SIZE = 8
STATE_DIM = 6
ACTION_DIM = 3


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
        "assignment": np.random.default_rng(30_000 + seed),
    }


def make_data(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(50_000 + seed)
    size = 32
    return {
        "state": rng.normal(size=(size, STATE_DIM)).astype(np.float32),
        "action": np.tanh(
            rng.normal(size=(size, ACTION_DIM))
        ).astype(np.float32),
        "reward": rng.normal(size=size).astype(np.float32),
        "next_state": rng.normal(
            size=(size, STATE_DIM)
        ).astype(np.float32),
        "nonterminal": rng.integers(
            0,
            2,
            size=size,
        ).astype(np.float32),
    }


def run_update(
    agent: DDPGAgent,
    streams: dict[str, np.random.Generator],
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, tuple[float, float]]:
    indices = streams["sample"].choice(
        data["state"].shape[0],
        size=BATCH_SIZE,
        replace=False,
    )
    eligible = (indices % 4) != 0
    main = np.logical_and(eligible, (indices % 3) == 0)
    control = context_eligible_count_matched_assignment(
        eligible,
        main,
        streams["assignment"],
    )
    raw = np.ones(BATCH_SIZE, dtype=np.float32)
    raw[control] = 2.0
    omega = (raw / float(np.mean(raw))).astype(np.float32)

    # Advance every non-assignment stream as a real run would. These draws are
    # not used to choose control identities.
    streams["noise"].normal(size=ACTION_DIM)
    streams["weight"].random()
    random.random()
    np.random.random()
    torch.rand(1)

    update = agent.update(
        state=torch.as_tensor(data["state"][indices]),
        action_exec=torch.as_tensor(data["action"][indices]),
        reward=torch.as_tensor(data["reward"][indices]),
        next_state=torch.as_tensor(data["next_state"][indices]),
        nonterminal=torch.as_tensor(data["nonterminal"][indices]),
        omega_critic=torch.as_tensor(omega),
        omega_actor=torch.ones(BATCH_SIZE, dtype=torch.float32),
    )
    return control.copy(), (update.critic_loss, update.actor_loss)


def run_updates(
    agent: DDPGAgent,
    streams: dict[str, np.random.Generator],
    data: dict[str, np.ndarray],
    count: int,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    assignments = []
    losses = []
    for _ in range(count):
        assignment, loss = run_update(agent, streams, data)
        assignments.append(assignment)
        losses.append(loss)
    return assignments, losses


def equal_state(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            equal_state(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(
        right,
        (list, tuple),
    ):
        return len(left) == len(right) and all(
            equal_state(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


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


def assignment_isolation(seed: int) -> bool:
    streams = make_streams(seed)
    before = {
        name: copy.deepcopy(streams[name].bit_generator.state)
        for name in ("sample", "noise", "weight")
    }
    global_before = capture_global_rng_state()
    assignment_before = copy.deepcopy(
        streams["assignment"].bit_generator.state
    )
    eligible = np.ones(BATCH_SIZE, dtype=np.bool_)
    main = np.zeros(BATCH_SIZE, dtype=np.bool_)
    main[:3] = True
    context_eligible_count_matched_assignment(
        eligible,
        main,
        streams["assignment"],
    )
    global_after = capture_global_rng_state()
    return (
        all(
            equal_state(before[name], streams[name].bit_generator.state)
            for name in before
        )
        and equal_state(global_before, global_after)
        and not equal_state(
            assignment_before,
            streams["assignment"].bit_generator.state,
        )
    )


def audit_seed(seed: int, temp_root: Path) -> dict[str, object]:
    data = make_data(seed)

    continuous_agent = make_agent(seed)
    continuous_streams = make_streams(seed)
    continuous_assignments, continuous_losses = run_updates(
        continuous_agent,
        continuous_streams,
        data,
        UPDATES,
    )
    continuous_learning_state = copy.deepcopy(
        continuous_agent.state_dict()
    )
    continuous_stream_states = generator_states(continuous_streams)
    continuous_global_state = capture_global_rng_state()

    split_agent = make_agent(seed)
    split_streams = make_streams(seed)
    first_assignments, first_losses = run_updates(
        split_agent,
        split_streams,
        data,
        SPLIT_UPDATE,
    )
    checkpoint = {
        "agent": split_agent.state_dict(),
        "stream_states": generator_states(split_streams),
        **capture_global_rng_state(),
    }
    checkpoint_path = temp_root / f"seed_{seed}.pt"
    save_checkpoint(checkpoint_path, checkpoint)

    # Simulate a fresh process whose construction has perturbed every state.
    resumed_agent = make_agent(seed + 1_000)
    resumed_streams = make_streams(seed + 1_000)
    loaded = load_checkpoint(checkpoint_path)
    resumed_agent.load_state_dict(loaded["agent"])
    restore_generator_states(
        resumed_streams,
        loaded["stream_states"],
    )
    restore_global_rng_state(
        loaded,
        strict=True,
        device=torch.device("cpu"),
    )
    second_assignments, second_losses = run_updates(
        resumed_agent,
        resumed_streams,
        data,
        UPDATES - SPLIT_UPDATE,
    )

    resumed_assignments = first_assignments + second_assignments
    resumed_losses = first_losses + second_losses
    assignment_reproduced = all(
        np.array_equal(expected, observed)
        for expected, observed in zip(
            continuous_assignments,
            resumed_assignments,
        )
    )
    learning_state_reproduced = equal_state(
        continuous_learning_state,
        resumed_agent.state_dict(),
    )
    losses_reproduced = equal_state(
        continuous_losses,
        resumed_losses,
    )
    streams_reproduced = equal_state(
        continuous_stream_states,
        generator_states(resumed_streams),
    )
    global_rng_reproduced = equal_state(
        continuous_global_state,
        capture_global_rng_state(),
    )
    isolation = assignment_isolation(seed)
    result = {
        "seed": seed,
        "assignment_reproduced": assignment_reproduced,
        "learning_result_reproduced": (
            learning_state_reproduced and losses_reproduced
        ),
        "other_rng_streams_unchanged": isolation,
        "checkpoint_rng_state_restored": (
            streams_reproduced and global_rng_reproduced
        ),
        "model_optimizer_state_exact": learning_state_reproduced,
        "loss_trace_exact": losses_reproduced,
        "dedicated_rng_streams_exact": streams_reproduced,
        "global_rng_streams_exact": global_rng_reproduced,
        "updates": UPDATES,
        "split_update": SPLIT_UPDATE,
    }
    result["passed"] = all(
        bool(result[field])
        for field in (
            "assignment_reproduced",
            "learning_result_reproduced",
            "other_rng_streams_unchanged",
            "checkpoint_rng_state_restored",
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit deterministic strict-control assignment and fake DDPG "
            "learning across checkpoint/resume."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(
        prefix="drawer_strict_resume_audit_"
    ) as temp:
        rows = [
            audit_seed(seed, Path(temp))
            for seed in SEEDS
        ]
    payload = {
        "schema_version": 1,
        "audit": "deterministic_fake_ddpg_split_resume",
        "seeds": rows,
        "seed_coverage": list(SEEDS),
        "passed": all(bool(row["passed"]) for row in rows),
        "assignment_seed_convention": "30000 + seed",
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
