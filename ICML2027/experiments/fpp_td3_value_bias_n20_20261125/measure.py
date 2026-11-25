#!/usr/bin/env python3
"""Value bias at the policy action on saved FPP TD3 checkpoints (no training). See PROTOCOL_ko.md.

For each checkpoint, at noise-free on-policy states s with goal g:
    bias = Q(s || g, pi(s || g)) - G(s, g),   G = sum_{k<300} gamma^k r_k under the frozen deterministic actor
Two goal banks on the same states: O = the episode's original goal, H = a hindsight goal (achieved goal of a
later step of the same episode). Training bootstraps through the 50-step time limit, so the value the critic
is trained toward is the return of the continuing task; G is rolled out on the unwrapped env for 300 steps
(gamma^300 = 0.0023).

  .venv/bin/python <this dir>/measure.py [--worker 0 --n-workers 2] [--seeds 11 ...] [--stages 400000 ...]

Each checkpoint writes outputs/measure/<arm>/seed_<s>/step_<n>.npz; existing files are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FOU = HERE.parent / "draweropen_ddpg_first_order_usefulness_20261109"
sys.path.insert(0, str(FOU))
from fou import actor_signal as AS, core, protocol as P  # noqa: E402
from run import load  # noqa: E402

# ---- fixed 2026-11-25 before any measurement (PROTOCOL_ko.md)
STAGES = (100_000, 150_000, 200_000, 300_000, 400_000, 500_000, 750_000, 1_000_000)
HORIZON = 300
HINDSIGHT_RNG_SEED = 972_000
OUT = HERE / "outputs" / "measure"


def build_queries(src: core.Source, sim: core.FetchSim):
    """Same states as actor_signal.build_queries (same reset seeds and time draws), plus each episode's
    achieved goals so that a hindsight goal can be drawn."""
    rng = np.random.default_rng(AS.QUERY_RNG_SEED)
    rng_h = np.random.default_rng(HINDSIGHT_RNG_SEED)
    T = src.cfg.max_episode_steps
    obs, goals, hgoals, snaps, eps, ts = [], [], [], [], [], []
    for ep in range(AS.QUERY_EPISODES):
        chosen = sorted(int(t) for t in rng.choice(T, size=AS.QUERIES_PER_EPISODE, replace=False))
        o = sim.reset(AS.QUERY_RESET0 + ep)
        goal = o.desired_goal.copy()
        achieved, picked = [o.achieved_goal.copy()], []
        for t in range(T):
            if t in chosen:
                picked.append((t, o.observation.copy(), sim.snapshot()))
            o, truncated = sim.step(core._policy(src, np.concatenate([o.observation, goal]).astype(np.float32)))
            achieved.append(o.achieved_goal.copy())
            if truncated:
                break
        core.require(len(achieved) == T + 1, "query episode shorter than the horizon")
        for t, ob, snap in picked:
            k = int(rng_h.integers(t + 1, T + 1))           # future strategy: achieved goal of a later next-state
            obs.append(ob); goals.append(goal.copy()); hgoals.append(achieved[k]); snaps.append(snap); eps.append(ep); ts.append(t)
    return (np.asarray(obs, np.float32), np.asarray(goals, np.float32), np.asarray(hgoals, np.float32), snaps,
            np.asarray(eps), np.asarray(ts))


def rollout_rewards(src: core.Source, sim: core.FetchSim, snap, goal: np.ndarray, a_first: np.ndarray) -> np.ndarray:
    """Restore, take a_first, then follow the frozen actor; the package's sparse reward on each next state."""
    sim.restore(snap)
    thr, a, rs = src.success_threshold, np.asarray(a_first, np.float32), np.empty(HORIZON, np.float32)
    for k in range(HORIZON):
        o = sim.step_raw(a)
        rs[k] = 0.0 if np.linalg.norm(o.achieved_goal - goal) <= thr else -1.0
        a = core._policy(src, np.concatenate([o.observation, goal]).astype(np.float32))
    return rs


def measure_one(path: Path, dest: Path, expect: tuple[str, int, int]) -> dict:
    t0 = time.perf_counter()
    src = load("fpp_n20", path)
    core.require((src.arm, src.seed, src.step) == expect, f"checkpoint content does not match {expect}: {path}")
    sim = core.FetchSim(src)
    obs, goals, hgoals, snaps, eps, ts = build_queries(src, sim)
    disc = src.cfg.gamma ** np.arange(HORIZON)
    arrays = dict(episode=eps, t=ts)
    for bank, G in (("O", goals), ("H", hgoals)):
        S = np.concatenate([obs, G], axis=1).astype(np.float32)
        with torch.no_grad():
            St = torch.as_tensor(S)
            A = src.agent.actor(St)
            Q = src.heads(St, A).numpy().astype(np.float64)            # [2, n]
        A = A.numpy().copy()
        R = np.stack([rollout_rewards(src, sim, snaps[i], G[i], A[i]) for i in range(len(S))])
        arrays.update({f"{bank}_Q1": Q[0], f"{bank}_Q2": Q[1], f"{bank}_G300": (R * disc).sum(1),
                       f"{bank}_G50": (R[:, :50] * disc[:50]).sum(1), f"{bank}_success300": (R[:, -1] == 0.0),
                       f"{bank}_goal": G, f"{bank}_a": A})
    src.verify_unchanged()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.rename(dest)
    meta = dict(checkpoint=str(path), arm=src.arm, seed=src.seed, step=src.step, gamma=src.cfg.gamma, horizon=HORIZON,
                last_eval_success=src.last_eval_success, n=int(len(obs)), seconds=time.perf_counter() - t0)
    dest.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--stages", nargs="+", type=int, default=None)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(1)
    todo = [t for t in P.checkpoints("fpp_n20") if t[2] in (a.stages or STAGES) and (a.seeds is None or t[1] in a.seeds)
            and (a.arms is None or t[0] in a.arms)]
    todo.sort(key=lambda t: (t[2] not in (150_000, 400_000), t[2], t[1], t[0]))     # the two reference stages first
    for arm, s, st, path in todo[a.worker::a.n_workers]:
        dest = OUT / arm / f"seed_{s}" / f"step_{st:07d}.npz"
        if dest.exists():
            continue
        m = measure_one(path, dest, (arm, s, st))
        print(f"done {arm} seed {s} step {st}: {m['seconds']:.0f}s (eval {m['last_eval_success']:.2f})", flush=True)


if __name__ == "__main__":
    main()
