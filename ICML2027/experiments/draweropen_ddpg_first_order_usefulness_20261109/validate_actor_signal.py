#!/usr/bin/env python3
"""Checks that must pass before the actor-signal measurement (PROTOCOL_actor_signal_ko.md).

  .venv/bin/python <this dir>/validate_actor_signal.py [--seed 11 --step 500000 --arm saber]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fou import actor_signal as AS, core, protocol as P  # noqa: E402
from run import load  # noqa: E402

N_PASS = 0


def check(cond: bool, msg: str) -> None:
    global N_PASS
    core.require(bool(cond), "FAILED: " + msg)
    N_PASS += 1
    print("ok  ", msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="fpp_n20")
    ap.add_argument("--arm", default="saber")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--step", type=int, default=500_000)
    a = ap.parse_args()
    torch.set_num_threads(1)
    (arm, s, st, path), = [t for t in P.checkpoints(a.set) if t[:3] == (a.arm, a.seed, a.step)]
    src = load(a.set, path)
    sim = core.FetchSim(src)
    q = AS.build_queries(src, sim)
    n = len(q.a_pi)
    check(n == AS.QUERY_EPISODES * AS.QUERIES_PER_EPISODE, f"{n} query states")
    check(np.array_equal(q.a_pi, np.stack([core._policy(src, x) for x in q.S.numpy()])), "a_pi is the actor's action")

    # G is deterministic and its first rewards equal the original episode's rewards (wrapped env, from reset)
    G1 = [AS.rollout_return(src, sim, q.snaps[i], q.goals[i], q.a_pi[i]) for i in range(8)]
    G2 = [AS.rollout_return(src, sim, q.snaps[i], q.goals[i], q.a_pi[i]) for i in range(8)]
    check(G1 == G2, "G(s, a) is deterministic (restore + rollout twice)")
    for i in range(0, n, 9):
        ep, t = int(q.episode[i]), int(q.t[i])
        o = sim.reset(AS.QUERY_RESET0 + ep)
        goal = o.desired_goal.copy()
        for _ in range(t):
            o, _ = sim.step(core._policy(src, np.concatenate([o.observation, goal]).astype(np.float32)))
        check(np.array_equal(np.concatenate([o.observation, goal]).astype(np.float32), q.S[i].numpy()),
              f"query {i} (episode {ep}, t={t}) state reproduced from reset")
        ref = []
        for _ in range(src.cfg.max_episode_steps - t):
            o, _ = sim.step(core._policy(src, np.concatenate([o.observation, goal]).astype(np.float32)))
            ref.append(0.0 if np.linalg.norm(o.achieved_goal - goal) <= src.success_threshold else -1.0)
        _, rs = AS.rollout_return(src, sim, q.snaps[i], q.goals[i], q.a_pi[i], return_rewards=True)
        check(rs[:len(ref)] == ref, f"query {i}: rollout rewards match the episode's remaining {len(ref)} rewards")
        rs_g = sum(src.cfg.gamma ** k * r for k, r in enumerate(rs))
        check(abs(rs_g - AS.rollout_return(src, sim, q.snaps[i], q.goals[i], q.a_pi[i])) < 1e-12,
              f"query {i}: G equals the discounted reward sum")

    Z = AS.directions(n)
    check(np.allclose(np.linalg.norm(Z, axis=-1), 1, atol=1e-6), "directions are unit vectors")
    Ap, Am = AS.perturbed_actions(q, Z)
    check(Ap.max() <= 1 and Am.min() >= -1, "perturbed actions inside [-1, 1]")
    free = np.all(np.abs(q.a_pi) < 1 - AS.SIGMAS[0], axis=1)
    if free.any():
        i = int(np.flatnonzero(free)[0])
        check(np.allclose(Ap[i, 0] - q.a_pi[i], AS.SIGMAS[0] * Z[i], atol=1e-6), "unclipped perturbation is a + sigma z")

    # critic signal: head 0, grad matches central finite differences
    sig = AS.critic_signal(src.heads, q, Ap, Am)
    with torch.no_grad():
        direct = (src.agent.critic1(q.S[:4], torch.as_tensor(Ap[:4, 0, 0])).reshape(-1)
                  - src.agent.critic1(q.S[:4], torch.as_tensor(Am[:4, 0, 0])).reshape(-1)).numpy()
    check(np.allclose(sig["dq"][:4, 0, 0], direct, atol=1e-5), "dq uses head 0 (critic1, read by the actor)")
    h = 1e-2
    fd = np.zeros((n, 4))
    with torch.no_grad():
        for d in range(4):
            e = np.zeros(4, np.float32); e[d] = h
            fd[:, d] = ((src.agent.critic1(q.S, torch.as_tensor(q.a_pi + e)) - src.agent.critic1(q.S, torch.as_tensor(q.a_pi - e)))
                        .reshape(-1).numpy().astype(np.float64) / (2 * h))
    rel = np.linalg.norm(fd - sig["grad"], axis=1) / np.maximum(np.linalg.norm(sig["grad"], axis=1), 1e-6)
    check(np.median(rel) < 0.05, f"grad_a Q matches finite differences (median rel. error {np.median(rel):.3g})")

    # branch critics equal core.run_branches at K = 100 on a small R1 bank
    batches = [core.make_batch(src, P.KSTEP_SEED0 + k, check_production=k < 3) for k in range(AS.K_BRANCH)]
    params = core.Params(src.heads)
    bank = core.bank_r1(src, 4)
    ref = core.run_branches(src, params, batches, [bank], (AS.K_BRANCH,), P.BRANCH_RNG_SEED)[0]
    crit = AS.branch_critics(src, batches, P.BRANCH_RNG_SEED)

    def R(c):
        v = torch.cat([p.detach().reshape(-1) for p in c.parameters()])
        with torch.no_grad():
            return float(core.risk(params, v, bank))
    ru = R(crit["uniform"])
    check(ru == ref["R1_R_uniform"], "uniform branch critic equals run_branches exactly")
    for arm in ("saber", "inverse", "td_topk"):
        check(ru - R(crit[arm]) == ref[f"R1_K_{arm}"], f"{arm} branch critic equals run_branches exactly")
    rr = float(np.mean([ru - R(crit[f"random{j}"]) for j in range(core.N_RANDOM_SEQUENCES)]))
    check(rr == ref["R1_K_random"], "random branch critics equal run_branches exactly")
    check(not torch.equal(next(crit["saber"].parameters()), next(src.heads.parameters())), "branches moved the critic")
    src.verify_unchanged()
    check(True, "checkpoint state unchanged (hash)")
    print(f"all {N_PASS} checks passed")


if __name__ == "__main__":
    main()
