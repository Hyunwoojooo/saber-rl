#!/usr/bin/env python3
"""Checks that must pass before any measurement is reported. Plain asserts (no pytest in .venv).

  .venv/bin/python <this dir>/validate.py --out outputs/validation_<date> [--set a1|late|fpp]
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fou import core, protocol as P  # noqa: E402
import run as runner  # noqa: E402

# (tag, checkpoint set, trained arm, seed, stage, path)
FIXTURES = {
    "a1": (("saber101@100k", "a1", "saber", 101, 100_000, P.checkpoint_path("saber", 101, 100_000)),
           ("her101@300k", "a1", "her", 101, 300_000, P.checkpoint_path("her", 101, 300_000))),
    # late phase: a successful and a failed (saturated-actor) SABER run, a failed HER run whose
    # replay ring pointer is mid-buffer, and one run trained with another assignment rule
    "late": (("ext-saber11@2M(success)", "ext2m", "saber", 11, 2_000_000, P.ext_path("saber", 11)),
             ("ext-saber13@2M(failed)", "ext2m", "saber", 13, 2_000_000, P.ext_path("saber", 13)),
             ("ext-td12@2M", "ext2m", "td_topk", 12, 2_000_000, P.ext_path("td_topk", 12)),
             ("mature-saber1@1.5M", "mature", "saber", 1, 1_500_000, P.mature_path("saber", 1_500_000)),
             ("mature-her1@2M(failed)", "mature", "her", 1, 2_000_000, P.mature_path("her", 2_000_000))),
    # FetchPickAndPlace TD3: early, middle and final checkpoints of both arms
    "fpp": (("fpp-saber6@500k", "fpp_incl", "saber", 6, 500_000, P.fpp_path("fpp_incl", "saber", 6, 500_000)),
            ("fpp-her7@1M", "fpp_incl", "her", 7, 1_000_000, P.fpp_path("fpp_incl", "her", 7, 1_000_000)),
            ("fpp-saber9@50k", "fpp_incl", "saber", 9, 50_000, P.fpp_path("fpp_incl", "saber", 9, 50_000))),
}
results: dict[str, str] = {}


def check(name: str, cond: bool, detail: str) -> None:
    results[name] = ("PASS " if cond else "FAIL ") + detail
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}", flush=True)
    if not cond:
        raise SystemExit(f"validation failed: {name}")


def t_counterexample() -> None:
    # 11/5 analysis §8.1: context A supported & already exact, context B unsupported but its
    # replay-action update transfers fully to the policy query. SABER must come out worse.
    u = np.array([0.0, 1.0])                       # u_A = 0 (delta_A = 0), u_B = <(0, .5), (0, 2)> = 1
    m = np.array([True, False])
    g1 = float(((core.weights(m) - 1.0) * u).mean())
    p = m.mean()
    closed = p * (1 - p) / (1 + p) * (u[m].mean() - u[~m].mean())
    check("counterexample_sign", g1 < 0 and abs(g1 - closed) < 1e-6 and abs(g1 + 1 / 6) < 1e-6,
          f"G1={g1:.6f} (expected -1/6): the measurement can report SABER worse than uniform")


def weighted_loss(Pm: core.Params, v: torch.Tensor, S, A, y, w) -> torch.Tensor:
    """The production critic loss: sum over heads of mean(w * residual^2)."""
    return (w * (Pm.q_all(v, S, A) - y).pow(2)).mean(1).sum()


def critic_params(agent) -> torch.Tensor:
    mods = [agent.critic] if hasattr(agent, "critic") else [agent.critic1, agent.critic2]
    return torch.cat([p.detach().reshape(-1) for m in mods for p in m.parameters()])


def t_fixture(tag: str, set_name: str, arm: str, seed: int, stage: int, path: Path) -> None:
    src = runner.load(set_name, path)
    check(f"{tag}:labels", (src.arm, src.seed, src.step) == (arm, seed, stage),
          f"checkpoint says arm={src.arm} seed={src.seed} step={src.step} kind={src.kind} family={src.family}; "
          f"evaluation success at this step {src.last_eval_success:.2f}, {src.total_updates:,} updates")
    Pm = core.Params(src.heads)
    v0 = Pm.flat0
    bank = core.bank_r1(src, 3 if src.family == "drawer_ddpg" else 12)
    gR = core.risk_grad(Pm, v0, bank)

    # parity with production support/eligibility/omega on 8 batches (make_batch raises otherwise)
    batches = [core.make_batch(src, P.BATCH_SEED0 + b, check_production=True) for b in range(8)]
    check(f"{tag}:production_parity", True, "psi, eligibility and omega equal LiteWeightComputer on 8 batches "
          f"(p = {[round(float(b.support.mean()), 3) for b in batches]})")
    # sample age must order transitions the same way their episode ids do (ring-buffer arithmetic)
    for b in batches:
        order = np.argsort(b.age, kind="stable")
        ep = src.replay.episode_id[b.slot][order]
        core.require(b.age.min() >= 0 and b.age.max() < src.replay.size and np.all(np.diff(ep) <= 0),
                     "sample age disagrees with episode order")
    check(f"{tag}:sample_age", True, f"age is consistent with episode order on 8 batches "
          f"(replay ptr {src.replay.ptr:,}, size {src.replay.size:,})")
    live = [b for b in batches if 0 < b.support.sum() < len(b.support)]
    bt = live[0] if live else batches[0]
    m, e = bt.support, bt.eligible
    w = torch.as_tensor(core.weights(m))

    # one real production update (the training package's agent.update) vs our critic step
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12345)
        z = torch.randn_like(bt.A)                          # TD3Agent.update draws exactly this first
    noise = None
    if src.family == "fpp_td3":
        noise = torch.clamp(z * src.agent.target_policy_noise, -src.agent.target_noise_clip, src.agent.target_noise_clip)
    with torch.no_grad():
        y_prod = bt.reward + src.cfg.gamma * bt.nonterm * core.target_q(src, bt.S2, noise)
    ours = core.torch_step(src, w, dataclasses.replace(bt, y=y_prod), return_params=True)
    ag = copy.deepcopy(src.agent)
    for mod in vars(ag).values():
        if isinstance(mod, torch.nn.Module):
            mod.requires_grad_(True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12345)
        ag.update(bt.S, bt.A, bt.reward, bt.S2, bt.nonterm, w, torch.ones(len(w)))
    theirs = critic_params(ag)
    step = float((theirs - v0).norm())
    err = float((ours - theirs).norm()) / step
    check(f"{tag}:production_update", err <= 1e-6,
          f"our critic step vs agent.update of the training code: difference {err:.1e} of the update norm "
          f"({'bit-identical' if torch.equal(ours, theirs) else 'not bit-identical'})")

    # per-sample Jacobian vs autograd; u via Jacobian vs jvp
    Jr = Pm.per_sample_jacobian(v0, bt.S, bt.A)
    errs = []
    for i in (0, 1, 2):
        vv = v0.clone().requires_grad_(True)
        gi = torch.autograd.grad(Pm.q(vv, bt.S[i:i + 1], bt.A[i:i + 1]).sum(), vv)[0]
        errs.append(float((gi - Jr[i]).abs().max() / max(1e-12, float(gi.abs().max()))))
    _, jv = torch.func.jvp(lambda pv: Pm.q(pv, bt.S, bt.A), (v0,), (gR,))
    rel = float((jv - Jr @ gR).abs().max() / max(1e-30, float((Jr @ gR).abs().max())))
    check(f"{tag}:jacobian", max(errs) < 1e-5 and rel < 1e-3, f"max rel err J {max(errs):.1e}, jvp {rel:.1e}")

    # R depends on head 0 only, so the other heads' loss gradients are orthogonal to grad R
    with torch.no_grad():
        delta = Pm.q(v0, bt.S, bt.A) - bt.y
    u = (2 * delta * (Jr @ gR)).numpy()
    p = m.mean()
    g1 = float(((core.weights(m) - 1) * u).mean())
    if not live:
        check(f"{tag}:identities", True, "all 8 batches have p = 0 or 1, so every arm equals uniform; identity checks skipped")
    closed = p * (1 - p) / (1 + p) * (u[m].mean() - u[~m].mean()) if live else 0.0
    check(f"{tag}:identity_uniform", abs(g1 - closed) <= 1e-5 * max(1.0, abs(g1)), f"G1 {g1:.6e} vs closed form {closed:.6e}")

    rng = np.random.default_rng(0)
    draws = [float(((core.weights(core.arm_mask("random", bt, None, rng)) - 1) * u).mean()) for _ in range(4000)]
    k = m.sum()
    pred_gap = p / (1 + p) * (u[m].mean() - u[e].mean()) if k < e.sum() else 0.0
    mc_gap = g1 - float(np.mean(draws))
    se = float(np.std(draws) / np.sqrt(len(draws)))
    check(f"{tag}:identity_random", abs(mc_gap - pred_gap) <= 4 * se + 1e-9,
          f"SABER-Random MC {mc_gap:.4e} vs closed form {pred_gap:.4e} (4 se = {4 * se:.1e})")

    # first-order value vs float64 finite difference of the full (all-heads) weighted loss
    v64 = v0.double()
    S64, A64, y64 = bt.S.double(), bt.A.double(), bt.y.double()
    bS, bA, by = bank.S.double(), bank.A.double(), bank.y.double()

    def loss_grad64(wts):
        vv = v64.clone().requires_grad_(True)
        return torch.autograd.grad(weighted_loss(Pm, vv, S64, A64, y64, torch.as_tensor(wts, dtype=torch.float64)), vv)[0]

    def sgd_risk(wts, eta):
        return float((Pm.q(v64 - eta * loss_grad64(wts), bS, bA) - by).pow(2).mean())

    vv = v64.clone().requires_grad_(True)
    gR64 = torch.autograd.grad((Pm.q(vv, bS, bA) - by).pow(2).mean(), vv)[0]
    g1_64 = float(gR64 @ (loss_grad64(core.weights(m)) - loss_grad64(np.ones(len(m)))))

    # The O(eta) curvature term is not negligible at late-phase checkpoints even for eta = 1e-8
    # (large residuals -> large gradients), so remove it by Richardson extrapolation.
    def fd_at(eta):
        return (sgd_risk(np.ones(len(m)), eta) - sgd_risk(core.weights(m), eta)) / eta
    fd_a, fd_b = fd_at(1e-9), fd_at(1e-10)
    fd = (10.0 * fd_b - fd_a) / 9.0
    check(f"{tag}:finite_difference", abs(fd - g1_64) <= 0.02 * abs(g1_64) + 1e-300,
          f"float64 first-order value {g1_64:.4e} vs float64 finite difference {fd:.4e} (eta 1e-9: {fd_a:.4e}, 1e-10: {fd_b:.4e})")
    rms = float(delta.pow(2).mean().sqrt())
    with torch.no_grad():
        bank_rms = float(core.risk(Pm, v0, bank).sqrt())
    if bank_rms < P.FLOOR_RMS:
        check(f"{tag}:float32_pipeline", True,
              f"policy-query RMS residual {bank_rms:.1e} is below the float32 floor {P.FLOOR_RMS:.0e} (batch RMS {rms:.1e}): "
              f"float32 G1 {g1:.4e} vs float64 {g1_64:.4e}; such checkpoints are reported as numerically null, not signed")
    else:
        check(f"{tag}:float32_pipeline", abs(g1 - g1_64) <= 0.02 * abs(g1_64) + 1e-4 * float(np.abs(u).mean()),
              f"float32 G1 {g1:.4e} vs float64 {g1_64:.4e} (policy-query RMS residual {bank_rms:.1e}, mean |u| {float(np.abs(u).mean()):.2e})")

    # analytic Adam + clip step vs the real torch step from the saved optimizer state
    vv = v0.clone().requires_grad_(True)
    g = torch.autograd.grad(weighted_loss(Pm, vv, bt.S, bt.A, bt.y, w), vv)[0].detach()
    st = src.agent.critic_opt.state_dict()
    d32 = core.adam_delta(st, src.cfg.critic_lr, core.clip(g), Pm.sizes, dtype=torch.float32)
    d64 = core.adam_delta(st, src.cfg.critic_lr, core.clip(g), Pm.sizes).float()
    p_torch = core.torch_step(src, w, bt, return_params=True)
    # Compare parameters, not p_new - p_old: that float32 subtraction loses ~1e-4 relative
    # precision because parameters are ~1e3 times larger than one update.
    scale = float((p_torch - v0).norm())
    rel32 = float(((v0 + d32) - p_torch).norm()) / scale
    with torch.no_grad():
        dr = abs(float(core.risk(Pm, v0 + d64, bank) - core.risk(Pm, p_torch, bank)))
        dr_common = abs(float(core.risk(Pm, p_torch, bank) - core.risk(Pm, v0, bank)))
    check(f"{tag}:adam_step", rel32 < 1e-4 and dr <= 1e-3 * dr_common,
          f"float32 emulation of Adam+clip lands on torch's parameters (err {rel32:.1e} of the update norm); "
          f"float64 step |R diff| {dr:.1e} vs |R change| {dr_common:.1e}")

    if src.family == "fpp_td3":
        t_fetch(tag, src)
    src.verify_unchanged()
    check(f"{tag}:source_unchanged", True, "actor/targets/critic(s)/Adam/replay hash unchanged")


def t_fetch(tag: str, src: core.Source) -> None:
    """FetchPickAndPlace-only checks: reward formula, HER convention, simulator state restore."""
    b = src.replay.sample(4096, src.cfg.her_ratio, np.random.default_rng(5))
    ours = np.where(np.linalg.norm(b.next_achieved_goal - b.desired_goal_train, axis=1) <= src.success_threshold, 0.0, -1.0)
    theirs = src.env.compute_reward(b.next_achieved_goal, b.desired_goal_train, {})
    check(f"{tag}:reward", np.array_equal(ours, theirs),
          f"relabelled sparse reward equals env.compute_reward on 4,096 samples (zero-reward share {float((ours == 0).mean()):.3f}, "
          f"relabelled {b.her_relabel_ratio:.3f}); HER convention of the code: {'inclusive' if src.her_inclusive else 'exclusive'} future")
    sim, rng, ok, n = core.FetchSim(src), np.random.default_rng(7), True, 0
    for ep in range(3):
        seed = 71_000 + ep
        o = sim.reset(seed)
        goal0, acts, snaps, obs = o.desired_goal.copy(), [], {}, []
        for t in range(src.cfg.max_episode_steps):
            obs.append(o.observation.copy())
            if t in (0, 7, 23, 49):
                snaps[t] = sim.snapshot()
            a = np.clip(core._policy(src, np.concatenate([o.observation, goal0]).astype(np.float32))
                        + rng.normal(0, 0.2, 4), -1, 1).astype(np.float32)
            acts.append(a)
            o, _ = sim.step(a)
        for t, snap in snaps.items():
            q = rng.uniform(-1, 1, 4).astype(np.float32)
            sim.restore(snap)
            same = np.array_equal(sim.observe().observation, obs[t])
            o_a = sim.step_raw(q)
            o = sim.reset(seed)
            for k in range(t):
                o, _ = sim.step(acts[k])
            o_b, _ = sim.step(q)
            ok &= same and np.array_equal(o_a.observation, o_b.observation) and np.array_equal(o_a.achieved_goal, o_b.achieved_goal)
            n += 1
    check(f"{tag}:state_restore", bool(ok), f"restore + step equals replay-from-reset + step bit-for-bit on {n} queries")


def t_restore_drawer(path: Path) -> None:
    src = core.load_source(path)
    sim = core.DrawerSimulator(src.cfg, 12345)
    sim.reset(12345)
    for _ in range(37):
        sim.step(core._policy(src, sim.state()))
    snap = sim.snapshot()
    a = np.array([0.3, -0.7, 0.1, 0.9], np.float32)
    traj1 = []
    for _ in range(5):
        sim.step(a); traj1.append(sim.state().copy())
    sim.restore(snap)
    traj2 = []
    for _ in range(5):
        sim.step(a); traj2.append(sim.state().copy())
    sim.close()
    check("restore_determinism", all(np.array_equal(x, y) for x, y in zip(traj1, traj2)),
          "restore + same 5 actions reproduces the trajectory bit-for-bit")


def t_end_to_end(out: Path, fixtures) -> None:
    # tiny settings: exercises the whole pipeline without producing reportable numbers
    P.R1_EPISODES, P.R2_EPISODES, P.R2_QUERIES, P.K_RECORD = 2, 2, 32, (1, 3)
    P.FPP_R1_EPISODES, P.FPP_R2_EPISODES = 8, 2
    for tag, set_name, arm, seed, stage, path in fixtures:
        dest = out / "end_to_end" / tag.replace("@", "_at_").replace("(", "_").replace(")", "")
        meta = runner.measure_one(path, dest, 2, True, True, expect=(arm, seed, stage), set_name=set_name)
        files = sorted(p.name for p in dest.iterdir())
        ok = files == ["batches.csv", "ksteps.csv", "meta.json", "per_sample.npz"] and meta["banks"]["R2"]["n"] == 32
        check(f"{tag}:end_to_end", ok, f"measure_one wrote {len(files)} files in {meta['seconds']['total']:.1f}s; "
              f"query phases R1 {meta['banks']['R1']['phase_frac']}, R2 {meta['banks']['R2']['phase_frac']}; source unchanged")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="a1", choices=sorted(FIXTURES))
    a = ap.parse_args()
    fixtures = FIXTURES[a.set]
    out = Path(a.out)
    if not out.is_absolute():
        out = HERE / out
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    t_counterexample()
    for fx in fixtures:
        t_fixture(*fx)
    if a.set != "fpp":
        t_restore_drawer(fixtures[0][5])
    t_end_to_end(out, fixtures[:1] if a.set == "a1" else fixtures)
    (out / "validation.json").write_text(json.dumps(dict(results=results, seconds=time.perf_counter() - t0), indent=2) + "\n")
    print(f"all {len(results)} checks passed in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
