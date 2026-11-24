#!/usr/bin/env python3
"""Pre-registered cumulative-effect analysis (PROTOCOL_ko.md, fixed 2026-11-18 before computing any
SABER-HER difference of the candidate indicators).

  .venv/bin/python <this dir>/analyze.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = HERE.parent / "fpp_td3_primary_checkpoint_rerun_20261110" / "outputs" / "runs_n20"
LOG_DIRS = {"her": RUNS / "td3_her_baseline_ls512", "saber": RUNS / "td3_saber_critic_only_v4_ls512"}
FOU = HERE.parent / "draweropen_ddpg_first_order_usefulness_20261109" / "outputs"
PERF = REPO / "ICML2027" / "paper_artifacts" / "fpp_td3_primary_rerun_20261110" / "n20_seeds11_30" / "per_seed.csv"
OUT = REPO / "ICML2027" / "paper_artifacts" / "fpp_td3_cumulative_effect_n20_20261118"
SEEDS = tuple(range(11, 31))
STAGES = tuple(range(100_000, 1_000_001, 50_000))
EARLY = tuple(range(100_000, 400_001, 50_000))
MEAS_EARLY = (100_000, 150_000, 200_000, 300_000, 400_000)
MEAS_STAGES = (100_000, 150_000, 200_000, 300_000, 400_000, 500_000, 750_000, 1_000_000)
WINDOW = 50_000
ALPHA = 0.05
N_PERM, PERM_SEED = 10_000, 20261118
LOG = {"L1": ["train/episode_success"], "L2": ["critic/q_policy_mean"], "L3": ["loss/critic"],
       "L4": ["loss/td_error_abs_mean"], "L5": ["lite/phase_lift_count_rate", "lite/phase_place_count_rate"]}
NAMES = {"L1": "train episode success", "L2": "Q(s, pi(s)) mean", "L3": "critic loss", "L4": "|TD error| mean",
         "L5": "batch lift+place rate", "D1": "replay lift+place fraction", "D2": "distinct contexts / 1k",
         "D3": "actor saturation", "D4": "median |grad_a Q1|", "D5": "object moved >2cm",
         "M1": "PA (K0, sigma 0.3)", "M2": "GC (K0)", "M3": "R1 sqrt(R0)"}
ORDER = list(NAMES)


def sign_p(pos, neg):
    n = pos + neg
    if n == 0:
        return float("nan")
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(pos, neg) + 1)) / 2 ** n)


def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: (math.isnan(ps[i]), ps[i]))
    out, running = [float("nan")] * len(ps), 0.0
    m = sum(not math.isnan(p) for p in ps)
    for rank, i in enumerate(idx):
        if math.isnan(ps[i]):
            continue
        running = max(running, min(1.0, (m - rank) * ps[i]))
        out[i] = running
    return out


def ranks(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman_perm(x, y, rng):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = ranks(x[ok]), ranks(y[ok])
    r = float(np.corrcoef(x, y)[0, 1])
    null = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(N_PERM)])
    return r, float((np.sum(np.abs(null) >= abs(r) - 1e-12) + 1) / (N_PERM + 1)), int(ok.sum())


def values() -> dict:
    """v[(indicator, arm, seed, stage)] -> float"""
    v = {}
    for arm, d in LOG_DIRS.items():
        for s in SEEDS:
            (f,) = glob.glob(str(d / f"*_s{s}_1000000_ls512" / "metrics.csv"))
            rows = list(csv.DictReader(open(f)))
            steps = np.array([int(r["step"]) for r in rows])
            for key, cols in LOG.items():
                x = np.array([sum(float(r[c]) for c in cols) if all(r[c] not in ("", "nan") for c in cols) else np.nan
                              for r in rows])
                for st in STAGES:
                    sel = (steps > st - WINDOW) & (steps <= st)
                    v[(key, arm, s, st)] = float(np.nanmean(x[sel]))
    for f in (HERE / "outputs" / "replay_stats").glob("*.json"):
        r = json.loads(f.read_text())
        for key in ("D1", "D2", "D3", "D4", "D5"):
            v[(key, r["arm"], r["seed"], r["stage"])] = r[key]
    for arm in ("saber", "her"):
        for s in SEEDS:
            for st in MEAS_STAGES:
                d = FOU / "actor_signal_n20" / arm / f"seed_{s}" / f"step_{st:07d}" / "arms.csv"
                k0 = next(r for r in csv.DictReader(open(d)) if r["critic"] == "K0")
                v[("M1", arm, s, st)] = float(k0["PA_s0.3"]) if float(k0["n_inf_s0.3"]) >= 32 else float("nan")
                v[("M2", arm, s, st)] = float(k0["GC"]) if float(k0["n_gstar"]) >= 8 else float("nan")
                m = json.loads((FOU / "fpp_n20" / arm / f"seed_{s}" / f"step_{st:07d}" / "meta.json").read_text())
                v[("M3", arm, s, st)] = math.sqrt(m["banks"]["R1"]["R0"])
    return v


def main() -> None:
    v = values()
    n_replay = len(list((HERE / "outputs" / "replay_stats").glob("*.json")))
    if n_replay != 2 * len(SEEDS) * len(STAGES):
        raise SystemExit(f"replay stats incomplete: {n_replay}")
    perf = {int(r["seed"]): r for r in csv.DictReader(open(PERF))}
    fq_gain = [float(perf[s]["saber_fq"]) - float(perf[s]["her_fq"]) for s in SEEDS]
    rng = np.random.default_rng(PERM_SEED)
    rows = []
    for key in ORDER:
        stages = MEAS_EARLY if key.startswith("M") else EARLY
        score = []
        for s in SEEDS:
            d = [v[(key, "saber", s, st)] - v[(key, "her", s, st)] for st in stages]
            d = [x for x in d if not math.isnan(x)]
            score.append(float(np.median(d)) if d else float("nan"))
        sc = np.array(score)
        pos, neg = int(np.sum(sc > 0)), int(np.sum(sc < 0))
        r, p_r, n_r = spearman_perm(sc, fq_gain, rng)
        rows.append(dict(indicator=key, name=NAMES[key], early_median_diff=float(np.nanmedian(sc)),
                         early_mean_her=float(np.nanmean([v[(key, "her", s, st)] for s in SEEDS for st in stages])),
                         seeds_pos=pos, seeds_neg=neg, T1_sign_p=sign_p(pos, neg),
                         T2_spearman_vs_fq_gain=r, T2_perm_p=p_r, T2_n=n_r, **{f"s{s}": x for s, x in zip(SEEDS, score)}))
    for k, adj in zip(("T1_sign_p", "T2_perm_p"), ("T1_holm_p", "T2_holm_p")):
        for r, p in zip(rows, holm([r[k] for r in rows])):
            r[adj] = p
    for r in rows:
        r["T1"] = (("+" if r["seeds_pos"] > r["seeds_neg"] else "-") if r["T1_holm_p"] <= ALPHA else "ns")
        r["T2"] = (("+" if r["T2_spearman_vs_fq_gain"] > 0 else "-") if r["T2_holm_p"] <= ALPHA else "ns")
    # descriptive profile over all stages, with the evaluation-success difference for reference
    profile = []
    for key in ORDER:
        for st in (MEAS_STAGES if key.startswith("M") else STAGES):
            d = np.array([v[(key, "saber", s, st)] - v[(key, "her", s, st)] for s in SEEDS])
            ok = d[~np.isnan(d)]
            profile.append(dict(indicator=key, stage=st, n=len(ok), mean_diff=float(ok.mean()) if len(ok) else float("nan"),
                                mean_her=float(np.nanmean([v[(key, "her", s, st)] for s in SEEDS])),
                                pos=int((ok > 0).sum()), neg=int((ok < 0).sum())))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("early_tests.csv", rows), ("profile.csv", profile)):
        with (OUT / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
            w.writeheader()
            w.writerows(data)
    print(f"{'ind':4s} {'name':28s} {'median d':>10s} {'+/-':>7s} {'T1 p':>7s} {'holm':>6s} {'rho':>6s} {'p':>6s} {'holm':>6s}")
    for r in rows:
        print(f"{r['indicator']:4s} {r['name']:28s} {r['early_median_diff']:+10.4f} {r['seeds_pos']:3d}/{r['seeds_neg']:<3d} "
              f"{r['T1_sign_p']:7.3f} {r['T1_holm_p']:6.3f} {r['T2_spearman_vs_fq_gain']:+6.2f} {r['T2_perm_p']:6.3f} "
              f"{r['T2_holm_p']:6.3f}  T1 {r['T1']} T2 {r['T2']}")


if __name__ == "__main__":
    main()
