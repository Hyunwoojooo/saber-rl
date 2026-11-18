#!/usr/bin/env python3
"""Pre-registered analysis of the actor-signal measurement (PROTOCOL_actor_signal_ko.md, fixed 2026-11-17
before any fpp_n20 checkpoint was measured).

  .venv/bin/python <this dir>/analyze_actor_signal.py [--out outputs/actor_signal_n20] [--partial]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fou import protocol as P  # noqa: E402
from analyze_fpp_n20 import PERF, two_sided_sign_p, spearman, write_csv  # noqa: E402

SEEDS = P.FPP_SETS["fpp_n20"]["seeds"]
STAGES = P.FPP_STAGES
LATE = (500_000, 750_000, 1_000_000)
ALPHA = 0.05
MIN_INF_PAIRS = 32          # a checkpoint counts for a pair-accuracy test only with >= this many informative pairs
MIN_GSTAR = 8               # ... and for the gradient-cosine test only with >= this many queries with g* != 0
RANDOM = [f"random{j}" for j in range(4)]


def load(root: Path) -> list[dict]:
    out = []
    for d in sorted(root.glob("*/seed_*/step_*/meta.json")):
        meta = json.loads(d.read_text())
        with (d.parent / "arms.csv").open() as f:
            rows = {r["critic"]: {k: float(v) for k, v in r.items() if k != "critic"} for r in csv.DictReader(f)}
        rand = {k: float(np.mean([rows[r][k] for r in RANDOM])) for k in rows["uniform"]}
        rows["random"] = rand
        out.append(dict(arm=meta["trained_arm"], seed=meta["seed"], stage=meta["step"], meta=meta, c=rows))
    return out


def metric(name: str, sigma: str = "0.3"):
    """value(checkpoint, critic) and its validity rule."""
    if name == "PA":
        return (lambda c, k: c[k][f"PA_s{sigma}"]), (lambda c: c["K0"][f"n_inf_s{sigma}"] >= MIN_INF_PAIRS)
    if name == "GC":
        return (lambda c, k: c[k]["GC"]), (lambda c: c["K0"]["n_gstar"] >= MIN_GSTAR)
    if name == "AI":
        return (lambda c, k: c[k]["AI"]), (lambda c: True)
    raise ValueError(name)


# name -> (metric, sigma, first critic, second critic); score = value(first) - value(second)
WITHIN = {
    "PA: SABER > uniform (K=100, sigma 0.3)": ("PA", "0.3", "saber", "uniform"),
    "PA: SABER > eligible Random (K=100, sigma 0.3)": ("PA", "0.3", "saber", "random"),
    "PA: SABER > uniform (K=100, sigma 0.1)": ("PA", "0.1", "saber", "uniform"),
    "PA: inverse > uniform (K=100, sigma 0.3)": ("PA", "0.3", "inverse", "uniform"),
    "PA: TD top-k > uniform (K=100, sigma 0.3)": ("PA", "0.3", "td_topk", "uniform"),
    "GC: SABER > uniform (K=100)": ("GC", "0.3", "saber", "uniform"),
    "AI: SABER > uniform (K=100)": ("AI", "0.3", "saber", "uniform"),
    "AI: SABER > eligible Random (K=100)": ("AI", "0.3", "saber", "random"),
}
PRIMARY = "PA: SABER > uniform (K=100, sigma 0.3)"
CROSS = {   # checkpoint critic (K0) of the SABER-trained run minus that of the HER-trained run, same seed and stage
    "cross-run PA (K0, sigma 0.3): SABER run > HER run": ("PA", "0.3"),
    "cross-run PA (K0, sigma 0.1): SABER run > HER run": ("PA", "0.1"),
    "cross-run GC (K0): SABER run > HER run": ("GC", "0.3"),
    "cross-run AI (K0): SABER run > HER run": ("AI", "0.3"),
}


def verdict(name, arm, window, scores):
    vals = [v for v in scores.values() if not math.isnan(v)]
    pos, neg = sum(v > 0 for v in vals), sum(v < 0 for v in vals)
    p = two_sided_sign_p(pos, neg)
    v = ("consistent +" if pos > neg else "consistent -") if (not math.isnan(p) and p <= ALPHA) else "mixed"
    return dict(test=name, trained_arm=arm, window=window, verdict=v, seeds_pos=pos, seeds_neg=neg,
                seeds_tie=len(SEEDS) - pos - neg, sign_p=p,
                median_score=float(np.median(vals)) if vals else float("nan"),
                **{f"s{s}": scores[s] for s in SEEDS})


def within_scores(ck, arm, spec, stages):
    m, sig, a, b = spec
    f, ok = metric(m, sig)
    scores = {}
    for s in SEEDS:
        vals = [f(r["c"], a) - f(r["c"], b) for r in ck
                if r["arm"] == arm and r["seed"] == s and r["stage"] in stages and ok(r["c"])]
        vals = [v for v in vals if not math.isnan(v)]
        scores[s] = float(np.median(vals)) if vals else float("nan")
    return scores


def cross_scores(ck, spec, stages):
    m, sig = spec
    f, ok = metric(m, sig)
    idx = {(r["arm"], r["seed"], r["stage"]): r for r in ck}
    scores = {}
    for s in SEEDS:
        vals = []
        for st in stages:
            a, b = idx.get(("saber", s, st)), idx.get(("her", s, st))
            if a and b and ok(a["c"]) and ok(b["c"]):
                v = f(a["c"], "K0") - f(b["c"], "K0")
                if not math.isnan(v):
                    vals.append(v)
        scores[s] = float(np.median(vals)) if vals else float("nan")
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/actor_signal_n20")
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    root = Path(a.out) if Path(a.out).is_absolute() else HERE / a.out
    ck = load(root)
    expected = 2 * len(SEEDS) * len(STAGES)
    if len(ck) != expected and not a.partial:
        raise SystemExit(f"{len(ck)}/{expected} checkpoints measured; pass --partial to analyse anyway")
    rows = []
    for arm in ("saber", "her"):
        for name, spec in WITHIN.items():
            for window, stages in (("all", STAGES), ("late", LATE)):
                rows.append(verdict(name, arm, window, within_scores(ck, arm, spec, stages)))
    for name, spec in CROSS.items():
        for window, stages in (("all", STAGES), ("late", LATE)):
            rows.append(verdict(name, "saber-her", window, cross_scores(ck, spec, stages)))
    # descriptive levels by stage
    levels = []
    for arm in ("saber", "her"):
        for st in STAGES:
            sel = [r for r in ck if r["arm"] == arm and r["stage"] == st]
            if not sel:
                continue
            row = dict(trained_arm=arm, stage=st, n=len(sel),
                       eval_success=float(np.mean([r["meta"]["last_eval_success"] for r in sel])),
                       G0=float(np.mean([r["meta"]["G0_mean"] for r in sel])),
                       saturated_frac=float(np.mean([r["meta"]["saturated_frac"] for r in sel])),
                       n_inf=float(np.mean([r["c"]["K0"]["n_inf_s0.3"] for r in sel])))
            for k in ("K0", "uniform", "saber", "random", "inverse", "td_topk"):
                for m in ("PA_s0.3", "GC", "AI"):
                    row[f"{k}_{m}"] = float(np.nanmean([r["c"][k][m] for r in sel]))
            levels.append(row)
    # performance link: primary seed score vs the n20 AUC gain
    link = {}
    if PERF.exists():
        with PERF.open() as f:
            perf = {int(r["seed"]): r for r in csv.DictReader(f)}
        auc_gain = [float(perf[s]["saber_auc"]) - float(perf[s]["her_auc"]) for s in SEEDS]
        prim = within_scores(ck, "saber", WITHIN[PRIMARY], STAGES)
        cross = cross_scores(ck, CROSS["cross-run PA (K0, sigma 0.3): SABER run > HER run"], STAGES)
        link = dict(spearman_primary_vs_auc_gain=spearman([prim[s] for s in SEEDS], auc_gain),
                    spearman_crossPA_vs_auc_gain=spearman([cross[s] for s in SEEDS], auc_gain))
    dest = root / "seed_analysis"
    dest.mkdir(exist_ok=True)
    write_csv(dest / "seed_verdicts.csv", rows)
    write_csv(dest / "levels_by_stage.csv", levels)
    summary = dict(partial=len(ck) != expected, n_checkpoints=len(ck), expected=expected,
                   primary=next(r for r in rows if r["test"] == PRIMARY and r["trained_arm"] == "saber" and r["window"] == "all"),
                   performance_link=link)
    (dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for r in rows:
        print(f"{r['trained_arm']:9s} {r['window']:4s} {r['test']:55s} {r['verdict']:13s} "
              f"+{r['seeds_pos']}/-{r['seeds_neg']} p={r['sign_p']:.3f} median {r['median_score']:+.4f}")
    if link:
        print("performance link:", link)


if __name__ == "__main__":
    main()
