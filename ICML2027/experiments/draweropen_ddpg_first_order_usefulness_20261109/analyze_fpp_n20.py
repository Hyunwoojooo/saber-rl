#!/usr/bin/env python3
"""Pre-registered seed-level analysis of the fpp_n20 measurement (PROTOCOL_fpp_n20_ko.md, fixed 2026-11-16).

Written before any fpp_n20 checkpoint was measured. Uses aggregate.summarise_checkpoint unchanged.

  .venv/bin/python <this dir>/analyze_fpp_n20.py [--out outputs/fpp_n20] [--name seed_analysis] [--partial]

Seed score = median over counted stages of gain / R0 (gain > 0: SABER reduced the policy-query error more
than uniform). Ties: sqrt(R0) < FLOOR_RMS, or no live batch (p = 0 or 1 in every batch).
Verdict: exact two-sided sign test over seeds with non-zero score; p <= 0.05 -> consistent +/-; else mixed.
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
import aggregate as G  # noqa: E402

SEEDS = P.FPP_SETS["fpp_n20"]["seeds"]
LATE = (500_000, 750_000, 1_000_000)
ALPHA = 0.05
PERF = HERE.parents[1] / "paper_artifacts" / "fpp_td3_primary_rerun_20261110" / "n20_seeds11_30" / "per_seed.csv"

TESTS = {   # name -> (bank, value(r)); every value is a gain, positive favours SABER / the first arm
    "SABER > uniform, K=100, R1": ("R1", lambda r: r.get("R1_K100_saber", float("nan"))),
    "SABER > uniform, K=100, R2": ("R2", lambda r: r.get("R2_K100_saber", float("nan"))),
    "SABER > uniform, real 1-step, R1": ("R1", lambda r: r["R1_A1_saber"]),
    "SABER > uniform, G3 prediction, R1": ("R1", lambda r: r["R1_G3_saber"]),
    "SABER > eligible Random, K=100, R1": ("R1", lambda r: r.get("R1_K100_saber", float("nan")) - r.get("R1_K100_random", float("nan"))),
    "SABER > eligible Random, K=100, R2": ("R2", lambda r: r.get("R2_K100_saber", float("nan")) - r.get("R2_K100_random", float("nan"))),
    "inverse > uniform, K=100, R1": ("R1", lambda r: r.get("R1_K100_inverse", float("nan"))),
}
KAPPA = "kappa(M) > kappa(E\\M), dist-matched"


def two_sided_sign_p(n_pos: int, n_neg: int) -> float:
    n = n_pos + n_neg
    if n == 0:
        return float("nan")
    x = min(n_pos, n_neg)
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(0, x + 1)) / 2 ** n)


def seed_scores(ck, arm, bank, f, stages, relative=True):
    scores = {}
    for s in SEEDS:
        vals = []
        for r in ck:
            if r["trained_arm"] != arm or r["seed"] != s or r["stage"] not in stages:
                continue
            if bank and (f"{bank}_R0" not in r or r[f"{bank}_null"]):
                continue
            if r["n_live"] == 0:
                continue
            v = f(r)
            if v is None or math.isnan(v):
                continue
            vals.append(v / r[f"{bank}_R0"] if (relative and bank) else v)
        scores[s] = float(np.median(vals)) if vals else float("nan")
    return scores


def verdict_row(name, arm, window, scores):
    vals = [v for v in scores.values() if not math.isnan(v)]
    pos, neg = sum(v > 0 for v in vals), sum(v < 0 for v in vals)
    p = two_sided_sign_p(pos, neg)
    v = ("consistent +" if pos > neg else "consistent -") if (not math.isnan(p) and p <= ALPHA) else "mixed"
    return dict(trained_arm=arm, window=window, test=name, verdict=v, seeds_pos=pos, seeds_neg=neg,
                seeds_tie=len(SEEDS) - pos - neg, sign_p=p, median_score=float(np.median(vals)) if vals else float("nan"),
                **{f"s{s}": scores[s] for s in SEEDS})


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 4:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def roc_auc(score, label) -> float:
    score, label = np.asarray(score, float), np.asarray(label, bool)
    if label.all() or not label.any():
        return float("nan")
    ranks = np.argsort(np.argsort(score, kind="mergesort"), kind="mergesort") + 1.0
    # average ranks for ties
    order = np.argsort(score, kind="mergesort")
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1, n0 = label.sum(), (~label).sum()
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logistic_auc(X, y, iters=25) -> float:
    """In-sample ROC AUC of a Newton-fitted logistic regression on standardised features (descriptive)."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    X = np.c_[np.ones(len(X)), X]
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(X.shape[1])
        w -= np.linalg.solve(H, X.T @ (p - y))
    return roc_auc(X @ w, y.astype(bool))


def psi_proxy(dirs) -> list[dict]:
    rows = []
    for d in dirs:
        meta = json.loads((d / "meta.json").read_text())
        ps = np.load(d / "per_sample.npz")
        if "age" not in ps:
            continue
        m, e = ps["support"].astype(bool).ravel(), ps["eligible"].astype(bool).ravel()
        dist, age = ps["dist"].ravel().astype(float), ps["age"].ravel().astype(float)
        sel = e
        y = m[sel]
        r = dict(trained_arm=meta["trained_arm"], seed=meta["seed"], stage=meta["step"],
                 n_eligible=int(sel.sum()), psi_rate_in_E=float(y.mean()) if sel.any() else float("nan"))
        if sel.any() and 0 < y.sum() < y.size:
            r.update(dist_psi1=float(dist[sel][y].mean()), dist_psi0=float(dist[sel][~y].mean()),
                     age_psi1=float(age[sel][y].mean()), age_psi0=float(age[sel][~y].mean()),
                     auc_neg_dist=roc_auc(-dist[sel], y), auc_neg_age=roc_auc(-age[sel], y),
                     auc_logit_dist_age=logistic_auc(np.c_[dist[sel], age[sel]], y.astype(float)))
        rows.append(r)
    return rows


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/fpp_n20")
    ap.add_argument("--name", default="seed_analysis")
    ap.add_argument("--partial", action="store_true", help="allow an incomplete set (labelled as such)")
    a = ap.parse_args()
    root = Path(a.out) if Path(a.out).is_absolute() else HERE / a.out
    dirs = sorted(p.parent for p in root.glob("*/seed_*/step_*/meta.json"))
    ck = [G.summarise_checkpoint(d) for d in dirs]
    expected = {(arm, s, st) for arm in ("saber", "her") for s in SEEDS for st in P.FPP_STAGES}
    have = {(r["trained_arm"], r["seed"], r["stage"]) for r in ck}
    complete = {arm: all((arm, s, st) in have for s in SEEDS for st in P.FPP_STAGES) for arm in ("saber", "her")}
    if not a.partial and not complete["saber"]:
        raise SystemExit(f"SABER-trained set incomplete ({len([h for h in have if h[0] == 'saber'])}/180); use --partial")
    dest = root / a.name
    dest.mkdir(exist_ok=False)
    write_csv(dest / "checkpoints.csv", ck)

    verdicts, score_tab = [], {}
    for arm in ("saber", "her"):
        if not any(r["trained_arm"] == arm for r in ck):
            continue
        for window, stages in (("all 9 stages", P.FPP_STAGES), ("late 500k-1M", LATE)):
            for name, (bank, f) in TESTS.items():
                sc = seed_scores(ck, arm, bank, f, stages)
                score_tab[(arm, window, name)] = sc
                row = verdict_row(name, arm, window, sc)
                row["complete"] = complete[arm]
                verdicts.append(row)
            sc = seed_scores(ck, arm, None, lambda r: r["kappa_gap_by_dist"], stages, relative=False)
            row = verdict_row(KAPPA, arm, window, sc)
            row["complete"] = complete[arm]
            verdicts.append(row)
    write_csv(dest / "seed_verdicts.csv", verdicts)

    # link to performance (descriptive)
    link = []
    if PERF.exists():
        perf = {int(float(r["seed"])): {k: float(v) for k, v in r.items()} for r in csv.DictReader(PERF.open())}
        for window in ("all 9 stages", "late 500k-1M"):
            for name in ("SABER > uniform, K=100, R1", "SABER > uniform, K=100, R2"):
                sc = score_tab.get(("saber", window, name))
                if sc is None:
                    continue
                xs = [sc[s] for s in SEEDS]
                for target, f in (("pair AUC diff", lambda q: q["saber_auc"] - q["her_auc"]),
                                  ("pair final-quarter diff", lambda q: q["saber_fq"] - q["her_fq"]),
                                  ("SABER final success", lambda q: q["saber_final"])):
                    link.append(dict(window=window, test=name, target=target,
                                     spearman=spearman(xs, [f(perf[s]) for s in SEEDS])))
                conv = [sc[s] for s in SEEDS if perf[s]["saber_final"] >= 0.9]
                fail = [sc[s] for s in SEEDS if perf[s]["saber_final"] <= 0.1]
                link.append(dict(window=window, test=name, target="median score: SABER final>=0.9 vs <=0.1",
                                 converged=float(np.nanmedian(conv)) if conv else float("nan"), n_converged=len(conv),
                                 failed=float(np.nanmedian(fail)) if fail else float("nan"), n_failed=len(fail)))
        write_csv(dest / "performance_link.csv", link)

    # p over time (descriptive)
    ptab = []
    for arm in ("saber", "her"):
        for st in P.FPP_STAGES:
            ps_ = [r["p"] for r in ck if r["trained_arm"] == arm and r["stage"] == st]
            if ps_:
                ptab.append(dict(trained_arm=arm, stage=st, n=len(ps_), p_mean=float(np.mean(ps_)),
                                 p_max=float(np.max(ps_)), n_degenerate=sum(r["n_live"] == 0 for r in ck
                                                                            if r["trained_arm"] == arm and r["stage"] == st)))
    write_csv(dest / "positive_rate_by_stage.csv", ptab)

    proxy = psi_proxy(dirs)
    if proxy:
        write_csv(dest / "psi_proxy.csv", proxy)

    summary = dict(complete=complete, n_checkpoints=len(ck),
                   verdicts=[{k: v for k, v in r.items() if not k.startswith("s1") and not k.startswith("s2")
                              and not k.startswith("s3")} for r in verdicts])
    if proxy:
        cols = ("psi_rate_in_E", "dist_psi1", "dist_psi0", "age_psi1", "age_psi0", "auc_neg_dist", "auc_neg_age",
                "auc_logit_dist_age")
        summary["psi_proxy_median"] = {arm: {c: float(np.nanmedian([r.get(c, float("nan")) for r in proxy
                                                                     if r["trained_arm"] == arm]))
                                             for c in cols} for arm in ("saber", "her")
                                       if any(r["trained_arm"] == arm for r in proxy)}
    (dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for r in verdicts:
        print(f"[{r['trained_arm']:5s} | {r['window']:13s}] {r['test']:40s} {r['verdict']:13s} "
              f"+{r['seeds_pos']}/-{r['seeds_neg']} (tie {r['seeds_tie']}) p={r['sign_p']:.4f} "
              f"median {r['median_score']:+.4f}" + ("" if r["complete"] else "  [INCOMPLETE]"))
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
