#!/usr/bin/env python3
"""Pre-registered analysis of the 20-pair FPP TD3 HER vs SABER extension (seeds 11-30, cuda).

Fixed 2026-11-13 before any of these runs started (see PROTOCOL_ko.md, "추가 2").

  .venv/bin/python <this dir>/analyze_n20.py
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = HERE / "outputs" / "runs_n20"
SLUG = {"her": "td3_her_baseline_ls512", "saber": "td3_saber_critic_only_v4_ls512"}
SEEDS = tuple(range(11, 31))
OUT = REPO / "ICML2027" / "paper_artifacts" / "fpp_td3_primary_rerun_20261110" / "n20_seeds11_30"
BOOT, RNG_SEED = 100_000, 20261113


def curve(path: Path) -> dict[int, float]:
    h = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            v = row.get("eval/success_mean", "")
            if v not in ("", None):
                h[int(float(row["step"]))] = float(v)
    return h


def auc(h: dict[int, float], lo: int, hi: int = 1_000_000) -> float:
    ks = sorted(k for k in h if lo <= k <= hi)
    xs, ys = np.array(ks, float), np.array([h[k] for k in ks])
    return float(np.trapezoid(ys, xs) / (xs[-1] - xs[0]))


def stats(d: np.ndarray) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    boot = d[rng.integers(0, len(d), size=(BOOT, len(d)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # exact two-sided sign-flip permutation test of the mean (2^20 patterns)
    signs = np.array(list(itertools.product((1, -1), repeat=len(d))), dtype=float) if len(d) <= 20 else None
    p_flip = float(np.mean(np.abs(signs @ np.abs(d)) >= abs(d.sum()) - 1e-12)) if signs is not None else float("nan")
    return dict(n=len(d), mean=float(d.mean()), ci_low=float(lo), ci_high=float(hi), positive=int((d > 0).sum()),
                negative=int((d < 0).sum()), sign_flip_p=p_flip)


def verdict(st: dict) -> str:
    if st["ci_low"] > 0:
        return "SABER > HER (95% CI excludes 0)"
    if st["ci_high"] < 0:
        return "SABER < HER (95% CI excludes 0)"
    if st["ci_high"] < 0.035:
        return "no gain of half the original size: CI includes 0 and its upper bound is below +0.035"
    return "inconclusive: CI includes 0 and reaches +0.035"


def main() -> None:
    rows, missing = [], []
    for s in SEEDS:
        r = {"seed": s}
        for arm, slug in SLUG.items():
            d = BASE / slug / f"fetch_pick_and_place_{slug}_s{s}_1000000_ls512"
            if not (d / "summary.json").exists():
                missing.append(f"{arm}{s}")
                continue
            h = curve(d / "metrics.csv")
            r[f"{arm}_auc"], r[f"{arm}_fq"], r[f"{arm}_final"] = auc(h, 10_000), auc(h, 750_000), h[1_000_000]
        rows.append(r)
    if missing:
        raise SystemExit(f"incomplete runs, refusing to analyse: {missing}")
    report = {}
    for key, label in (("auc", "AUC 10k-1M (primary)"), ("fq", "final-quarter AUC"), ("final", "final success")):
        d = np.array([r[f"saber_{key}"] - r[f"her_{key}"] for r in rows])
        st = stats(d)
        st.update(her_mean=float(np.mean([r[f"her_{key}"] for r in rows])),
                  saber_mean=float(np.mean([r[f"saber_{key}"] for r in rows])))
        if key == "auc":
            st["verdict"] = verdict(st)
        report[label] = st
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_seed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    for label, st in report.items():
        print(f"{label}: HER {st['her_mean']:.4f} SABER {st['saber_mean']:.4f} diff {st['mean']:+.4f} "
              f"CI [{st['ci_low']:+.4f}, {st['ci_high']:+.4f}] +{st['positive']}/-{st['negative']} "
              f"sign-flip p {st['sign_flip_p']:.4f}" + (f"  => {st['verdict']}" if "verdict" in st else ""))


if __name__ == "__main__":
    main()
