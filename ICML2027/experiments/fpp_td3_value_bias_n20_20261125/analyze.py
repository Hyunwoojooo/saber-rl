#!/usr/bin/env python3
"""Analysis of the value-bias measurement (PROTOCOL_ko.md, fixed 2026-11-25 before the full measurement).

  .venv/bin/python <this dir>/analyze.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MEAS = HERE / "outputs" / "measure"
OUT = REPO / "ICML2027" / "paper_artifacts" / "fpp_td3_value_bias_n20_20261125"
SEEDS = tuple(range(11, 31))
EARLY = (100_000, 150_000, 200_000, 300_000, 400_000)
STAGES = EARLY + (500_000, 750_000, 1_000_000)
METRICS = ("Q1", "G", "bias", "abs_err", "gap", "bias_min", "success300")

_spec = importlib.util.spec_from_file_location(
    "analyze_n20", HERE.parent / "fpp_td3_primary_checkpoint_rerun_20261110" / "analyze_n20.py")
A = importlib.util.module_from_spec(_spec)          # type: ignore[arg-type]
_spec.loader.exec_module(A)                         # type: ignore[union-attr]


def checkpoint_values(arm: str, seed: int, stage: int) -> dict:
    z = np.load(MEAS / arm / f"seed_{seed}" / f"step_{stage:07d}.npz")
    v = {}
    for b in ("O", "H"):
        q1, q2, g = z[f"{b}_Q1"], z[f"{b}_Q2"], z[f"{b}_G300"]
        v[b] = dict(Q1=q1.mean(), G=g.mean(), bias=(q1 - g).mean(), abs_err=np.abs(q1 - g).mean(), gap=np.abs(q1 - q2).mean(),
                    bias_min=(np.minimum(q1, q2) - g).mean(), success300=z[f"{b}_success300"].mean())
    return v


def arm_ci(x: np.ndarray) -> dict:
    rng = np.random.default_rng(A.RNG_SEED)
    boot = x[rng.integers(0, len(x), size=(A.BOOT, len(x)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(mean=float(x.mean()), ci_low=float(lo), ci_high=float(hi))


def sign(st: dict) -> str:
    return "positive" if st["ci_low"] > 0 else "negative" if st["ci_high"] < 0 else "includes 0"


def verdict(bias: str, err: str, g: str) -> str:
    if bias == "includes 0":
        return ("the Q-level difference is not a difference in critic bias"
                + ("; the SABER policy's true value is already higher, which accounts for it" if g == "positive" else ""))
    if bias == "positive":
        return ("SABER's critic sits higher relative to the true value and is more accurate (less underestimation)"
                if err == "negative" else "SABER's critic sits higher relative to the true value but is not more accurate")
    return "SABER's critic sits lower relative to the true value: opposite to the Q-level hypothesis"


def main() -> None:
    missing = [f"{a}{s}@{st}" for a in ("her", "saber") for s in SEEDS for st in STAGES
               if not (MEAS / a / f"seed_{s}" / f"step_{st:07d}.npz").exists()]
    if missing:
        raise SystemExit(f"incomplete measurement, refusing to analyse: {missing[:6]} ... ({len(missing)})")
    val = {(a, s, st): checkpoint_values(a, s, st) for a in ("her", "saber") for s in SEEDS for st in STAGES}
    report, rows = {}, []
    for b, bank in (("O", "original goal (primary)"), ("H", "hindsight goal")):
        sec = {}
        for window, stages in (("early 100k-400k", EARLY), *((f"{st // 1000}k", (st,)) for st in STAGES)):
            per = {m: {a: np.array([np.mean([val[(a, s, st)][b][m] for st in stages]) for s in SEEDS]) for a in ("her", "saber")}
                   for m in METRICS}
            w = {}
            for m in METRICS:
                st = A.stats(per[m]["saber"] - per[m]["her"])
                st.update(her=arm_ci(per[m]["her"]), saber=arm_ci(per[m]["saber"]))
                w[m] = st
                rows.append(dict(bank=b, window=window, metric=m, her=st["her"]["mean"], saber=st["saber"]["mean"], diff=st["mean"],
                                 ci_low=st["ci_low"], ci_high=st["ci_high"], positive=st["positive"], negative=st["negative"],
                                 sign_flip_p=st["sign_flip_p"]))
            w["reading"] = dict(G=sign(w["G"]), bias=sign(w["bias"]), abs_err=sign(w["abs_err"]),
                                verdict=verdict(sign(w["bias"]), sign(w["abs_err"]), sign(w["G"])),
                                more_accurate_vs_real_return=bool(w["abs_err"]["ci_high"] < 0))
            sec[window] = w
        report[bank] = sec
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "contrasts.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with (OUT / "per_checkpoint.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["arm", "seed", "stage", "bank", *METRICS])
        for (a, s, st), v in sorted(val.items()):
            for b in ("O", "H"):
                wr.writerow([a, s, st, b, *(f"{float(v[b][m]):.6f}" for m in METRICS)])
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    for bank, sec in report.items():
        print(f"== {bank}")
        for window, w in sec.items():
            if window != "early 100k-400k" and window not in ("150k", "400k", "1000k"):
                continue
            print(f" [{window}]")
            for m in ("Q1", "G", "bias", "abs_err", "gap"):
                st = w[m]
                print(f"   {m:8s} HER {st['her']['mean']:+8.3f} [{st['her']['ci_low']:+.2f},{st['her']['ci_high']:+.2f}]  SABER {st['saber']['mean']:+8.3f}"
                      f"  diff {st['mean']:+.3f} CI [{st['ci_low']:+.3f}, {st['ci_high']:+.3f}] +{st['positive']}/-{st['negative']} p {st['sign_flip_p']:.4f}")
            print(f"   => {w['reading']['verdict']} | more accurate vs real return: {w['reading']['more_accurate_vs_real_return']}")


if __name__ == "__main__":
    main()
