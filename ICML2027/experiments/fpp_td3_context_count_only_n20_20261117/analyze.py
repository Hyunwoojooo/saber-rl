#!/usr/bin/env python3
"""Pre-registered analysis of the context-count-only control on FPP TD3 seeds 11-30 (PROTOCOL_ko.md).

Fixed 2026-11-17 before any of these runs started. Pairs each context-count-only run with the n20 HER and
SABER runs of the same seed (fpp_td3_primary_checkpoint_rerun_20261110/outputs/runs_n20).

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
N20 = HERE.parent / "fpp_td3_primary_checkpoint_rerun_20261110"
DIRS = {
    "her": N20 / "outputs" / "runs_n20" / "td3_her_baseline_ls512" / "fetch_pick_and_place_td3_her_baseline_ls512_s{s}_1000000_ls512",
    "saber": N20 / "outputs" / "runs_n20" / "td3_saber_critic_only_v4_ls512"
             / "fetch_pick_and_place_td3_saber_critic_only_v4_ls512_s{s}_1000000_ls512",
    "cco": HERE / "outputs" / "runs" / "fetch_pick_and_place_td3_saber_context_count_only_ls512_s{s}_1000000_ls512",
}
SEEDS = tuple(range(11, 31))
OUT = REPO / "ICML2027" / "paper_artifacts" / "fpp_td3_context_count_only_n20_20261117"
HALF_SABER_EFFECT = 0.0225       # half of the n20 SABER - HER AUC gain (+0.045)

# reuse the exact AUC and statistics code of the n20 analysis (same bootstrap seed and resamples)
_spec = importlib.util.spec_from_file_location("analyze_n20", N20 / "analyze_n20.py")
A = importlib.util.module_from_spec(_spec)          # type: ignore[arg-type]
_spec.loader.exec_module(A)                         # type: ignore[union-attr]


def verdict_saber_vs_cco(st: dict) -> str:
    if st["ci_low"] > 0:
        return ("SABER > context-count-only: the action-membership query (is a^pi's bin in the local support?) "
                "adds to context counts alone (95% CI excludes 0)")
    if st["ci_high"] < 0:
        return "context-count-only > SABER (95% CI excludes 0)"
    if st["ci_high"] < HALF_SABER_EFFECT:
        return ("no action-query effect of half the SABER gain: CI includes 0 and its upper bound is below +0.0225; "
                "the SABER gain is mostly reproduced by the context-count gate without the a^pi query")
    return "inconclusive: CI includes 0 and reaches +0.0225"


def main() -> None:
    rows, missing = [], []
    for s in SEEDS:
        r = {"seed": s}
        for arm, pattern in DIRS.items():
            d = Path(str(pattern).format(s=s))
            if not (d / "summary.json").exists():
                missing.append(f"{arm}{s}")
                continue
            h = A.curve(d / "metrics.csv")
            r[f"{arm}_auc"], r[f"{arm}_fq"], r[f"{arm}_final"] = A.auc(h, 10_000), A.auc(h, 750_000), h[1_000_000]
        rows.append(r)
    if missing:
        raise SystemExit(f"incomplete runs, refusing to analyse: {missing}")
    report = {}
    contrasts = (("saber", "cco", "SABER - context-count-only"), ("cco", "her", "context-count-only - HER"),
                 ("saber", "her", "SABER - HER (n20, for reference)"))
    for a, b, name in contrasts:
        for key, label in (("auc", "AUC 10k-1M"), ("fq", "final-quarter AUC"), ("final", "final success")):
            d = np.array([r[f"{a}_{key}"] - r[f"{b}_{key}"] for r in rows])
            st = A.stats(d)
            st.update({f"{a}_mean": float(np.mean([r[f"{a}_{key}"] for r in rows])),
                       f"{b}_mean": float(np.mean([r[f"{b}_{key}"] for r in rows]))})
            if key == "auc" and (a, b) == ("saber", "cco"):
                st["verdict"] = verdict_saber_vs_cco(st)
            report[f"{name} | {label}"] = st
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_seed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    for label, st in report.items():
        print(f"{label}: diff {st['mean']:+.4f} CI [{st['ci_low']:+.4f}, {st['ci_high']:+.4f}] "
              f"+{st['positive']}/-{st['negative']} sign-flip p {st['sign_flip_p']:.4f}"
              + (f"\n  => {st['verdict']}" if "verdict" in st else ""))


if __name__ == "__main__":
    main()
