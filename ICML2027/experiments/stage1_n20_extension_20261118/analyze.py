#!/usr/bin/env python3
"""Pre-registered analysis of stage 1 (PROTOCOL_ko.md, fixed 2026-11-18 before any of these runs started).

  .venv/bin/python <this dir>/analyze.py --cells fpp_ddpg fpp_sac
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "ICML2027" / "paper_artifacts" / "stage1_n20_extension_20261118"
SEEDS = tuple(range(11, 31))
STEPS = {"fpp_ddpg": 1_000_000, "fpp_sac": 1_000_000,
         # Drawer cells added 2026-11-22 before any Drawer stage-1 run started (PROTOCOL_ko.md section 7)
         "draweropen_sac": 2_000_000, "drawerclose_sac": 2_000_000, "drawerclose_ddpg": 2_000_000}


def run_glob(cell: str, arm: str, s: int, T: int) -> str:
    if cell.endswith("_sac") and cell.startswith("drawer"):
        return f"*_sac_{arm}_s{s}_2m/summary.json"
    if cell == "drawerclose_ddpg":
        return f"seed_{s:02d}/summary.json"
    return f"*_s{s}_{T}_*/summary.json"

# same AUC and statistics code as the FPP TD3 n20 analysis (bootstrap 100,000, seed 20261113, exact sign-flip)
_spec = importlib.util.spec_from_file_location(
    "analyze_n20", HERE.parent / "fpp_td3_primary_checkpoint_rerun_20261110" / "analyze_n20.py")
A = importlib.util.module_from_spec(_spec)          # type: ignore[arg-type]
_spec.loader.exec_module(A)                         # type: ignore[union-attr]


def verdict(st: dict) -> str:
    if st["ci_low"] > 0:
        return "SABER > HER (95% CI excludes 0)"
    if st["ci_high"] < 0:
        return "HER > SABER (95% CI excludes 0)"
    return "inconclusive (95% CI includes 0); report the effect size"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True, choices=list(STEPS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for cell in a.cells:
        T = STEPS[cell]
        rows, missing = [], []
        for s in SEEDS:
            r = {"seed": s}
            for arm in ("her", "saber"):
                found = list((HERE / "outputs" / "runs" / cell / arm).glob(run_glob(cell, arm, s, T)))
                if len(found) != 1:
                    missing.append(f"{arm}{s}")
                    continue
                h = A.curve(found[0].parent / "metrics.csv")
                r[f"{arm}_auc"], r[f"{arm}_fq"], r[f"{arm}_final"] = A.auc(h, 10_000, T), A.auc(h, int(0.75 * T), T), h[T]
            rows.append(r)
        if missing:
            print(f"{cell}: incomplete, refusing to analyse: {missing}")
            continue
        report = {}
        for key, label in (("auc", "AUC 10k-end"), ("fq", "final-quarter AUC"), ("final", "final success")):
            d = np.array([r[f"saber_{key}"] - r[f"her_{key}"] for r in rows])
            st = A.stats(d)
            st.update(her_mean=float(np.mean([r[f"her_{key}"] for r in rows])),
                      saber_mean=float(np.mean([r[f"saber_{key}"] for r in rows])))
            if key == "auc":
                st["verdict"] = verdict(st)
            report[label] = st
        dest = OUT / cell
        dest.mkdir(exist_ok=True)
        with (dest / "per_seed.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        (dest / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
        for label, st in report.items():
            print(f"{cell} {label}: HER {st['her_mean']:.4f} SABER {st['saber_mean']:.4f} diff {st['mean']:+.4f} "
                  f"CI [{st['ci_low']:+.4f}, {st['ci_high']:+.4f}] +{st['positive']}/-{st['negative']} p {st['sign_flip_p']:.4f}"
                  + (f"\n  => {st['verdict']}" if "verdict" in st else ""))


if __name__ == "__main__":
    main()
