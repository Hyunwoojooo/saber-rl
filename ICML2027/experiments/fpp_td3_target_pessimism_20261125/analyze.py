#!/usr/bin/env python3
"""Pre-registered analysis of the FPP TD3 target-pessimism test (PROTOCOL_ko.md). Fixed 2026-11-25 before any run started.

Pairs every new run with the n20 HER / SABER run of the same seed (alpha 0) and uses the n20 AUC and
statistics code unchanged (same bootstrap seed and resamples).

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
SLUG = {"her": "td3_her_baseline_ls512", "saber": "td3_saber_critic_only_v4_ls512"}
SEEDS = tuple(range(11, 31))
TAGS = ("a000", "a025", "a050")
OUT = REPO / "ICML2027" / "paper_artifacts" / "fpp_td3_target_pessimism_20261125"
HALF_SABER_EFFECT = 0.0225       # half of the n20 SABER - HER AUC gain (+0.045)
EARLY = tuple(range(100_000, 400_001, 50_000))      # same early window as the cumulative-effect analysis (L2)
WINDOW = 50_000

_spec = importlib.util.spec_from_file_location("analyze_n20", N20 / "analyze_n20.py")
A = importlib.util.module_from_spec(_spec)          # type: ignore[arg-type]
_spec.loader.exec_module(A)                         # type: ignore[union-attr]


def run_dir(arm: str, tag: str, seed: int) -> Path:
    name = f"fetch_pick_and_place_{SLUG[arm]}_s{seed}_1000000_ls512"
    if tag == "a000":
        return N20 / "outputs" / "runs_n20" / SLUG[arm] / name
    return HERE / "outputs" / "runs" / f"{arm}_{tag}" / name


def check_config(d: Path, tag: str, seed: int) -> None:
    cfg = json.loads((d / "config.json").read_text())
    want = {"a000": 0.0, "a025": 0.25, "a050": 0.5}[tag]
    assert float(cfg.get("td3_target_mix_alpha", 0.0)) == want and int(cfg["env_seed"]) == seed, d


def early_q(path: Path) -> float:
    """Mean of critic/q_policy_mean over the early stages, each stage a 50k-step window (indicator L2)."""
    rows = list(csv.DictReader(path.open()))
    steps = np.array([int(float(r["step"])) for r in rows])
    x = np.array([float(r["critic/q_policy_mean"]) if r["critic/q_policy_mean"] not in ("", "nan") else np.nan for r in rows])
    return float(np.mean([np.nanmean(x[(steps > st - WINDOW) & (steps <= st)]) for st in EARLY]))


def read_her_dose(st: dict) -> str:
    if st["ci_low"] > 0:
        return "raised"
    if st["ci_high"] < 0:
        return "lowered"
    return "unchanged"


def read_saber_gain(st: dict) -> str:
    if st["ci_low"] > 0:
        return "remains"
    if st["ci_high"] < HALF_SABER_EFFECT:
        return "less than half remains"       # includes CI entirely below 0
    return "inconclusive"


def verdict(her_dose: str, saber_gain: str) -> str:
    if saber_gain == "remains":
        return ("NOT explained by target pessimism: the SABER gain remains (95% CI excludes 0) when the target is "
                "less pessimistic")
    if her_dose == "raised" and saber_gain == "less than half remains":
        return ("consistent with the pessimism-relief route: the less pessimistic target alone raises HER and less "
                "than half of the SABER gain remains on top of it")
    if her_dose != "raised":
        return ("no test: the less pessimistic target did not raise HER, so this dose cannot show whether a higher "
                "Q level reproduces the gain")
    return "inconclusive: HER is raised but the remaining SABER gain is neither excluded nor established"


def main() -> None:
    rows, missing = [], []
    for s in SEEDS:
        r = {"seed": s}
        for arm in SLUG:
            for tag in TAGS:
                d = run_dir(arm, tag, s)
                if not (d / "summary.json").exists():
                    missing.append(f"{arm}_{tag}_s{s}")
                    continue
                check_config(d, tag, s)
                h = A.curve(d / "metrics.csv")
                k = f"{arm}_{tag}"
                r[f"{k}_auc"], r[f"{k}_fq"], r[f"{k}_final"] = A.auc(h, 10_000), A.auc(h, 750_000), h[1_000_000]
                r[f"{k}_earlyq"] = early_q(d / "metrics.csv")
        rows.append(r)
    if missing:
        raise SystemExit(f"incomplete runs, refusing to analyse: {missing[:8]} ... ({len(missing)})")

    def contrast(a: str, b: str, key: str) -> dict:
        d = np.array([r[f"{a}_{key}"] - r[f"{b}_{key}"] for r in rows])
        st = A.stats(d)
        st.update({f"{a}_mean": float(np.mean([r[f"{a}_{key}"] for r in rows])), f"{b}_mean": float(np.mean([r[f"{b}_{key}"] for r in rows]))})
        return st

    report = {"reference": {"SABER - HER, alpha 0 | AUC": contrast("saber_a000", "her_a000", "auc"),
                            "SABER - HER, alpha 0 | early Q": contrast("saber_a000", "her_a000", "earlyq")}}
    for tag in ("a050", "a025"):
        sec = {}
        for key, label in (("auc", "AUC 10k-1M"), ("fq", "final-quarter AUC"), ("final", "final success")):
            sec[f"C1 HER dose: HER {tag} - HER a000 | {label}"] = contrast(f"her_{tag}", "her_a000", key)
            sec[f"C2 SABER gain at dose: SABER {tag} - HER {tag} | {label}"] = contrast(f"saber_{tag}", f"her_{tag}", key)
            sec[f"C4 HER {tag} - SABER a000 | {label}"] = contrast(f"her_{tag}", "saber_a000", key)
            d = np.array([(r[f"saber_a000_{key}"] - r[f"her_a000_{key}"]) - (r[f"saber_{tag}_{key}"] - r[f"her_{tag}_{key}"]) for r in rows])
            sec[f"C3 interaction: gain at a000 - gain at {tag} | {label}"] = A.stats(d)
        sec[f"manipulation check: HER {tag} - HER a000 | early Q"] = contrast(f"her_{tag}", "her_a000", "earlyq")
        sec[f"manipulation check: SABER {tag} - HER {tag} | early Q"] = contrast(f"saber_{tag}", f"her_{tag}", "earlyq")
        hd = read_her_dose(sec[f"C1 HER dose: HER {tag} - HER a000 | AUC 10k-1M"])
        sg = read_saber_gain(sec[f"C2 SABER gain at dose: SABER {tag} - HER {tag} | AUC 10k-1M"])
        sec["reading"] = dict(her_dose=hd, saber_gain=sg, verdict=verdict(hd, sg))
        report[tag + (" (primary)" if tag == "a050" else " (secondary)")] = sec
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_seed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, sec in report.items():
        print(f"== {name}")
        for label, st in sec.items():
            if label == "reading":
                print(f"   => HER dose: {st['her_dose']}; SABER gain: {st['saber_gain']}\n   => {st['verdict']}")
            else:
                print(f"   {label}: {st['mean']:+.4f} CI [{st['ci_low']:+.4f}, {st['ci_high']:+.4f}] +{st['positive']}/-{st['negative']} p {st['sign_flip_p']:.4f}")


if __name__ == "__main__":
    main()
