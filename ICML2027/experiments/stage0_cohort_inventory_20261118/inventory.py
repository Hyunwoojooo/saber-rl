#!/usr/bin/env python3
"""Stage 0: inventory of every existing HER / SABER (critic-only main) pair, per task x backbone.

No training. Sources are this working tree and origin/main (which holds seed extensions and Drawer SAC
artifacts that are not in the working tree). AUC is recomputed from each run's evaluation curve with one
rule (the n20 rule): eval/success_mean (or `success` in evaluation.csv), trapezoid over [10k, total steps]
divided by the span. Runs with identical curves are one run (paper_artifacts keep copies).

  .venv/bin/python ICML2027/experiments/stage0_cohort_inventory_20261118/inventory.py
"""
from __future__ import annotations

import csv
import glob
import hashlib
import itertools
import json
import os
import re
import subprocess
import tarfile
import io
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "ICML2027" / "paper_artifacts" / "stage0_cohort_inventory_20261118"
CACHE = HERE / "outputs"
LO = 10_000
BOOT, BOOT_SEED = 100_000, 20261113

# (cell, cohort label, role, root ("L" working tree / "M" origin/main), glob of run dirs)
S = []
def add(cell, cohort, her, saber, root="L"):
    S.append((cell, cohort, "her", root, her))
    S.append((cell, cohort, "saber", root, saber))

add("FPP DDPG", "s1-5", "experiment_FPP/runs/B0/*_1000000_*", "experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512/*_1000000_*")
add("FPP TD3", "s1-5 (June)", "experiment_FPP/runs/td3_her_baseline_ls512/*", "experiment_FPP/runs/td3_saber_critic_only_v4_ls512/*")
add("FPP TD3", "s1-10 (n10 artifact)", "experiment_FPP/paper_artifacts/fpp_td3_her_vs_saber_n10_1m/data/td3_her/*",
    "experiment_FPP/paper_artifacts/fpp_td3_her_vs_saber_n10_1m/data/td3_saber/*", root="M")
add("FPP TD3", "s1-5 rerun cpu (2026-11-10)", "ICML2027/experiments/fpp_td3_primary_checkpoint_rerun_20261110/outputs/runs/td3_her_baseline_ls512/*",
    "ICML2027/experiments/fpp_td3_primary_checkpoint_rerun_20261110/outputs/runs/td3_saber_critic_only_v4_ls512/*")
add("FPP TD3", "s1-5 rerun cuda (2026-11-12)", "ICML2027/experiments/fpp_td3_primary_checkpoint_rerun_20261110/outputs/runs_cuda/td3_her_baseline_ls512/*",
    "ICML2027/experiments/fpp_td3_primary_checkpoint_rerun_20261110/outputs/runs_cuda/td3_saber_critic_only_v4_ls512/*")
add("FPP TD3", "s11-30 (n20, 2026-11-14)", "ICML2027/experiments/fpp_td3_primary_checkpoint_rerun_20261110/outputs/runs_n20/td3_her_baseline_ls512/*",
    "ICML2027/experiments/fpp_td3_primary_checkpoint_rerun_20261110/outputs/runs_n20/td3_saber_critic_only_v4_ls512/*")
add("FPP SAC", "s1-5", "experiment_FPP/runs/sac_her_baseline_ls512/*", "experiment_FPP/runs/sac_saber_critic_only_v4_ls512/*")
add("FPP SAC", "s1-5 (paper artifact)", "experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_baseline_ls512/*",
    "experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_saber_critic_only_v4_ls512/*", root="M")
for task, T in (("Open", "DrawerOpen"), ("Close", "DrawerClose")):
    for bk in ("DDPG", "TD3"):
        add(f"{T} {bk}", "s1-5", f"metaworld_Drawer_{task}_{bk}/results_2m/{bk}-HER-B0-2M/*",
            f"metaworld_Drawer_{task}_{bk}/results_2m/{bk}-HER-B3-Critic-Only-2M/*")
for p in ("metaworld_Drawer_Open_DDPG/paper_artifacts/drawer_open_ddpg_her_saber_seed_extension_2m",
          "metaworld_Drawer_Open_DDPG/paper_artifacts/drawer_open_ddpg_main_seed_extension_2m"):
    add("DrawerOpen DDPG", "s6-10 (extension)", p + "/data/extension_6_10/her/*", p + "/data/extension_6_10/saber/*", root="M")
    add("DrawerOpen DDPG", "s1-5", p + "/data/historical_1_5/her/*", p + "/data/historical_1_5/saber/*", root="M")
p = "metaworld_Drawer_Open_TD3/paper_artifacts/drawer_open_td3_main_seed_extension_2m"
add("DrawerOpen TD3", "s6-10 (extension)", p + "/data/extension_6_10/her/*", p + "/data/extension_6_10/saber/*", root="M")
add("DrawerOpen TD3", "s1-5", p + "/data/historical_1_5/her/*", p + "/data/historical_1_5/saber/*", root="M")
p = "metaworld_Drawer_Open_SAC/paper_artifacts/drawer_open_sac_main_seed_extension_2m"
add("DrawerOpen SAC", "s6-10 (extension)", p + "/data/extension_6_10/her/*", p + "/data/extension_6_10/saber/*", root="M")
add("DrawerOpen SAC", "s1-5", p + "/data/historical_1_5/her/*", p + "/data/historical_1_5/saber/*", root="M")
p = "metaworld_Drawer_Open_SAC/paper_artifacts/drawer_open_sac_her_saber_hgr_per_2m"
add("DrawerOpen SAC", "s1-5", p + "/data/her/*", p + "/data/saber/*", root="M")
p = "metaworld_Drawer_Close_DDPG/paper_artifacts/drawer_close_ddpg_her_saber_seed_extension_2m"
add("DrawerClose DDPG", "s6-10 (extension)", p + "/data/extension_6_10/her/*", p + "/data/extension_6_10/saber/*", root="M")
add("DrawerClose DDPG", "s1-5", p + "/data/historical_1_5/her/*", p + "/data/historical_1_5/saber/*", root="M")
p = "metaworld_Drawer_Close_SAC/paper_artifacts/drawer_close_sac_her_saber_hgr_per_2m"
add("DrawerClose SAC", "s1-5", p + "/data/her/*", p + "/data/saber/*", root="M")
CELLS = [f"{t} {b}" for t in ("FPP", "DrawerOpen", "DrawerClose") for b in ("DDPG", "TD3", "SAC")]
# rerun cohorts repeat seeds that already have a first run; they are reported but not pooled
NOT_POOLED = {"s1-5 rerun cpu (2026-11-10)", "s1-5 rerun cuda (2026-11-12)"}


def remote_root() -> tuple[Path, str]:
    sha = subprocess.run(["git", "rev-parse", "origin/main"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    dest = CACHE / f"origin_main_{sha[:12]}"
    if not dest.exists():
        dirs = sorted({s[4].split("/paper_artifacts/")[0] + "/paper_artifacts" for s in S if s[3] == "M"})
        tar = subprocess.run(["git", "archive", "--format=tar", sha, *dirs], cwd=REPO, capture_output=True, check=True).stdout
        dest.mkdir(parents=True)
        tarfile.open(fileobj=io.BytesIO(tar)).extractall(dest)
    return dest, sha


def curve(d: Path) -> dict[int, float]:
    for name, col in (("metrics.csv", "eval/success_mean"), ("metrics_selected_10k.csv", "eval/success_mean"),
                      ("evaluations.csv", "eval/success_mean"), ("evaluation.csv", "success")):
        f = d / name
        if f.exists():
            h = {}
            for r in csv.DictReader(f.open()):
                v = r.get(col, "")
                if v not in ("", None, "nan"):
                    h[int(float(r["step"]))] = float(v)
            if h:
                return h
    return {}


def auc(h, lo, hi):
    ks = sorted(k for k in h if lo <= k <= hi)
    x, y = np.array(ks, float), np.array([h[k] for k in ks])
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def provenance(d: Path) -> dict:
    out = {}
    for name in ("config.json", "summary.json"):
        f = d / name
        if f.exists():
            try:
                c = json.loads(f.read_text())
            except Exception:
                continue
            for k in ("output_root", "run_dir"):
                v = str(c.get(k, ""))
                m = re.match(r"(/home/[^/]+)", v)
                if m and "home" not in out:
                    out["home"] = m.group(1)
                elif v.startswith("<WORKSPACE>") and "home" not in out:
                    out["home"] = "<WORKSPACE>"
            if c.get("device") and "device" not in out:
                out["device"] = c["device"]
    if (d / "launch_record.json").exists():
        out["started"] = json.loads((d / "launch_record.json").read_text()).get("started_utc", "")[:10]
    elif (d / "run.log").exists():
        out["started"] = (d / "run.log").open().readline()[:10]
    if "device" not in out and d.name.endswith("_cuda"):
        out["device"] = "cuda"
    return out


def seed_of(d: Path) -> int | None:
    m = re.search(r"_s(\d+)_", d.name) or re.fullmatch(r"seed_0*(\d+)", d.name) or re.fullmatch(r"(\d+)", d.name)
    return int(m.group(1)) if m else None


def stats(d):
    rng = np.random.default_rng(BOOT_SEED)
    bs = d[rng.integers(0, len(d), (BOOT, len(d)))].mean(1)
    if len(d) <= 22:
        flips = np.array(list(itertools.product([1, -1], repeat=len(d))))
        p = float(np.mean(np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12))
    else:                                                       # Monte Carlo sign flips (exact count is 2^n)
        flips = np.where(np.random.default_rng(BOOT_SEED + 1).random((200_000, len(d))) < 0.5, 1, -1)
        p = float(np.mean(np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12))
    return dict(n=len(d), mean=float(d.mean()), ci_low=float(np.percentile(bs, 2.5)), ci_high=float(np.percentile(bs, 97.5)),
                pos=int((d > 0).sum()), neg=int((d < 0).sum()), sign_flip_p=p)


def main() -> None:
    M, sha = remote_root()
    runs = {}                                                   # (cell, role, seed, curve hash) -> record
    for cell, cohort, role, root, pat in S:
        base = REPO if root == "L" else M
        for d in sorted(Path(x) for x in glob.glob(str(base / pat))):
            if not d.is_dir() or not (d / "summary.json").exists():
                continue
            s = seed_of(d)
            h = curve(d)
            if s is None or not h:
                continue
            hi = max(h)
            if hi < (1_000_000 if cell.startswith("FPP") else 2_000_000):
                continue                                        # shorter pilots
            key = (cell, role, s, hashlib.sha256(json.dumps(sorted(h.items())).encode()).hexdigest()[:16])
            rec = runs.setdefault(key, dict(cell=cell, role=role, seed=s, curve_hash=key[3], steps=hi,
                                            auc=auc(h, LO, hi), n_eval=len(h), cohorts=set(), paths=[], **provenance(d)))
            rec["cohorts"].add(cohort)
            rec["paths"].append(("origin/main:" if root == "M" else "") + str(d.relative_to(base)))
            pv = provenance(d)
            rec.setdefault("prov", set()).add(f"{pv.get('home', '?')}|{pv.get('device', '?')}|{pv.get('started', '?')}")
            for k, v in pv.items():
                rec.setdefault(k, v)
    recs = list(runs.values())
    # cohort membership is the set of labels whose glob found the run; a run's canonical cohort is the
    # first label in S order (copies in paper_artifacts are found by several labels)
    order = list(dict.fromkeys(s[1] for s in S))
    for r in recs:
        r["cohort"] = min(r["cohorts"], key=order.index)
    rows = []
    for cell in CELLS:
        sel = [r for r in recs if r["cell"] == cell]
        if not sel:
            rows.append(dict(cell=cell, cohort="-", n=0, note="no runs found in working tree or origin/main"))
            continue
        cohorts = [c for c in order if any(c in r["cohorts"] for r in sel)]

        def first_runs(role):
            """seed -> earliest non-rerun run; conflicts = seeds with more than one distinct non-rerun run"""
            out, conflicts = {}, []
            for sd in sorted({r["seed"] for r in sel if r["role"] == role}):
                cand = [r for r in sel if r["role"] == role and r["seed"] == sd and not r["cohorts"] <= NOT_POOLED]
                if not cand:
                    continue
                cand.sort(key=lambda r: min(order.index(c) for c in r["cohorts"]))
                out[sd] = cand[0]
                if len(cand) > 1:
                    conflicts.append(sd)
            return out, conflicts
        for coh in cohorts + ["POOLED (first run per seed)"]:
            note = ""
            if coh.startswith("POOLED"):
                H, ch = first_runs("her")
                Sb, cs = first_runs("saber")
                if ch or cs:
                    note = f"distinct non-rerun runs for the same seed: HER {ch} SABER {cs} (earliest used)"
            else:
                H = {r["seed"]: r for r in sel if r["role"] == "her" and coh in r["cohorts"]}
                Sb = {r["seed"]: r for r in sel if r["role"] == "saber" and coh in r["cohorts"]}
            pairs = [(H[k], Sb[k]) for k in sorted(set(H) & set(Sb))]
            unpaired = sorted(set(H) ^ set(Sb))
            if unpaired:
                note = (note + "; " if note else "") + f"unpaired seeds {unpaired}"
            if not pairs:
                rows.append(dict(cell=cell, cohort=coh, n=0, note="no HER/SABER pair; " + note))
                continue
            d = np.array([b["auc"] - a["auc"] for a, b in pairs])
            prov = sorted({x for pr in pairs for p in pr for x in p["prov"]})
            rows.append(dict(cell=cell, cohort=coh, seeds=",".join(str(a["seed"]) for a, _ in pairs),
                             her_auc=float(np.mean([a["auc"] for a, _ in pairs])), saber_auc=float(np.mean([b["auc"] for _, b in pairs])),
                             **stats(d), provenance=" ; ".join(prov), note=note))
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / "pairs_by_cohort.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with (OUT / "runs.csv").open("w", newline="") as f:
        fields = ["cell", "role", "seed", "cohort", "auc", "steps", "n_eval", "prov", "curve_hash", "paths"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(recs, key=lambda r: (CELLS.index(r["cell"]), r["role"], r["seed"])):
            w.writerow({**r, "prov": " ; ".join(sorted(r["prov"])), "paths": " | ".join(r["paths"])})
    (OUT / "source.json").write_text(json.dumps(dict(origin_main=sha, rule=f"eval/success_mean trapezoid AUC over [{LO}, total]",
                                                     bootstrap=BOOT, bootstrap_seed=BOOT_SEED), indent=2) + "\n")
    for r in rows:
        if r.get("n", 0):
            print(f"{r['cell']:17s} {r['cohort']:30s} n={r['n']:2d} HER {r['her_auc']:.3f} SABER {r['saber_auc']:.3f} "
                  f"d {r['mean']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}] {r['pos']}/{r['neg']} p={r['sign_flip_p']:.3f} {r['note']}")
        else:
            print(f"{r['cell']:17s} {r['cohort']:30s} {r['note']}")


if __name__ == "__main__":
    main()
