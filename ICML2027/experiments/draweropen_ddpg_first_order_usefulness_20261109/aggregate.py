#!/usr/bin/env python3
"""Summarise measured checkpoints and apply the pre-registered hypothesis rule.

  .venv/bin/python <this dir>/aggregate.py --out outputs/main            (set a1: early phase)
  .venv/bin/python <this dir>/aggregate.py --out outputs/late_ext2m      (2M, seeds 11-20)
  .venv/bin/python <this dir>/aggregate.py --out outputs/late_mature     (seed 1, 500k-2M, descriptive)

The rule (a1 | ext2m | mature) is read from <out>/run_environment.json.
Writes <out>/analysis/{checkpoints.csv, hypotheses.csv, REPORT_ko.md}. Refuses to overwrite.
Sign convention: every gain is R(after uniform) - R(after arm); positive = arm reduced the
policy-query error more than uniform (HER) weights did.
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

BANKS = ("R1", "R2")


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]


def _num(v: str):
    if v in ("True", "False"):
        return v == "True"
    try:
        return float(v)
    except ValueError:
        return v


def nanmean(x) -> float:
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], float)
    return float(x.mean()) if x.size else float("nan")


def stratified_gap(values: np.ndarray, in_a: np.ndarray, in_b: np.ndarray, strata_by: np.ndarray, pool: np.ndarray,
                   n_strata: int = 4) -> float:
    """Mean over strata (quartiles of strata_by within pool) of mean(values|a) - mean(values|b)."""
    if pool.sum() < n_strata * 2:
        return float("nan")
    edges = np.quantile(strata_by[pool], np.linspace(0, 1, n_strata + 1))
    gaps = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = pool & (strata_by >= lo) & (strata_by <= hi)
        a, b = s & in_a, s & in_b
        if a.any() and b.any():
            gaps.append(values[a].mean() - values[b].mean())
    return float(np.mean(gaps)) if gaps else float("nan")


def summarise_checkpoint(d: Path) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    rows = [r for r in read_csv(d / "batches.csv")]
    live = [r for r in rows if not r["degenerate"]]
    ps = np.load(d / "per_sample.npz")
    last_eval = float(meta.get("last_eval_success", float("nan")))
    outcome = ("success" if last_eval >= P.SUCCESS_MIN else "failed" if last_eval <= P.FAILURE_MAX
               else "partial" if not math.isnan(last_eval) else "unknown")
    out = dict(set=meta.get("set", "a1"), trained_arm=meta["trained_arm"], seed=meta["seed"], stage=meta["step"],
               last_eval_success=last_eval, outcome=outcome,
               n_batches=len(rows), n_live=len(live), p=nanmean(r["p"] for r in rows),
               clip_rate_uniform=nanmean(float(r["clipped_uniform"]) for r in rows),
               clip_rate_saber=nanmean(float(r["clipped_saber"]) for r in rows))
    K = read_csv(d / "ksteps.csv") if (d / "ksteps.csv").exists() else []
    for bank in BANKS:
        if bank not in meta["banks"]:
            continue
        R0 = meta["banks"][bank]["R0"]
        out[f"{bank}_R0"] = R0
        out[f"{bank}_rms"] = math.sqrt(max(R0, 0.0))
        out[f"{bank}_null"] = bool(math.sqrt(max(R0, 0.0)) < P.FLOOR_RMS)
        out[f"{bank}_phase_frac"] = "/".join(str(x) for x in meta["banks"][bank]["phase_frac"])
        out[f"{bank}_common_change"] = nanmean(r[f"{bank}_A1_common_change"] for r in live)
        for arm in ("saber", "random", "td_topk", "inverse"):
            for geo in ("G1", "G2", "G3", "G3mid", "A1"):
                out[f"{bank}_{geo}_{arm}"] = nanmean(r[f"{bank}_{geo}_{arm}"] for r in live)
        # H4: does each predictor get the sign of the real one-step SABER-vs-uniform gain right?
        for geo in ("G1", "G2", "G3", "G3mid"):
            agree = [np.sign(r[f"{bank}_{geo}_saber"]) == np.sign(r[f"{bank}_A1_saber"]) for r in live]
            out[f"{bank}_agree_{geo}"] = float(np.mean(agree)) if agree else float("nan")
        for kr in K:
            k = int(kr["K"])
            for arm in ("saber", "random", "td_topk", "inverse"):
                out[f"{bank}_K{k}_{arm}"] = kr[f"{bank}_K_{arm}"]
            out[f"{bank}_K{k}_R_uniform"] = kr[f"{bank}_R_uniform"]
        # decomposition from per-sample arrays, batch by batch
        u, cos = ps[f"u_{bank}"], ps[f"cos_{bank}"]
        m, e = ps["support"].astype(bool), ps["eligible"].astype(bool)
        gaps_u, gaps_cos, gaps_cos_d = [], [], []
        for b in range(u.shape[0]):
            mb, eb = m[b], e[b]
            if not (0 < mb.sum() < mb.size):
                continue
            scale = np.abs(u[b]).mean()
            gaps_u.append((u[b][mb].mean() - u[b][~mb].mean()) / scale)
            gaps_cos.append(cos[b][mb].mean() - cos[b][~mb].mean())
            gaps_cos_d.append(stratified_gap(cos[b], mb, ~mb, np.abs(ps["delta"][b]), np.ones_like(mb)))
        out[f"{bank}_u_gap_norm"] = nanmean(gaps_u)                # (u+ - u-) / mean|u|
        out[f"{bank}_cos_gap"] = nanmean(gaps_cos)                  # direction only
        out[f"{bank}_cos_gap_by_absdelta"] = nanmean(gaps_cos_d)    # direction, |delta| quartiles
    kap, dist, m, e = ps["kappa"], ps["dist"], ps["support"].astype(bool), ps["eligible"].astype(bool)
    gk = []
    for b in range(kap.shape[0]):
        mb, eb = m[b], e[b]
        if mb.any() and (eb & ~mb).any():
            gk.append(stratified_gap(kap[b], mb, eb & ~mb, dist[b], eb))
    out["kappa_gap_by_dist"] = nanmean(gk)                          # H3, within E, matched on |a^r - a^pi|
    out["kappa_M"] = float(kap[m].mean()) if m.any() else float("nan")
    out["kappa_E_not_M"] = float(kap[e & ~m].mean()) if (e & ~m).any() else float("nan")
    out["self_support_share"] = float(ps["self_support"][m].mean()) if m.any() else float("nan")
    return out


def hypotheses(ck: list[dict], seeds_expected=None, stages_expected=None, min_stages=None) -> list[dict]:
    seeds_expected = tuple(P.SEEDS if seeds_expected is None else seeds_expected)
    stages_expected = tuple(P.STAGES if stages_expected is None else stages_expected)
    min_stages = P.CONSISTENT_MIN_STAGES if min_stages is None else min_stages
    tests = {
        "H1 SABER > uniform (G3)": lambda r, b: r[f"{b}_G3_saber"],
        "H1 SABER > uniform (real 1-step)": lambda r, b: r[f"{b}_A1_saber"],
        "H1 SABER > uniform (K=100)": lambda r, b: r.get(f"{b}_K100_saber", float("nan")),
        "H2 SABER > Random (G3)": lambda r, b: r[f"{b}_G3_saber"] - r[f"{b}_G3_random"],
        "H2 SABER > TD top-k (G3)": lambda r, b: r[f"{b}_G3_saber"] - r[f"{b}_G3_td_topk"],
        "H2 SABER > Random (K=100)": lambda r, b: r.get(f"{b}_K100_saber", float("nan")) - r.get(f"{b}_K100_random", float("nan")),
        "H3 kappa(M) > kappa(E\\M), dist-matched": lambda r, b: r["kappa_gap_by_dist"],
        "dir: cos(M) > cos(not M), abs(delta)-matched": lambda r, b: r[f"{b}_cos_gap_by_absdelta"],
        "H4 G3 sign agreement - 0.8": lambda r, b: r[f"{b}_agree_G3"] - P.H4_MIN_AGREEMENT,
    }
    out = []
    for arm in sorted({r["trained_arm"] for r in ck}):
        for bank in BANKS:
            rows = [r for r in ck if r["trained_arm"] == arm and f"{bank}_R0" in r]
            if not rows:
                continue
            seeds = sorted({r["seed"] for r in rows})
            for name, f in tests.items():
                per_seed = {}
                n_rows = {}
                for s in seeds:
                    mine = [r for r in rows if r["seed"] == s]
                    n_rows[s] = len(mine)
                    # numerically null checkpoints and checkpoints where SABER equals uniform are ties
                    vals = [f(r, bank) for r in mine if not r[f"{bank}_null"] and r["n_live"] > 0]
                    vals = [v for v in vals if not math.isnan(v)]
                    per_seed[s] = (sum(v > 0 for v in vals), sum(v < 0 for v in vals), len(vals))
                full = len(seeds) == len(seeds_expected) and all(n_rows[s] == len(stages_expected) for s in seeds)
                pos = all(p_ >= min_stages for p_, _, _ in per_seed.values())
                neg = all(n_ >= min_stages for _, n_, _ in per_seed.values())
                verdict = ("consistent +" if pos else "consistent -" if neg else "mixed") if full else \
                    ("incomplete: " + ("+" if pos else "-" if neg else "mixed") + " so far")
                out.append(dict(trained_arm=arm, bank=bank, test=name, verdict=verdict,
                                **{f"seed_{s}(+/-/n)": f"{a}/{b}/{c}" for s, (a, b, c) in per_seed.items()}))
    return out


LATE_TESTS = {
    "H1 SABER > uniform (K=100)": lambda r, b: r.get(f"{b}_K100_saber", float("nan")),
    "H1 SABER > uniform (K=10)": lambda r, b: r.get(f"{b}_K10_saber", float("nan")),
    "H1 SABER > uniform (real 1-step)": lambda r, b: r[f"{b}_A1_saber"],
    "H1 SABER > uniform (G3 prediction)": lambda r, b: r[f"{b}_G3_saber"],
    "H2 SABER > Random (K=100)": lambda r, b: r.get(f"{b}_K100_saber", float("nan")) - r.get(f"{b}_K100_random", float("nan")),
    "H2 SABER > TD top-k (K=100)": lambda r, b: r.get(f"{b}_K100_saber", float("nan")) - r.get(f"{b}_K100_td_topk", float("nan")),
    "inverse (psi=0 upweighted) > uniform (K=100)": lambda r, b: r.get(f"{b}_K100_inverse", float("nan")),
    "H3 kappa(M) > kappa(E\\M), dist-matched": lambda r, b: r["kappa_gap_by_dist"],
    "dir: cos(M) > cos(not M), abs(delta)-matched": lambda r, b: r[f"{b}_cos_gap_by_absdelta"],
}
LATE_STRATA = {
    "success, all arms": lambda r: r["outcome"] == "success",
    "success, SABER-trained": lambda r: r["outcome"] == "success" and r["trained_arm"] == "saber",
    "success, Random-trained": lambda r: r["outcome"] == "success" and r["trained_arm"] == "random",
    "success, TD-trained": lambda r: r["outcome"] == "success" and r["trained_arm"] == "td_topk",
    "failed, all arms": lambda r: r["outcome"] == "failed",
}


def sign_test(n_pos: int, n_neg: int) -> float:
    """Exact one-sided binomial p-value for the majority side (ties already removed)."""
    n, x = n_pos + n_neg, max(n_pos, n_neg)
    return sum(math.comb(n, k) for k in range(x, n + 1)) / 2 ** n if n else float("nan")


def late_verdicts(ck: list[dict]) -> list[dict]:
    out = []
    for bank in BANKS:
        rows = [r for r in ck if f"{bank}_R0" in r]
        for sname, sel in LATE_STRATA.items():
            runs = [r for r in rows if sel(r)]
            for tname, f in LATE_TESTS.items():
                pos = neg = ties = 0
                for r in runs:
                    v = f(r, bank)
                    if r[f"{bank}_null"] or r["n_live"] == 0 or math.isnan(v) or v == 0:
                        ties += 1
                    elif v > 0:
                        pos += 1
                    else:
                        neg += 1
                pval = sign_test(pos, neg)
                if pos + neg == 0:
                    verdict = "no counted runs"
                elif pval <= P.LATE_ALPHA:
                    verdict = "consistent +" if pos > neg else "consistent -"
                else:
                    verdict = "mixed"
                out.append(dict(bank=bank, stratum=sname, test=tname, verdict=verdict, runs=len(runs),
                                positive=pos, negative=neg, ties=ties,
                                sign_test_p=round(pval, 4) if pos + neg else float("nan")))
    return out


def rel(r: dict, key: str, bank: str) -> str:
    v, R0 = r.get(key, float("nan")), r[f"{bank}_R0"]
    if v is None or (isinstance(v, float) and math.isnan(v)) or R0 <= 0:
        return "–"
    return f"{100.0 * v / R0:+.2f}%"


def report_late(ck: list[dict], hy: list[dict], rule: str) -> str:
    arms = ", ".join(f"{a} {sum(r['trained_arm'] == a for r in ck)}" for a in sorted({r["trained_arm"] for r in ck}))
    lines = ["# 1차 유용성 측정 보고서: 후반부", "",
             "설계와 판정 규칙: 이 폴더의 `PROTOCOL_late_ko.md`. "
             "부호 규칙: 모든 gain은 `R(uniform 갱신 후) − R(arm 갱신 후)`이며 양수면 그 arm이 정책 조회 오차를 더 줄였다. "
             "표의 gain은 R0 대비 백분율이다.", "",
             f"측정한 checkpoint: {len(ck)}개 (학습 arm별: {arms}). "
             f"실행 결과: success {sum(r['outcome'] == 'success' for r in ck)}, failed {sum(r['outcome'] == 'failed' for r in ck)}, "
             f"partial {sum(r['outcome'] == 'partial' for r in ck)}.", ""]
    if rule == "ext2m":
        lines += ["## 사전 고정 판정", "",
                  f"단위는 실행(arm, seed) 하나다. 정확 단측 부호 검정 p ≤ {P.LATE_ALPHA}이면 consistent. "
                  "수치적으로 null이거나 모든 배치에서 p = 0 또는 1이라 두 가중이 같은 실행은 tie로 세지 않는다.", "",
                  "| R | 층 | 검사 | 판정 | + | − | tie | p |", "|---|---|---|---|---|---|---|---|"]
        for h in hy:
            lines.append(f"| {h['bank']} | {h['stratum']} | {h['test']} | {h['verdict']} | {h['positive']} | {h['negative']} | "
                         f"{h['ties']} | {h['sign_test_p'] if not (isinstance(h['sign_test_p'], float) and math.isnan(h['sign_test_p'])) else '–'} |")
    else:
        lines += ["seed 1개의 시점별 기록이라 판정하지 않는다. 추세만 본다.", ""]
    for bank in BANKS:
        lines += ["", f"## Checkpoint별 값 ({bank}, gain / R0)", "",
                  "| arm | seed | step | 마지막 평가 | p | live | clip | √R0 | 1-step SABER | K=10 SABER | K=100 SABER | K=100 Random | K=100 TD | K=100 역방향 | κ gap | G3 부호 일치 | query phase |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(ck, key=lambda r: (r["trained_arm"], r["seed"], r["stage"])):
            if f"{bank}_R0" not in r:
                continue
            null = " (null)" if r[f"{bank}_null"] else ""
            agree = r[f"{bank}_agree_G3"]
            lines.append("| " + " | ".join([
                r["trained_arm"], str(int(r["seed"])), f"{int(r['stage']):,}", f"{r['last_eval_success']:.2f}",
                f"{r['p']:.3f}", f"{int(r['n_live'])}/{int(r['n_batches'])}", f"{r['clip_rate_saber']:.2f}",
                f"{r[f'{bank}_rms']:.2e}{null}", rel(r, f"{bank}_A1_saber", bank), rel(r, f"{bank}_K10_saber", bank),
                rel(r, f"{bank}_K100_saber", bank), rel(r, f"{bank}_K100_random", bank), rel(r, f"{bank}_K100_td_topk", bank),
                rel(r, f"{bank}_K100_inverse", bank), fmt(r["kappa_gap_by_dist"]),
                "–" if math.isnan(agree) else f"{agree:.2f}", r[f"{bank}_phase_frac"]]) + " |")
    lines += ["", "## 읽을 때 주의", "",
              "- 독립 단위는 학습 실행이다. 같은 seed의 세 arm은 초기화와 첫 10k 무작위 행동을 공유한다.",
              "- Random·TD로 학습한 checkpoint의 ψ는 'SABER였다면 무엇을 키웠을까'라는 반사실 값이다.",
              "- live는 0 < p < 1인 배치 수다. 0이면 SABER 가중이 uniform과 같아 모든 gain이 0이다.",
              "- (null)은 정책 조회 RMS residual이 float32 한계 아래라 부호를 세지 않은 checkpoint다.",
              "- 고정 target 기준이고 Q^π 정확도가 아니다. 값 오차 개선은 ∇_a Q 개선을 뜻하지 않는다.",
              "- actor와 target을 고정하고 critic만 갱신한 결과다."]
    return "\n".join(lines) + "\n"


def report_fpp(ck: list[dict], hy: list[dict], rule: str) -> str:
    runs = sorted({(r["trained_arm"], int(r["seed"])) for r in ck})
    lines = [f"# 1차 유용성 측정 보고서: FetchPickAndPlace TD3 ({rule})", "",
             "설계와 판정 규칙: 이 폴더의 `PROTOCOL_fpp_ko.md`. "
             "부호 규칙: 모든 gain은 `R(uniform 갱신 후) − R(arm 갱신 후)`이며 양수면 그 arm이 정책 조회 오차를 더 줄였다. "
             "R은 actor가 읽는 critic1의 정책 조회 오차다. checkpoint 표의 gain은 R0 대비 백분율이다.", "",
             f"측정한 checkpoint: {len(ck)}개, 실행 {len(runs)}개 "
             f"(HER 학습 {sum(a == 'her' for a, _ in runs)}, SABER 학습 {sum(a == 'saber' for a, _ in runs)}).", "",
             "## 사전 고정 가설 판정", "",
             f"규칙: seed마다 {len(P.FPP_STAGES)}개 시점 중 {P.FPP_CONSISTENT_MIN_STAGES}개 이상이 같은 부호이고 이것이 "
             f"{len(P.FPP_SETS[rule]['seeds'])}개 seed 모두에서 성립하면 consistent. 칸의 숫자는 양수/음수/센 시점 수다.", ""]
    seed_cols = sorted({k for h in hy for k in h if k.startswith("seed_")}, key=lambda c: int(c.split("_")[1].split("(")[0]))
    lines.append("| 학습 arm | R | 검사 | 판정 | " + " | ".join(seed_cols) + " |")
    lines.append("|---|---|---|---|" + "---|" * len(seed_cols))
    for h in hy:
        lines.append(f"| {h['trained_arm']} | {h['bank']} | {h['test']} | {h['verdict']} | "
                     + " | ".join(h.get(c, "–") for c in seed_cols) + " |")
    for bank in BANKS:
        lines += ["", f"## Checkpoint별 값 ({bank}, gain / R0)", "",
                  "| arm | seed | step | 평가 성공률 | p | clip | √R0 | G3 SABER | 1-step SABER | K=10 SABER | K=100 SABER | K=100 Random | K=100 TD | K=100 역방향 | κ gap | G3 부호 일치 | query phase |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(ck, key=lambda r: (r["trained_arm"], r["seed"], r["stage"])):
            if f"{bank}_R0" not in r:
                continue
            agree = r[f"{bank}_agree_G3"]
            lines.append("| " + " | ".join([
                r["trained_arm"], str(int(r["seed"])), f"{int(r['stage']):,}", f"{r['last_eval_success']:.2f}",
                f"{r['p']:.3f}", f"{r['clip_rate_saber']:.2f}", f"{r[f'{bank}_rms']:.2e}{' (null)' if r[f'{bank}_null'] else ''}",
                rel(r, f"{bank}_G3_saber", bank), rel(r, f"{bank}_A1_saber", bank), rel(r, f"{bank}_K10_saber", bank),
                rel(r, f"{bank}_K100_saber", bank), rel(r, f"{bank}_K100_random", bank), rel(r, f"{bank}_K100_td_topk", bank),
                rel(r, f"{bank}_K100_inverse", bank), fmt(r["kappa_gap_by_dist"]),
                "–" if math.isnan(agree) else f"{agree:.2f}", r[f"{bank}_phase_frac"]]) + " |")
    lines += ["", "## 읽을 때 주의", "",
              "- 독립 단위는 학습 seed다. 배치와 query, 한 실행의 여러 시점은 그 안의 반복이다.",
              "- HER로 학습한 checkpoint의 ψ는 'SABER였다면 무엇을 키웠을까'라는 반사실 값이다.",
              "- query phase 열은 approach/grasp/lift/place 개수다.",
              "- 고정 target 기준이고 Q^π 정확도가 아니다. 값 오차 개선은 ∇_a Q 개선을 뜻하지 않는다.",
              "- actor와 target을 고정하고 critic 두 개만 갱신한 결과다. TD3의 지연된 actor 갱신은 포함하지 않는다."]
    return "\n".join(lines) + "\n"


def fmt(x, nd=3) -> str:
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if abs(x) != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4):
        return f"{x:+.2e}"
    return f"{x:+.{nd}f}"


def report(ck: list[dict], hy: list[dict], out_dir: Path) -> str:
    lines = ["# 1차 유용성 측정 보고서", "",
             "설계: `ICML2027/experment_design/04_first_order_usefulness/README_ko.md`. "
             "부호 규칙: 모든 gain은 `R(uniform 갱신 후) − R(arm 갱신 후)`이며 양수면 그 arm이 정책 조회 오차를 더 줄였다.", "",
             f"측정한 checkpoint: {len(ck)}개 "
             f"({', '.join(sorted({r['trained_arm'] + ' seed ' + str(int(r['seed'])) for r in ck}))}).", "",
             "## 사전 고정 가설 판정", "",
             f"규칙: seed마다 {len(P.STAGES)}개 시점 중 {P.CONSISTENT_MIN_STAGES}개 이상이 같은 부호이고 이것이 세 seed 모두에서 성립하면 consistent. "
             "seed나 시점이 빠져 있으면 incomplete.", ""]
    seed_cols = sorted({k for h in hy for k in h if k.startswith("seed_")})
    lines.append("| 학습 arm | R | 검사 | 판정 | " + " | ".join(seed_cols) + " |")
    lines.append("|---|---|---|---|" + "---|" * len(seed_cols))
    for h in hy:
        lines.append(f"| {h['trained_arm']} | {h['bank']} | {h['test']} | {h['verdict']} | "
                     + " | ".join(h.get(c, "–") for c in seed_cols) + " |")
    for bank in BANKS:
        lines += ["", f"## Checkpoint별 값 ({bank})", "",
                  "| arm | seed | step | p | clip | R0 | 공통 1-step 변화 | G3 SABER | 실제 1-step SABER | K=100 SABER | K=100 Random | (ū₊−ū₋)/mean\\|u\\| | κ gap | G3 부호 일치 | query phase |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(ck, key=lambda r: (r["trained_arm"], r["seed"], r["stage"])):
            if f"{bank}_R0" not in r:
                continue
            lines.append("| " + " | ".join([r["trained_arm"], str(int(r["seed"])), f"{int(r['stage']):,}",
                         f"{r['p']:.3f}", f"{r['clip_rate_saber']:.2f}", f"{r[f'{bank}_R0']:.4f}",
                         fmt(r[f"{bank}_common_change"]), fmt(r[f"{bank}_G3_saber"]), fmt(r[f"{bank}_A1_saber"]),
                         fmt(r.get(f"{bank}_K100_saber", float('nan'))), fmt(r.get(f"{bank}_K100_random", float('nan'))),
                         fmt(r[f"{bank}_u_gap_norm"]), fmt(r["kappa_gap_by_dist"]), f"{r[f'{bank}_agree_G3']:.2f}",
                         r[f"{bank}_phase_frac"]]) + " |")
    lines += ["", "## 읽을 때 주의", "",
              "- 배치(64개)와 query는 checkpoint 안의 반복이다. 독립 표본은 학습 seed뿐이다.",
              "- R1은 노이즈 없는 정책의 원래-goal 상태, R2는 behavior 상태 + HER goal이다. 둘 다 고정 target 기준이고 Q^π 정확도가 아니다.",
              "- query phase 열은 approach/contact/opening/open 개수다. 한 phase에 몰려 있으면 그 상황에 대한 결론이다.",
              "- 값 오차 개선은 ∇_a Q 개선을 뜻하지 않는다.",
              "- A1은 3-seed pilot이다. consistent 판정도 확증이 아니다."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="analysis")
    a = ap.parse_args()
    root = Path(a.out)
    if not root.is_absolute():
        root = HERE / root
    env = root / "run_environment.json"
    rule = json.loads(env.read_text()).get("set", "a1") if env.exists() else "a1"
    dirs = sorted(p.parent for p in root.glob("*/seed_*/step_*/meta.json"))
    ck = [summarise_checkpoint(d) for d in dirs]
    if rule == "a1":
        hy = hypotheses(ck)
    elif rule in P.FPP_SETS:
        hy = hypotheses(ck, P.FPP_SETS[rule]["seeds"], P.FPP_STAGES, P.FPP_CONSISTENT_MIN_STAGES)
    else:
        hy = late_verdicts(ck) if rule == "ext2m" else []
    dest = root / a.name
    dest.mkdir(exist_ok=False)
    for name, rows in (("checkpoints.csv", ck), ("hypotheses.csv", hy)):
        if not rows:
            continue
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with (dest / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    text = report(ck, hy, dest) if rule == "a1" else report_fpp(ck, hy, rule) if rule in P.FPP_SETS else report_late(ck, hy, rule)
    (dest / "REPORT_ko.md").write_text(text)
    print(f"{len(ck)} checkpoints ({rule}) -> {dest}")


if __name__ == "__main__":
    main()
