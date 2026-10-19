"""Require the complete fixed checkpoint grid; never silently shorten AUC."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from protocol import ROOT, SPEC, make_config


def auc(curve):
    if len(curve) < 2:
        raise ValueError("AUC needs at least two checkpoints")
    return sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(curve, curve[1:])) / (curve[-1][0] - curve[0][0])


def read_curve(path, expected_steps):
    curve = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            value = row.get("eval/env_success_mean", "")
            if value == "":
                continue
            point = (int(row["step"]), float(value))
            if not math.isfinite(point[1]) or not 0 <= point[1] <= 1:
                raise ValueError(f"Invalid evaluation: {point}")
            curve.append(point)
    if [x for x, _ in curve] != list(expected_steps):
        raise ValueError("Missing, duplicated, reordered, or unexpected evaluation checkpoints")
    return curve


def validate_run(run_dir, config):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    recorded = json.loads((run_dir / "config.json").read_text())
    for key, value in json.loads(json.dumps(config.as_dict())).items():
        if recorded.get(key) != value:
            raise ValueError(f"Run config mismatch: {key}")
    if summary["total_steps"] != config.total_steps:
        raise ValueError("Incomplete training")
    expected_updates = max(0, config.total_steps - max(config.batch_size, config.warmup_steps) + 1) * config.updates_per_step
    if summary["total_updates"] != expected_updates:
        raise ValueError("Unexpected number of gradient updates")
    if any(not math.isfinite(float(x)) for x in summary["final_losses"].values()):
        raise ValueError("Non-finite final loss")
    steps = range(config.eval_freq, config.total_steps + 1, config.eval_freq)
    curve = read_curve(run_dir / "metrics.csv", steps)
    if summary["final_eval_env_success"] != curve[-1][1]:
        raise ValueError("Summary/final evaluation disagreement")
    full_auc = auc(curve)
    if not math.isclose(full_auc, summary["eval_AUC_success"], abs_tol=1e-12):
        raise ValueError("Online/offline AUC disagreement")
    fq = [p for p in curve if p[0] >= config.total_steps * 0.75]
    hit = next((step for step, value in curve if value >= 0.9), None)
    return {"auc": full_auc, "fq_auc": auc(fq) if len(fq) >= 2 else None,
            "final": curve[-1][1], "hit90": hit, "updates": expected_updates,
            "runtime_seconds": summary["runtime_total_sec"],
            "steps_per_second": summary["final_steps_per_sec"]}


def paired_intervals(differences):
    import numpy as np
    d = np.asarray(differences, dtype=np.float64)
    n = len(SPEC["seeds"])
    if d.shape != (n,) or not np.isfinite(d).all():
        raise ValueError(f"The declared comparison requires all {n} paired seeds")
    rng = np.random.default_rng(SPEC["statistics"]["bootstrap_seed"])
    ids = rng.integers(0, n, size=(SPEC["statistics"]["bootstrap_resamples"], n))
    means = d[ids].mean(axis=1)
    return {"n_pairs": n, "mean": float(d.mean()), "positive_pairs": int((d > 0).sum()),
            "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
            "ci97_5": np.quantile(means, [0.0125, 0.9875]).tolist(),
            "ci98_75": np.quantile(means, [0.00625, 0.99375]).tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "analysis/final.json")
    p.add_argument("--tasks", nargs="+", choices=SPEC["tasks"], default=SPEC["tasks"])
    args = p.parse_args()
    if len(args.tasks) != len(set(args.tasks)):
        p.error("Tasks must be unique")
    runs, incomplete = [], []
    for task in args.tasks:
        for method in SPEC["methods"]:
            for seed in SPEC["seeds"]:
                cfg = make_config(task, method, seed)
                claim = ROOT / ".production_claims" / cfg.run_name / "status.json"
                try:
                    status = json.loads(claim.read_text())
                    if status["status"] != "completed":
                        raise ValueError(f"status={status['status']}")
                    result = validate_run(ROOT / "runs" / cfg.run_name, cfg)
                    runs.append({"task": task, "method": method, "seed": seed, **result})
                except (OSError, ValueError, KeyError) as exc:
                    incomplete.append({"run": cfg.run_name, "reason": str(exc)})
    payload = {"tasks": args.tasks, "seeds": SPEC["seeds"], "cohort_stage": SPEC["cohort_stage"],
               "automatic_seed_extension": False, "inference_scope": "exploratory_five_seed_baseline_extension",
               "runs": runs, "incomplete": incomplete, "status": "incomplete" if incomplete else "completed"}
    if not incomplete:
        from dependency import read_references, predecessor_state
        if predecessor_state()["state"] != "ready":
            raise RuntimeError("Reference cohort is not finalized")
        runs += [r for r in read_references() if r["task"] in args.tasks]
        comparisons, aggregates = {}, []
        for task in args.tasks:
            by_method = {m: sorted([r for r in runs if r["task"] == task and r["method"] == m], key=lambda r: r["seed"])
                         for m in ("her", "saber", "hgr", "per")}
            comparisons[task] = {}
            for control in ("hgr", "per", "her"):
                comparisons[task][f"saber-{control}"] = paired_intervals([
                    s["auc"] - c["auc"] for s, c in zip(by_method["saber"], by_method[control])])
            for method, rows in by_method.items():
                aggregates.append({"task": task, "method": method,
                    **{f"{metric}_{stat}": fn([r[metric] for r in rows])
                       for metric in ("auc", "fq_auc", "final", "runtime_seconds", "steps_per_second")
                       for stat, fn in (("mean", statistics.mean), ("sd", statistics.stdev))},
                    "hit90_count": sum(r["hit90"] is not None for r in rows)})
        payload.update(comparisons=comparisons, aggregates=aggregates,
            primary_contrasts=SPEC["statistics"]["primary_contrasts"],
            simultaneous_interval="98.75% per contrast for four primary contrasts; approximate bootstrap coverage",
            reference_source_manifest_sha256=SPEC["amendment"]["parent_source_manifest_sha256"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    expected = len(args.tasks) * 4 * len(SPEC["seeds"])
    print(f"{len(runs)}/{expected} validated, {len(incomplete)} missing/failed; {args.output}")
    if incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
