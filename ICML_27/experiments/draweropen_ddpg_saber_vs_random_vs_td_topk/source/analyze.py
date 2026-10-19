"""Strict completion checks and joint seed-row percentile bootstrap."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from protocol import (ARMS, SEEDS, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED,
                      ANALYSIS_VERSION, make_config)


def read_json(path):
    return json.loads(path.read_text())


def evaluation_curve(path, grid):
    values, steps = [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            value = row.get("eval/env_success_mean", "")
            if value == "":
                continue
            steps.append(int(row["step"]))
            values.append(float(value))
    if steps != list(grid):
        raise ValueError("Evaluation grid missing, duplicated, reordered, or extended")
    y = np.asarray(values, dtype=np.float64)
    if not np.isfinite(y).all() or np.any(y < 0) or np.any(y > 1):
        raise ValueError("Invalid evaluation success values")
    return y


def curve_metrics(grid, success, fq_start):
    x, y = np.asarray(grid), np.asarray(success)
    if not np.isfinite(y).all() or len(x) != len(y) or np.any(np.diff(x) <= 0):
        raise ValueError("Invalid curve")
    if fq_start not in x:
        raise ValueError("Missing final-quarter boundary")
    late = x >= fq_start
    hit = x[y >= 0.9]
    return {"auc": float(np.trapezoid(y, x) / (x[-1] - x[0])),
            "fq_auc": float(np.trapezoid(y[late], x[late]) / (x[-1] - fq_start)),
            "final_success": float(y[-1]),
            "hit90": int(hit[0]) if len(hit) else None}


def validate_assignment_trace(path, total_updates):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    all_rows = [r for r in rows if r["subset"] == "all"]
    info_rows = [r for r in rows if r["subset"] == "informative"]
    if len(rows) != 2 * len(all_rows) or len(info_rows) != len(all_rows):
        raise ValueError("Missing diagnostic subset/window")
    last = 0
    sums, counts = {"all": {}, "informative": {}}, {"all": {}, "informative": {}}
    for a, i in zip(all_rows, info_rows):
        if (int(a["start_update"]) != last + 1 or
                a["start_update"] != i["start_update"] or a["end_update"] != i["end_update"]):
            raise ValueError("Noncontiguous assignment windows")
        last = int(a["end_update"])
        if int(a["n_updates"]) != last - int(a["start_update"]) + 1:
            raise ValueError("Wrong diagnostic denominator")
        expected_info = round(float(a["informative_update_mean"]) * int(a["n_updates"]))
        if int(i["n_updates"]) != expected_info:
            raise ValueError("Informative-update count mismatch")
        for row in (a, i):
            group = row["subset"]
            for key, val in row.items():
                if not key.endswith("_mean"):
                    continue
                metric = key[:-5]
                n = int(row[metric + "_n"])
                if n < 0 or n > int(row["n_updates"]) or ((val == "") != (n == 0)):
                    raise ValueError("Invalid NA or valid-count diagnostic")
                if n:
                    value = float(val)
                    if not np.isfinite(value):
                        raise ValueError("Non-finite assignment diagnostic")
                    sums[group][metric] = sums[group].get(metric, 0) + value * n
                    counts[group][metric] = counts[group].get(metric, 0) + n
    if last != total_updates:
        raise ValueError("Incomplete assignment diagnostic trace")
    keys = [k[:-5] for k in all_rows[0] if k.endswith("_mean")]
    return {g: {k: {"mean": sums[g][k] / counts[g][k] if counts[g].get(k, 0) else None,
                    "valid_updates": counts[g].get(k, 0)} for k in keys} for g in sums}


def validate_run(attempt, profile="production", require_status=True):
    status = read_json(attempt / "status.json")
    if require_status and status["status"] != "complete":
        raise ValueError("Run not complete")
    if status["profile"] != profile or status.get("checkpoint_continuation") is not False:
        raise ValueError("Profile/continuation mismatch")
    cfg = read_json(attempt / "run" / "config.json")
    expected = make_config(status["arm"], status["seed"], attempt, profile, cfg["device"]).as_dict()
    expected = json.loads(json.dumps(expected))
    for key, value in expected.items():
        if key == "output_root":
            continue  # Moving an intact run between machines is not a protocol change.
        if cfg.get(key) != value:
            raise ValueError(f"Protocol mismatch for {key}: {cfg.get(key)} != {value}")
    summary = read_json(attempt / "run" / "summary.json")
    steps, freq = cfg["total_steps"], cfg["eval_freq"]
    updates = (steps - max(cfg["warmup_steps"], cfg["batch_size"]) + 1) * cfg["updates_per_step"]
    if summary["total_steps"] != steps or summary["total_updates"] != updates:
        raise ValueError("Training budget incomplete")
    grid = list(range(freq, steps + 1, freq))
    y = evaluation_curve(attempt / "run" / "metrics.csv", grid)
    fq_start = 1_500_000 if profile == "production" else 4_000
    metrics = curve_metrics(grid, y, fq_start)
    if not np.isclose(summary["final_eval_env_success"], metrics["final_success"], atol=1e-12):
        raise ValueError("Summary/curve disagreement")
    runtime = float(summary["runtime_total_sec"])
    if not np.isfinite(runtime) or runtime <= 0:
        raise ValueError("Invalid runtime")
    metrics["steps_per_second"] = steps / runtime
    metrics["diagnostics"] = validate_assignment_trace(attempt / "run" / "assignment_windows.csv", updates)
    if not (attempt / "run" / "checkpoint.pt").is_file():
        raise ValueError("Final checkpoint missing")
    return metrics


def joint_bootstrap(values, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (10, 3) or not np.isfinite(values).all():
        raise ValueError("Primary inference requires all ten complete three-arm seed rows")
    differences = values[:, [0]] - values[:, 1:]
    rows = np.random.default_rng(seed).integers(0, 10, size=(resamples, 10))
    draws = differences[rows].mean(axis=1)
    result = {}
    for j, name in enumerate(("support_minus_random", "support_minus_td_topk")):
        result[name] = {"mean": float(differences[:, j].mean()),
                        "positive_pairs": int((differences[:, j] > 0).sum()),
                        "seed_differences": differences[:, j].tolist(),
                        "ci95": np.quantile(draws[:, j], [0.025, 0.975]).tolist(),
                        "ci97_5_bonferroni": np.quantile(draws[:, j], [0.0125, 0.9875]).tolist()}
    return result


def list_attempts(seed_dir):
    return sorted(p for p in seed_dir.glob("attempt_*") if p.is_dir())


def accepted_attempt(seed_dir):
    attempts = list_attempts(seed_dir)
    if not attempts:
        raise ValueError("Missing run")
    for old in attempts[:-1]:
        state = read_json(old / "status.json")
        if state["status"] not in {"failed_infrastructure", "interrupted"}:
            raise ValueError("Unapproved replacement of a scientific run")
    current = attempts[-1]
    if len(attempts) > 1 and not read_json(current / "status.json").get("retry_reason"):
        raise ValueError("Missing infrastructure retry reason")
    return current


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.output_root.resolve()
    plan = read_json(root / "production" / "plan.json")
    report = {"analysis_version": ANALYSIS_VERSION, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
              "bootstrap_seed": BOOTSTRAP_SEED, "interval_coverage": "approximate, not exact",
              "runs": [], "failures_or_missing": [], "inference": None}
    metrics_by_key, env_by_key = {}, {}
    for seed in SEEDS:
        for arm in ARMS:
            directory = root / "production" / arm / f"seed_{seed:02d}"
            attempts = []
            for path in list_attempts(directory):
                try:
                    attempts.append(read_json(path / "status.json"))
                except (OSError, ValueError):
                    attempts.append({"status": "missing_status", "attempt": path.name})
            row = {"arm": arm, "seed": seed, "attempts": attempts}
            try:
                attempt = accepted_attempt(directory)
                status = read_json(attempt / "status.json")
                if (status["arm"], status["seed"]) != (arm, seed):
                    raise ValueError("Arm/seed identity does not match the cohort catalog")
                if status["source_manifest_sha256"] != plan["source_manifest_sha256"]:
                    raise ValueError("Mixed source snapshots in one cohort")
                metrics = validate_run(attempt)
                metrics_by_key[seed, arm] = metrics
                env_by_key[seed, arm] = read_json(attempt / "environment.json")
                row["metrics"] = metrics
            except (OSError, ValueError, KeyError) as exc:
                row["problem"] = str(exc)
                report["failures_or_missing"].append({"arm": arm, "seed": seed, "problem": str(exc)})
            report["runs"].append(row)
    for seed in SEEDS:
        if all((seed, a) in env_by_key for a in ARMS):
            envs = [env_by_key[seed, a] for a in ARMS]
            for field in ("python", "packages", "torch_cuda_build", "cudnn", "gpu", "threads"):
                if any(e[field] != envs[0][field] for e in envs[1:]):
                    report["failures_or_missing"].append({"seed": seed, "problem": f"Paired environment mismatch: {field}"})
    if not report["failures_or_missing"]:
        matrix = [[metrics_by_key[s, a]["auc"] for a in ARMS] for s in SEEDS]
        report["inference"] = joint_bootstrap(matrix)
        report["arm_summary"] = {}
        for arm in ARMS:
            data = [metrics_by_key[s, arm] for s in SEEDS]
            report["arm_summary"][arm] = {
                key: {"mean": float(np.mean([d[key] for d in data])),
                      "sd": float(np.std([d[key] for d in data], ddof=1))}
                for key in ("auc", "fq_auc", "final_success", "steps_per_second")}
            hit = [d["hit90"] for d in data if d["hit90"] is not None]
            report["arm_summary"][arm]["hit90"] = {"coverage": len(hit), "of": 10,
                                                           "median_among_hits": float(np.median(hit)) if hit else None}
    destination = root / "analysis"
    destination.mkdir(exist_ok=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (destination / "seed_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "arm", "auc", "fq_auc", "final_success", "hit90", "steps_per_second"])
        writer.writeheader()
        for (seed, arm), metrics in sorted(metrics_by_key.items()):
            writer.writerow({"seed": seed, "arm": arm, **{k: v for k, v in metrics.items() if k != "diagnostics"}})
    if report["failures_or_missing"]:
        raise SystemExit("Incomplete/noncompliant cohort: status report written; primary inference withheld")
    print(json.dumps(report["inference"], indent=2))


if __name__ == "__main__":
    main()
