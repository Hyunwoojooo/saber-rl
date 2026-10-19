"""Validate PAL runs and audit all ten fixed SABER references before inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from freeze import verify
from protocol import (ROOT, SEEDS, BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES,
                      ANALYSIS_VERSION, REFERENCE_MANIFEST_SHA256,
                      make_config, expected_reference_config, scientific_config)
from records import accepted_attempt, list_attempts, local_output, read_json, write_json

ENVIRONMENT_FIELDS = ("python", "packages", "torch_cuda_build", "cudnn", "gpu",
                      "threads", "host", "architecture", "nvidia_driver", "platform")


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def evaluation_curve(path, grid, *, pal=False, warmup=5000):
    points, times = [], []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"step", "eval/env_success_mean", "runtime/total_sec", "loss/critic",
                    "loss/actor", "loss/td_error_abs_mean", "value/q_mean", "value/target_mean"}
        if pal:
            required |= {"pal/normalizer", "pal/base_loss", "pal/large_error_fraction"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing metric columns")
        for row in reader:
            step = int(row["step"])
            for key in ("loss/critic", "loss/actor", "loss/td_error_abs_mean",
                        "value/q_mean", "value/target_mean"):
                if step >= warmup and (not row.get(key) or not math.isfinite(float(row[key]))):
                    raise ValueError(f"Invalid {key} at {step}")
            if pal and step >= warmup:
                z, base, fraction = (float(row[key]) for key in
                                     ("pal/normalizer", "pal/base_loss", "pal/large_error_fraction"))
                if not all(math.isfinite(v) for v in (z, base, fraction)) or z < 1 or base < 0 or not 0 <= fraction <= 1:
                    raise ValueError("Invalid PAL loss diagnostics")
                if not math.isclose(float(row["loss/critic"]), base / z, rel_tol=1e-5, abs_tol=1e-7):
                    raise ValueError("PAL loss/normalizer disagreement")
            value = row.get("eval/env_success_mean")
            if value in (None, ""):
                continue
            points.append((step, float(value)))
            times.append(float(row["runtime/total_sec"]))
    if [s for s, _ in points] != list(grid):
        raise ValueError("Evaluation grid missing, duplicated, reordered, or extended")
    values = np.asarray([v for _, v in points], dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid evaluation success values")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or min(times) <= 0:
        raise ValueError("Invalid evaluation timestamps")
    return values, times


def curve_metrics(grid, success, fq_start):
    x, y = np.asarray(grid), np.asarray(success, dtype=np.float64)
    if len(x) < 2 or x.shape != y.shape or not np.isfinite(y).all() or np.any(np.diff(x) <= 0):
        raise ValueError("Invalid curve")
    if fq_start not in x or x[-1] <= fq_start:
        raise ValueError("Missing final-quarter boundary")
    late = x >= fq_start
    hit = x[y >= 0.9]
    return {"auc": float(np.trapezoid(y, x) / (x[-1] - x[0])),
            "fq_auc": float(np.trapezoid(y[late], x[late]) / (x[-1] - fq_start)),
            "final_success": float(y[-1]), "hit90": int(hit[0]) if len(hit) else None}


def compare_config(actual, expected):
    a, b = canonical(scientific_config(actual)), canonical(scientific_config(expected))
    if a != b:
        changed = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
        raise ValueError(f"Protocol mismatch: {changed}")


def validate_run(attempt, profile="production", require_status=True, reference=False):
    attempt = Path(attempt)
    status = read_json(attempt / "status.json")
    if require_status and status["status"] != "complete":
        raise ValueError(f"Run not complete: {status['status']}")
    if (status["profile"] != profile or status.get("checkpoint_continuation") is not False
            or status["seed"] not in SEEDS or status["arm"] != ("support" if reference else "pal")):
        raise ValueError("Identity/profile/continuation mismatch")
    cfg = read_json(attempt / "run/config.json")
    builder = expected_reference_config if reference else lambda *a: make_config("pal", *a).as_dict()
    expected = builder(status["seed"], attempt, profile, cfg["device"])
    compare_config(cfg, expected)
    compare_config(read_json(attempt / "requested_config.json"), expected)
    summary = read_json(attempt / "run/summary.json")
    steps = cfg["total_steps"]
    updates = (steps - max(cfg["warmup_steps"], cfg["batch_size"]) + 1) * cfg["updates_per_step"]
    if summary["total_steps"] != steps or summary["total_updates"] != updates:
        raise ValueError("Training budget incomplete")
    grid = list(range(cfg["eval_freq"], steps + 1, cfg["eval_freq"]))
    y, elapsed = evaluation_curve(attempt / "run/metrics.csv", grid, pal=not reference,
                                  warmup=cfg["warmup_steps"])
    metrics = curve_metrics(grid, y, 1_500_000 if profile == "production" else 4_000)
    if not math.isclose(float(summary["final_eval_env_success"]), metrics["final_success"], abs_tol=1e-12):
        raise ValueError("Summary/curve disagreement")
    if summary.get("legacy_live_target_diagnostic_enabled") is not False:
        raise ValueError("Unexpected legacy residual diagnostic")
    if not reference:
        required = {"method": "pal", "pal_experiment": True, "critic_objective": "pal",
                    "pal_alpha": 0.4, "pal_min_priority": 1.0,
                    "support_computation_enabled": False, "replay_support_index_slots": 0,
                    "assignment_experiment": False}
        if any(summary.get(k) != v for k, v in required.items()):
            raise ValueError("PAL/support-scope summary mismatch")
        if (attempt / "run/assignment_windows.csv").exists():
            raise ValueError("PAL must not compute assignment-control masks")
    runtime = float(summary["runtime_total_sec"])
    if not math.isfinite(runtime) or runtime <= 0 or runtime < elapsed[-1]:
        raise ValueError("Invalid run runtime")
    if not (attempt / "run/checkpoint.pt").is_file():
        raise ValueError("Final checkpoint missing")
    metrics.update(runtime_sec=runtime, steps_per_second=steps / runtime)
    return metrics


def paired_bootstrap(values, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (10, 2) or not np.isfinite(matrix).all():
        raise ValueError("Inference requires all ten SABER/PAL seed pairs")
    delta = matrix[:, 0] - matrix[:, 1]
    rows = np.random.default_rng(seed).integers(0, 10, size=(resamples, 10))
    draws = delta[rows].mean(axis=1)
    return {"contrast": "SABER-minus-PAL full-span AUC", "mean": float(delta.mean()),
            "median": float(np.median(delta)), "positive_pairs": int((delta > 0).sum()),
            "negative_pairs": int((delta < 0).sum()), "tied_pairs": int((delta == 0).sum()),
            "seed_differences": delta.tolist(),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "coverage": "approximate percentile bootstrap; not an equivalence test"}


def audit_reference_source(root):
    root = Path(root).resolve()
    manifest = root / "source_manifest.json"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != REFERENCE_MANIFEST_SHA256:
        raise ValueError("Reference must be the frozen 2026-09-25 assignment cohort")
    data = read_json(manifest)
    for rel, wanted in data["sha256"].items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != wanted:
            raise ValueError(f"Reference source changed: {rel}")
    return root


def audit_environment(a, b):
    missing = [k for k in ENVIRONMENT_FIELDS if k not in a or k not in b]
    changed = [k for k in ENVIRONMENT_FIELDS if a.get(k) != b.get(k)]
    if missing or changed:
        raise ValueError(f"Reference/PAL environment mismatch: missing={missing}, changed={changed}")


def collect_cohort(production, arm, source_hash, reference=False):
    rows, accepted, problems = [], {}, []
    for seed in SEEDS:
        directory = Path(production) / arm / f"seed_{seed:02d}"
        row = {"seed": seed, "arm": arm, "attempts": []}
        try:
            row["attempts"] = [read_json(p / "status.json") for p in list_attempts(directory)]
            attempt = accepted_attempt(directory)
            status = read_json(attempt / "status.json")
            if status["source_manifest_sha256"] != source_hash or status["seed"] != seed or status["arm"] != arm:
                raise ValueError("Cohort identity/source mismatch")
            metrics = validate_run(attempt, reference=reference)
            accepted[seed] = {"metrics": metrics, "environment": read_json(attempt / "environment.json")}
            row["metrics"] = metrics
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row["problem"] = str(exc)
            problems.append({"arm": arm, "seed": seed, "problem": str(exc)})
        rows.append(row)
    return rows, accepted, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    ap.add_argument("--saber-root", type=Path, help="Read-only 2026-09-25 experiment directory")
    args = ap.parse_args()
    root = local_output(args.output_root)
    snapshot = verify()
    plan = read_json(root / "production/plan.json")
    if plan.get("source_manifest_sha256") != snapshot or plan.get("arms") != ["pal"] or plan.get("seeds") != list(SEEDS):
        raise ValueError("PAL plan/source mismatch")
    rows, pal, problems = collect_cohort(root / "production", "pal", snapshot)
    report = {"analysis_version": ANALYSIS_VERSION, "runs": rows, "problems": problems,
              "inference": None, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
              "bootstrap_seed": BOOTSTRAP_SEED, "source_manifest_sha256": snapshot,
              "runtime_comparison_valid": False,
              "runtime_note": "SABER has assignment instrumentation; do not rank methods by these runtimes"}
    cohorts = {"pal": pal}
    if args.saber_root:
        try:
            ref = audit_reference_source(args.saber_root)
            ref_plan = read_json(ref / "outputs/production/plan.json")
            if (ref_plan.get("source_manifest_sha256") != REFERENCE_MANIFEST_SHA256
                    or ref_plan.get("arms") != ["support", "random", "td_topk"]
                    or ref_plan.get("seeds") != list(SEEDS)):
                raise ValueError("Reference plan mismatch")
            ref_rows, saber, ref_problems = collect_cohort(
                ref / "outputs/production", "support", REFERENCE_MANIFEST_SHA256, reference=True)
            report["reference_runs"] = ref_rows
            problems.extend(ref_problems)
            cohorts["support"] = saber
            for seed in sorted(pal.keys() & saber.keys()):
                try:
                    audit_environment(pal[seed]["environment"], saber[seed]["environment"])
                except ValueError as exc:
                    problems.append({"seed": seed, "problem": str(exc)})
            if not problems and len(pal) == len(saber) == 10:
                report["inference"] = paired_bootstrap(
                    [[saber[s]["metrics"]["auc"], pal[s]["metrics"]["auc"]] for s in SEEDS])
                report["reference_reuse_audit_passed"] = True
        except (OSError, ValueError, KeyError) as exc:
            problems.append({"problem": f"Reference audit: {exc}"})
    report["arm_summary"] = {}
    for name, runs in cohorts.items():
        if len(runs) == 10:
            report["arm_summary"][name] = {
                key: {"mean": float(np.mean([r["metrics"][key] for r in runs.values()])),
                      "sd": float(np.std([r["metrics"][key] for r in runs.values()], ddof=1))}
                for key in ("auc", "fq_auc", "final_success", "runtime_sec", "steps_per_second")}
            hit = [r["metrics"]["hit90"] for r in runs.values() if r["metrics"]["hit90"] is not None]
            report["arm_summary"][name]["hit90"] = {"coverage": len(hit), "of": 10,
                                                          "median_among_hits": float(np.median(hit)) if hit else None}
    destination = root / "analysis"
    write_json(destination / "report.json", report)
    fields = ["arm", "seed", "auc", "fq_auc", "final_success", "hit90", "runtime_sec", "steps_per_second"]
    with (destination / "seed_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, runs in cohorts.items():
            for seed, row in sorted(runs.items()):
                writer.writerow({"arm": name, "seed": seed, **row["metrics"]})
    print(json.dumps({"completed_pal": len(pal), "problems": problems, "inference": report["inference"]}, indent=2))
    if problems:
        raise SystemExit("Incomplete/noncompliant cohort: all statuses retained; paired inference withheld")


if __name__ == "__main__":
    main()
