"""Publish/verify completed HER-SABER seed extension; never run a trainer."""
import argparse
import csv
import importlib
import json
import math
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

import numpy as np

from export_local_experiments import scrub, sha, write_json
from sac_results import assert_close, csv_text, inventory, within
from verify_archive import check_hashes, check_links

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "metaworld_Drawer_Open_DDPG/paper_artifacts/drawer_open_ddpg_her_saber_seed_extension_2m"
EXPERIMENT = "experiments_post_review/2026-10-24_draweropen_ddpg_main_seed_extension"
ARMS = ("her", "saber")
FIELDS = ("auc", "final_quarter_auc", "final_success")
COHORTS = {"historical_1_5": list(range(1, 6)), "extension_6_10": list(range(6, 11)), "pooled_1_10": list(range(1, 11))}


def read(path):
    return json.loads(path.read_text())


def measure(points):
    steps = np.asarray([p[0] for p in points], dtype=np.int64)
    values = np.asarray([p[1] for p in points], dtype=np.float64)
    if not np.array_equal(steps, np.arange(10000, 2000001, 10000)):
        raise ValueError("Require every checkpoint on the exact 10k..2M grid")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid evaluation success")
    last_quarter = steps >= 1500000
    hit = steps[values >= .9]
    return {"auc": float(np.trapezoid(values, steps) / 1990000),
            "final_quarter_auc": float(np.trapezoid(values[last_quarter], steps[last_quarter]) / 500000),
            "final_success": float(values[-1]), "hit90_step": int(hit[0]) if hit.size else None}


def paired(left, right):
    if len(left) != len(right) or len(left) not in (5, 10):
        raise ValueError("Require five or ten complete seed pairs")
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite paired difference")
    indices = np.random.default_rng(2027).integers(0, len(delta), size=(10000, len(delta)))
    means = delta[indices].mean(axis=1)
    return {"n": len(delta), "mean_difference": float(delta.mean()),
            "ci95": np.quantile(means, [.025, .975]).tolist(),
            "positive_pairs": int((delta > 0).sum()), "seed_differences": delta.tolist()}


def summarize(rows):
    keys = [(r["arm"], r["seed"]) for r in rows]
    if len(keys) != 20 or set(keys) != {(a, s) for a in ARMS for s in range(1, 11)}:
        raise ValueError("Missing or duplicate run; require HER and SABER seeds 1-10")
    indexed = {(r["arm"], r["seed"]): r for r in rows}
    cohorts = {}
    for label, seeds in COHORTS.items():
        groups = {a: [indexed[a, s] for s in seeds] for a in ARMS}
        cohorts[label] = {"summary": {a: {metric: {"mean": float(np.mean([r[metric] for r in rr])),
                                                  "sd": float(np.std([r[metric] for r in rr], ddof=1))}
                                          for metric in FIELDS} for a, rr in groups.items()},
                          "paired": {metric: paired([r[metric] for r in groups["saber"]],
                                                     [r[metric] for r in groups["her"]]) for metric in FIELDS}}
    return {"status": "HER_SABER_complete_remaining_queue_incomplete", "new_completed_runs": 10,
            "reused_historical_runs": 10, "cohorts": cohorts, "per_seed": rows}


def products(root):
    provenance = read(root / "provenance.json")
    expected = read(root / "provenance/expected_configs.json")
    queue = read(root / "provenance/queue_snapshot.json")
    rows, curves = [], []
    for entry in provenance["runs"]:
        arm, seed = entry["arm"], entry["seed"]
        folder = within(root, entry["published_directory"])
        with (folder / "evaluation.csv").open(newline="") as stream:
            points = [(int(r["step"]), float(r["success"])) for r in csv.DictReader(stream)]
        metrics = measure(points)
        if seed >= 6:
            config, summary = read(folder / "config.json"), read(folder / "summary.json")
            claim = read(folder / "launch_record.json")
            state = queue["arms"][arm]["seeds"][str(seed)]
            if state["status"] != "completed" or state["exit_code"] != 0 or config["resume"] or config.get("resume_path"):
                raise ValueError("Incomplete or resumed extension run")
            for key, value in expected[f"{arm}:{seed}"].items():
                if key not in config and key in {"numba_support_scan", "support_predicate_early_exit", "support_diagnostic_audit_interval"}:
                    continue
                assert_close(config.get(key), value)
            assert_close(claim["arm"], arm)
            assert_close(claim["seed"], seed)
            assert_close(claim["resume"], False)
            assert_close(summary["total_steps"], 2000000)
            assert_close(summary["total_updates"], 1995001)
            assert_close(metrics["auc"], summary["eval_AUC_success"])
            assert_close(metrics["final_success"], summary["final_eval_env_success"])
            if any(v is None or not math.isfinite(v) for v in summary["final_losses"].values()):
                raise ValueError("Nonfinite final training loss")
            assert_close({k: state["result"][k] for k in metrics}, metrics)
        rows.append({"arm": arm, "seed": seed, **metrics})
        curves.extend({"arm": arm, "seed": seed, "step": step, "success": success} for step, success in points)
    report = summarize(rows)
    assert_close(report, read(root / "reference_report.json"))
    aggregate, intervals = [], []
    for cohort, result in report["cohorts"].items():
        for arm, values in result["summary"].items():
            aggregate.append({"cohort": cohort, "arm": arm,
                              **{f"{metric}_{stat}": values[metric][stat] for metric in FIELDS for stat in ("mean", "sd")}})
        for metric, value in result["paired"].items():
            intervals.append({"cohort": cohort, "contrast": "SABER-minus-HER", "metric": metric,
                              "n": value["n"], "mean_difference": value["mean_difference"],
                              "ci95_low": value["ci95"][0], "ci95_high": value["ci95"][1],
                              "positive_pairs": value["positive_pairs"]})
    return {"analysis/report.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
            "analysis/seed_metrics.csv": csv_text(rows), "analysis/aggregate_metrics.csv": csv_text(aggregate),
            "analysis/paired_differences.csv": csv_text(intervals), "analysis/learning_curves.csv": csv_text(curves)}


def verify(root=ARTIFACT):
    provenance = read(root / "provenance.json")
    check_hashes(root, provenance["source_files_sha256"])
    for name, expected in products(root).items():
        if (root / name).read_text() != expected:
            raise ValueError(f"Derived result mismatch: {name}")
    if (root / "MANIFEST.sha256").read_text() != inventory(root):
        raise ValueError("Artifact manifest mismatch")
    check_links(root / "README.md")
    return {"passed": True, "new_runs": 10, "historical_runs": 10, "evaluation_points": 4000,
            "source_files": len(provenance["source_files_sha256"]), "paired_intervals": 9}


def make_readme(report, queue):
    lines = ["# DrawerOpen DDPG: HER vs SABER Seed Extension", "",
             "HER and SABER seeds 6-10 are complete: 10 new executions, 2M steps per run, UTD=1.",
             "They are combined with the 10 original main-paper executions (two methods x seeds 1-5).",
             "All outcomes, including HER seed 10 and SABER seed 7 with zero success, are retained.", "",
             "## Results", "",
             "| Cohort | HER mean AUC | SABER mean AUC | SABER - HER | Paired 95% CI | Positive pairs |",
             "| --- | ---: | ---: | ---: | --- | ---: |"]
    for label, c in report["cohorts"].items():
        diff = c["paired"]["auc"]
        lines.append(f"| {label} | {c['summary']['her']['auc']['mean']:.6f} | {c['summary']['saber']['auc']['mean']:.6f} | {diff['mean_difference']:+.6f} | [{diff['ci95'][0]:+.6f}, {diff['ci95'][1]:+.6f}] | {diff['positive_pairs']}/{diff['n']} |")
    lines += ["", "The positive pooled sample mean is smaller than the original five-seed estimate.",
              "The extension-only and pooled AUC intervals include zero: superiority, equivalence, and consistent benefit are not established.",
              "The added seeds favor SABER in only one of five pairs. Do not report the original five-seed CI as the ten-seed result.", "",
              "| New seed | HER AUC | SABER AUC | HER final success | SABER final success |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    index = {(r["arm"], r["seed"]): r for r in report["per_seed"]}
    for seed in range(6, 11):
        h, s = index["her", seed], index["saber", seed]
        lines.append(f"| {seed} | {h['auc']:.6f} | {s['auc']:.6f} | {h['final_success']:.2f} | {s['final_success']:.2f} |")
    lines += ["", "## Remaining Queue Is Not Complete", "",
              f"Timestamped snapshot: {queue['updated_utc']}. This is not live monitoring.", "",
              "| Condition | Status | Completed seeds | Minimum / maximum step |", "| --- | --- | ---: | --- |"]
    for arm in queue["protocol"]["order"]:
        a = queue["arms"][arm]
        values = list(a["seeds"].values())
        completed = sum(r["status"] == "completed" for r in values)
        lines.append(f"| {arm} | {a['status']} | {completed}/5 | {min(r['step'] for r in values):,} / {max(r['step'] for r in values):,} |")
    lines += ["", "The full planned extension is eight conditions x five new seeds = 40 new runs.",
              "HGR and the remaining controls are not included in final performance aggregates or comparisons in this artifact.",
              "Their status appears only in [the queue snapshot](provenance/queue_snapshot.json).", "",
              "## Protocol and Cohort Identity", "",
              "- DrawerOpen-v3, DDPG, sparse reward v2, 2M steps, HER ratio=0.8, horizon=200, 20 evaluation episodes per checkpoint.",
              "- Every final curve contains exactly 200 evaluations from 10k to 2M. Full AUC is normalized by 1,990,000; FQ-AUC covers 1.5M..2M, divided by 500,000. No smoothing, interpolation or truncated horizon.",
              "- Success is `eval/env_success_mean`; final success is that metric at 2M. Hit90 is the first evaluation >=0.90, not sustained success.",
              "- Each extension run completed 1,995,001 updates. Same-seed pairing is used; bootstrap seed rows are resampled 10,000 times with NumPy default_rng(2027). The 95% percentile endpoints are 2.5% and 97.5%.",
              "- AUC is the primary endpoint. FQ-AUC and final-success intervals are secondary, unadjusted intervals, not simultaneous guarantees.",
              "- Historical 1-5, extension 6-10 and pooled 1-10 are reported separately. Cross-period pooling is not proof of identical historical execution environments.",
              "- Original HER/SABER full configs, exact execution commits and dependency locks were not retained. New runs use the earliest retained canonical trainer and launcher-contract reconstruction, not bitwise-verified historical execution.",
              "- These are the October extension executions, not the post-review assignment experiment or earlier diagnostic executions. Same-number diagnostic reruns are not additional independent seeds and must not be counted again.",
              "- All completed failures-to-learn remain included; no checkpoint continuation, outcome-based restart, or significance-based stopping was used.", "",
              "## Included Files", "",
              "- [Seed metrics](analysis/seed_metrics.csv), [aggregate metrics](analysis/aggregate_metrics.csv), [paired intervals](analysis/paired_differences.csv), [full curves](analysis/learning_curves.csv), [report](analysis/report.json).",
              "- `data/historical_1_5/`: retained evaluation CSVs only. No reconstructed configuration is presented as an original execution config.",
              "- `data/extension_6_10/`: evaluation CSVs plus recorded configs, summaries, simulator descriptions and actual launch arguments.",
              "- `source_snapshot/`: byte-identical trainers for the two completed conditions. `original_launcher.sh` is the historical launcher, not the executed seed-extension command; use per-run `launch_record.json` to inspect actual arguments.",
              "- [Provenance](provenance.json), [source origins](provenance/source_origins.json), [recorded new environment](provenance/environment.json), [full queue protocol](protocol.json).",
              "- JSON workspace prefixes are redacted; nonfinite unavailable diagnostics become null. Required performance/loss values must remain finite. Original hashes and export transformations are recorded.",
              "- Checkpoints, replay buffers, raw training logs, caches, process locks and unfinished training outputs are omitted. This is a performance-result artifact, not a standalone training bundle.", "",
              "## Verification", "",
              "From the repository root with Python 3.12 and NumPy (recorded version 2.3.5):", "",
              "```bash", "python3 ICML_27/tools/ddpg_seed_extension_results.py verify", "```", "",
              "Verification is read-only and CPU-only; no trainer is imported. It recomputes all metrics and nine intervals from the included curves, checks completion/configs/source hashes, and verifies the manifest.", ""]
    return "\n".join(lines)


def export(workspace):
    origin = within(workspace, EXPERIMENT)
    if ARTIFACT.exists() or workspace.resolve() == REPO.resolve():
        raise ValueError("Use a separate publication checkout and a new artifact destination")
    sys.path.insert(0, str(origin))
    api = importlib.import_module("analysis")
    common = importlib.import_module("common")
    common.verify_snapshots()
    api.verify_reference_metrics()
    expected = read(origin / "outputs/validation/expected_configs.json")
    old = [r for r in api.reference_rows() if r["arm"] in ARMS]
    new = [api.read_run(a, s, expected_config=expected[f"{a}:{s}"]) for a in ARMS for s in range(6, 11)]
    queue_bytes = (origin / "outputs/queue_status.json").read_bytes()
    queue = json.loads(queue_bytes)
    if any(queue["arms"][a]["status"] != "completed" for a in ARMS):
        raise ValueError("HER and SABER must both be complete")
    rows = [{k: r[k] for k in ("arm", "seed", *FIELDS, "hit90_step")} for r in old + new]
    report = summarize(rows)
    for label, seeds in COHORTS.items():
        for metric in FIELDS:
            indexed = {(r["arm"], r["seed"]): r for r in rows}
            assert_close(report["cohorts"][label]["paired"][metric], api.paired(
                [indexed["saber", s][metric] for s in seeds], [indexed["her", s][metric] for s in seeds]))
    ARTIFACT.mkdir(parents=True)
    sources, entries = {}, []
    snapshot = read(origin / "snapshot_manifest.json")
    for name, digest in snapshot.items():
        if not any(name.startswith(f"sources/{a}/") for a in ARMS):
            continue
        target = ARTIFACT / "source_snapshot" / Path(name).relative_to("sources")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin / name, target)
        sources[str(target.relative_to(ARTIFACT))] = digest
    for row in rows:
        arm, seed = row["arm"], row["seed"]
        cohort = "historical_1_5" if seed <= 5 else "extension_6_10"
        dest = ARTIFACT / "data" / cohort / arm / f"seed_{seed}"
        dest.mkdir(parents=True)
        hashes = {}
        if seed <= 5:
            source = origin / f"references/{arm}/seed_{seed}/evaluation.csv"
            shutil.copy2(source, dest / "evaluation.csv")
            hashes[str(source.relative_to(workspace))] = sha(source)
        else:
            run = origin / f"outputs/production/{arm}/seed_{seed:02d}"
            for source, filename in [(run / n, n) for n in ("config.json", "summary.json", "env_snapshot.json")] + [(run.with_suffix(".claim.json"), "launch_record.json")]:
                write_json(dest / filename, scrub(read(source), workspace))
                hashes[str(source.relative_to(workspace))] = sha(source)
            with (run / "metrics.csv").open(newline="") as stream:
                evaluations = [{"step": int(float(r["step"])), "success": r["eval/env_success_mean"]}
                               for r in csv.DictReader(stream) if r.get("eval/env_success_mean", "") != ""]
            (dest / "evaluation.csv").write_text(csv_text(evaluations))
            hashes[str((run / "metrics.csv").relative_to(workspace))] = sha(run / "metrics.csv")
        entries.append({"arm": arm, "seed": seed, "cohort": cohort,
                        "published_directory": str(dest.relative_to(ARTIFACT)), "original_files_sha256": hashes})
    origins = read(origin / "provenance/source_origins.json")
    origins["arms"] = {k: v for k, v in origins["arms"].items() if k in ARMS}
    origins["curves"] = [r for r in origins["curves"] if r["arm"] in ARMS]
    write_json(ARTIFACT / "provenance/source_origins.json", origins)
    write_json(ARTIFACT / "provenance/expected_configs.json", scrub({k: v for k, v in expected.items() if k.split(":")[0] in ARMS}, workspace))
    write_json(ARTIFACT / "provenance/queue_snapshot.json", scrub(queue, workspace))
    write_json(ARTIFACT / "provenance/environment.json", read(origin / "outputs/validation/environment.json"))
    shutil.copy2(origin / "protocol.json", ARTIFACT / "protocol.json")
    import hashlib
    write_json(ARTIFACT / "provenance.json", {"exported_utc": datetime.now(timezone.utc).isoformat(),
               "experiment": EXPERIMENT, "source_files_sha256": sources, "runs": entries,
               "original_snapshot_manifest_sha256": sha(origin / "snapshot_manifest.json"),
               "original_queue_snapshot_sha256": hashlib.sha256(queue_bytes).hexdigest(),
               "historical_limit": origins["historical_limit"],
               "transformations": ["workspace path replaced with <WORKSPACE> in JSON", "nonfinite optional diagnostics serialized as null",
                                   "new CSVs retain every evaluation step and env-success value; training rows omitted"],
               "omitted": ["weights/checkpoints", "replay buffers", "raw training logs", "unfinished result files", "unexecuted trainer variants"]})
    write_json(ARTIFACT / "reference_report.json", report)
    for filename, value in products(ARTIFACT).items():
        path = ARTIFACT / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    (ARTIFACT / "README.md").write_text(make_readme(report, queue))
    (ARTIFACT / "MANIFEST.sha256").write_text(inventory(ARTIFACT))
    print(json.dumps(verify(), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("export").add_argument("--workspace", type=Path, required=True)
    sub.add_parser("verify")
    args = parser.parse_args()
    if args.command == "export":
        export(args.workspace.resolve())
    else:
        print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    main()
