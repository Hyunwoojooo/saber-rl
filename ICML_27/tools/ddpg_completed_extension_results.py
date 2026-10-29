"""Export completed DrawerOpen DDPG runs; aggregate only complete seed cohorts."""
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

import numpy as np

from ddpg_seed_extension_results import COHORTS, EXPERIMENT, FIELDS, paired, published_run, read
from export_local_experiments import scrub, sha, write_json
from sac_results import assert_close, csv_text, inventory, within
from verify_archive import check_hashes, check_links

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "metaworld_Drawer_Open_DDPG/paper_artifacts/drawer_open_ddpg_main_seed_extension_2m"
ARMS = ("her", "saber", "hgr", "per", "replay_query", "shuffled_query", "random_support", "strict_random")
LABELS = {"her": "HER", "saber": "SABER (Ours)", "hgr": "HGR", "per": "PER",
          "replay_query": "Replay-Action Query", "shuffled_query": "Shuffled-Policy Query",
          "random_support": "Random-Support", "strict_random": "Context-Eligible Count-Matched"}


def completed_keys(queue):
    if tuple(queue["protocol"]["order"]) != ARMS or queue["protocol"]["seeds"] != list(range(6, 11)):
        raise ValueError("Unexpected experiment or seed set")
    if set(queue["arms"]) != set(ARMS):
        raise ValueError("Missing or unexpected queue arm")
    keys = set()
    for arm in ARMS:
        states = queue["arms"][arm]["seeds"]
        if set(states) != {str(s) for s in range(6, 11)}:
            raise ValueError("Incomplete queue population")
        for seed, state in states.items():
            if state["status"] == "completed":
                if state.get("exit_code") != 0 or state["step"] != 2000000 or state.get("nonfinite_loss"):
                    raise ValueError("Invalid completion record")
                keys.add((arm, int(seed)))
    if not keys:
        raise ValueError("No completed production runs")
    return keys


def summarize(rows, queue):
    selected = completed_keys(queue)
    included_arms = [a for a in ARMS if any(k[0] == a for k in selected)]
    expected = selected | {(a, s) for a in included_arms for s in range(1, 6)}
    keys = [(r["arm"], r["seed"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Missing, duplicate or unfinished result included")
    rows = sorted(rows, key=lambda r: (ARMS.index(r["arm"]), r["seed"]))
    indexed = {(r["arm"], r["seed"]): r for r in rows}
    complete = [a for a in ARMS if all((a, s) in selected for s in range(6, 11))]
    cohorts = {}
    for label, seeds in COHORTS.items():
        summary, contrasts = {}, {}
        for arm in complete:
            rr = [indexed[arm, s] for s in seeds]
            summary[arm] = {m: {"mean": float(np.mean([r[m] for r in rr])),
                                "sd": float(np.std([r[m] for r in rr], ddof=1))} for m in FIELDS}
            summary[arm]["hit90_reached"] = sum(r["hit90_step"] is not None for r in rr)
            if arm != "saber" and "saber" in complete:
                contrasts[f"saber_minus_{arm}"] = {m: paired([indexed["saber", s][m] for s in seeds],
                                                            [indexed[arm, s][m] for s in seeds]) for m in FIELDS}
        cohorts[label] = {"summary": summary, "paired": contrasts}
    return {"snapshot_utc": queue["updated_utc"], "full_queue_completed": len(selected) == 40,
            "new_completed_runs": len(selected), "historical_reference_runs": len(expected) - len(selected),
            "complete_arms": complete, "partial_arms": [a for a in included_arms if a not in complete],
            "cohorts": cohorts, "per_seed": rows}


def products(root):
    provenance = read(root / "provenance.json")
    queue = read(root / "provenance/queue_snapshot.json")
    expected = read(root / "provenance/expected_configs.json")
    assert_close(read(root / "protocol.json"), queue["protocol"])
    selected = completed_keys(queue)
    assert_close(set(expected), {f"{a}:{s}" for a, s in selected})
    rows, curves = [], []
    for entry in provenance["runs"]:
        row, points = published_run(root, entry, expected, queue)
        rows.append(row)
        curves.extend({"arm": row["arm"], "seed": row["seed"], "step": s, "success": v} for s, v in points)
    report = summarize(rows, queue)
    assert_close(report, read(root / "reference_report.json"))
    actual_dirs = {p.parent.relative_to(root).as_posix() for p in (root / "data").rglob("evaluation.csv")}
    assert_close(actual_dirs, {e["published_directory"] for e in provenance["runs"]})
    aggregates, intervals = [], []
    for label, cohort in report["cohorts"].items():
        for arm, values in cohort["summary"].items():
            aggregates.append({"cohort": label, "arm": arm, "n": len(COHORTS[label]),
                               **{f"{m}_{stat}": values[m][stat] for m in FIELDS for stat in ("mean", "sd")},
                               "hit90_reached": values["hit90_reached"]})
        for contrast, metrics in cohort["paired"].items():
            for metric, values in metrics.items():
                intervals.append({"cohort": label, "contrast": contrast, "metric": metric,
                                  "n": values["n"], "mean_difference": values["mean_difference"],
                                  "ci95_low": values["ci95"][0], "ci95_high": values["ci95"][1],
                                  "positive_pairs": values["positive_pairs"]})
    if not aggregates or not intervals:
        raise ValueError("At least SABER and one complete comparison cohort are required")
    return {"analysis/report.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
            "analysis/seed_metrics.csv": csv_text(report["per_seed"]),
            "analysis/aggregate_metrics.csv": csv_text(aggregates),
            "analysis/paired_differences.csv": csv_text(intervals),
            "analysis/learning_curves.csv": csv_text(curves)}


def make_readme(report, queue):
    lines = ["# DrawerOpen DDPG: Completed Main-Paper Seed Extensions", "",
             f"Snapshot: **{report['snapshot_utc']}**. This is not a live dashboard.", "",
             f"This artifact contains **{report['new_completed_runs']} completed new runs** and "
             f"{report['historical_reference_runs']} historical seed-level references. Each new run has 2M steps.",
             "The earlier HER/SABER artifact's 10 new runs are included again for comparison, not counted as new independent observations.",
             "Only completed production outputs are exported. A partially completed method has seed-level data only, not a final aggregate or paired CI.", "",
             "## Queue Coverage", "", "| Method | Completed new seeds | State at capture |", "| --- | --- | --- |"]
    selected = completed_keys(queue)
    for arm in ARMS:
        seeds = [str(s) for s in range(6, 11) if (arm, s) in selected]
        lines.append(f"| {LABELS[arm]} | {', '.join(seeds) if seeds else 'None'} | {queue['arms'][arm]['status']} |")
    for label in COHORTS:
        lines += ["", f"## {label}", "",
                  "| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        c = report["cohorts"][label]
        for arm, values in c["summary"].items():
            cells = [f"{values[m]['mean']:.6f} +/- {values[m]['sd']:.6f}" for m in FIELDS]
            lines.append(f"| {LABELS[arm]} | {' | '.join(cells)} | {values['hit90_reached']}/{len(COHORTS[label])} |")
        lines += ["", "| AUC contrast | Mean difference | Paired 95% CI | Positive pairs |",
                  "| --- | ---: | --- | ---: |"]
        for contrast, values in c["paired"].items():
            v = values["auc"]
            lines.append(f"| {contrast} | {v['mean_difference']:+.6f} | [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}] | {v['positive_pairs']}/{v['n']} |")
    lines += ["", "## Completed New Seed Results", "",
              "No row below is an intermediate checkpoint. Missing seeds are intentionally not imputed or dropped to form a smaller final cohort.", "",
              "| Method | Seed | AUC | FQ-AUC | Final success | Hit90 step |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in report["per_seed"]:
        if r["seed"] >= 6:
            hit = str(r["hit90_step"]) if r["hit90_step"] is not None else "Not reached"
            lines.append(f"| {LABELS[r['arm']]} | {r['seed']} | {r['auc']:.6f} | {r['final_quarter_auc']:.6f} | {r['final_success']:.2f} | {hit} |")
    lines += ["", "## Interpretation and Protocol", "",
              "- Higher sample means with a CI including zero do not establish superiority or equivalence. Pooled SABER/PER AUC means are very close; no general superiority claim follows.",
              "- The historical, added-seed and pooled estimates are reported separately. Their differing directions and failure-to-learn outcomes must remain visible.",
              "- Only methods with all five new seeds complete enter aggregate/paired tables. Partial methods remain explicitly incomplete even though finished seeds can be inspected.",
              "- DrawerOpen-v3 / DDPG / sparse reward v2; 2M steps, UTD=1, 1,995,001 optimizer updates, HER ratio 0.8, horizon 200, 20 evaluation episodes every 10k steps.",
              "- AUC uses exactly 200 evaluation points from 10k to 2M, divided by 1,990,000. FQ-AUC covers 1.5M..2M, divided by 500,000. Success is eval/env_success_mean. No smoothing, interpolation or silently shortened horizons.",
              "- Final success is evaluated at 2M. Hit90 is the first evaluation >=0.90, not sustained task mastery. SD is across seeds with ddof=1.",
              "- Paired percentile bootstrap resamples seed rows 10,000 times using NumPy default_rng(2027); endpoints are 2.5% and 97.5%. These are approximate, pointwise, unadjusted intervals, not simultaneous guarantees. AUC is primary; other metrics are secondary.",
              "- Every completed zero/low-success run is retained. New runs began from scratch; no checkpoint continuation or outcome-based exclusions.",
              "- Historical exact 3,125-assignment mechanism intervals are not overwritten: any later pooled control comparison will use the declared Monte Carlo procedure.",
              "- Original HER/SABER/Random full configs and exact historical environment locks were not retained. New configs follow the audited canonical-source/launcher reconstruction. Retained configs were checked where available; bitwise historical environment identity is not asserted.",
              "- Earlier diagnostic and assignment executions are not extra independent seeds in these cohorts. Source/result overlap across artifacts must not be counted twice.",
              "- Runtime records include instrumentation and are not evidence of a matched-cost algorithm ranking.", "",
              "## Included and Omitted", "",
              "- [Seed metrics](analysis/seed_metrics.csv), [aggregates](analysis/aggregate_metrics.csv), [paired intervals](analysis/paired_differences.csv), [curves](analysis/learning_curves.csv), [report](analysis/report.json).",
              "- data/historical_1_5 contains retained evaluation curves, not reconstructed historical execution configs.",
              "- data/extension_6_10 contains every completed run's evaluation curve, config, summary, simulator metadata and actual launch record.",
              "- source_snapshot contains byte-identical executed trainer sources for included methods. Original launchers are historical presets; launch_record.json records the actual extension command. This is not a standalone training package.",
              "- [Provenance](provenance.json) records original file hashes and export transformations. [Queue snapshot](provenance/queue_snapshot.json) records status only for unfinished runs; their result directories are omitted.",
              "- Checkpoints, replay buffers, raw training logs, caches, locks and unfinished output files are excluded. Workspace prefixes are redacted in exported JSON; unavailable optional diagnostics become null, never valid performance values.", "",
              "## Verification", "", "From the repository root with Python 3.12 and NumPy:", "", "```bash",
              "python3 ICML_27/tools/ddpg_completed_extension_results.py verify", "```", "",
              "Read-only CPU verification recomputes all seed metrics and complete-cohort intervals, checks configs/completion/source hashes, and verifies every manifest entry. It does not import trainers or modify the running queue.", ""]
    return "\n".join(lines)


def verify(root=ARTIFACT):
    provenance = read(root / "provenance.json")
    check_hashes(root, provenance["source_files_sha256"])
    for name, expected in products(root).items():
        if (root / name).read_text() != expected:
            raise ValueError(f"Derived result mismatch: {name}")
    report = read(root / "analysis/report.json")
    if (root / "README.md").read_text() != make_readme(report, read(root / "provenance/queue_snapshot.json")):
        raise ValueError("README differs from verified metrics/status")
    if (root / "MANIFEST.sha256").read_text() != inventory(root):
        raise ValueError("Artifact manifest mismatch")
    check_links(root / "README.md")
    return {"passed": True, "new_runs": report["new_completed_runs"],
            "historical_runs": report["historical_reference_runs"],
            "evaluation_points": len(report["per_seed"]) * 200,
            "complete_arms": report["complete_arms"], "partial_arms": report["partial_arms"],
            "source_files": len(provenance["source_files_sha256"])}


def export(workspace):
    workspace = workspace.resolve()
    origin = within(workspace, EXPERIMENT)
    if ARTIFACT.exists() or workspace == REPO.resolve():
        raise ValueError("Use a separate publication checkout and a new artifact destination")
    sys.path.insert(0, str(origin))
    api = importlib.import_module("analysis")
    common = importlib.import_module("common")
    common.verify_snapshots()
    api.verify_reference_metrics()
    queue_bytes = (origin / "outputs/queue_status.json").read_bytes()
    queue = json.loads(queue_bytes)
    selected = completed_keys(queue)
    arms = [a for a in ARMS if any(k[0] == a for k in selected)]
    expected = read(origin / "outputs/validation/expected_configs.json")
    old = [r for r in api.reference_rows() if r["arm"] in arms]
    new = [api.read_run(a, s, expected_config=expected[f"{a}:{s}"])
           for a, s in sorted(selected, key=lambda k: (ARMS.index(k[0]), k[1]))]
    rows = [{k: r[k] for k in ("arm", "seed", *FIELDS, "hit90_step")} for r in old + new]
    report = summarize(rows, queue)
    indexed = {(r["arm"], r["seed"]): r for r in rows}
    for label, c in report["cohorts"].items():
        for contrast, metrics in c["paired"].items():
            arm = contrast.removeprefix("saber_minus_")
            for metric, value in metrics.items():
                assert_close(value, api.paired([indexed["saber", s][metric] for s in COHORTS[label]],
                                               [indexed[arm, s][metric] for s in COHORTS[label]]))
    ARTIFACT.mkdir(parents=True)
    sources, entries = {}, []
    for name, digest in read(origin / "snapshot_manifest.json").items():
        if not any(name.startswith(f"sources/{a}/") for a in arms):
            continue
        target = ARTIFACT / "source_snapshot" / Path(name).relative_to("sources")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin / name, target)
        sources[str(target.relative_to(ARTIFACT))] = digest
    for row in report["per_seed"]:
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
            files = [(run / n, n) for n in ("config.json", "summary.json", "env_snapshot.json")]
            files.append((run.with_suffix(".claim.json"), "launch_record.json"))
            for source, name in files:
                write_json(dest / name, scrub(read(source), workspace))
                hashes[str(source.relative_to(workspace))] = sha(source)
            with (run / "metrics.csv").open(newline="") as stream:
                evaluations = [{"step": int(float(r["step"])), "success": r["eval/env_success_mean"]}
                               for r in csv.DictReader(stream) if r.get("eval/env_success_mean", "") != ""]
            (dest / "evaluation.csv").write_text(csv_text(evaluations))
            hashes[str((run / "metrics.csv").relative_to(workspace))] = sha(run / "metrics.csv")
        entries.append({"arm": arm, "seed": seed, "cohort": cohort,
                        "published_directory": str(dest.relative_to(ARTIFACT)), "original_files_sha256": hashes})
    origins = read(origin / "provenance/source_origins.json")
    origins["arms"] = {k: v for k, v in origins["arms"].items() if k in arms}
    origins["curves"] = [r for r in origins["curves"] if r["arm"] in arms]
    write_json(ARTIFACT / "provenance/source_origins.json", scrub(origins, workspace))
    write_json(ARTIFACT / "provenance/expected_configs.json", scrub({f"{a}:{s}": expected[f"{a}:{s}"] for a, s in sorted(selected)}, workspace))
    write_json(ARTIFACT / "provenance/queue_snapshot.json", scrub(queue, workspace))
    write_json(ARTIFACT / "provenance/environment.json", scrub(read(origin / "outputs/validation/environment.json"), workspace))
    shutil.copy2(origin / "protocol.json", ARTIFACT / "protocol.json")
    write_json(ARTIFACT / "provenance.json", {"exported_utc": datetime.now(timezone.utc).isoformat(),
               "experiment": EXPERIMENT, "source_files_sha256": sources, "runs": entries,
               "original_snapshot_manifest_sha256": sha(origin / "snapshot_manifest.json"),
               "original_queue_snapshot_sha256": hashlib.sha256(queue_bytes).hexdigest(),
               "historical_limit": origins["historical_limit"],
               "transformations": ["workspace path replaced with <WORKSPACE> in JSON", "unavailable optional nonfinite diagnostics serialized as null",
                                   "all 200 evaluation steps retained per completed run; training rows omitted"],
               "omitted": ["checkpoints", "replay buffers", "raw training logs", "unfinished result directories", "unexecuted trainers"]})
    write_json(ARTIFACT / "reference_report.json", report)
    for name, value in products(ARTIFACT).items():
        path = ARTIFACT / name
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
        export(args.workspace)
    else:
        print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    main()
