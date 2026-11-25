"""Export completed Drawer TD3 cohorts without importing a trainer."""
import argparse
import csv
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

import numpy as np

from ddpg_seed_extension_results import COHORTS, FIELDS, paired, published_run, read
from export_local_experiments import scrub, sha, write_json
from sac_results import assert_close, csv_text, inventory, within
from verify_archive import check_hashes, check_links

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiments_post_review/2026-11-04_draweropen_td3_main_seed_extension"
ARTIFACT = REPO / "metaworld_Drawer_Open_TD3/paper_artifacts/drawer_open_td3_main_seed_extension_2m"
DRAWERCLOSE_EXPERIMENT = "experiments_post_review/2026-11-19_drawerclose_td3_main_seed_extension"
DRAWERCLOSE_ARTIFACTS = REPO / "metaworld_Drawer_Close_TD3/paper_artifacts"
ARMS = ("her", "saber", "hgr", "per")
LABELS = {"her": "HER", "saber": "SABER (Ours)", "hgr": "HGR", "per": "PER-full"}
INTERVAL_COLUMNS = "cohort,contrast,metric,n,mean_difference,ci95_low,ci95_high,positive_pairs\n"


@dataclass(frozen=True)
class Publication:
    task: str
    experiment: str
    artifact: Path
    verifier: str
    required_arms: tuple[str, ...] = ()


DRAWEROPEN = Publication("DrawerOpen-v3", EXPERIMENT, ARTIFACT, "td3_seed_extension_results.py")
DRAWERCLOSE_HER_SABER = Publication(
    "DrawerClose-v3", DRAWERCLOSE_EXPERIMENT,
    DRAWERCLOSE_ARTIFACTS / "drawer_close_td3_her_saber_seed_extension_2m",
    "drawerclose_td3_her_saber_results.py", ("her", "saber"),
)
DRAWERCLOSE_MAIN = Publication(
    "DrawerClose-v3", DRAWERCLOSE_EXPERIMENT,
    DRAWERCLOSE_ARTIFACTS / "drawer_close_td3_main_seed_extension_2m",
    "drawerclose_td3_main_results.py", ARMS,
)


def status_snapshot(queue, publication=DRAWEROPEN):
    if not publication.required_arms:
        return queue
    return {**queue, "arms": {
        arm: group if arm in publication.required_arms else {"status": group["status"]}
        for arm, group in queue["arms"].items()
    }}


def completed_keys(queue, publication=DRAWEROPEN):
    protocol = queue["protocol"]
    required = {"task": publication.task, "backbone": "TD3", "order": list(ARMS),
                "seeds": list(range(6, 11)), "steps_per_run": 2000000,
                "total_new_runs": 20, "bootstrap_resamples": 10000, "bootstrap_seed": 2027}
    for key, value in required.items():
        assert_close(protocol[key], value)
    if set(queue["arms"]) != set(ARMS):
        raise ValueError("Unexpected queue arms")
    if publication.required_arms == ARMS and queue.get("status") != "completed":
        raise ValueError("Require completed final production queue")
    result = set()
    for arm in publication.required_arms or ARMS:
        states = queue["arms"][arm]["seeds"]
        if set(states) != {str(s) for s in range(6, 11)}:
            raise ValueError("Incomplete queue population")
        for seed, state in states.items():
            if state["status"] == "completed":
                if state.get("exit_code") != 0 or state["step"] != 2000000 or state.get("nonfinite_loss"):
                    raise ValueError("Invalid completion record")
                result.add((arm, int(seed)))
        if queue["arms"][arm]["status"] == "completed" and not all((arm, s) in result for s in range(6, 11)):
            raise ValueError("Arm marked complete with unfinished seeds")
        if publication.required_arms and queue["arms"][arm]["status"] != "completed":
            raise ValueError("Require all five completed seeds of each included method")
    if not result:
        raise ValueError("No completed production runs")
    return result


def summarize(rows, queue, publication=DRAWEROPEN):
    selected = completed_keys(queue, publication)
    included = [a for a in ARMS if any(arm == a for arm, _ in selected)]
    expected = selected | {(a, s) for a in included for s in range(1, 6)}
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
                contrasts[f"saber_minus_{arm}"] = {
                    m: paired([indexed["saber", s][m] for s in seeds], [indexed[arm, s][m] for s in seeds])
                    for m in FIELDS}
        cohorts[label] = {"summary": summary, "paired": contrasts}
    full_queue = (all(queue["arms"][a]["status"] == "completed" for a in ARMS)
                  if publication.required_arms else len(selected) == 20)
    return {"snapshot_utc": queue["updated_utc"], "full_queue_completed": full_queue,
            "new_completed_runs": len(selected), "historical_reference_runs": len(expected) - len(selected),
            "complete_arms": complete, "partial_arms": [a for a in included if a not in complete],
            "cohorts": cohorts, "per_seed": rows}


def products(root, publication=DRAWEROPEN):
    provenance = read(root / "provenance.json")
    queue = read(root / "provenance/queue_snapshot.json")
    expected = read(root / "provenance/expected_configs.json")
    assert_close(provenance["experiment"], publication.experiment)
    assert_close(queue, status_snapshot(queue, publication))
    assert_close(read(root / "protocol.json"), queue["protocol"])
    selected = completed_keys(queue, publication)
    assert_close(set(expected), {f"{a}:{s}" for a, s in selected})
    rows, curves, runtimes = [], [], []
    for entry in provenance["runs"]:
        cohort = "historical_1_5" if entry["seed"] <= 5 else "extension_6_10"
        assert_close(entry["cohort"], cohort)
        assert_close(entry["published_directory"], f"data/{cohort}/{entry['arm']}/seed_{entry['seed']}")
        row, points = published_run(root, entry, expected, queue)
        if publication.required_arms and row["seed"] >= 6:
            folder = within(root, entry["published_directory"])
            claim = read(folder / "launch_record.json")
            assert_close(claim["phase"], "production")
            if any(arg.split("=", 1)[0] in {"--resume", "--resume_path", "--smoke"} for arg in claim["argv"]):
                raise ValueError("Unexpected resume/smoke production argument")
            summary = read(folder / "summary.json")
            seconds, speed = summary["runtime_total_sec"], summary["final_steps_per_sec"]
            if (not all(math.isfinite(v) and v > 0 for v in (seconds, speed)) or
                    not math.isclose(seconds * speed, 2000000, rel_tol=1e-6)):
                raise ValueError("Invalid runtime or throughput")
            runtimes.append({"arm": row["arm"], "seed": row["seed"], "runtime_seconds": seconds,
                             "runtime_hours": seconds / 3600, "steps_per_second": speed})
        rows.append(row)
        curves.extend({"arm": row["arm"], "seed": row["seed"], "step": s, "success": v} for s, v in points)
    report = summarize(rows, queue, publication)
    assert_close(report, read(root / "reference_report.json"))
    actual_dirs = {p.parent.relative_to(root).as_posix() for p in (root / "data").rglob("evaluation.csv")}
    assert_close(actual_dirs, {e["published_directory"] for e in provenance["runs"]})
    included = {r["arm"] for r in rows}
    sources = provenance["source_files_sha256"]
    if {Path(p).parts[1] for p in sources} != included:
        raise ValueError("Source coverage differs from included methods")
    actual_sources = {p.relative_to(root).as_posix() for p in (root / "source_snapshot").rglob("*") if p.is_file()}
    assert_close(actual_sources, set(sources))
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
    if not aggregates:
        raise ValueError("At least one complete five-seed extension is required")
    result = {"analysis/report.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
              "analysis/seed_metrics.csv": csv_text(report["per_seed"]),
              "analysis/aggregate_metrics.csv": csv_text(aggregates),
              "analysis/paired_differences.csv": csv_text(intervals) if intervals else INTERVAL_COLUMNS,
              "analysis/learning_curves.csv": csv_text(curves)}
    if publication.required_arms:
        result["analysis/runtime_metrics.csv"] = csv_text(runtimes)
    return result


def make_readme(report, queue, publication=DRAWEROPEN):
    task = publication.task.removesuffix("-v3")
    full_publication = publication.required_arms == ARMS
    title = ("HER and SABER Seed Extension" if publication.required_arms == ("her", "saber")
             else "Completed Main-Paper Seed Extensions")
    lines = [f"# {task} TD3: {title}", "",
             f"Snapshot: **{report['snapshot_utc']}**. This is not a live dashboard.", "",
             f"This artifact contains **{report['new_completed_runs']} completed new runs** and "
             f"{report['historical_reference_runs']} historical seed-level references.",
             "Each new run has 2M environment steps. Only completed runs are exported."]
    if full_publication:
        lines += ["HGR and PER-full seeds 6-10 are newly published. The ten HER/SABER extension runs "
                  "from the earlier comparison at fb4e55a are reused, not new independent runs.",
                  "Seeds 1-5 are the retained main-paper references. There are 40 distinct runs "
                  "across the four methods and ten matched seeds.",
                  "The complete extension includes zero-final-success runs.",
                  "[Runtime records](analysis/runtime_metrics.csv) cover all twenty extension runs."]
    else:
        lines.append("Unfinished methods are listed as status only; their intermediate curves and results are omitted.")
    if publication.required_arms and not full_publication:
        lines += ["Only HER and SABER results are included; other methods appear as timestamped status only.",
                  "Seeds 1-5 are reused, not additional executions. The two methods are compared on paired seeds.",
                  "[Runtime records](analysis/runtime_metrics.csv) cover all ten extension runs."]
    lines += ["", "## Queue Coverage", "", "| Method | Completed new seeds | State at capture |", "| --- | --- | --- |"]
    selected = completed_keys(queue, publication)
    for arm in ARMS:
        seeds = [str(s) for s in range(6, 11) if (arm, s) in selected]
        lines.append(f"| {LABELS[arm]} | {', '.join(seeds) if seeds else 'None'} | {queue['arms'][arm]['status']} |")
    for label, cohort in report["cohorts"].items():
        lines += ["", f"## {label}", "",
                  "| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for arm, values in cohort["summary"].items():
            cells = [f"{values[m]['mean']:.6f} +/- {values[m]['sd']:.6f}" for m in FIELDS]
            lines.append(f"| {LABELS[arm]} | {' | '.join(cells)} | {values['hit90_reached']}/{len(COHORTS[label])} |")
        if cohort["paired"]:
            lines += ["", "| AUC contrast | Mean difference | Paired 95% CI | Positive pairs |",
                      "| --- | ---: | --- | ---: |"]
            for contrast, values in cohort["paired"].items():
                v = values["auc"]
                lines.append(f"| {contrast} | {v['mean_difference']:+.6f} | [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}] | {v['positive_pairs']}/{v['n']} |")
    if not any(c["paired"] for c in report["cohorts"].values()):
        lines += ["", "**No new paired comparison is available.** SABER and a comparator must both have all five",
                  "extension seeds complete before paired contrasts are reported. The paired CSV contains a header only.",
                  "Do not compare ten HER seeds against five historical SABER seeds as a paired ten-seed result."]
    lines += ["", "## Completed New Seed Results", "",
              "| Method | Seed | AUC | FQ-AUC | Final success | Hit90 step |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["per_seed"]:
        if row["seed"] >= 6:
            hit = row["hit90_step"] if row["hit90_step"] is not None else "Not reached"
            lines.append(f"| {LABELS[row['arm']]} | {row['seed']} | {row['auc']:.6f} | {row['final_quarter_auc']:.6f} | {row['final_success']:.2f} | {hit} |")
    lines += ["", "## Protocol and Interpretation", "",
              f"- {publication.task} / TD3 / sparse reward v2; UTD=1, HER ratio 0.8, horizon 200, 20 evaluation episodes every 10k steps.",
              "- Every completed run has 2M environment steps and 1,995,001 recorded update iterations. TD3 retains its delayed actor updates.",
              "- Success is eval/env_success_mean. AUC uses all 200 checkpoints from 10k to 2M, divided by 1,990,000; FQ-AUC covers 1.5M..2M, divided by 500,000. No smoothing, interpolation or silently shortened horizon.",
              "- Final success is evaluated at 2M. Hit90 means the first evaluation >=0.90, not sustained success. SD is across seeds with ddof=1.",
              "- Historical seeds 1-5, new seeds 6-10 and pooled seeds 1-10 are separate. Pooling does not establish historical environment identity.",
              "- All completed zero/low-success runs are retained. Zero success is a learning outcome, not an omitted or crashed run.",
              "- No checkpoint continuation or outcome-based exclusion is used. Only methods with all five new seeds complete receive aggregate results.",
              "- Paired contrasts, when available, use 10,000 seed-row bootstrap resamples with default_rng(2027), endpoints 2.5% and 97.5%. These are approximate, pointwise, unadjusted intervals, not simultaneous guarantees.",
              "- Retained source/preset and historical metric parity were audited. Original per-run full configs, execution commits and exact historical dependency locks were not retained; byte-for-byte historical environment identity is not asserted.",
              "- Original internal B0 names may contain ddpg, but the retained agent backend is TD3. Actual launch arguments are recorded per run.",
              "- Throughput records reflect five-worker concurrency, not a dedicated compute-matched comparison.", "",
              "## Included and Omitted", "",
              "- [Seed metrics](analysis/seed_metrics.csv), [aggregates](analysis/aggregate_metrics.csv), [paired intervals](analysis/paired_differences.csv), [curves](analysis/learning_curves.csv), [report](analysis/report.json).",
              "- data/historical_1_5 contains original retained evaluation curves only, not reconstructed historical execution configs.",
              "- data/extension_6_10 contains complete evaluation curves, configs, summaries, simulator metadata and actual launch records.",
              "- source_snapshot contains byte-identical executed trainer sources for included methods only. Original launchers are historical presets, not the actual extension commands.",
              "- [Provenance](provenance.json), [source origins](provenance/source_origins.json), [queue snapshot](provenance/queue_snapshot.json) and [protocol](protocol.json) record scope and omissions.",
              "- Workspace prefixes are redacted in JSON; unavailable optional diagnostics become null. Required loss and evaluation values are validated as finite before export.",
              "- Checkpoints, replay buffers, raw training logs, caches and unfinished result directories are omitted. This is a compact result artifact, not a standalone training bundle.", "",
              "## Verification", "", "From the repository root with Python 3.12 and NumPy:", "", "```bash",
              f"python3 ICML_27/tools/{publication.verifier} verify", "```", "",
              "This read-only CPU check regenerates metrics from every included evaluation point, checks completion/config/source hashes and verifies the manifest. It does not import trainers or modify the live queue.", ""]
    return "\n".join(lines)


def verify_native_report(report, native, protocol):
    """Match the final queue analysis, excluding fields not exported in the compact table."""
    assert_close(native["protocol"], protocol)
    fields = ("arm", "seed", *FIELDS, "hit90_step")
    expected_rows = sorted(native["per_seed"], key=lambda row: (ARMS.index(row["arm"]), row["seed"]))
    assert_close(report["per_seed"], [{key: row[key] for key in fields} for row in expected_rows])
    assert_close(set(report["cohorts"]), set(native["cohorts"]))
    for label, cohort in report["cohorts"].items():
        ref = native["cohorts"][label]
        assert_close(cohort["paired"], ref["paired"])
        assert_close(set(cohort["summary"]), set(ref["summary"]))
        for arm, values in cohort["summary"].items():
            other = ref["summary"][arm]
            assert_close(values["hit90_reached"], other["hit90_reached"])
            for metric in FIELDS:
                assert_close(values[metric], {"mean": other[metric]["mean"], "sd": other[metric]["std"]})


def verify(root=None, *, publication=DRAWEROPEN):
    root = publication.artifact if root is None else root
    check_hashes(root, read(root / "provenance.json")["source_files_sha256"])
    for name, value in products(root, publication).items():
        if (root / name).read_text() != value:
            raise ValueError(f"Derived result mismatch: {name}")
    report = read(root / "analysis/report.json")
    if (root / "README.md").read_text() != make_readme(report, read(root / "provenance/queue_snapshot.json"), publication):
        raise ValueError("README differs from verified results/status")
    if (root / "MANIFEST.sha256").read_text() != inventory(root):
        raise ValueError("Artifact manifest mismatch")
    check_links(root / "README.md")
    return {"passed": True, "new_runs": report["new_completed_runs"],
            "historical_runs": report["historical_reference_runs"],
            "evaluation_points": len(report["per_seed"]) * 200, "complete_arms": report["complete_arms"],
            "paired_contrasts": sum(len(c["paired"]) for c in report["cohorts"].values()),
            "source_files": len(read(root / "provenance.json")["source_files_sha256"])}


def export(workspace, destination=None, *, publication=DRAWEROPEN):
    destination = publication.artifact if destination is None else destination
    workspace, destination = workspace.resolve(), destination.resolve()
    origin = within(workspace, publication.experiment)
    if destination.exists() or workspace == REPO.resolve() or destination == workspace or workspace in destination.parents:
        raise ValueError("Use a separate publication checkout and a new destination")
    # Import only the experiment's read-only analysis helpers, never worker/trainer modules.
    sys.path.insert(0, str(origin))
    api, common = importlib.import_module("analysis"), importlib.import_module("common")
    common.verify_snapshots()
    api.verify_reference_metrics()
    if publication.required_arms:
        validation = origin / "outputs/validation"
        approval = read(validation / "approved.json")
        assert_close(approval["passed"], True)
        assert_close(approval["smoke_arms"], list(ARMS))
        assert_close(approval["package_hashes"], common.package_hashes())
        for name in ("expected_configs", "preflight", "smoke_results"):
            assert_close(approval[f"{name}_sha256"], sha(validation / f"{name}.json"))
    queue_bytes = (origin / "outputs/queue_status.json").read_bytes()
    queue = status_snapshot(json.loads(queue_bytes), publication)
    selected = completed_keys(queue, publication)
    arms = [a for a in ARMS if any(k[0] == a for k in selected)]
    expected = read(origin / "outputs/validation/expected_configs.json")
    old = [r for r in api.reference_rows() if r["arm"] in arms]
    new = [api.read_run(a, s, expected_config=expected[f"{a}:{s}"])
           for a, s in sorted(selected, key=lambda k: (ARMS.index(k[0]), k[1]))]
    rows = [{k: r[k] for k in ("arm", "seed", *FIELDS, "hit90_step")} for r in old + new]
    report = summarize(rows, queue, publication)
    native_report = origin / "outputs/analysis/report.json"
    if publication.required_arms == ARMS:
        verify_native_report(report, read(native_report), queue["protocol"])
    elif publication.required_arms:
        indexed = {(r["arm"], r["seed"]): r for r in rows}
        for cohort, cs in COHORTS.items():
            for metric, value in report["cohorts"][cohort]["paired"]["saber_minus_her"].items():
                assert_close(value, api.paired([indexed["saber", s][metric] for s in cs],
                                               [indexed["her", s][metric] for s in cs]))
    destination.mkdir(parents=True)
    sources, entries = {}, []
    for name, digest in read(origin / "snapshot_manifest.json").items():
        if not any(name.startswith(f"sources/{a}/") for a in arms):
            continue
        target = destination / "source_snapshot" / Path(name).relative_to("sources")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin / name, target)
        sources[target.relative_to(destination).as_posix()] = digest
    for row in report["per_seed"]:
        arm, seed = row["arm"], row["seed"]
        cohort = "historical_1_5" if seed <= 5 else "extension_6_10"
        dest = destination / "data" / cohort / arm / f"seed_{seed}"
        dest.mkdir(parents=True)
        hashes = {}
        if seed <= 5:
            source = origin / f"references/{arm}/seed_{seed}/evaluation.csv"
            shutil.copy2(source, dest / "evaluation.csv")
            hashes[source.relative_to(workspace).as_posix()] = sha(source)
        else:
            run = origin / f"outputs/production/{arm}/seed_{seed:02d}"
            files = [(run / n, n) for n in ("config.json", "summary.json", "env_snapshot.json")]
            files.append((run.with_suffix(".claim.json"), "launch_record.json"))
            for source, name in files:
                write_json(dest / name, scrub(read(source), workspace))
                hashes[source.relative_to(workspace).as_posix()] = sha(source)
            with (run / "metrics.csv").open(newline="") as stream:
                evaluations = [{"step": int(float(r["step"])), "success": r["eval/env_success_mean"]}
                               for r in csv.DictReader(stream) if r.get("eval/env_success_mean", "") != ""]
            (dest / "evaluation.csv").write_text(csv_text(evaluations))
            hashes[(run / "metrics.csv").relative_to(workspace).as_posix()] = sha(run / "metrics.csv")
        entries.append({"arm": arm, "seed": seed, "cohort": cohort,
                        "published_directory": dest.relative_to(destination).as_posix(), "original_files_sha256": hashes})
    origins = read(origin / "provenance/source_origins.json")
    origins["arms"] = {k: v for k, v in origins["arms"].items() if k in arms}
    origins["curves"] = [r for r in origins["curves"] if r["arm"] in arms]
    write_json(destination / "provenance/source_origins.json", scrub(origins, workspace))
    write_json(destination / "provenance/expected_configs.json", scrub({f"{a}:{s}": expected[f"{a}:{s}"] for a, s in sorted(selected)}, workspace))
    write_json(destination / "provenance/queue_snapshot.json", scrub(queue, workspace))
    write_json(destination / "provenance/environment.json", scrub(read(origin / "outputs/validation/environment.json"), workspace))
    shutil.copy2(origin / "protocol.json", destination / "protocol.json")
    provenance = {"exported_utc": datetime.now(timezone.utc).isoformat(),
               "experiment": publication.experiment, "source_files_sha256": sources, "runs": entries,
               "original_snapshot_manifest_sha256": sha(origin / "snapshot_manifest.json"),
               "original_queue_snapshot_sha256": hashlib.sha256(queue_bytes).hexdigest(),
               "historical_limit": origins["historical_limit"],
               "transformations": ["workspace path replaced with <WORKSPACE> in JSON",
                                   "unavailable optional nonfinite diagnostics serialized as null",
                                   "all 200 evaluation steps retained per completed run; training rows omitted"],
               "omitted": ["checkpoints", "replay buffers", "raw training logs", "unfinished results", "unexecuted trainers"]}
    if publication.required_arms == ARMS:
        provenance["native_final_report_sha256"] = sha(native_report)
    write_json(destination / "provenance.json", provenance)
    write_json(destination / "reference_report.json", report)
    for name, value in products(destination, publication).items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    (destination / "README.md").write_text(make_readme(report, queue, publication))
    (destination / "MANIFEST.sha256").write_text(inventory(destination))
    print(json.dumps(verify(destination, publication=publication), indent=2))


def main(publication=DRAWEROPEN):
    parser = argparse.ArgumentParser(description=f"Export completed {publication.task} TD3 results.")
    sub = parser.add_subparsers(dest="command", required=True)
    exporter = sub.add_parser("export")
    exporter.add_argument("--workspace", type=Path, required=True)
    exporter.add_argument("--output", type=Path, default=publication.artifact)
    sub.add_parser("verify").add_argument("--root", type=Path, default=publication.artifact)
    args = parser.parse_args()
    if args.command == "export":
        export(args.workspace, args.output, publication=publication)
    else:
        print(json.dumps(verify(args.root.resolve(), publication=publication), indent=2))


if __name__ == "__main__":
    main()
