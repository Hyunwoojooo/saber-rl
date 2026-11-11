"""Publish only the authorized, completed DrawerOpen SAC HER seed extension."""
import argparse
import csv
import json
import math
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path

from export_local_experiments import scrub, sha, write_json
from sac_results import (
    METRICS, artifact_path, assert_close, csv_text, inventory, read,
    run_metrics, within,
)
from verify_archive import check_hashes, check_links

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiments_post_review/2026-11-09_draweropen_sac_main_seed_extension"
ARTIFACT = Path("metaworld_Drawer_Open_SAC/paper_artifacts/drawer_open_sac_main_seed_extension_2m")
SOURCE = "ICML_27/experiments/draweropen_drawerclose_sac_her_vs_saber/source"
COHORTS = {"historical_1_5": range(1, 6), "extension_6_10": range(6, 11),
           "pooled_1_10": range(1, 11)}
LOSS_FIELDS = ("loss/critic", "loss/actor", "loss/td_error_abs_mean", "sac/alpha", "sac/alpha_loss")
CONFIG_METADATA = {"seeds", "run_name", "output_root", "experiment_id", "thread_hint",
                   "terminal_convention", "sac_target_entropy_effective"}


def require_complete_her(state):
    if state["status"] != "completed" or set(state["seeds"]) != {str(s) for s in range(6, 11)}:
        raise ValueError("All five HER extension seeds must be complete")
    for run in state["seeds"].values():
        if (run["status"] != "completed" or run.get("exit_code") != 0
                or run["step"] != 2000000 or run.get("nonfinite_loss")):
            raise ValueError("Invalid HER completion record")


def science(config):
    return {k: v for k, v in config.items() if k not in CONFIG_METADATA}


def summarize(rows):
    keys = [(r["method"], r["seed"]) for r in rows]
    if len(keys) != 10 or set(keys) != {("her", s) for s in range(1, 11)}:
        raise ValueError("Publish exactly HER seeds 1-10, including all zero-success runs")
    rows = sorted(rows, key=lambda r: r["seed"])
    aggregates = []
    for cohort, seeds in COHORTS.items():
        selected = [r for r in rows if r["seed"] in seeds]
        aggregates.append({"cohort": cohort, "method": "her", "n": len(selected),
                           **{f"{metric}_{stat}": fn([r[metric] for r in selected])
                              for metric in METRICS
                              for stat, fn in (("mean", statistics.mean), ("sd", statistics.stdev))},
                           "hit90_count": sum(r["hit90"] is not None for r in selected)})
    return {"task": "drawer_open", "published_methods": ["her"], "new_completed_runs": 5,
            "historical_reference_runs": 5, "runs": rows, "aggregates": aggregates,
            "comparisons": {}, "comparison_status": "Not published: this snapshot contains HER only"}


def scan_training_csv(path):
    counts = {k: 0 for k in LOSS_FIELDS}
    evaluations = []
    previous = (0, 0.0)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = [k for k in reader.fieldnames if k == "step" or k.startswith("eval/")
                  or k in {"runtime/total_sec", "runtime/steps_per_sec"}]
        for row in reader:
            step, elapsed = int(row["step"]), float(row["runtime/total_sec"])
            if not math.isfinite(elapsed) or step < previous[0] or elapsed < previous[1]:
                raise ValueError("Training step/runtime reset or nonfinite runtime")
            previous = step, elapsed
            if step >= 5000:
                for key in LOSS_FIELDS:
                    value = row.get(key, "")
                    if value != "":
                        number = float(value)
                        if not math.isfinite(number) or (key == "sac/alpha" and number <= 0):
                            raise ValueError(f"Invalid {key} at step {step}")
                        counts[key] += 1
            if row.get("eval/env_success_mean", "") != "":
                evaluations.append({k: row[k] for k in fields})
    if previous[0] != 2000000 or not all(counts.values()):
        raise ValueError("Incomplete training or missing optimizer diagnostics")
    return evaluations, {"finite_values_checked": counts, "last_step": previous[0],
                         "monotonic_step_and_runtime": True}


def products(root):
    provenance = read(root / "provenance.json")
    assert_close(provenance["published_methods"], ["her"])
    assert_close(provenance["sources"], {"her_saber": SOURCE})
    state = read(root / "provenance/her_completion.json")
    require_complete_her(state)
    expected = read(root / "provenance/expected_configs.json")
    assert_close(set(expected), {f"her:{s}" for s in range(6, 11)})
    protocol = read(root / "protocol.json")
    for key, value in {"task": "drawer-open-v3", "seeds": list(range(6, 11)),
                       "steps_per_run": 2000000,
                       "backbone": "SAC, FPP-reference automatic entropy temperature"}.items():
        assert_close(protocol[key], value)
    rows, learning, configs = [], [], []
    for entry in sorted(provenance["runs"], key=lambda r: r["seed"]):
        seed = entry["seed"]
        cohort = "historical_1_5" if seed <= 5 else "extension_6_10"
        assert_close(entry["cohort"], cohort)
        assert_close(entry["method"], "her")
        assert_close(entry["task"], "drawer_open")
        assert_close(entry["checkpoint_continuation"], False)
        assert_close(entry["published_directory"], f"data/{cohort}/her/seed_{seed}")
        folder = within(root, entry["published_directory"])
        config = read(folder / "config.json")
        configs.append(science(config))
        metrics, evaluations = run_metrics(root, entry)
        rows.append(metrics)
        completion = read(folder / "completion.json")
        if seed >= 6:
            assert_close(completion, state["seeds"][str(seed)])
            wanted = expected[f"her:{seed}"]
            assert_close({k: config[k] for k in wanted}, wanted)
            assert_close(config["sac_target_entropy_effective"], -4.0)
            assert_close(config["weight_scope"], "none")
            launch = read(folder / "launch_record.json")
            for key, value in {"arm": "her", "seed": seed, "phase": "production", "resume": False}.items():
                assert_close(launch[key], value)
            assert_close(launch["config"], wanted)
            result = completion["result"]
            mapping = {"auc": "auc", "final_quarter_auc": "fq_auc", "final_success": "final",
                       "hit90_step": "hit90", "total_updates": "updates",
                       "runtime_seconds": "runtime_seconds", "steps_per_second": "steps_per_second"}
            assert_close(result["arm"], "her")
            assert_close(result["seed"], seed)
            for original, exported in mapping.items():
                assert_close(result[original], metrics[exported])
            audit = read(folder / "training_validation.json")
            assert_close(audit["last_step"], 2000000)
            assert_close(audit["monotonic_step_and_runtime"], True)
            if set(audit["finite_values_checked"]) != set(LOSS_FIELDS) or not all(
                    v > 0 for v in audit["finite_values_checked"].values()):
                raise ValueError("Invalid training validation record")
        for row in evaluations:
            elapsed = float(row["runtime/total_sec"])
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("Invalid evaluation runtime")
            learning.append({"cohort": cohort, "method": "her", "seed": seed,
                             "step": int(row["step"]), "env_success": float(row["eval/env_success_mean"]),
                             "runtime_seconds": elapsed})
    for config in configs[1:]:
        assert_close(configs[0], config)
    report = summarize(rows)
    assert_close(report, read(root / "reference_report.json"))
    actual = {p.parent.relative_to(root).as_posix() for p in (root / "data").rglob("evaluations.csv")}
    assert_close(actual, {r["published_directory"] for r in provenance["runs"]})
    return {"analysis/report.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
            "analysis/seed_metrics.csv": csv_text(report["runs"]),
            "analysis/aggregate_metrics.csv": csv_text(report["aggregates"]),
            "analysis/learning_curves.csv": csv_text(learning)}


def make_readme(report, provenance, state):
    lines = ["# DrawerOpen SAC: HER Seed Extension", "",
             f"Published snapshot: **{provenance['exported_utc']}**.",
             f"HER seeds 6-10 completed on **{state['finished_utc']}**.", "",
             "**Only HER is included in this publication.** Five new seeds (6-10), each at 2M steps,",
             "are combined descriptively with five retained historical HER runs (1-5).",
             "No new SABER, HGR, PER or CPU-pilot results are exported. This is not a completed four-method comparison.", "",
             "## Results", "", "Values are mean +/- sample SD (ddof=1).", "",
             "| HER cohort | Seeds | AUC | FQ-AUC | Final success | Hit90 | Steps/s |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in report["aggregates"]:
        seeds = COHORTS[r["cohort"]]
        cells = [f"{r[m + '_mean']:.6f} +/- {r[m + '_sd']:.6f}"
                 for m in ("auc", "fq_auc", "final")]
        lines.append(f"| {r['cohort']} | {seeds.start}-{seeds.stop - 1} | {' | '.join(cells)} | "
                     f"{r['hit90_count']}/{r['n']} | {r['steps_per_second_mean']:.3f} +/- {r['steps_per_second_sd']:.3f} |")
    lines += ["", "### New Seed-Level Results", "",
              "| Seed | AUC | FQ-AUC | Final success | Hit90 step | Runtime (h) |",
              "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in report["runs"]:
        if r["seed"] >= 6:
            hit = str(r["hit90"]) if r["hit90"] is not None else "Not reached"
            lines.append(f"| {r['seed']} | {r['auc']:.6f} | {r['fq_auc']:.6f} | {r['final']:.3f} | {hit} | {r['runtime_seconds']/3600:.3f} |")
    lines += ["", "## Interpretation and Scope", "",
              "Four of five new seeds reach final success 1.0; seed 8 has zero success throughout and is retained.",
              "The pooled result is descriptive. No SABER-minus-HER interval is computed by this artifact:",
              "comparing ten HER seeds against five historical or selectively completed SABER seeds would not be the planned paired analysis.",
              "The historical runs overlap the [original five-seed artifact](../drawer_open_sac_her_saber_hgr_per_2m/README.md); do not count them twice.", "",
              "## Protocol and Provenance", "",
              "- FPP-reference SAC with automatic entropy temperature, initial alpha=0.2, target entropy=-4, twin critics and stochastic actor. This is not the older fixed-alpha prototype.",
              "- DrawerOpen-v3 sparse reward v2, 2M steps, UTD=1, HER ratio=0.8, horizon=200, gamma=0.98, batch=256, replay capacity=300000.",
              "- Each run has 1,995,001 optimizer updates and 20 evaluation episodes at each of 200 checkpoints.",
              "- AUC uses the exact 10k..2M grid divided by 1,990,000. FQ-AUC uses 1.5M..2M divided by 500,000. Curves are unsmoothed and retain all checkpoints.",
              "- Success is eval/env_success_mean (ever-success within each episode). Hit90 is the first crossing, not sustained success.",
              "- New seeds started from scratch; no checkpoint continuation, replacement or splicing. Historical HER seeds 1-3 were retained complete pre-outage runs; seeds 4-5 were complete fresh recovery runs.",
              "- Exact recorded scientific configurations match across all ten HER runs after removing documented seed/output metadata. Frozen sources are byte-identical to the linked previously published source.",
              "- The inherited original-goal observation components plus separately relabeled HER goal are unchanged. Their causal effect is not established.",
              "- Recorded core package versions were matched in the prelaunch audit; this does not assert byte-for-byte identity of every historical machine component.",
              "- Whole-run runtime includes evaluation, checkpoints and concurrent workloads. It is not a controlled wall-clock comparison.",
              "- The predeclared four-method protocol is retained for provenance, but its other methods are not results in this snapshot. The fixed-size extension followed inspection of the historical outcomes.", "",
              "## Files", "",
              "- [Seed metrics](analysis/seed_metrics.csv), [aggregate metrics](analysis/aggregate_metrics.csv), [full evaluation curves](analysis/learning_curves.csv), [report](analysis/report.json).",
              "- [Provenance](provenance.json), [HER completion](provenance/her_completion.json), [protocol](protocol.json), [environment audit](provenance/environment.json).",
              "- [Executed scientific source](../../../ICML_27/experiments/draweropen_drawerclose_sac_her_vs_saber/source/) is already archived. Source hashes are checked against the extension snapshot; no generic trainer is substituted.",
              "- data/ retains configs, summaries, evaluation rows, environment records and completion records. New runs also retain launch records and the full-log finite-loss check counts.",
              "- Absolute workspace prefixes are redacted in JSON; unavailable optional diagnostics become null. Evaluation strings are not rounded.",
              "- Model weights, replay buffers, full training logs and other methods' result directories are excluded. This compact artifact is not a standalone training bundle.", "",
              "## Verification", "", "From the repository root using Python 3.12:", "", "```bash",
              "python3 ICML_27/tools/sac_seed_extension_results.py verify", "```", "",
              "Verification is read-only and CPU-only. It recomputes all 2,000 evaluation points, checks source/config/completion records and manifest hashes, and never imports a trainer or starts a run.", ""]
    return "\n".join(lines)


def verify(repo=REPO, root=None):
    root = root or repo / ARTIFACT
    provenance = read(root / "provenance.json")
    sources = provenance["source_files_sha256"]
    if not sources or any(not p.startswith(SOURCE + "/") for p in sources):
        raise ValueError("Unexpected or empty source snapshot")
    check_hashes(repo, sources)
    for name, value in products(root).items():
        if (root / name).read_text() != value:
            raise ValueError(f"Derived result differs: {name}")
    report = read(root / "analysis/report.json")
    if (root / "README.md").read_text() != make_readme(report, provenance, read(root / "provenance/her_completion.json")):
        raise ValueError("README differs from verified results")
    if (root / "MANIFEST.sha256").read_text() != inventory(root):
        raise ValueError("Manifest mismatch")
    check_links(root / "README.md")
    return {"passed": True, "new_runs": 5, "historical_runs": 5, "evaluation_points": 2000,
            "published_methods": ["her"], "paired_comparisons": 0, "source_files": len(sources)}


def export(workspace, repo=REPO):
    workspace, repo = workspace.resolve(), repo.resolve()
    root, origin = repo / ARTIFACT, within(workspace, EXPERIMENT)
    if root.exists() or repo == workspace or workspace in root.parents:
        raise ValueError("Use a separate publication checkout and a new artifact")
    protocol = read(origin / "protocol.json")
    queue = read(origin / "outputs/queue_status.json")
    state = queue["arms"]["her"]
    require_complete_her(state)
    assert_close(queue["protocol"], protocol)
    snapshot = read(origin / "snapshot_manifest.json")
    check_hashes(origin, snapshot)
    approval = read(origin / "outputs/validation/approved.json")
    assert_close(approval["passed"], True)
    check_hashes(origin, approval["package_hashes"])
    expected_path = origin / "outputs/validation/expected_configs.json"
    assert_close(sha(expected_path), approval["expected_configs_sha256"])
    expected = {k: v for k, v in read(expected_path).items() if k.startswith("her:")}
    sources = {f"{SOURCE}/{Path(p).relative_to('sources/her_saber').as_posix()}": h
               for p, h in snapshot.items() if p.startswith("sources/her_saber/")}
    check_hashes(repo, sources)
    historical = artifact_path(repo, "drawer_open")
    old_provenance = read(historical / "provenance.json")
    old_entries = {r["seed"]: r for r in old_provenance["runs"] if r["method"] == "her"}
    assert_close(set(old_entries), set(range(1, 6)))
    reference_metrics = {r["seed"]: r for r in read(historical / "reference_report.json")["runs"] if r["method"] == "her"}
    root.mkdir(parents=True)
    entries, rows = [], []
    for seed in range(1, 11):
        cohort = "historical_1_5" if seed <= 5 else "extension_6_10"
        dest = root / "data" / cohort / "her" / f"seed_{seed}"
        dest.mkdir(parents=True)
        hashes = {}
        if seed <= 5:
            previous = within(historical, old_entries[seed]["published_directory"])
            for source in sorted(previous.iterdir()):
                if not source.is_file():
                    raise ValueError("Unexpected historical run directory")
                shutil.copy2(source, dest / source.name)
                hashes[source.relative_to(repo).as_posix()] = sha(source)
            original_directory = previous.relative_to(repo).as_posix()
        else:
            previous = origin / f"outputs/production/her/seed_{seed:02d}"
            original_directory = previous.relative_to(workspace).as_posix()
            files = [(previous / n, n) for n in ("config.json", "summary.json", "env_snapshot.json")]
            files.append((previous.with_suffix(".claim.json"), "launch_record.json"))
            for source, name in files:
                hashes[source.relative_to(workspace).as_posix()] = sha(source)
                write_json(dest / name, scrub(read(source), workspace))
            evaluations, audit = scan_training_csv(previous / "metrics.csv")
            hashes[(previous / "metrics.csv").relative_to(workspace).as_posix()] = sha(previous / "metrics.csv")
            (dest / "evaluations.csv").write_text(csv_text(evaluations))
            write_json(dest / "training_validation.json", audit)
            write_json(dest / "completion.json", scrub(state["seeds"][str(seed)], workspace))
        entry = {"task": "drawer_open", "method": "her", "seed": seed, "cohort": cohort,
                 "checkpoint_continuation": False, "original_directory": original_directory,
                 "published_directory": dest.relative_to(root).as_posix(), "original_files_sha256": hashes}
        metrics, _ = run_metrics(root, entry)
        if seed <= 5:
            assert_close(metrics, reference_metrics[seed])
        rows.append(metrics)
        entries.append(entry)
    environment = read(origin / "outputs/validation/environment.json")
    environment["historical_checks"] = [r for r in environment["historical_checks"] if r["arm"] == "her"]
    write_json(root / "provenance/environment.json", environment)
    write_json(root / "provenance/her_completion.json", scrub(state, workspace))
    write_json(root / "provenance/expected_configs.json", scrub(expected, workspace))
    shutil.copy2(origin / "protocol.json", root / "protocol.json")
    provenance = {"exported_utc": datetime.now(timezone.utc).isoformat(), "experiment": EXPERIMENT,
                  "published_methods": ["her"], "sources": {"her_saber": SOURCE},
                  "source_files_sha256": sources, "runs": entries,
                  "original_snapshot_manifest_sha256": sha(origin / "snapshot_manifest.json"),
                  "approved_launcher_hashes": approval["package_hashes"],
                  "historical_artifact": historical.relative_to(repo).as_posix(),
                  "transformations": ["workspace prefixes redacted", "unavailable JSON diagnostics become null",
                                      "training CSV restricted to every evaluation row and eval/runtime columns"],
                  "omitted": ["new SABER/HGR/PER results", "CPU pilot", "checkpoints", "replay buffers", "full training logs"]}
    write_json(root / "provenance.json", provenance)
    report = summarize(rows)
    write_json(root / "reference_report.json", report)
    for name, value in products(root).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    (root / "README.md").write_text(make_readme(report, provenance, state))
    (root / "MANIFEST.sha256").write_text(inventory(root))
    return verify(repo, root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("export").add_argument("--workspace", type=Path, required=True)
    commands.add_parser("verify")
    args = parser.parse_args()
    print(json.dumps(export(args.workspace) if args.command == "export" else verify(), indent=2))
