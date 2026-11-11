"""Publish completed DrawerOpen SAC HER/SABER extensions without training."""
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


def require_complete(state):
    if state["status"] != "completed" or set(state["seeds"]) != {str(s) for s in range(6, 11)}:
        raise ValueError("All five extension seeds must be complete")
    for run in state["seeds"].values():
        if (run["status"] != "completed" or run.get("exit_code") != 0
                or run["step"] != 2000000 or run.get("nonfinite_loss")):
            raise ValueError("Invalid completion record")


def require_complete_her(state):
    require_complete(state)


def science(config):
    return {k: v for k, v in config.items() if k not in CONFIG_METADATA}


def paired_metrics(rows, seeds):
    import numpy as np

    indexed = {(r["method"], r["seed"]): r for r in rows}
    metrics = ("auc", "fq_auc", "final")
    diff = np.asarray([[indexed["saber", s][m] - indexed["her", s][m]
                        for m in metrics] for s in seeds], dtype=np.float64)
    if not np.isfinite(diff).all():
        raise ValueError("Nonfinite paired difference")
    # Reuse seed-row indices across metrics; retain the planned three-AUC family.
    indices = np.random.default_rng(2027).integers(0, len(seeds), size=(10000, len(seeds)))
    means = diff[indices].mean(axis=1)
    return {m: {"n": len(seeds), "mean_difference": float(diff[:, i].mean()),
                "ci95": np.quantile(means[:, i], [.025, .975]).tolist(),
                "ci_bonferroni_three": np.quantile(means[:, i], [.05 / 6, 1 - .05 / 6]).tolist(),
                "positive_pairs": int((diff[:, i] > 0).sum()),
                "seed_differences": diff[:, i].tolist()} for i, m in enumerate(metrics)}


def summarize(rows, methods=("her",)):
    methods = tuple(methods)
    if methods not in (("her",), ("her", "saber")):
        raise ValueError("Only explicitly selected HER or HER/SABER may be published")
    keys = [(r["method"], r["seed"]) for r in rows]
    if len(keys) != 10 * len(methods) or set(keys) != {(m, s) for m in methods for s in range(1, 11)}:
        raise ValueError("Publish all ten seeds per selected method, including zero-success runs")
    rows = sorted(rows, key=lambda r: (methods.index(r["method"]), r["seed"]))
    aggregates = []
    for cohort, seeds in COHORTS.items():
        for method in methods:
            selected = [r for r in rows if r["seed"] in seeds and r["method"] == method]
            aggregates.append({"cohort": cohort, "method": method, "n": len(selected),
                               **{f"{metric}_{stat}": fn([r[metric] for r in selected])
                                  for metric in METRICS
                                  for stat, fn in (("mean", statistics.mean), ("sd", statistics.stdev))},
                               "hit90_count": sum(r["hit90"] is not None for r in selected)})
    comparisons = {c: paired_metrics(rows, seeds) for c, seeds in COHORTS.items()} if "saber" in methods else {}
    return {"task": "drawer_open", "published_methods": list(methods), "new_completed_runs": 5 * len(methods),
            "historical_reference_runs": 5 * len(methods), "runs": rows, "aggregates": aggregates,
            "comparisons": comparisons,
            "comparison_status": "SABER minus HER; HGR/PER not published" if comparisons else
            "Not published: this snapshot contains HER only"}


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
    methods = provenance["published_methods"]
    if methods not in (["her"], ["her", "saber"]):
        raise ValueError("Unexpected publication scope")
    assert_close(provenance["sources"], {"her_saber": SOURCE})
    states = {m: read(root / f"provenance/{m}_completion.json") for m in methods}
    for state in states.values():
        require_complete(state)
    expected = read(root / "provenance/expected_configs.json")
    assert_close(set(expected), {f"{m}:{s}" for m in methods for s in range(6, 11)})
    protocol = read(root / "protocol.json")
    for key, value in {"task": "drawer-open-v3", "seeds": list(range(6, 11)),
                       "steps_per_run": 2000000,
                       "backbone": "SAC, FPP-reference automatic entropy temperature",
                       "bootstrap_seed": 2027, "bootstrap_resamples": 10000}.items():
        assert_close(protocol[key], value)
    rows, learning, configs = [], [], {m: [] for m in methods}
    for entry in sorted(provenance["runs"], key=lambda r: (methods.index(r["method"]), r["seed"])):
        seed, method = entry["seed"], entry["method"]
        cohort = "historical_1_5" if seed <= 5 else "extension_6_10"
        assert_close(entry["cohort"], cohort)
        assert_close(entry["task"], "drawer_open")
        assert_close(entry["checkpoint_continuation"], False)
        assert_close(entry["published_directory"], f"data/{cohort}/{method}/seed_{seed}")
        folder = within(root, entry["published_directory"])
        config = read(folder / "config.json")
        configs[method].append(science(config))
        metrics, evaluations = run_metrics(root, entry)
        rows.append(metrics)
        completion = read(folder / "completion.json")
        if seed >= 6:
            assert_close(completion, states[method]["seeds"][str(seed)])
            wanted = expected[f"{method}:{seed}"]
            assert_close({k: config[k] for k in wanted}, wanted)
            assert_close(config["sac_target_entropy_effective"], -4.0)
            assert_close(config["weight_scope"], "critic_only" if method == "saber" else "none")
            launch = read(folder / "launch_record.json")
            for key, value in {"arm": method, "seed": seed, "phase": "production", "resume": False}.items():
                assert_close(launch[key], value)
            assert_close(launch["config"], wanted)
            result = completion["result"]
            mapping = {"auc": "auc", "final_quarter_auc": "fq_auc", "final_success": "final",
                       "hit90_step": "hit90", "total_updates": "updates",
                       "runtime_seconds": "runtime_seconds", "steps_per_second": "steps_per_second"}
            assert_close(result["arm"], method)
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
            learning.append({"cohort": cohort, "method": method, "seed": seed,
                             "step": int(row["step"]), "env_success": float(row["eval/env_success_mean"]),
                             "runtime_seconds": elapsed})
    for values in configs.values():
        for config in values[1:]:
            assert_close(values[0], config)
    report = summarize(rows, methods)
    assert_close(report, read(root / "reference_report.json"))
    actual = {p.parent.relative_to(root).as_posix() for p in (root / "data").rglob("evaluations.csv")}
    assert_close(actual, {r["published_directory"] for r in provenance["runs"]})
    result = {"analysis/report.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
              "analysis/seed_metrics.csv": csv_text(report["runs"]),
              "analysis/aggregate_metrics.csv": csv_text(report["aggregates"]),
              "analysis/learning_curves.csv": csv_text(learning)}
    if report["comparisons"]:
        intervals, differences = [], []
        for cohort, metrics in report["comparisons"].items():
            for metric, value in metrics.items():
                intervals.append({"cohort": cohort, "contrast": "saber_minus_her", "metric": metric,
                                  "n": value["n"], "mean_difference": value["mean_difference"],
                                  "positive_pairs": value["positive_pairs"],
                                  **{f"{level}_{side}": value[level][i]
                                     for level in ("ci95", "ci_bonferroni_three")
                                     for i, side in enumerate(("low", "high"))}})
                differences.extend({"cohort": cohort, "contrast": "saber_minus_her", "metric": metric,
                                    "seed": seed, "difference": difference}
                                   for seed, difference in zip(COHORTS[cohort], value["seed_differences"]))
        result["analysis/paired_statistics.csv"] = csv_text(intervals)
        result["analysis/paired_seed_differences.csv"] = csv_text(differences)
    return result


def make_readme(report, provenance, state):
    if "saber" in report["published_methods"]:
        return comparison_readme(report, provenance)
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


def comparison_readme(report, provenance):
    labels = {"her": "HER", "saber": "SABER (Ours)"}
    lines = ["# DrawerOpen SAC: HER and SABER Seed Extensions", "",
             f"Published snapshot: **{provenance['exported_utc']}**.",
             f"SABER seeds 6-10 completed on **{provenance['saber_finished_utc']}**.", "",
             "**HER and SABER only:** ten completed new runs (five per method, seeds 6-10),",
             "each at 2M steps, plus ten historical seed-level references (1-5).",
             "New HGR, PER and CPU-pilot results are not included. The original HER run files are unchanged.", "",
             "## Results", "", "Values are mean +/- sample SD (ddof=1).", "",
             "| Cohort | Method | Seeds | AUC | FQ-AUC | Final success | Hit90 | Steps/s |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["aggregates"]:
        seeds = COHORTS[row["cohort"]]
        cells = [f"{row[m + '_mean']:.6f} +/- {row[m + '_sd']:.6f}"
                 for m in ("auc", "fq_auc", "final")]
        lines.append(f"| {row['cohort']} | {labels[row['method']]} | {seeds.start}-{seeds.stop - 1} | "
                     f"{' | '.join(cells)} | {row['hit90_count']}/{row['n']} | "
                     f"{row['steps_per_second_mean']:.3f} +/- {row['steps_per_second_sd']:.3f} |")
    lines += ["", "### Paired AUC: SABER Minus HER", "",
              "| Cohort | Mean difference | Pointwise 95% CI | Nominal 98.333333% CI | Positive pairs |",
              "| --- | ---: | --- | --- | ---: |"]
    for cohort, metrics in report["comparisons"].items():
        c = metrics["auc"]
        lines.append(f"| {cohort} | {c['mean_difference']:+.6f} | "
                     f"[{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] | "
                     f"[{c['ci_bonferroni_three'][0]:+.6f}, {c['ci_bonferroni_three'][1]:+.6f}] | "
                     f"{c['positive_pairs']}/{c['n']} |")
    lines += ["", "### New Seed-Level Results", "",
              "| Method | Seed | AUC | FQ-AUC | Final success | Hit90 step | Runtime (h) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in report["runs"]:
        if r["seed"] >= 6:
            hit = str(r["hit90"]) if r["hit90"] is not None else "Not reached"
            lines.append(f"| {labels[r['method']]} | {r['seed']} | {r['auc']:.6f} | {r['fq_auc']:.6f} | "
                         f"{r['final']:.3f} | {hit} | {r['runtime_seconds']/3600:.3f} |")
    pooled = report["comparisons"]["pooled_1_10"]["auc"]
    lower = sum(d < 0 for d in pooled["seed_differences"])
    tied = sum(d == 0 for d in pooled["seed_differences"])
    lines += ["", "## Interpretation and Scope", "",
              f"Across ten paired seeds, SABER AUC is higher in {pooled['positive_pairs']}, lower in {lower}, and equal in {tied}.",
              "SABER has a slightly higher mean AUC, but both the pointwise and adjusted intervals include zero.",
              "These data establish neither superiority nor equivalence. They do not demonstrate a causal support mechanism.",
              "Both methods have mean final success 0.700 and Hit90 coverage 7/10. The final-zero seeds are 2, 3 and 8 in both methods.",
              "All zero/low-success seeds are retained. Do not pool the historical, extension and combined summaries as independent cohorts.",
              "The ten historical references overlap the [original five-seed artifact](../drawer_open_sac_her_saber_hgr_per_2m/README.md).",
              "Only five new SABER executions are added by this update; the previously published HER runs are not new executions.", "",
              "## Protocol, Statistics and Caveats", "",
              "- FPP-reference automatic-temperature SAC: twin critics, stochastic actor, initial alpha=0.2, target entropy=-4. This is not the legacy fixed-alpha prototype.",
              "- DrawerOpen-v3 sparse reward v2, 2M steps, UTD=1, HER ratio=0.8, horizon=200, gamma=0.98, batch=256, replay capacity=300000.",
              "- SABER reweights critic losses only and queries support at the squashed policy mean: K=5, minimum local count=4, tail-512. Actor objectives and Bellman targets remain the inherited SAC definitions.",
              "- Each run has 1,995,001 optimizer updates and 20 evaluation episodes at each of 200 checkpoints.",
              "- Success is eval/env_success_mean, ever-success within an evaluation episode. AUC integrates the exact 10k..2M grid divided by 1,990,000; FQ-AUC integrates 1.5M..2M divided by 500,000. No missing-grid truncation or smoothing.",
              "- Hit90 is the first crossing, not sustained success. Missing Hit90 stays missing.",
              "- Pair by seed, not row order. NumPy default_rng(2027), 10,000 seed-row bootstrap resamples; all metrics share the same sampled indices within each cohort. Percentile 95% endpoints are 0.025 and 0.975.",
              "- Adjusted endpoints are 0.05/6 and 1-0.05/6, retaining the predeclared three-AUC-comparison family (SABER versus HER/HGR/PER), although only HER is published here. Small-sample coverage is approximate, not an exact simultaneous guarantee.",
              "- FQ-AUC/final-success intervals are secondary descriptive analyses. The three-contrast adjustment does not cover every metric and all three overlapping cohort summaries jointly.",
              "- All new seeds began from scratch without continuation, splicing, replacement or outcome-based exclusion. Historical HER seeds 1-3 were complete pre-outage runs; the other retained HER/SABER references are complete fresh recovery runs.",
              "- Recorded scientific configs match across historical and new seeds within each method after documented seed/output metadata is removed. Source files match the already published executed snapshot byte-for-byte.",
              "- The inherited original-goal observation components plus the separately HER-relabeled goal are unchanged in both methods; their causal effect is unresolved.",
              "- Core recorded package versions matched the prelaunch audit. This is not a claim of byte-for-byte historical host identity. Runtime includes evaluation, checkpoints and concurrent workloads, not a controlled wall-clock comparison.",
              "- The fixed-size extension was selected after historical outcomes were observed. The four-method protocol is retained, but HGR/PER extension results are not included here.", "",
              "## Files and Verification", "",
              "- [Seed metrics](analysis/seed_metrics.csv), [aggregates](analysis/aggregate_metrics.csv), [evaluation curves](analysis/learning_curves.csv), [report](analysis/report.json).",
              "- [Paired statistics](analysis/paired_statistics.csv) and [seed-level differences](analysis/paired_seed_differences.csv).",
              "- [Provenance](provenance.json), [HER completion](provenance/her_completion.json), [SABER completion](provenance/saber_completion.json), [protocol](protocol.json), [environment](provenance/environment.json).",
              "- [Executed source](../../../ICML_27/experiments/draweropen_drawerclose_sac_her_vs_saber/source/) is unchanged and hash-verified. Configs, summaries, evaluation rows, completion and launch records are included.",
              "- Full-log finite-loss counts are retained for new runs. Workspace prefixes are redacted; optional unavailable JSON diagnostics become null. Evaluation strings are not rounded.",
              "- Checkpoints, replay buffers, full raw logs, new HGR/PER results and CPU-pilot results are omitted. This compact result artifact is not a standalone training bundle.", "",
              "From the repository root with Python 3.12 and NumPy:", "", "```bash",
              "python3 ICML_27/tools/sac_seed_extension_results.py verify", "```", "",
              "This read-only CPU check regenerates metrics and paired intervals from 4,000 evaluation points, checks completion/config/source hashes and verifies the manifest. It does not import trainers or change the live queue.", ""]
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
    return {"passed": True, "new_runs": report["new_completed_runs"],
            "historical_runs": report["historical_reference_runs"],
            "evaluation_points": len(report["runs"]) * 200,
            "published_methods": report["published_methods"],
            "paired_comparisons": len(report["comparisons"]), "source_files": len(sources)}


def export_method(workspace, repo, root, origin, method, state):
    require_complete(state)
    historical = artifact_path(repo, "drawer_open")
    old_provenance = read(historical / "provenance.json")
    old_entries = {r["seed"]: r for r in old_provenance["runs"] if r["method"] == method}
    assert_close(set(old_entries), set(range(1, 6)))
    reference_metrics = {r["seed"]: r for r in read(historical / "reference_report.json")["runs"] if r["method"] == method}
    entries, rows = [], []
    for seed in range(1, 11):
        cohort = "historical_1_5" if seed <= 5 else "extension_6_10"
        dest = root / "data" / cohort / method / f"seed_{seed}"
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
            previous = origin / f"outputs/production/{method}/seed_{seed:02d}"
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
        entry = {"task": "drawer_open", "method": method, "seed": seed, "cohort": cohort,
                 "checkpoint_continuation": False, "original_directory": original_directory,
                 "published_directory": dest.relative_to(root).as_posix(), "original_files_sha256": hashes}
        metrics, _ = run_metrics(root, entry)
        if seed <= 5:
            assert_close(metrics, reference_metrics[seed])
        rows.append(metrics)
        entries.append(entry)
    return entries, rows


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
    root.mkdir(parents=True)
    entries, rows = export_method(workspace, repo, root, origin, "her", state)
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
                  "historical_artifact": artifact_path(repo, "drawer_open").relative_to(repo).as_posix(),
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


def append_saber(workspace, repo=REPO):
    workspace, repo = workspace.resolve(), repo.resolve()
    root, origin = repo / ARTIFACT, within(workspace, EXPERIMENT)
    if repo == workspace or workspace in root.parents:
        raise ValueError("Never publish into the live experiment workspace")
    verify(repo, root)
    provenance = read(root / "provenance.json")
    assert_close(provenance["published_methods"], ["her"])
    protected = {p.relative_to(root).as_posix(): sha(p) for cohort in ("historical_1_5", "extension_6_10")
                 for p in (root / "data" / cohort / "her").rglob("*") if p.is_file()}
    protected.update({p: sha(root / p) for p in ("protocol.json", "provenance/her_completion.json")})
    queue = read(origin / "outputs/queue_status.json")
    state = queue["arms"]["saber"]
    require_complete(state)
    assert_close(queue["protocol"], read(root / "protocol.json"))
    assert_close(sha(origin / "snapshot_manifest.json"), provenance["original_snapshot_manifest_sha256"])
    check_hashes(origin, read(origin / "snapshot_manifest.json"))
    approval = read(origin / "outputs/validation/approved.json")
    assert_close(approval["passed"], True)
    assert_close(approval["package_hashes"], provenance["approved_launcher_hashes"])
    check_hashes(origin, approval["package_hashes"])
    path = origin / "outputs/validation/expected_configs.json"
    assert_close(sha(path), approval["expected_configs_sha256"])
    expected = read(root / "provenance/expected_configs.json")
    for key, value in read(path).items():
        if key.startswith("saber:"):
            expected[key] = scrub(value, workspace)
        elif key.startswith("her:"):
            assert_close(expected[key], scrub(value, workspace))
    entries, rows = export_method(workspace, repo, root, origin, "saber", state)
    old_rows = read(root / "reference_report.json")["runs"]
    provenance["runs"].extend(entries)
    provenance["her_publication_utc"] = provenance["exported_utc"]
    provenance["exported_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["saber_finished_utc"] = state["finished_utc"]
    provenance["published_methods"] = ["her", "saber"]
    provenance["omitted"] = ["new HGR/PER results", "CPU pilot", "checkpoints", "replay buffers", "full training logs"]
    environment = read(origin / "outputs/validation/environment.json")
    environment["historical_checks"] = [r for r in environment["historical_checks"] if r["arm"] in ("her", "saber")]
    write_json(root / "provenance/environment.json", environment)
    write_json(root / "provenance/expected_configs.json", expected)
    write_json(root / "provenance/saber_completion.json", scrub(state, workspace))
    write_json(root / "provenance.json", provenance)
    report = summarize(old_rows + rows, ("her", "saber"))
    write_json(root / "reference_report.json", report)
    for name, value in products(root).items():
        (root / name).write_text(value)
    (root / "README.md").write_text(comparison_readme(report, provenance))
    check_hashes(root, protected)
    (root / "MANIFEST.sha256").write_text(inventory(root))
    return verify(repo, root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("export").add_argument("--workspace", type=Path, required=True)
    commands.add_parser("append-saber").add_argument("--workspace", type=Path, required=True)
    commands.add_parser("verify")
    args = parser.parse_args()
    if args.command == "export":
        result = export(args.workspace)
    elif args.command == "append-saber":
        result = append_saber(args.workspace)
    else:
        result = verify()
    print(json.dumps(result, indent=2))
