"""Export or verify completed Drawer SAC results, without importing training code."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics
from datetime import datetime, timezone

from export_local_experiments import BASELINES, RECOVERY, PUBLISHED_NAMES, scrub, sha, write_json
from verify_archive import check_hashes, check_links

REPO = Path(__file__).resolve().parents[2]
TASKS = {"drawer_open": "metaworld_Drawer_Open_SAC", "drawer_close": "metaworld_Drawer_Close_SAC"}
METHODS = ("her", "saber", "hgr", "per")
CONFIG_METHODS = {"her": "sac", "saber": "sac_saber", "hgr": "sac_hgr", "per": "sac_her_per_full"}
LABELS = {"her": "HER", "saber": "SABER (Ours)", "hgr": "HGR", "per": "PER (full replay)"}
SEEDS = list(range(1, 6))
STEPS = list(range(10000, 2000001, 10000))
METRICS = ("auc", "fq_auc", "final", "runtime_seconds", "steps_per_second")


def read(path):
    return json.loads(path.read_text())


def within(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or root.resolve() not in path.parents:
        raise ValueError(f"Path outside package: {relative}")
    return path


def artifact_path(repo, task):
    return repo / TASKS[task] / "paper_artifacts" / f"{task}_sac_her_saber_hgr_per_2m"


def assert_close(a, b):
    if isinstance(a, dict):
        if set(a) != set(b):
            raise ValueError(f"Different keys: {set(a) ^ set(b)}")
        for key in a:
            assert_close(a[key], b[key])
    elif isinstance(a, list):
        if len(a) != len(b):
            raise ValueError("Different lengths")
        for x, y in zip(a, b):
            assert_close(x, y)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        if not math.isfinite(a) or not math.isclose(a, b, rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"Numeric mismatch: {a} != {b}")
    elif a != b:
        raise ValueError(f"Mismatch: {a} != {b}")


def curve(rows):
    points = [(int(r["step"]), float(r["eval/env_success_mean"])) for r in rows]
    if [s for s, _ in points] != STEPS:
        raise ValueError("Missing, duplicate, reordered or unexpected checkpoint")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for _, v in points):
        raise ValueError("Invalid evaluation success")
    return points


def auc(points):
    return sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(points, points[1:])) / (points[-1][0] - points[0][0])


def paired(differences):
    import numpy as np
    d = np.asarray(differences, dtype=np.float64)
    if d.shape != (5,) or not np.isfinite(d).all():
        raise ValueError("All five finite paired differences are required")
    ids = np.random.default_rng(2027).integers(0, 5, size=(10000, 5))
    means = d[ids].mean(axis=1)
    return {"n_pairs": 5, "mean": float(d.mean()), "positive_pairs": int((d > 0).sum()),
            "ci95": np.quantile(means, [.025, .975]).tolist(),
            "ci97_5": np.quantile(means, [.0125, .9875]).tolist(),
            "ci98_75": np.quantile(means, [.00625, .99375]).tolist()}


def run_metrics(root, entry):
    folder = within(root, entry["published_directory"])
    config, summary = read(folder / "config.json"), read(folder / "summary.json")
    completion = read(folder / "completion.json")
    if completion["status"] != "completed" or config["resume"] or config["resume_path"] is not None:
        raise ValueError("Incomplete or checkpoint-continued execution")
    expected = {"task_name": entry["task"], "method": CONFIG_METHODS[entry["method"]],
                "total_steps": 2000000, "updates_per_step": 1, "sac_auto_alpha": True,
                "sac_alpha": .2, "sac_target_entropy": -4.0, "eval_freq": 10000,
                "eval_episodes": 20, "max_episode_steps": 200, "her_ratio": .8}
    for key, value in expected.items():
        assert_close(config[key], value)
    for key in ("env_seed", "torch_seed", "numpy_seed"):
        assert_close(config["seeds"][key], entry["seed"])
    assert_close(config["seeds"]["eval_seed"], 10000 + entry["seed"])
    assert_close(config["lambda_support"], 1.0 if entry["method"] == "saber" else 0.0)
    if entry["method"] == "saber" and config["weight_scope"] != "critic_only":
        raise ValueError("Unexpected SABER scope")
    assert_close(summary["total_steps"], 2000000)
    assert_close(summary["total_updates"], 1995001)
    if not summary["final_losses"] or any(v is None or not math.isfinite(v) for v in summary["final_losses"].values()):
        raise ValueError("Invalid final training losses")
    with (folder / "evaluations.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    points = curve(rows)
    full_auc = auc(points)
    assert_close(full_auc, summary["eval_AUC_success"])
    assert_close(points[-1][1], summary["final_eval_env_success"])
    if summary["runtime_total_sec"] <= 0 or summary["final_steps_per_sec"] <= 0:
        raise ValueError("Invalid recorded runtime")
    metrics = {"task": entry["task"], "method": entry["method"], "seed": entry["seed"],
               "auc": full_auc, "fq_auc": auc([p for p in points if p[0] >= 1500000]),
               "final": points[-1][1], "hit90": next((s for s, v in points if v >= .9), None),
               "updates": 1995001, "runtime_seconds": summary["runtime_total_sec"],
               "steps_per_second": summary["final_steps_per_sec"]}
    return metrics, rows


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def products(root):
    provenance = read(root / "provenance.json")
    task = provenance["task"]
    entries = provenance["runs"]
    if len(entries) != 20 or {(r["task"], r["method"], r["seed"]) for r in entries} != {
            (task, m, s) for m in METHODS for s in SEEDS}:
        raise ValueError("Expected exactly four methods x five unique seeds")
    runs, learning = [], []
    for entry in sorted(entries, key=lambda r: (METHODS.index(r["method"]), r["seed"])):
        metrics, rows = run_metrics(root, entry)
        runs.append(metrics)
        for row in rows:
            learning.append({"task": task, "method": entry["method"], "seed": entry["seed"],
                             "step": int(row["step"]), "env_success": float(row["eval/env_success_mean"]),
                             "runtime_seconds": float(row["runtime/total_sec"])})
    aggregates, comparisons, pairs = [], {}, []
    by_method = {m: [r for r in runs if r["method"] == m] for m in METHODS}
    for m, rows in by_method.items():
        aggregates.append({"task": task, "method": m,
                           **{f"{metric}_{stat}": fn([r[metric] for r in rows])
                              for metric in METRICS for stat, fn in (("mean", statistics.mean), ("sd", statistics.stdev))},
                           "hit90_count": sum(r["hit90"] is not None for r in rows)})
    for control in ("her", "hgr", "per"):
        contrast = f"saber-{control}"
        c = paired([s["auc"] - b["auc"] for s, b in zip(by_method["saber"], by_method[control])])
        comparisons[contrast] = c
        pairs.append({"task": task, "contrast": contrast, "n_pairs": c["n_pairs"],
                      "mean_auc_difference": c["mean"], "positive_pairs": c["positive_pairs"],
                      **{f"{level}_{side}": c[level][i] for level in ("ci95", "ci97_5", "ci98_75")
                         for i, side in enumerate(("low", "high"))}})
    report = {"task": task, "status": "completed", "seeds": SEEDS, "runs": runs,
              "aggregates": aggregates, "comparisons": comparisons}
    assert_close(report, read(root / "reference_report.json"))
    return {"analysis/report.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
            "analysis/seed_metrics.csv": csv_text(runs), "analysis/aggregate_metrics.csv": csv_text(aggregates),
            "analysis/paired_auc.csv": csv_text(pairs), "analysis/learning_curves.csv": csv_text(learning)}


def inventory(root):
    result = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError(f"Symlink in artifact: {p}")
        if p.is_file() and p.name != "MANIFEST.sha256":
            if p.suffix in {".pt", ".pth", ".log", ".pyc"}:
                raise ValueError(f"Unexpected large/raw payload: {p}")
            result[str(p.relative_to(root))] = sha(p)
    return "".join(f"{h}  {p}\n" for p, h in sorted(result.items()))


def check_sources(repo, sources, workspace=None):
    for source in sources.values():
        path = within(repo, source["directory"])
        provenance = read(path.parent / "provenance.json")
        check_hashes(path, provenance["source_files_sha256"])
        assert_close(sha(path / "source_manifest.json"), source["source_manifest_sha256"])
        if workspace is not None:
            original = within(workspace, provenance["original_directory"])
            check_hashes(original, provenance["source_files_sha256"])


def verify(repo, task):
    root = artifact_path(repo, task)
    provenance = read(root / "provenance.json")
    assert_close(provenance["task"], task)
    check_sources(repo, provenance["sources"])
    for entry in provenance["runs"]:
        completion = read(within(root, entry["published_directory"]) / "completion.json")
        assert_close(completion["source_manifest_sha256"], provenance["sources"][entry["source_family"]]["source_manifest_sha256"])
    for name, expected in products(root).items():
        if (root / name).read_text() != expected:
            raise ValueError(f"Derived result differs: {task}/{name}")
    if (root / "MANIFEST.sha256").read_text() != inventory(root):
        raise ValueError(f"Manifest mismatch: {task}")
    check_links(root / "README.md")
    check_links(repo / TASKS[task] / "README.md")
    return {"task": task, "validated_runs": 20, "evaluation_points": 4000, "passed": True}


def artifact_readme(task, report):
    name = "DrawerOpen" if task == "drawer_open" else "DrawerClose"
    lines = [f"# {name} SAC: HER, SABER, HGR and PER", "",
             "Completed post-AAAI cohort: four methods x seeds 1-5 x 2M steps = 20 unique runs.",
             "All seeds, including zero-success and collapsed runs, are included. This is not a 10-seed cohort.", "",
             "## Results", "", "Values are mean +/- sample SD (ddof=1).", "",
             "| Method | Final success | AUC | FQ-AUC | Hit90 coverage | Steps/s |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["aggregates"]:
        fmt = lambda metric: f"{row[metric + '_mean']:.3f} +/- {row[metric + '_sd']:.3f}"
        lines.append(f"| {LABELS[row['method']]} | {fmt('final')} | {fmt('auc')} | {fmt('fq_auc')} | {row['hit90_count']}/5 | {fmt('steps_per_second')} |")
    lines += ["", "| Paired AUC contrast | Mean difference | 95% CI | Positive pairs |",
              "| --- | ---: | --- | ---: |"]
    for contrast, c in report["comparisons"].items():
        lines.append(f"| {contrast} | {c['mean']:+.6f} | [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] | {c['positive_pairs']}/5 |")
    lines += ["", "## Interpretation", ""]
    if task == "drawer_open":
        lines += ["HER and SABER have nearly identical mean AUC and final success; their paired CI includes zero.",
                  "HGR and PER solve all five runs at the final checkpoint, whereas HER and SABER solve three."]
    else:
        lines += ["SABER has a lower sample mean AUC than all three controls and zero final success in every seed.",
                  "All methods reach Hit90 at least once in each seed, but this does not imply sustained success:",
                  "the low final-quarter AUC and final success expose later collapse, particularly for SABER."]
    lines += ["SABER-minus-PER AUC is negative with a 95% interval below zero in this task.",
              "Intervals crossing zero establish neither superiority nor equivalence. Results are limited to this cohort.", "",
              "## Executed Protocol", "",
              "- FPP-reference SAC: stochastic Gaussian actor, twin critics, automatic temperature, initial alpha=0.2, target entropy=-4.",
              "- MetaWorld drawer v3, sparse reward version v2; 2M environment steps, UTD=1, HER ratio=0.8, horizon=200.",
              "- SABER applies support weights only to critic losses and queries support at the squashed policy mean.",
              "- HGR is the hierarchical prioritized-goal baseline; PER is full prioritized replay with IS critic weights, not loss-only TD weighting.",
              "- Each run has 1,995,001 updates and 20 evaluation episodes at each of 200 checkpoints.",
              "- AUC integrates exactly 10k..2M, divided by 1,990,000; FQ-AUC integrates 1.5M..2M, divided by 500,000. No smoothing or missing-grid truncation.",
              "- Success is `eval/env_success_mean`: ever-success within an evaluation episode. Final success means this metric at 2M, not final-state success.",
              "- Hit90 is the first evaluation with success >=0.90, not a stability criterion; a blank seed-level Hit90 means not reached.",
              "- Paired AUC bootstrap uses seed rows, 10,000 resamples and NumPy default_rng(2027). Quantiles: 95%=(2.5,97.5), 97.5%=(1.25,98.75), 98.75%=(0.625,99.375).",
              "- The two-task SABER/HER family uses 97.5% per contrast; the four SABER/HGR and SABER/PER contrasts use 98.75%. These are nominal Bonferroni adjustments with approximate small-sample bootstrap coverage.",
              "- HGR/PER are an exploratory extension after the HER/SABER outcomes were observed. Do not describe all four arms as one preregistered experiment.",
              "- Steps/s and runtime are recorded whole-run measurements, including instrumentation and varying concurrent workloads, not a controlled efficiency comparison.", "",
              "## Recovery and Known Caveats", "",
              "Three DrawerOpen HER runs (seeds 1-3) were already complete before the power interruption and were retained.",
              "Five interrupted attempts were rerun from step zero. The recovery's 17 fresh runs plus three retained runs",
              "form one 20-run HER/SABER cohort across the two tasks. HGR/PER add 20 new runs. No checkpoint continuation or spliced curves.",
              "The task-specific attempt inventory and run origins are in [provenance.json](provenance.json).", "",
              "The inherited MetaWorld observation includes the original goal in its last three components while HER",
              "changes the separately concatenated goal. All compared SAC methods share this representation.",
              "Its causal effect has not been established; the executed implementation is preserved rather than silently corrected.",
              "Older fixed-temperature DrawerOpen prototype results are separate and are not pooled here.", "",
              "## Files and Verification", "",
              "- [Seed-level results](analysis/seed_metrics.csv), [aggregate results](analysis/aggregate_metrics.csv), [paired CIs](analysis/paired_auc.csv), [full curves](analysis/learning_curves.csv), [report](analysis/report.json).",
              "- `data/<method>/seed_<n>/`: recorded config, summary, simulator description, execution environment, completion record and all evaluation rows/columns plus elapsed time and throughput.",
              "- JSON exports redact the workspace prefix and replace unavailable nonfinite diagnostic fields with null; required loss/performance fields must remain finite. CSV evaluation strings are retained without rounding.",
              "- `reference_report.json` is the task's original validated analysis, used to cross-check independently recomputed results. Original input and exported hashes are recorded.",
              "- Checkpoints, replay buffers, full training logs and live control files are not published. These compact data regenerate performance metrics, not full training trajectories or every diagnostic.",
              "- Exact executed [HER/SABER source](../../../ICML_27/experiments/draweropen_drawerclose_sac_her_vs_saber/source/) and [HGR/PER source](../../../ICML_27/experiments/draweropen_drawerclose_sac_hgr_per/source/) remain in ICML_27; no generic/legacy trainer is substituted.", "",
              "From the repository root, with Python 3.12 and NumPy (recorded analysis version: 2.3.5):", "",
              "```bash", f"python3 ICML_27/tools/sac_results.py verify --task {task}", "```", "",
              "Verification is read-only and CPU-only. It checks every checkpoint, completion, source snapshot, metric, CI, derived CSV and artifact SHA256; it never launches training.", ""]
    return "\n".join(lines)


def export(workspace, repo):
    origin = workspace / "experiments_post_review"
    final = read(origin / BASELINES / "analysis/final.json")
    recovery = read(origin / RECOVERY / "analysis/final.json")
    if final["status"] != "completed" or final["incomplete"] or len(final["runs"]) != 40:
        raise ValueError("Require all 40 completed SAC runs")
    sources = {family: {"directory": f"ICML_27/experiments/{PUBLISHED_NAMES[name]}/source",
                        "source_manifest_sha256": sha(origin / name / "source_manifest.json")}
               for family, name in (("her_saber", RECOVERY), ("hgr_per", BASELINES))}
    check_sources(repo, sources, workspace)
    if any(artifact_path(repo, task).exists() for task in TASKS):
        raise ValueError("Refuse to overwrite an existing result artifact")
    for task in TASKS:
        root = artifact_path(repo, task)
        root.mkdir(parents=True)
        entries = []
        reference_runs = []
        for row in sorted((r for r in final["runs"] if r["task"] == task), key=lambda r: (METHODS.index(r["method"]), r["seed"])):
            method, seed = row["method"], row["seed"]
            run_name = f"{task}_sac_{method}_s{seed}_2m"
            run_dir = Path(row["run_dir"]) if "run_dir" in row else origin / BASELINES / "runs" / run_name
            run_dir.relative_to(workspace)
            claim_dir = run_dir.parent.parent / ".production_claims" / run_name
            family = "her_saber" if method in {"her", "saber"} else "hgr_per"
            dest = root / "data" / method / f"seed_{seed}"
            dest.mkdir(parents=True)
            original_hashes = {}
            for source, filename in [(run_dir / n, n) for n in ("config.json", "summary.json", "env_snapshot.json")] + [
                    (claim_dir / "environment.json", "environment.json"), (claim_dir / "status.json", "completion.json")]:
                original_hashes[str(source.relative_to(workspace))] = sha(source)
                write_json(dest / filename, scrub(read(source), workspace))
            with (run_dir / "metrics.csv").open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields = [k for k in reader.fieldnames if k == "step" or k.startswith("eval/") or k in {"runtime/total_sec", "runtime/steps_per_sec"}]
                evaluations = [{k: row[k] for k in fields} for row in reader if row["eval/env_success_mean"] != ""]
            curve(evaluations)
            (dest / "evaluations.csv").write_text(csv_text(evaluations))
            original_hashes[str((run_dir / "metrics.csv").relative_to(workspace))] = sha(run_dir / "metrics.csv")
            entry = {"task": task, "method": method, "seed": seed, "source_family": family,
                     "original_directory": str(run_dir.relative_to(workspace)),
                     "published_directory": str(dest.relative_to(root)),
                     "checkpoint_continuation": False, "reused_completed": row.get("reused_completed", False),
                     "original_files_sha256": original_hashes}
            metrics, _ = run_metrics(root, entry)
            expected = {k: row[k] for k in metrics}
            assert_close(metrics, expected)
            reference_runs.append(expected)
            entries.append(entry)
        report = {"task": task, "status": "completed", "seeds": SEEDS, "runs": reference_runs,
                  "aggregates": [a for a in final["aggregates"] if a["task"] == task],
                  "comparisons": final["comparisons"][task]}
        write_json(root / "reference_report.json", report)
        write_json(root / "provenance.json", {"task": task, "exported_utc": datetime.now(timezone.utc).isoformat(),
                   "sources": sources, "runs": entries,
                   "original_analysis_sha256": sha(origin / BASELINES / "analysis/final.json"),
                   "original_attempts": [a for a in recovery["original_attempts"] if a["job"][0] == task],
                   "transformations": ["workspace prefix replaced with <WORKSPACE>", "JSON nonfinite diagnostic fields replaced with null",
                                       "CSV restricted to all evaluation rows and eval/runtime columns; numeric strings not rounded"],
                   "omitted": ["checkpoint weights and replay buffers", "full training logs", "process locks", "virtual environment"]})
        for name, value in products(root).items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
        (root / "README.md").write_text(artifact_readme(task, report))
        (root / "MANIFEST.sha256").write_text(inventory(root))
        print(json.dumps(verify(repo, task)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("export").add_argument("--workspace", type=Path, required=True)
    commands.add_parser("verify").add_argument("--task", choices=list(TASKS) + ["all"], default="all")
    args = parser.parse_args()
    if args.command == "export":
        if args.workspace.resolve() == REPO.resolve():
            parser.error("Export into a separate publication checkout, not the live workspace")
        export(args.workspace.resolve(), REPO)
    else:
        for task in TASKS if args.task == "all" else [args.task]:
            print(json.dumps(verify(REPO, task)))


if __name__ == "__main__":
    main()
