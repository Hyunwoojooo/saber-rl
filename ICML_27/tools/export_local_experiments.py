"""Publish executed experiment snapshots only; never import or launch training."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
from datetime import datetime, timezone, timedelta


ASSIGN = "2026-09-25_draweropen_ddpg_assignment"
PAL = "2026-10-01_draweropen_ddpg_pal_v2"
DIAG = "2026-10-04_draweropen_retrieval_failure_diagnostics_v2"
RETRIEVAL = "2026-10-04_draweropen_ddpg_retrieval_v1"
RECOVERY = "2026-10-12_drawer_sac_fpp_v2_recovery_s1_5"
BASELINES = "2026-10-16_drawer_sac_hgr_per_s1_5"
PUBLISHED_NAMES = {
    ASSIGN: "draweropen_ddpg_saber_vs_random_vs_td_topk",
    PAL: "draweropen_ddpg_saber_vs_pal",
    DIAG: "draweropen_ddpg_retrieval_and_failure_diagnostics",
    RETRIEVAL: "draweropen_ddpg_saber_tail_vs_random512",
    RECOVERY: "draweropen_drawerclose_sac_her_vs_saber",
    BASELINES: "draweropen_drawerclose_sac_hgr_per",
}
SPECS = {
    ASSIGN: ("DrawerOpen / DDPG: SABER vs Random vs TD-error Top-k", "completed", 30),
    PAL: ("DrawerOpen / DDPG: SABER vs PAL", "completed", 10),
    DIAG: ("DrawerOpen / DDPG: Retrieval and Failure Diagnostics", "completed_offline", 0),
    RETRIEVAL: ("DrawerOpen / DDPG: SABER Tail vs Random-512 Retrieval", "completed", 10),
    RECOVERY: ("DrawerOpen and DrawerClose / SAC: HER vs SABER", "completed", 20),
    BASELINES: ("DrawerOpen and DrawerClose / SAC: HGR and PER", "in_progress", 20),
}
REPORTS = {
    ASSIGN: ["outputs/analysis/report.json", "outputs/analysis/seed_metrics.csv",
             "outputs/analysis/final_details_20261002/extended_summary.json"],
    PAL: ["outputs/analysis/report.json", "outputs/analysis/seed_metrics.csv"],
    DIAG: ["outputs/final/REPORT.md", "outputs/final/report.json",
           "outputs/retrieval/report.json", "outputs/retrieval/aggregate_summary.csv",
           "outputs/retrieval/seed_summary.csv", "outputs/retrieval/random_repeat_summary.csv",
           "outputs/failure_analysis/report.json", "outputs/failure_analysis/run_summary.csv",
           "outputs/failure_analysis/posthoc_groups.csv", "outputs/failure_analysis/time_bands.csv",
           "outputs/failure_analysis/evaluation_curves.csv", "outputs/recovery_verification/report.json"],
    RETRIEVAL: ["outputs/analysis/report.json", "outputs/analysis/seed_metrics.csv",
                "outputs/final_retrieval/report.json", "outputs/final_retrieval/seed_summary.csv"],
    RECOVERY: ["analysis/final.json", "analysis/drawer_open.json", "analysis/drawer_close.json"],
}
SOURCE_DIRS = ("code", "tests", "upstream", "third_party", "recovery")
SOURCE_EXTS = {".py", ".md", ".json", ".sh", ".txt", ".toml", ".yaml", ".yml", ".ini", ".cfg"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def scrub(value, workspace):
    if isinstance(value, str):
        return value.replace(str(workspace), "<WORKSPACE>")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: scrub(v, workspace) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v, workspace) for v in value]
    return value


def source_paths(root):
    paths = {p for p in root.iterdir() if p.is_file() and
             (p.suffix in SOURCE_EXTS or p.name in {".gitignore", "LICENSE", "NOTICE"})}
    for directory in SOURCE_DIRS:
        for p in (root / directory).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and (
                    p.suffix in SOURCE_EXTS or p.name in {".gitignore", "LICENSE", "NOTICE"}):
                paths.add(p)
    return sorted(paths)


def copy_sources(root, dest):
    copied = {}
    for p in source_paths(root):
        if p.is_symlink():
            raise ValueError(f"Unexpected symlink: {p}")
        relative = p.relative_to(root)
        target = dest / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        copied[str(relative)] = sha(p)
        if copied[str(relative)] != sha(target):
            raise ValueError(f"Copy mismatch: {p}")
    for manifest in ("source_manifest.json", "recovery_manifest.json"):
        if (root / manifest).exists():
            for name, expected in read_json(root / manifest)["sha256"].items():
                if copied.get(name) != expected:
                    raise ValueError(f"Frozen source mismatch: {root.name}/{name}")
    if (root / "upstream_manifest.json").exists():
        for row in read_json(root / "upstream_manifest.json")["files"]:
            if copied.get(row["path"]) != row["sha256"]:
                raise ValueError(f"Upstream mismatch: {root.name}/{row['path']}")
    return copied


def copy_reports(root, dest, workspace):
    paths = set()
    for name in REPORTS.get(root.name, []):
        p = root / name
        if p.is_dir():
            paths.update(x for x in p.rglob("*") if x.is_file() and x.suffix in {".json", ".csv", ".md"})
        elif p.is_file():
            paths.add(p)
        else:
            raise FileNotFoundError(p)
    for name in ("outputs/automation/status.json", "automation/status.json", "recovery_state/status.json"):
        if (root / name).is_file():
            paths.add(root / name)
    if root.name == BASELINES and (root / "analysis/final.json").is_file():
        paths.update(root / "analysis" / name for name in ("final.json", "drawer_open.json", "drawer_close.json"))
    provenance = []
    for p in sorted(paths):
        if p.stat().st_size > 8 * 1024 * 1024:
            raise ValueError(f"Unexpectedly large report: {p}")
        relative = p.relative_to(root)
        target = dest / "results" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = p.read_bytes()
        if p.suffix == ".json":
            write_json(target, scrub(json.loads(raw), workspace))
        else:
            target.write_text(raw.decode().replace(str(workspace), "<WORKSPACE>"))
        provenance.append({"original_path": str(relative), "original_sha256": hashlib.sha256(raw).hexdigest(),
                           "published_path": str(target.relative_to(dest)), "published_sha256": sha(target)})
    return provenance


def auc(curve):
    if len(curve) < 2:
        return None
    return sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(curve, curve[1:])) / (curve[-1][0] - curve[0][0])


def capture_progress(root, timestamp):
    rows = []
    for task in ("drawer_open", "drawer_close"):
        for method in ("hgr", "per"):
            for seed in range(1, 6):
                name = f"{task}_sac_{method}_s{seed}_2m"
                claim = root / ".production_claims" / name / "status.json"
                state = read_json(claim)["status"] if claim.exists() else "pending"
                row = dict(task=task, method=method, seed=seed, status=state, step=0)
                metrics = root / "runs" / name / "metrics.csv"
                if metrics.exists():
                    # Ignore only a concurrently-written partial CSV line, never a missing checkpoint.
                    with metrics.open(newline="") as f:
                        data = [r for r in csv.DictReader(f) if None not in r and None not in r.values()]
                    curve = [(int(r["step"]), float(r["eval/env_success_mean"])) for r in data
                             if r.get("eval/env_success_mean") not in (None, "")]
                    if [x for x, y in curve] != list(range(10000, curve[-1][0] + 1, 10000)):
                        raise ValueError(f"Invalid checkpoint grid: {name}")
                    if not all(math.isfinite(y) and 0 <= y <= 1 for x, y in curve):
                        raise ValueError(f"Invalid success values: {name}")
                    row.update(step=int(data[-1]["step"]), eval_step=curve[-1][0],
                               last_success=curve[-1][1], auc_to_eval_step=auc(curve),
                               runtime_seconds=float(data[-1]["runtime/total_sec"]))
                    if state == "completed":
                        summary = read_json(root / "runs" / name / "summary.json")
                        if (row["step"] != 2000000 or len(curve) != 200 or
                                summary["total_updates"] != 1995001 or
                                abs(summary["eval_AUC_success"] - row["auc_to_eval_step"]) > 1e-12):
                            raise ValueError(f"Completion mismatch: {name}")
                rows.append(row)
    return {"captured_utc": timestamp, "automatic_live_update": False, "total_planned": 20,
            "counts": {state: sum(r["status"] == state for r in rows)
                       for state in sorted({r["status"] for r in rows})},
            "total_steps_observed": sum(r["step"] for r in rows), "total_steps_planned": 40000000,
            "runs": rows, "note": "Running AUC ends at eval_step, not at 2M; do not pool unequal horizons."}


def result_text(name, root, progress):
    if name in (ASSIGN, PAL, RETRIEVAL):
        r = read_json(root / "outputs/analysis/report.json")
        text = ["| Method | Mean AUC | AUC SD | Mean final success |", "| --- | ---: | ---: | ---: |"]
        labels = {"support": "SABER", "random": "Context-eligible random assignment",
                  "td_topk": "Context-eligible TD-error Top-k", "pal": "HER + PAL",
                  "tail": "SABER Tail-512 (reused)", "random512": "SABER Random-512 retrieval"}
        for method, a in r["arm_summary"].items():
            text.append(f"| {labels[method]} | {a['auc']['mean']:.6f} | {a['auc']['sd']:.6f} | {a['final_success']['mean']:.3f} |")
        text += ["", "All results use seeds 1-10, DrawerOpen-v3 / DDPG / HER, 2M steps per run, UTD=1.", ""]
        if name == ASSIGN:
            text += ["Question: does support-based high-weight placement add value when context eligibility,",
                     "the per-update high-weight count, 2:1 raw weights, and mean-one normalization are controlled?",
                     "Each online run computes its own eligibility and count; buffers and counts are not shared between runs.", ""]
            for key, c in r["inference"].items():
                text.append(f"- {key}: delta AUC {c['mean']:+.6f}; 95% CI [{c['ci95'][0]:.6f}, {c['ci95'][1]:.6f}]; "
                            f"Bonferroni 97.5% CI [{c['ci97_5_bonferroni'][0]:.6f}, {c['ci97_5_bonferroni'][1]:.6f}].")
        elif name == PAL:
            c = r["inference"]
            text += ["Question: does SABER offer value over a published uniform-replay loss-side baseline?",
                     "PAL uses alpha=0.4, min_priority=1, and the detached minibatch normalizer.",
                     "Only PAL's ten runs are new; the audited SABER reference is the assignment cohort.",
                     f"SABER minus PAL: {c['mean']:+.6f}; 95% CI [{c['ci95'][0]:.6f}, {c['ci95'][1]:.6f}].",
                     "Instrumentation differs; recorded runtime does not establish algorithmic time efficiency."]
        else:
            c = r["inference"]["random512_minus_tail"]
            text += ["Question: does changing coarse-bucket candidate retrieval from tail to uniform random",
                     "selection, with the same 512-candidate cap, change online learning?",
                     "This random operation selects replay candidates, not high-weight minibatch positions.",
                     "Ten Random-512 runs are new; all ten Tail references are reused, including the failed-to-learn seed.",
                     f"Random-512 minus Tail: {c['mean']:+.6f}; 95% CI [{c['ci95'][0]:.6f}, {c['ci95'][1]:.6f}].",
                     "Final checkpoint retrieval audits for both arms are included in results/outputs/final_retrieval/."]
        text += ["", "The intervals include zero: these descriptive mean differences do not establish",
                 "statistical superiority, equivalence, universal necessity of support, or generality beyond this cohort.",
                 "AUC uses the exact 10k-2M grid and denominator 2M-10k. See the full paired analysis in",
                 "[report.json](results/outputs/analysis/report.json)."]
        return "\n".join(text)
    if name == DIAG:
        return ("Completed offline analyses; no new training runs.\n\n"
                "- Ten SABER final checkpoints: 1,024 fixed HER queries per seed, five random-retrieval repeats.\n"
                "- Tail / random / full support rates: 0.537109 / 0.625352 / 0.902734.\n"
                "- Miss rates conditional on full-positive support: 0.410458 / 0.311727 / 0.\n"
                "- Failure/saturation analysis covers all 30 assignment runs. Seven had zero evaluation success;\n"
                "  saturation occurred in 7/7 of these and 0/23 of the nonzero-success runs.\n\n"
                "These are final-checkpoint and observational diagnostics, not causal failure explanations.\n"
                "Full search is a reference for the same discretized context, not true continuous support.\n"
                "Timing is CPU reference implementation timing, not production GPU/Numba speed.\n"
                "See [the detailed report](results/outputs/final/REPORT.md) and the seed-level CSVs.")
    if name == RECOVERY:
        r = read_json(root / "analysis/final.json")
        text = ["Completed: 20 runs, seeds 1-5, 2M steps each; auto-temperature SAC follows the FPP reference.", "",
                "| Task | Method | Mean AUC | AUC SD | Mean final success |", "| --- | --- | ---: | ---: | ---: |"]
        for a in r["aggregates"]:
            text.append(f"| {a['task']} | {a['method']} | {a['auc_mean']:.6f} | {a['auc_sd']:.6f} | {a['final_mean']:.3f} |")
        for task, c in r["comparisons"].items():
            text.append(f"\n{task}, SABER minus HER: {c['mean']:+.6f}, 95% CI [{c['ci95'][0]:.6f}, {c['ci95'][1]:.6f}].")
        text += ["", "Neither interval excludes zero. DrawerClose has a negative mean difference; unfavorable outcomes are retained.",
                 "Three completed DrawerOpen HER seeds were reused after the power outage; five interrupted runs",
                 "were rerun from step zero. Seventeen runs in this recovery plus three retained runs form ONE",
                 "20-run cohort, not an additional independent replication. No curves were spliced and no checkpoint continuation was used.",
                 "This is the single published SAC HER/SABER source folder. Its frozen scientific source also covers the three retained pre-outage runs.",
                 "See [final.json](results/analysis/final.json) for seed-level outcomes and the interruption inventory."]
        return "\n".join(text)
    if name == BASELINES:
        if progress["counts"].get("completed", 0) == 20:
            report = read_json(root / "analysis/final.json")
            text = ["Completed: 20 new HGR/PER runs plus 20 reused HER/SABER references; 40 unique SAC runs total.",
                    "Both tasks use seeds 1-5 and 2M steps, with FPP-reference auto-temperature SAC.",
                    "This is an exploratory extension after observing HER/SABER outcomes; no automatic seeds 6-10 extension.",
                    "PER is full prioritized replay with IS critic weights, not loss-only TD weighting.", "",
                    "| Task | Method | Mean AUC | AUC SD | Mean final success |",
                    "| --- | --- | ---: | ---: | ---: |"]
            for row in report["aggregates"]:
                text.append(f"| {row['task']} | {row['method']} | {row['auc_mean']:.6f} | {row['auc_sd']:.6f} | {row['final_mean']:.3f} |")
            text += ["", "SABER-minus-PER paired AUC intervals lie below zero for both tasks; unfavorable results are retained.",
                     "Both SABER-minus-HER intervals include zero. This neither establishes superiority nor equivalence.",
                     "DrawerClose's early Hit90 events must not be confused with sustained or final success.",
                     "See [final.json](results/analysis/final.json) for all seed metrics and adjusted paired intervals.",
                     "Compact evaluation data: [DrawerOpen](../../../metaworld_Drawer_Open_SAC/) and [DrawerClose](../../../metaworld_Drawer_Close_SAC/).",
                     "Recorded runtimes include instrumentation and varying concurrent workloads, not controlled wall-clock comparisons."]
            return "\n".join(text)
        text = ["Question: how do HER/SABER compare with HGR and full replay-prioritized PER under the matched Drawer SAC setup?",
                "Seeds 1-5, 2M steps per run, two tasks x two new methods = 20 planned runs.",
                "PER changes replay sampling; it is not the older PER-style loss-only control.", "",
                "| Task | Method | Seed | Status | Latest step | Eval step | Success at eval | AUC to eval step |",
                "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
        for r in progress["runs"]:
            values = [r["task"], r["method"], str(r["seed"]), r["status"], str(r["step"]),
                      str(r.get("eval_step", "NA")), f"{r['last_success']:.3f}" if "last_success" in r else "NA",
                      f"{r['auc_to_eval_step']:.6f}" if "auc_to_eval_step" in r else "NA"]
            text.append("| " + " | ".join(values) + " |")
        text += ["", "Running AUC is partial-horizon, NOT a final 2M result. Do not average it with completed-run AUCs.",
                 "The five-seed HGR cohort and the DrawerClose extension are not yet complete at this snapshot.",
                 "DrawerClose is queued after DrawerOpen completion and analysis; maximum concurrency is five.",
                 "There is no automatic expansion to seeds 6-10. This GitHub status is static, not live monitoring.",
                 "The HER/SABER reference is the completed recovery cohort; see [progress.json](results/progress.json)."]
        return "\n".join(text)
    raise ValueError(f"Not a published executed experiment: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path, help="New ICML_27/experiments directory")
    args = parser.parse_args()
    workspace, destination = args.workspace.resolve(), args.destination.resolve()
    origin = workspace / "experiments_post_review"
    if destination == origin or origin in destination.parents or destination.exists():
        parser.error("Destination must be new and outside the live experiments directory")
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    timestamp_kst = now.astimezone(timezone(timedelta(hours=9))).isoformat()
    destination.mkdir(parents=True)
    progress = capture_progress(origin / BASELINES, timestamp)
    entries = []
    for name, (title, status, planned) in SPECS.items():
        if name == BASELINES and progress["counts"].get("completed", 0) == 20:
            status = "completed"
        published_name = PUBLISHED_NAMES[name]
        root, dest = origin / name, destination / published_name
        sources = copy_sources(root, dest)
        reports = copy_reports(root, dest, workspace)
        if name == BASELINES:
            write_json(dest / "results/progress.json", progress)
        notice = ""
        if "drawer_sac" in name:
            notice = ("\n## Known observation-representation caveat\n\n"
                      "The inherited MetaWorld observation includes the original goal in its final three components, "
                      "while HER changes the separately concatenated goal. The executed source is preserved, not silently fixed. "
                      "This is shared by the compared SAC methods; its causal effect on performance has not been established. "
                      "A protocol-changing correction would require a separate matched experiment.\n")
        readme = (f"# {title}\n\nStatus: **{status}**. Snapshot: **{timestamp} UTC / {timestamp_kst} KST**.\n\n"
                  + result_text(name, root, progress) + notice + "\n\n## Code and provenance\n\n"
                  "- [source/](source/) contains the frozen executed source, protocol, tests, upstream copies and manifests.\n"
                  "- [source/README.md](source/README.md) is the ORIGINAL historical instruction file; its old pending/status text is not this snapshot's status.\n"
                  "- [provenance.json](provenance.json) records byte-identical source hashes and exported report hashes.\n"
                  "- Checkpoints, replay buffers, raw training logs, virtual environments and live process control files are omitted.\n"
                  "- Source dependencies and original local-path instructions are retained. Use the root materialization instructions before running tests or launchers; this is not a standalone reproduction bundle.\n"
                  "- No training is launched by browsing, exporting or verifying this archive. See [the archive guide](../../README.md).\n")
        (dest / "README.md").write_text(readme)
        write_json(dest / "provenance.json", {"snapshot_utc": timestamp, "original_directory": f"experiments_post_review/{name}",
                   "source_policy": "byte-for-byte copies, including original README and manifests",
                   "source_files_sha256": sources, "reports": reports,
                   "report_transformations": ["workspace absolute path replaced with <WORKSPACE>", "JSON reserialized; nonfinite values become null"],
                   "omitted": ["checkpoints/replay", "raw metrics and logs", "live locks and production claims", "venv/caches", "environment credentials"]})
        entries.append({"directory": published_name, "title": title, "status": status, "new_production_runs_planned": planned,
                        "source_file_count": len(sources)})
    write_json(destination / "index.json", {"snapshot_utc": timestamp, "snapshot_kst": timestamp_kst,
               "experiments": entries, "training_counts": {"planned": 90, "completed": 70 + progress["counts"].get("completed", 0),
               "running": progress["counts"].get("running", 0), "pending": progress["counts"].get("pending", 0)},
               "counting_note": "Exclude reused references, preproduction versions, offline diagnostics, smoke tests and interrupted attempts."})
    print(json.dumps({"destination": str(destination), "experiments": len(entries), "snapshot_utc": timestamp,
                      "counts": read_json(destination / "index.json")["training_counts"]}, indent=2))


if __name__ == "__main__":
    main()
