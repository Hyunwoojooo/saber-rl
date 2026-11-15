"""Export/verify completed DrawerClose DDPG HER seeds 6-10; never train."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

from ddpg_seed_extension_results import COHORTS, FIELDS, published_run, read
from export_local_experiments import scrub, sha, write_json
from sac_results import assert_close, csv_text, inventory, within
from verify_archive import check_hashes, check_links

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiments_post_review/2026-11-14_drawerclose_ddpg_main_seed_extension"
ARTIFACT = REPO / "metaworld_Drawer_Close_DDPG/paper_artifacts/drawer_close_ddpg_her_seed_extension_2m"


def require_completed_her(queue):
    p = queue["protocol"]
    if (p["task"] != "DrawerClose-v3" or p["backbone"] != "DDPG" or
            p["seeds"] != list(range(6, 11)) or p["steps_per_run"] != 2000000):
        raise ValueError("Wrong task, backbone, seed set or budget")
    arm = queue["arms"]["her"]
    if arm["status"] != "completed" or set(arm["seeds"]) != {str(s) for s in range(6, 11)}:
        raise ValueError("All five HER extension seeds must be completed")
    for run in arm["seeds"].values():
        if (run["status"] != "completed" or run.get("exit_code") != 0 or
                run["step"] != 2000000 or run.get("nonfinite_loss")):
            raise ValueError("Invalid HER completion record")


def require_her_config(config, seed):
    expected = {"task_name": "drawer_close", "metaworld_env_name": "drawer-close-v3",
                "metaworld_reward_function_version": "v2", "reward_type": "sparse",
                "method": "b0_ddpg_her", "weight_scope": "none", "total_steps": 2000000,
                "updates_per_step": 1, "her_ratio": .8, "max_episode_steps": 200,
                "eval_freq": 10000, "eval_episodes": 20, "resume": False, "resume_path": None}
    for key, value in expected.items():
        assert_close(config[key], value)
    for key in ("env_seed", "torch_seed", "numpy_seed", "action_noise_seed"):
        assert_close(config["seeds"][key], seed)
    assert_close(config["seeds"]["eval_seed"], 10000 + seed)
    assert_close(config["seeds"]["weight_seed"], 20000 + seed)


def summarize(rows):
    keys = [(r["arm"], r["seed"]) for r in rows]
    if len(keys) != 10 or set(keys) != {("her", s) for s in range(1, 11)}:
        raise ValueError("Require exactly HER seeds 1-10, without duplicates or other methods")
    rows = sorted(rows, key=lambda r: r["seed"])
    if any(not math.isfinite(r[m]) or not 0 <= r[m] <= 1 for r in rows for m in FIELDS):
        raise ValueError("Invalid performance metric")
    cohorts = {}
    for label, seeds in COHORTS.items():
        selected = [r for r in rows if r["seed"] in seeds]
        cohorts[label] = {"n": len(selected),
            **{m: {"mean": float(np.mean([r[m] for r in selected])),
                   "sd": float(np.std([r[m] for r in selected], ddof=1))} for m in FIELDS},
            "hit90_reached": sum(r["hit90_step"] is not None for r in selected)}
    return {"task": "DrawerClose-v3", "backbone": "DDPG", "included_arms": ["her"],
            "status": "HER_extension_completed_only", "new_completed_runs": 5,
            "historical_reference_runs": 5, "paired_comparisons": {},
            "cohorts": cohorts, "per_seed": rows}


def products(root):
    provenance = read(root / "provenance.json")
    queue = read(root / "provenance/queue_snapshot.json")
    expected = read(root / "provenance/expected_configs.json")
    require_completed_her(queue)
    assert_close(read(root / "protocol.json"), queue["protocol"])
    assert_close(provenance["experiment"], EXPERIMENT)
    assert_close(set(expected), {f"her:{s}" for s in range(6, 11)})
    rows, curves, runtimes = [], [], []
    entries = provenance["runs"]
    if len(entries) != 10 or {(e["arm"], e["seed"]) for e in entries} != {("her", s) for s in range(1, 11)}:
        raise ValueError("Only the ten declared HER curves may be included")
    for entry in sorted(entries, key=lambda e: e["seed"]):
        seed = entry["seed"]
        cohort = "historical_1_5" if seed <= 5 else "extension_6_10"
        assert_close(entry["published_directory"], f"data/{cohort}/her/seed_{seed}")
        row, points = published_run(root, entry, expected, queue)
        rows.append(row)
        curves.extend({"arm": "her", "seed": seed, "step": s, "success": v} for s, v in points)
        if seed >= 6:
            folder = within(root, entry["published_directory"])
            require_her_config(read(folder / "config.json"), seed)
            claim = read(folder / "launch_record.json")
            assert_close(claim["phase"], "production")
            if any(flag in claim["argv"] for flag in ("--resume", "--resume_path", "--smoke")):
                raise ValueError("Unexpected resume/smoke production argument")
            summary = read(folder / "summary.json")
            seconds, speed = summary["runtime_total_sec"], summary["final_steps_per_sec"]
            if not all(math.isfinite(x) and x > 0 for x in (seconds, speed)):
                raise ValueError("Invalid runtime")
            if not math.isclose(seconds * speed, 2000000, rel_tol=1e-6):
                raise ValueError("Runtime and throughput disagree")
            runtimes.append({"arm": "her", "seed": seed, "runtime_seconds": seconds,
                             "runtime_hours": seconds / 3600, "steps_per_second": speed})
    report = summarize(rows)
    assert_close(report, read(root / "reference_report.json"))
    actual_dirs = {p.parent.relative_to(root).as_posix() for p in (root / "data").rglob("evaluation.csv")}
    assert_close(actual_dirs, {e["published_directory"] for e in entries})
    aggregate = [{"cohort": label, "arm": "her", "n": c["n"],
                  **{f"{m}_{stat}": c[m][stat] for m in FIELDS for stat in ("mean", "sd")},
                  "hit90_reached": c["hit90_reached"]} for label, c in report["cohorts"].items()]
    return {"analysis/report.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
            "analysis/seed_metrics.csv": csv_text(rows),
            "analysis/aggregate_metrics.csv": csv_text(aggregate),
            "analysis/learning_curves.csv": csv_text(curves),
            "analysis/runtime_metrics.csv": csv_text(runtimes)}


def make_readme(report, queue):
    lines = ["# DrawerClose DDPG: Completed HER Seed Extension", "",
             f"Queue snapshot: **{queue['updated_utc']}** (not live monitoring).", "",
             "HER seeds 6-10 completed 2M steps each. This artifact includes these **five new runs**",
             "and five retained HER seed 1-5 evaluation curves. Historical runs are references, not new executions.",
             "**No SABER or other method's training results or paired confidence intervals are included.**", "",
             "## HER Results", "",
             "| Cohort | n | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, c in report["cohorts"].items():
        cells = [f"{c[m]['mean']:.6f} +/- {c[m]['sd']:.6f}" for m in FIELDS]
        lines.append(f"| {label} | {c['n']} | {' | '.join(cells)} | {c['hit90_reached']}/{c['n']} |")
    lines += ["", "## New Seeds 6-10", "",
              "| Seed | AUC | FQ-AUC | Final success | First Hit90 step |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for r in report["per_seed"]:
        if r["seed"] >= 6:
            hit = r["hit90_step"] if r["hit90_step"] is not None else "Not reached"
            lines.append(f"| {r['seed']} | {r['auc']:.6f} | {r['final_quarter_auc']:.6f} | {r['final_success']:.2f} | {hit} |")
    lines += ["", "## Scope and Interpretation", "",
              "These are final 2M HER results, not intermediate checkpoint estimates. They cannot establish a HER-SABER ranking while the matching SABER extension is incomplete.",
              "Hit90 records the first evaluation reaching 90%, not sustained success. Low-final-success runs remain included, without seed exclusions.",
              "The three cohort summaries separate historical 1-5, new 6-10, and pooled 1-10. Cross-period pooling does not prove identical historical execution environments.", "",
              "## Queue Status at Capture", "", "| Method | Status | Completed new seeds |", "| --- | --- | ---: |"]
    for arm in queue["protocol"]["order"]:
        a = queue["arms"][arm]
        count = sum(v["status"] == "completed" for v in a["seeds"].values())
        lines.append(f"| {arm} | {a['status']} | {count}/5 |")
    lines += ["", "Other methods appear only as timestamped queue status, not performance data.", "",
              "## Protocol and Reproducibility", "",
              "- DrawerClose-v3, DDPG+HER, sparse reward v2, 2M steps, UTD=1, HER ratio 0.8, horizon 200.",
              "- Each new run completed 1,995,001 updates from scratch. No checkpoint continuation or outcome-dependent restarts.",
              "- Environment success is evaluated with 20 episodes every 10k steps. Every curve contains all 200 checkpoints from 10k to 2M.",
              "- Normalized trapezoidal AUC uses denominator 1,990,000. FQ-AUC covers 1.5M-2M with denominator 500,000. Final success is measured at 2M; SD uses ddof=1.",
              "- No smoothing, interpolation, shortened horizons, or incomplete-run aggregation is used.",
              "- Original historical HER execution configs and dependency locks were not retained. New runs use the audited earliest retained canonical source and launcher contract. This is not bitwise verification of historical executions.",
              "- Recorded runtime is descriptive for these executions, not a matched-cost comparison with other algorithms.", "",
              "## Included Files", "",
              "- [Seed metrics](analysis/seed_metrics.csv), [cohort aggregates](analysis/aggregate_metrics.csv), [learning curves](analysis/learning_curves.csv), [runtime](analysis/runtime_metrics.csv), [report](analysis/report.json).",
              "- `data/extension_6_10/her/`: evaluations, config, summary, simulator metadata, and actual launch record for each completed run.",
              "- `data/historical_1_5/her/`: original evaluation curves only; no reconstructed config is represented as an original execution record.",
              "- `source_snapshot/her/`: byte-identical executed scientific source. Its original launcher targets historical seeds; actual new-run arguments are in each `launch_record.json`.",
              "- [Provenance](provenance.json), [source origins](provenance/source_origins.json), [environment](provenance/environment.json), [queue snapshot](provenance/queue_snapshot.json), [protocol](protocol.json), and [SHA-256 manifest](MANIFEST.sha256).",
              "- Checkpoints, replay buffers, raw training logs, caches, and unfinished run outputs are omitted. JSON workspace paths are redacted; unavailable optional nonfinite diagnostics become null. Required performance/loss values must remain finite.", "",
              "## Verification", "", "From the repository root with Python 3.12 and NumPy:", "", "```bash",
              "python3 ICML_27/tools/drawerclose_ddpg_seed_extension_results.py verify", "```", "",
              "Read-only CPU verification checks completed-run contracts, regenerates all tables from the included curves, verifies source hashes and the manifest, and does not import a trainer.", ""]
    return "\n".join(lines)


def verify(root=ARTIFACT):
    provenance = read(root / "provenance.json")
    if not provenance["source_files_sha256"] or any(not p.startswith("source_snapshot/her/") for p in provenance["source_files_sha256"]):
        raise ValueError("Only the completed HER source may be published")
    check_hashes(root, provenance["source_files_sha256"])
    for name, expected in products(root).items():
        if (root / name).read_text() != expected:
            raise ValueError(f"Derived result mismatch: {name}")
    if (root / "README.md").read_text() != make_readme(read(root / "analysis/report.json"), read(root / "provenance/queue_snapshot.json")):
        raise ValueError("README differs from verified results")
    if (root / "MANIFEST.sha256").read_text() != inventory(root):
        raise ValueError("Manifest mismatch")
    check_links(root / "README.md")
    return {"passed": True, "new_runs": 5, "historical_runs": 5, "evaluation_points": 2000,
            "source_files": len(provenance["source_files_sha256"]), "paired_intervals": 0}


def export(workspace, destination=ARTIFACT):
    workspace, destination = workspace.resolve(), destination.resolve()
    if destination.exists() or workspace == REPO.resolve() or destination == workspace or workspace in destination.parents:
        raise ValueError("Use a new artifact destination outside the live workspace")
    origin = within(workspace, EXPERIMENT)
    sys.path.insert(0, str(origin))
    api, common = importlib.import_module("analysis"), importlib.import_module("common")
    common.verify_snapshots()
    api.verify_reference_metrics()
    approval = read(origin / "outputs/validation/approved.json")
    assert_close(approval["passed"], True)
    assert_close(approval["package_hashes"], common.package_hashes())
    expected_path = origin / "outputs/validation/expected_configs.json"
    assert_close(approval["expected_configs_sha256"], sha(expected_path))
    expected = read(expected_path)
    queue_bytes = (origin / "outputs/queue_status.json").read_bytes()
    queue = json.loads(queue_bytes)
    require_completed_her(queue)
    # Retain only status for unexported methods, never their partial results.
    for arm, group in queue["arms"].items():
        if arm != "her":
            queue["arms"][arm] = {"status": group["status"], "seeds": {
                s: {"status": v["status"], "step": v["step"]} for s, v in group["seeds"].items()}}
    old = [r for r in api.reference_rows() if r["arm"] == "her"]
    new = [api.read_run("her", s, expected_config=expected[f"her:{s}"]) for s in range(6, 11)]
    rows = [{k: r[k] for k in ("arm", "seed", *FIELDS, "hit90_step")} for r in old + new]
    report = summarize(rows)
    destination.mkdir(parents=True)
    sources, entries = {}, []
    for name, digest in read(origin / "snapshot_manifest.json").items():
        if name.startswith("sources/her/"):
            target = destination / "source_snapshot" / Path(name).relative_to("sources")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin / name, target)
            sources[str(target.relative_to(destination))] = digest
    for row in report["per_seed"]:
        seed = row["seed"]
        cohort = "historical_1_5" if seed <= 5 else "extension_6_10"
        dest = destination / "data" / cohort / "her" / f"seed_{seed}"
        dest.mkdir(parents=True)
        hashes = {}
        if seed <= 5:
            source = origin / f"references/her/seed_{seed}/evaluation.csv"
            shutil.copy2(source, dest / "evaluation.csv")
            hashes[str(source.relative_to(workspace))] = sha(source)
        else:
            run = origin / f"outputs/production/her/seed_{seed:02d}"
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
        entries.append({"arm": "her", "seed": seed, "cohort": cohort,
                        "published_directory": str(dest.relative_to(destination)), "original_files_sha256": hashes})
    origins = read(origin / "provenance/source_origins.json")
    origins["arms"] = {"her": origins["arms"]["her"]}
    origins["curves"] = [r for r in origins["curves"] if r["arm"] == "her"]
    write_json(destination / "provenance/source_origins.json", scrub(origins, workspace))
    write_json(destination / "provenance/expected_configs.json", scrub({f"her:{s}": expected[f"her:{s}"] for s in range(6, 11)}, workspace))
    write_json(destination / "provenance/queue_snapshot.json", scrub(queue, workspace))
    write_json(destination / "provenance/environment.json", scrub(read(origin / "outputs/validation/environment.json"), workspace))
    shutil.copy2(origin / "protocol.json", destination / "protocol.json")
    write_json(destination / "provenance.json", {"exported_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT, "source_files_sha256": sources, "runs": entries,
        "original_snapshot_manifest_sha256": sha(origin / "snapshot_manifest.json"),
        "original_queue_snapshot_sha256": hashlib.sha256(queue_bytes).hexdigest(),
        "historical_limit": origins["historical_limit"],
        "transformations": ["workspace paths redacted in JSON", "optional nonfinite diagnostics become null",
                            "all evaluation checkpoints retained; training rows omitted",
                            "non-HER queue entries retain status only"],
        "omitted": ["checkpoints", "replay buffers", "raw logs", "unfinished outputs", "non-HER source/results"]})
    write_json(destination / "reference_report.json", report)
    for name, value in products(destination).items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    (destination / "README.md").write_text(make_readme(report, queue))
    (destination / "MANIFEST.sha256").write_text(inventory(destination))
    print(json.dumps(verify(destination), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exporter = sub.add_parser("export")
    exporter.add_argument("--workspace", type=Path, required=True)
    exporter.add_argument("--output", type=Path, default=ARTIFACT)
    sub.add_parser("verify").add_argument("--root", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    if args.command == "export":
        export(args.workspace, args.output)
    else:
        print(json.dumps(verify(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
