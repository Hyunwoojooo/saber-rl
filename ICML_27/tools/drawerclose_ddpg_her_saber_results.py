"""Publish and verify completed DrawerClose DDPG HER/SABER results."""
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

import ddpg_seed_extension_results as pair
from drawerclose_ddpg_seed_extension_results import EXPERIMENT, require_completed_her, require_her_config
from export_local_experiments import scrub, sha, write_json
from sac_results import assert_close, csv_text, inventory, within
from verify_archive import check_hashes, check_links

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "metaworld_Drawer_Close_DDPG/paper_artifacts/drawer_close_ddpg_her_saber_seed_extension_2m"
ARMS = pair.ARMS
read = pair.read


def require_completed(queue):
    require_completed_her(queue)
    group = queue["arms"]["saber"]
    if group["status"] != "completed" or set(group["seeds"]) != {str(s) for s in range(6, 11)}:
        raise ValueError("All five SABER extension seeds must be completed")
    for run in group["seeds"].values():
        if (run["status"] != "completed" or run.get("exit_code") != 0 or
                run["step"] != 2000000 or run.get("nonfinite_loss")):
            raise ValueError("Invalid SABER completion record")


def status_snapshot(queue):
    result = json.loads(json.dumps(queue))
    for arm, group in result["arms"].items():
        if arm not in ARMS:
            result["arms"][arm] = {"status": group["status"], "seeds": {
                s: {"status": v["status"], "step": v["step"]} for s, v in group["seeds"].items()}}
    return result


def require_config(config, arm, seed, *, support_method="b3_support_runtime_opt"):
    if arm not in ARMS or seed not in range(6, 11):
        raise ValueError("Unexpected production arm or seed")
    if arm == "her":
        require_her_config(config, seed)
        return
    expected = {"task_name": "drawer_close", "metaworld_env_name": "drawer-close-v3",
                "metaworld_reward_function_version": "v2", "reward_type": "sparse",
                "total_steps": 2000000, "updates_per_step": 1, "her_ratio": .8,
                "max_episode_steps": 200, "eval_freq": 10000, "eval_episodes": 20,
                "resume": False, "resume_path": None, "smoke": False,
                "method": support_method, "weight_scope": "critic_only",
                "lambda_support": 1., "weighting_mode": "batch_norm", "action_bins_per_dim": 5,
                "N_local_min": 4, "M_Q_min": 4, "M_B": 16, "local_search_max": 512,
                "policy_support_mode": "bin", "support_scan_mode": "deterministic_tail_v1"}
    for key, value in expected.items():
        assert_close(config[key], value)
    for key in ("env_seed", "torch_seed", "numpy_seed", "action_noise_seed"):
        assert_close(config["seeds"][key], seed)
    assert_close(config["seeds"]["eval_seed"], 10000 + seed)
    assert_close(config["seeds"]["weight_seed"], 20000 + seed)


def products(root):
    provenance = read(root / "provenance.json")
    queue = read(root / "provenance/queue_snapshot.json")
    expected = read(root / "provenance/expected_configs.json")
    require_completed(queue)
    assert_close(queue, status_snapshot(queue))
    assert_close(read(root / "protocol.json"), queue["protocol"])
    assert_close(provenance["experiment"], EXPERIMENT)
    assert_close(set(expected), {f"{a}:{s}" for a in ARMS for s in range(6, 11)})
    entries = provenance["runs"]
    if len(entries) != 20 or {(e["arm"], e["seed"]) for e in entries} != {(a, s) for a in ARMS for s in range(1, 11)}:
        raise ValueError("Require exactly HER and SABER seeds 1-10")
    runtimes = []
    for entry in entries:
        arm, seed = entry["arm"], entry["seed"]
        cohort = "historical_1_5" if seed <= 5 else "extension_6_10"
        assert_close(entry["cohort"], cohort)
        assert_close(entry["published_directory"], f"data/{cohort}/{arm}/seed_{seed}")
        if seed >= 6:
            folder = within(root, entry["published_directory"])
            require_config(read(folder / "config.json"), arm, seed)
            claim = read(folder / "launch_record.json")
            assert_close(claim["phase"], "production")
            if any(arg.split("=", 1)[0] in {"--resume", "--resume_path", "--smoke"} for arg in claim["argv"]):
                raise ValueError("Unexpected resume/smoke production argument")
            summary = read(folder / "summary.json")
            seconds, speed = summary["runtime_total_sec"], summary["final_steps_per_sec"]
            if not all(math.isfinite(v) and v > 0 for v in (seconds, speed)):
                raise ValueError("Invalid runtime")
            if not math.isclose(seconds * speed, 2000000, rel_tol=1e-6):
                raise ValueError("Runtime and throughput disagree")
            runtimes.append({"arm": arm, "seed": seed, "runtime_seconds": seconds,
                             "runtime_hours": seconds / 3600, "steps_per_second": speed})
    actual_dirs = {p.parent.relative_to(root).as_posix() for p in (root / "data").rglob("evaluation.csv")}
    assert_close(actual_dirs, {e["published_directory"] for e in entries})
    result = pair.products(root)
    result["analysis/runtime_metrics.csv"] = csv_text(runtimes)
    return result


def verify(root=ARTIFACT):
    sources = read(root / "provenance.json")["source_files_sha256"]
    if not sources or any(not any(p.startswith(f"source_snapshot/{a}/") for a in ARMS) for p in sources):
        raise ValueError("Only HER and SABER source may be published")
    if {Path(p).parts[1] for p in sources} != set(ARMS):
        raise ValueError("Both executed source snapshots are required")
    check_hashes(root, sources)
    for name, expected in products(root).items():
        if (root / name).read_text() != expected:
            raise ValueError(f"Derived result mismatch: {name}")
    if (root / "README.md").read_text() != make_readme(read(root / "analysis/report.json"), read(root / "provenance/queue_snapshot.json")):
        raise ValueError("README differs from verified results")
    if (root / "MANIFEST.sha256").read_text() != inventory(root):
        raise ValueError("Manifest mismatch")
    check_links(root / "README.md")
    return {"passed": True, "extension_runs": 10, "newly_published_saber_runs": 5,
            "historical_runs": 10, "evaluation_points": 4000,
            "source_files": len(sources), "paired_intervals": 9}


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
    queue = status_snapshot(json.loads(queue_bytes))
    require_completed(queue)
    old = [r for r in api.reference_rows() if r["arm"] in ARMS]
    new = [api.read_run(a, s, expected_config=expected[f"{a}:{s}"]) for a in ARMS for s in range(6, 11)]
    rows = [{k: r[k] for k in ("arm", "seed", *pair.FIELDS, "hit90_step")}
            for r in sorted(old + new, key=lambda r: (r["arm"], r["seed"]))]
    report = pair.summarize(rows)
    index = {(r["arm"], r["seed"]): r for r in rows}
    for label, seeds in pair.COHORTS.items():
        for metric in pair.FIELDS:
            assert_close(report["cohorts"][label]["paired"][metric], api.paired(
                [index["saber", s][metric] for s in seeds], [index["her", s][metric] for s in seeds]))
    destination.mkdir(parents=True)
    sources, entries = {}, []
    for name, digest in read(origin / "snapshot_manifest.json").items():
        if any(name.startswith(f"sources/{a}/") for a in ARMS):
            target = destination / "source_snapshot" / Path(name).relative_to("sources")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin / name, target)
            sources[str(target.relative_to(destination))] = digest
    for row in rows:
        arm, seed = row["arm"], row["seed"]
        cohort = "historical_1_5" if seed <= 5 else "extension_6_10"
        dest = destination / "data" / cohort / arm / f"seed_{seed}"
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
                        "published_directory": str(dest.relative_to(destination)), "original_files_sha256": hashes})
    origins = read(origin / "provenance/source_origins.json")
    origins["arms"] = {a: origins["arms"][a] for a in ARMS}
    origins["curves"] = [r for r in origins["curves"] if r["arm"] in ARMS]
    write_json(destination / "provenance/source_origins.json", scrub(origins, workspace))
    write_json(destination / "provenance/expected_configs.json", scrub({f"{a}:{s}": expected[f"{a}:{s}"] for a in ARMS for s in range(6, 11)}, workspace))
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
                            "non-HER/SABER queue entries retain status only"],
        "omitted": ["checkpoints", "replay buffers", "raw logs", "unfinished outputs", "other-method source/results"]})
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


def make_readme(report, queue):
    lines = ["# DrawerClose DDPG: HER vs SABER Seed Extension", "",
             f"Queue snapshot: **{queue['updated_utc']}** (not live monitoring).", "",
             "Both methods completed seeds 6-10, 2M steps per run: ten extension executions.",
             "This publication adds five SABER runs and reuses the five already-published HER runs.",
             "Historical seeds 1-5 are retained references, not new executions. Across both methods there are 20 distinct runs, not 25.", "",
             "## Results", "",
             "| Cohort | Metric | HER mean +/- SD | SABER mean +/- SD | SABER - HER | Paired 95% CI | Positive pairs |",
             "| --- | --- | ---: | ---: | ---: | --- | ---: |"]
    for label, c in report["cohorts"].items():
        for metric in pair.FIELDS:
            d = c["paired"][metric]
            cells = [f"{c['summary'][a][metric]['mean']:.6f} +/- {c['summary'][a][metric]['sd']:.6f}" for a in ARMS]
            lines.append(f"| {label} | {metric} | {' | '.join(cells)} | {d['mean_difference']:+.6f} | [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] | {d['positive_pairs']}/{d['n']} |")
    lines += ["", "The extension-only mean AUC difference is close to zero. The pooled mean favors SABER, but its paired AUC interval includes zero.",
              "These data do not establish superiority or equivalence. The original five-seed interval must not be presented as the ten-seed result.",
              "FQ-AUC and final success are secondary, unadjusted comparisons. Extension FQ-AUC favors SABER, but the pooled FQ-AUC interval includes zero; final success does not show a clear advantage.", "",
              "## New Seeds 6-10", "",
              "| Seed | HER AUC | SABER AUC | SABER - HER | HER final success | SABER final success |",
              "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    index = {(r["arm"], r["seed"]): r for r in report["per_seed"]}
    for seed in range(6, 11):
        h, s = index["her", seed], index["saber", seed]
        lines.append(f"| {seed} | {h['auc']:.6f} | {s['auc']:.6f} | {s['auc']-h['auc']:+.6f} | {h['final_success']:.2f} | {s['final_success']:.2f} |")
    lines += ["", "All seeds are retained, including SABER seed 8 with zero final success. Hit90 is first attainment, not sustained success; complete seed metrics include it.", "",
              "## Queue Status at Capture", "", "| Method | Status | Completed new seeds |", "| --- | --- | ---: |"]
    for arm in queue["protocol"]["order"]:
        group = queue["arms"][arm]
        count = sum(v["status"] == "completed" for v in group["seeds"].values())
        lines.append(f"| {arm} | {group['status']} | {count}/5 |")
    lines += ["", "Only HER and SABER performance is included. Other methods appear as timestamped status only, never performance aggregates.", "",
              "## Protocol and Limitations", "",
              "- DrawerClose-v3, DDPG+HER, sparse reward v2, 2M steps, UTD=1, HER ratio 0.8, horizon 200.",
              "- SABER is critic-only, with K=5, N_local_min=4, M_Q_min=4, M_B=16, scan limit 512, lambda_support=1 and batch-mean normalization.",
              "- All ten extension runs started from scratch and completed 1,995,001 updates. No checkpoint continuation or outcome-dependent exclusions.",
              "- Environment success uses 20 episodes every 10k steps. Each curve includes all 200 checkpoints from 10k to 2M.",
              "- Normalized trapezoidal AUC divides by 1,990,000; FQ-AUC covers 1.5M-2M and divides by 500,000. Final success is at 2M; SD uses ddof=1. No smoothing, interpolation or shortened horizons.",
              "- Pair by seed number. Bootstrap paired seed rows 10,000 times with NumPy default_rng(2027); use 2.5%/97.5% percentile endpoints. These approximate intervals are not simultaneous coverage guarantees.",
              "- AUC is primary. Historical 1-5, extension 6-10 and pooled 1-10 are reported separately to expose cohort variation.",
              "- Historical HER/SABER full configs, exact execution commits and dependency locks were not retained. Audited canonical source and launcher contracts are used, not a claim of bitwise historical execution identity.",
              "- This is the primary seed extension, not a diagnostic rerun. Previously published HER data are included for self-contained verification, not additional independent observations.",
              "- Runtime is descriptive for these executions, not proof of matched hardware load or a compute-normalized performance comparison.", "",
              "## Included Files", "",
              "- [Seed metrics](analysis/seed_metrics.csv), [aggregates](analysis/aggregate_metrics.csv), [paired intervals](analysis/paired_differences.csv), [curves](analysis/learning_curves.csv), [runtime](analysis/runtime_metrics.csv), [report](analysis/report.json).",
              "- `data/extension_6_10/`: evaluations, configs, summaries, simulator metadata and actual launch records.",
              "- `data/historical_1_5/`: retained evaluation CSVs only; no reconstructed configs presented as original execution records.",
              "- `source_snapshot/`: byte-identical executed scientific source. Original launchers target historical seeds; actual extension commands are in per-run `launch_record.json`.",
              "- [Provenance](provenance.json), [source origins](provenance/source_origins.json), [environment](provenance/environment.json), [queue](provenance/queue_snapshot.json), [protocol](protocol.json), [manifest](MANIFEST.sha256).",
              "- Checkpoints, replay buffers, raw logs, caches and unfinished outputs are omitted. JSON workspace prefixes are redacted; unavailable optional diagnostics become null. Required performance/loss values must be finite.", "",
              "## Verification", "", "From the repository root with Python 3.12 and NumPy:", "", "```bash",
              "python3 ICML_27/tools/drawerclose_ddpg_her_saber_results.py verify", "```", "",
              "Read-only CPU verification regenerates metrics and nine intervals, checks completion/config/source contracts and the manifest, and never imports a trainer.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
