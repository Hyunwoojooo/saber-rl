"""Publish the four complete DrawerClose DDPG primary seed-extension cohorts."""
import argparse
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

import numpy as np

from ddpg_seed_extension_results import COHORTS, FIELDS, paired, published_run, read
from drawerclose_ddpg_seed_extension_results import EXPERIMENT
from drawerclose_ddpg_her_saber_results import require_config as require_pair_config
from export_local_experiments import scrub, sha, write_json
from sac_results import assert_close, csv_text, inventory, within
from verify_archive import check_hashes, check_links

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "metaworld_Drawer_Close_DDPG/paper_artifacts/drawer_close_ddpg_primary_seed_extension_2m"
ARMS = ("her", "saber", "hgr", "per", "replay_query", "shuffled_query", "random_support", "strict_random")
PUBLISHED_ARMS = ARMS[:4]
LABELS = {"her": "HER", "saber": "SABER (Ours)", "hgr": "HGR", "per": "PER",
          "replay_query": "Replay-Action Query", "shuffled_query": "Shuffled-Policy Query",
          "random_support": "Random-Support", "strict_random": "Context-Eligible Count-Matched"}


def completed_keys(queue):
    protocol = queue["protocol"]
    if (tuple(protocol["order"]) != ARMS or protocol["seeds"] != list(range(6, 11)) or
            protocol["task"] != "DrawerClose-v3" or protocol["backbone"] != "DDPG" or
            protocol["steps_per_run"] != 2000000):
        raise ValueError("Unexpected experiment or seed set")
    if set(queue["arms"]) != set(ARMS):
        raise ValueError("Missing or unexpected queue arm")
    keys = set()
    for arm in PUBLISHED_ARMS:
        states = queue["arms"][arm]["seeds"]
        if (queue["arms"][arm]["status"] != "completed" or
                set(states) != {str(s) for s in range(6, 11)}):
            raise ValueError("Require all five completed seeds for every primary method")
        for seed, state in states.items():
            if (state["status"] != "completed" or state.get("exit_code") != 0 or
                    state["step"] != 2000000 or state.get("nonfinite_loss")):
                raise ValueError("Invalid completion record")
            keys.add((arm, int(seed)))
    return keys


def status_snapshot(queue):
    result = json.loads(json.dumps(queue))
    for arm in ARMS:
        if arm not in PUBLISHED_ARMS:
            group = result["arms"][arm]
            result["arms"][arm] = {"status": group["status"], "seeds": {
                s: {"status": r["status"], "step": r["step"]} for s, r in group["seeds"].items()}}
    return result


def require_config(config, arm, seed):
    if arm in ("her", "saber"):
        require_pair_config(config, arm, seed)
        return
    if arm not in ("hgr", "per") or seed not in range(6, 11):
        raise ValueError("Unexpected arm or seed")
    expected = {"task_name": "drawer_close", "metaworld_env_name": "drawer-close-v3",
                "metaworld_reward_function_version": "v2", "reward_type": "sparse",
                "total_steps": 2000000, "updates_per_step": 1, "her_ratio": .8,
                "max_episode_steps": 200, "eval_freq": 10000, "eval_episodes": 20,
                "resume": False, "resume_path": None, "smoke": False}
    if arm == "hgr":
        expected.update(method="hgr", weight_scope="none", hgr_alpha_episode=.6, hgr_alpha_goal=.6,
                        hgr_beta_episode_initial=.4, hgr_beta_episode_final=1.,
                        hgr_beta_goal_initial=.4, hgr_beta_goal_final=1., hgr_priority_eps=1e-6)
        assert_close(config["hgr_storage"]["enabled"], True)
        assert_close(config["hgr_storage"]["actor_loss_weight"], "all_ones")
    else:
        expected.update(method="ddpg_her_per_full", weight_scope="critic_only", lambda_support=0.,
                        per_alpha=.6, per_beta_start=.4, per_beta_end=1., per_beta_steps=2000000)
    for key, value in expected.items():
        assert_close(config[key], value)
    for key in ("env_seed", "torch_seed", "numpy_seed", "action_noise_seed"):
        assert_close(config["seeds"][key], seed)
    assert_close(config["seeds"]["eval_seed"], 10000 + seed)
    assert_close(config["seeds"]["weight_seed"], 20000 + seed)


def summarize(rows, queue):
    selected = completed_keys(queue)
    included_arms = [a for a in ARMS if any(k[0] == a for k in selected)]
    expected = selected | {(a, s) for a in included_arms for s in range(1, 6)}
    keys = [(r["arm"], r["seed"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Missing, duplicate or unfinished result included")
    if any(not math.isfinite(r[m]) or not 0 <= r[m] <= 1 for r in rows for m in FIELDS):
        raise ValueError("Invalid performance metric")
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
    return {"snapshot_utc": queue["updated_utc"],
            "full_queue_completed": all(queue["arms"][a]["status"] == "completed" for a in ARMS),
            "newly_published_runs": 10, "reused_previously_published_extension_runs": 10,
            "new_completed_runs": len(selected), "historical_reference_runs": len(expected) - len(selected),
            "complete_arms": complete, "partial_arms": [a for a in included_arms if a not in complete],
            "cohorts": cohorts, "per_seed": rows}


def products(root):
    provenance = read(root / "provenance.json")
    queue = read(root / "provenance/queue_snapshot.json")
    expected = read(root / "provenance/expected_configs.json")
    assert_close(provenance["experiment"], EXPERIMENT)
    assert_close(queue, status_snapshot(queue))
    assert_close(read(root / "protocol.json"), queue["protocol"])
    selected = completed_keys(queue)
    assert_close(set(expected), {f"{a}:{s}" for a, s in selected})
    rows, curves, runtimes = [], [], []
    for entry in provenance["runs"]:
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
            "analysis/learning_curves.csv": csv_text(curves),
            "analysis/runtime_metrics.csv": csv_text(runtimes)}


def make_readme(report, queue):
    lines = ["# DrawerClose DDPG: HER, SABER, HGR and PER Seed Extensions", "",
             f"Snapshot: **{report['snapshot_utc']}**. This is not a live dashboard.", "",
             f"This artifact contains **{report['new_completed_runs']} completed seed-extension runs** and "
             f"{report['historical_reference_runs']} historical seed-level references. Each new run has 2M steps.",
             "Ten HGR/PER runs are newly published. The earlier HER/SABER artifact's ten extension runs are included again for comparison, not additional independent observations.",
             "There are 40 distinct runs: four methods x ten seeds. The earlier HER-only and HER-SABER artifacts remain unchanged.",
             "Only HER, SABER, HGR and PER results are included. Other methods appear as timestamped queue status only.", "",
             "## Queue Coverage", "", "| Method | Included extension seeds | State at capture |", "| --- | --- | --- |"]
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
              "Every primary method includes all seeds 6-10 at 2M; these are not intermediate checkpoints.", "",
              "| Method | Seed | AUC | FQ-AUC | Final success | Hit90 step |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in report["per_seed"]:
        if r["seed"] >= 6:
            hit = str(r["hit90_step"]) if r["hit90_step"] is not None else "Not reached"
            lines.append(f"| {LABELS[r['arm']]} | {r['seed']} | {r['auc']:.6f} | {r['final_quarter_auc']:.6f} | {r['final_success']:.2f} | {hit} |")
    lines += ["", "## Interpretation and Protocol", "",
              "- All three pooled SABER-minus-comparator AUC intervals include zero. Higher sample means do not establish superiority, and overlapping uncertainty does not establish equivalence.",
              "- The historical, added-seed and pooled estimates are reported separately. Their differing directions and failure-to-learn outcomes must remain visible.",
              "- All four primary methods have five complete extension runs. Non-primary control results are excluded even if their queue status later becomes completed.",
              "- DrawerClose-v3 / DDPG / sparse reward v2; 2M steps, UTD=1, 1,995,001 optimizer updates, HER ratio 0.8, horizon 200, 20 evaluation episodes every 10k steps.",
              "- AUC uses exactly 200 evaluation points from 10k to 2M, divided by 1,990,000. FQ-AUC covers 1.5M..2M, divided by 500,000. Success is eval/env_success_mean. No smoothing, interpolation or silently shortened horizons.",
              "- Final success is evaluated at 2M. Hit90 is the first evaluation >=0.90, not sustained task mastery. SD is across seeds with ddof=1.",
              "- Paired percentile bootstrap resamples seed rows 10,000 times using NumPy default_rng(2027); endpoints are 2.5% and 97.5%. These are approximate, pointwise, unadjusted intervals, not simultaneous guarantees. AUC is primary; other metrics are secondary.",
              "- Every completed zero/low-success run is retained. New runs began from scratch; no checkpoint continuation or outcome-based exclusions.",
              "- HGR is the retained episode/transition-future-goal pair-priority implementation with importance weighting. PER is full prioritized replay (ddpg_her_per_full), not the uniform-replay TD-error loss-weighting control. Each method retains its own audited preset.",
              "- Original HER/SABER full configs and exact historical environment locks were not retained. New configs follow audited canonical-source/launcher contracts. Retained baseline configs were checked where available; bitwise historical environment identity is not asserted.",
              "- Earlier diagnostic and assignment executions are not extra independent seeds in these cohorts. Source/result overlap across artifacts must not be counted twice.",
              "- Runtime records include instrumentation and are not evidence of a matched-cost algorithm ranking.", "",
              "## Included and Omitted", "",
              "- [Seed metrics](analysis/seed_metrics.csv), [aggregates](analysis/aggregate_metrics.csv), [paired intervals](analysis/paired_differences.csv), [curves](analysis/learning_curves.csv), [runtime](analysis/runtime_metrics.csv), [report](analysis/report.json).",
              "- data/historical_1_5 contains retained evaluation curves, not reconstructed historical execution configs.",
              "- data/extension_6_10 contains every completed run's evaluation curve, config, summary, simulator metadata and actual launch record.",
              "- source_snapshot contains byte-identical executed trainer sources for included methods. Original launchers are historical presets; launch_record.json records the actual extension command. This is not a standalone training package.",
              "- [Provenance](provenance.json) records original file hashes and transformations. [Queue snapshot](provenance/queue_snapshot.json) records status only for non-primary methods; their result directories are omitted. [Manifest](MANIFEST.sha256) covers every artifact file.",
              "- Checkpoints, replay buffers, raw training logs, caches, locks and unfinished output files are excluded. Workspace prefixes are redacted in exported JSON; unavailable optional diagnostics become null, never valid performance values.", "",
              "## Verification", "", "From the repository root with Python 3.12 and NumPy:", "", "```bash",
              "python3 ICML_27/tools/drawerclose_ddpg_completed_extension_results.py verify", "```", "",
              "Read-only CPU verification recomputes all seed metrics and complete-cohort intervals, checks configs/completion/source hashes, and verifies every manifest entry. It does not import trainers or modify the training queue.", ""]
    return "\n".join(lines)


def verify(root=ARTIFACT):
    provenance = read(root / "provenance.json")
    sources = provenance["source_files_sha256"]
    if (not sources or any(not any(p.startswith(f"source_snapshot/{a}/") for a in PUBLISHED_ARMS) for p in sources)
            or {Path(p).parts[1] for p in sources} != set(PUBLISHED_ARMS)):
        raise ValueError("Require exactly the four published source families")
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
            "source_files": len(provenance["source_files_sha256"]), "paired_intervals": 27}


def export(workspace, destination=ARTIFACT):
    workspace = workspace.resolve()
    destination = destination.resolve()
    origin = within(workspace, EXPERIMENT)
    if (destination.exists() or workspace == REPO.resolve() or
            destination == workspace or workspace in destination.parents):
        raise ValueError("Use a separate publication checkout and a new artifact destination")
    sys.path.insert(0, str(origin))
    api = importlib.import_module("analysis")
    common = importlib.import_module("common")
    common.verify_snapshots()
    api.verify_reference_metrics()
    approval = read(origin / "outputs/validation/approved.json")
    assert_close(approval["passed"], True)
    assert_close(approval["package_hashes"], common.package_hashes())
    assert_close(approval["expected_configs_sha256"], sha(origin / "outputs/validation/expected_configs.json"))
    queue_bytes = (origin / "outputs/queue_status.json").read_bytes()
    queue = status_snapshot(json.loads(queue_bytes))
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
    destination.mkdir(parents=True)
    sources, entries = {}, []
    for name, digest in read(origin / "snapshot_manifest.json").items():
        if not any(name.startswith(f"sources/{a}/") for a in arms):
            continue
        target = destination / "source_snapshot" / Path(name).relative_to("sources")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin / name, target)
        sources[str(target.relative_to(destination))] = digest
    for row in report["per_seed"]:
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
    origins["arms"] = {k: v for k, v in origins["arms"].items() if k in arms}
    origins["curves"] = [r for r in origins["curves"] if r["arm"] in arms]
    write_json(destination / "provenance/source_origins.json", scrub(origins, workspace))
    write_json(destination / "provenance/expected_configs.json", scrub({f"{a}:{s}": expected[f"{a}:{s}"] for a, s in sorted(selected)}, workspace))
    write_json(destination / "provenance/queue_snapshot.json", scrub(queue, workspace))
    write_json(destination / "provenance/environment.json", scrub(read(origin / "outputs/validation/environment.json"), workspace))
    shutil.copy2(origin / "protocol.json", destination / "protocol.json")
    write_json(destination / "provenance.json", {"exported_utc": datetime.now(timezone.utc).isoformat(),
               "experiment": EXPERIMENT, "source_files_sha256": sources, "runs": entries,
               "original_snapshot_manifest_sha256": sha(origin / "snapshot_manifest.json"),
               "original_queue_snapshot_sha256": hashlib.sha256(queue_bytes).hexdigest(),
               "historical_limit": origins["historical_limit"],
               "transformations": ["workspace path replaced with <WORKSPACE> in JSON", "unavailable optional nonfinite diagnostics serialized as null",
                                   "all 200 evaluation steps retained per completed run; training rows omitted",
                                   "non-primary methods retain timestamped queue status only"],
               "omitted": ["checkpoints", "replay buffers", "raw training logs", "non-primary result directories", "non-primary trainers"]})
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
    exporter.add_argument("--output", type=Path, default=ARTIFACT,
                          help="New destination outside the experiment workspace; existing paths are refused")
    sub.add_parser("verify").add_argument("--root", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    if args.command == "export":
        export(args.workspace, args.output)
    else:
        print(json.dumps(verify(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
