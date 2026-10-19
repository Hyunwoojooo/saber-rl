"""Exploratory failure/saturation analysis of every retained assignment run."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

from common import (ASSIGNMENT, ASSIGNMENT_SHA, ARMS, SEEDS, TOTAL_UPDATES, TOTAL_STEPS,
                    completed_attempt, read_json, write_json, write_csv, output_directory,
                    digest, finite, finish_artifact, verify_source)

DIAGNOSTICS = ("eligible_rate", "support_rate", "empty_update", "saturated_nonempty_update",
               "informative_update", "random_expected_overlap_fraction",
               "random_overlap_fraction", "td_overlap_fraction")
LOG_METRICS = ("loss/critic", "loss/td_error_abs_mean", "value/q_mean",
               "value/target_mean", "critic/q_policy_mean", "critic/q_policy_std")


def weighted_metric(rows, name):
    count = 0
    total = 0.0
    for row in rows:
        n = int(row[name + "_n"])
        if n < 0 or n > int(row["n_updates"]):
            raise ValueError("Invalid diagnostic observation count")
        if n:
            value = finite(row[name + "_mean"])
            if not -1e-12 <= value <= 1 + 1e-12:
                raise ValueError("Invalid diagnostic fraction")
            total += value * n
            count += n
    return total / count if count else None


def validate_windows(rows, expected_updates=TOTAL_UPDATES):
    all_rows = [r for r in rows if r["subset"] == "all"]
    if any(r["subset"] not in {"all", "informative"} for r in rows):
        raise ValueError("Unknown diagnostic subset")
    cursor = 1
    for row in all_rows:
        first, last, count = (int(row[k]) for k in ("start_update", "end_update", "n_updates"))
        if first != cursor or count != last - first + 1 or count <= 0:
            raise ValueError("Missing, overlapping or invalid diagnostic windows")
        if int(row["step"]) != last + 4999:
            raise ValueError("Unexpected environment-step/update alignment")
        for key in DIAGNOSTICS:
            weighted_metric([row], key)
        for key in ("empty_update", "saturated_nonempty_update", "informative_update"):
            if int(row[key + "_n"]) != count:
                raise ValueError("Update partition denominator mismatch")
        partition = sum(finite(row[key + "_mean"]) for key in
                        ("empty_update", "saturated_nonempty_update", "informative_update"))
        if not math.isclose(partition, 1, abs_tol=1e-10):
            raise ValueError("Empty/saturated/informative do not partition updates")
        cursor = last + 1
    if cursor - 1 != expected_updates:
        raise ValueError("Incomplete diagnostic budget")
    informative = [r for r in rows if r["subset"] == "informative"]
    keys = lambda r: (int(r["start_update"]), int(r["end_update"]))
    if [keys(r) for r in informative] != [keys(r) for r in all_rows]:
        raise ValueError("Informative windows are missing or reordered")
    for whole, subset in zip(all_rows, informative):
        n = finite(whole["informative_update_mean"]) * int(whole["n_updates"])
        if not math.isclose(n, int(subset["n_updates"]), abs_tol=1e-6):
            raise ValueError("Informative-subset count mismatch")
        for key in DIAGNOSTICS:
            weighted_metric([subset], key)
    return all_rows, informative


def curve_metrics(points):
    if [step for step, _ in points] != list(range(10_000, TOTAL_STEPS + 1, 10_000)):
        raise ValueError("Evaluation grid incomplete, duplicated, or reordered")
    if any(not 0 <= finite(value) <= 1 for _, value in points):
        raise ValueError("Invalid success fraction")

    def area(values):
        return sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(values, values[1:]))

    late = [(step, value) for step, value in points if step >= 1_500_000]
    hits = [step for step, value in points if value >= .9]
    positive = [step for step, value in points if value > 0]
    return {"auc": area(points) / 1_990_000, "fq_auc": area(late) / 500_000,
            "final_success": points[-1][1], "hit90": min(hits) if hits else None,
            "first_positive_evaluation_step": min(positive) if positive else None,
            "zero_evaluation_success": not positive}


def temporal_relation(event, start, end):
    if start is None:
        return "no_saturation_observed"
    if event is None:
        return "no_positive_evaluation_observed"
    if event < start:
        return "positive_evaluation_before_first_saturated_window"
    if event <= end:
        return "same_window_order_unresolved"
    return "positive_evaluation_after_first_saturated_window"


def training_episode_rows(rows):
    # Evaluation rows repeat the training columns after the episode reset.
    # Count completed training episodes only, not these zero-length duplicates.
    episodes = [r for r in rows if r.get("eval/env_success_mean", "") == ""
                and r.get("train/episode_success", "") != ""
                and int(r.get("train/episode_len") or 0) > 0]
    if len({int(r["step"]) for r in episodes}) != len(episodes):
        raise ValueError("Duplicate completed training episode records")
    if any(not 0 <= finite(r["train/episode_success"]) <= 1 for r in episodes):
        raise ValueError("Invalid training success measurement")
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verify_source(ASSIGNMENT, ASSIGNMENT_SHA)
    source_report = ASSIGNMENT / "outputs/analysis/report.json"
    report = read_json(source_report)
    if report.get("failures_or_missing") != [] or report.get("inference") is None:
        raise ValueError("Complete frozen primary analysis required")
    reference = {(row["arm"], row["seed"]): row for row in report["runs"]}
    if set(reference) != {(arm, seed) for arm in ARMS for seed in SEEDS} or len(report["runs"]) != 30:
        raise ValueError("Unexpected assignment cohort")
    output = output_directory(args.output_dir)
    inputs = {str(source_report): digest(source_report)}
    summaries, window_rows, band_rows, curve_rows = [], [], [], []
    for arm in ARMS:
        for seed in SEEDS:
            attempt = completed_attempt(ASSIGNMENT, arm, seed, ASSIGNMENT_SHA)
            run = attempt / "run"
            for path in (attempt / "status.json", run / "config.json", run / "summary.json",
                         run / "metrics.csv", run / "assignment_windows.csv"):
                inputs[str(path)] = digest(path)
            with (run / "assignment_windows.csv").open(newline="") as handle:
                all_rows, informative = validate_windows(list(csv.DictReader(handle)))
            with (run / "metrics.csv").open(newline="") as handle:
                metrics = list(csv.DictReader(handle))
            points = [(int(row["step"]), finite(row["eval/env_success_mean"]))
                      for row in metrics if row.get("eval/env_success_mean", "") != ""]
            values = curve_metrics(points)
            for key in ("auc", "fq_auc", "final_success"):
                if not math.isclose(values[key], reference[arm, seed]["metrics"][key], abs_tol=1e-12):
                    raise ValueError("Reanalysis disagrees with retained primary results")
            train = training_episode_rows(metrics)
            if len(train) != int(read_json(run / "summary.json")["episodes"]):
                raise ValueError("Training episode log is incomplete")
            positive_train = [int(r["step"]) for r in train if finite(r["train/episode_success"]) > 0]
            saturated = [r for r in all_rows if finite(r["saturated_nonempty_update_mean"]) > 0]
            first_start = int(saturated[0]["start_update"]) + 4999 if saturated else None
            first_end = int(saturated[0]["step"]) if saturated else None
            entry = {"arm": arm, "seed": seed, **values,
                     "hit90_reached": values["hit90"] is not None,
                     "training_episodes_logged": len(train),
                     "positive_training_episodes_logged": len(positive_train),
                     "first_positive_training_episode_end_step": min(positive_train) if positive_train else None,
                     "first_saturated_window_start_step": first_start,
                     "first_saturated_window_end_step": first_end,
                     "evaluation_vs_saturation_order": temporal_relation(
                         values["first_positive_evaluation_step"], first_start, first_end)}
            for key in DIAGNOSTICS:
                entry[key] = weighted_metric(all_rows, key)
                saved = reference[arm, seed]["metrics"]["diagnostics"]["all"][key]["mean"]
                if saved is not None and not math.isclose(entry[key], saved, abs_tol=1e-10):
                    raise ValueError("Diagnostic aggregation changed")
            for prefix, stop in (("first_100k", 100_000), ("first_500k", 500_000)):
                chosen = [r for r in all_rows if int(r["step"]) <= stop]
                for key in ("support_rate", "eligible_rate", "saturated_nonempty_update"):
                    entry[prefix + "_" + key] = weighted_metric(chosen, key)
            summaries.append(entry)
            for subset, rows in (("all", all_rows), ("informative", informative)):
                for row in rows:
                    window_rows.append({"arm": arm, "seed": seed, "subset": subset,
                        **{key: int(row[key]) for key in
                           ("start_update", "end_update", "step", "n_updates")},
                        **{key: weighted_metric([row], key) for key in DIAGNOSTICS}})
            for band in range(1, 21):
                low, high = (band - 1) * 100_000, band * 100_000
                windows = [r for r in all_rows if low < int(r["step"]) <= high]
                logs = [r for r in train if low < int(r["step"]) <= high and int(r["step"]) >= 5000]
                row = {"arm": arm, "seed": seed, "end_step_band": band,
                       "band_low_exclusive": low, "band_high_inclusive": high,
                       "diagnostic_updates": sum(int(r["n_updates"]) for r in windows),
                       **{key: weighted_metric(windows, key) for key in DIAGNOSTICS}}
                for key in LOG_METRICS:
                    logged = [finite(r[key]) for r in logs if r.get(key, "") != ""]
                    row[key + "_logged_mean"] = statistics.mean(logged) if logged else None
                evals = [value for step, value in points if low < step <= high]
                row["evaluation_success_mean"] = statistics.mean(evals)
                band_rows.append(row)
            curve_rows.extend({"arm": arm, "seed": seed, "step": step, "success": value}
                              for step, value in points)
            print(f"failure analysis: {arm} seed {seed} complete", flush=True)
    groups = []
    for arm in ARMS:
        for label, condition in (("zero_evaluation", lambda r: r["zero_evaluation_success"]),
                                 ("nonzero_evaluation", lambda r: not r["zero_evaluation_success"]),
                                 ("hit90", lambda r: r["hit90_reached"]),
                                 ("not_hit90", lambda r: not r["hit90_reached"])):
            selected = [r for r in summaries if r["arm"] == arm and condition(r)]
            row = {"arm": arm, "posthoc_outcome_group": label, "n_runs": len(selected),
                   "seeds": ";".join(str(r["seed"]) for r in selected)}
            for key in ("auc", "support_rate", "eligible_rate", "saturated_nonempty_update",
                        "first_100k_saturated_nonempty_update", "first_500k_saturated_nonempty_update"):
                row[key + "_seed_mean"] = statistics.mean(r[key] for r in selected) if selected else None
            groups.append(row)
    notes = [
        "Exploratory analysis chosen after seeing outcomes; no confirmatory p-values or causal claims.",
        "All 30 runs and all evaluation checkpoints are retained, including zero-success runs.",
        "Zero evaluation success, failure to reach Hit90, and execution failure are different concepts.",
        "Saturation is k=|E|>0: all eligible positions are high-weight, not necessarily the entire minibatch.",
        "The three assignment rules coincide at saturation only conditional on the same batch and models.",
        "First saturation is interval-censored to a 1000-update window; its exact update is unknown.",
        "Window-end bins allocate whole windows, not fractional reconstructions of unlogged updates.",
        "Within-run diagnostic means use valid-update counts; group means give each seed equal weight.",
        "Critic log averages use completed-training-episode snapshots, excluding duplicate eval rows; not all updates or true Q errors.",
        "Early-window associations with final outcomes are not out-of-sample failure prediction.",
        "PAL has no support instrumentation and is intentionally not assigned fabricated saturation values.",
    ]
    for filename, rows in (("run_summary.csv", summaries), ("window_trajectories.csv", window_rows),
                           ("time_bands.csv", band_rows), ("evaluation_curves.csv", curve_rows),
                           ("posthoc_groups.csv", groups)):
        write_csv(output / filename, rows)
    write_json(output / "report.json", {"complete_runs": len(summaries), "notes": notes,
               "primary_inference_unchanged": report["inference"], "posthoc_groups": groups}, exclusive=True)
    finish_artifact(output, inputs, {"kind": "exploratory_failure_saturation", "complete_runs": 30})


if __name__ == "__main__":
    main()
