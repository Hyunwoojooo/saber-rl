"""Validate both artifacts and write a single descriptive handoff report."""
import csv

from common import ROOT, SEEDS, QUERIES, digest, read_json, write_json, output_directory, finish_artifact


def check_artifact(folder):
    manifest_path = folder / "manifest.json"
    manifest = read_json(manifest_path)
    from freeze import verify
    if manifest["source_manifest_sha256"] != verify():
        raise ValueError("Artifact was generated with different diagnostic source")
    for relative, sha in manifest["outputs_sha256"].items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder) or digest(path) != sha:
            raise ValueError(f"Artifact output verification failed: {path}")
    for path, sha in manifest["inputs_sha256"].items():
        if digest(path) != sha:
            raise ValueError(f"Artifact input verification failed: {path}")
    return {str(manifest_path): digest(manifest_path)}


def main():
    inputs = {}
    for name in ("failure_analysis", "retrieval_pilot", "retrieval"):
        inputs.update(check_artifact(ROOT / "outputs" / name))
    retrieval = read_json(ROOT / "outputs/retrieval/report.json")
    failure = read_json(ROOT / "outputs/failure_analysis/report.json")
    if (retrieval["profile"] != "full" or retrieval["seeds"] != list(SEEDS)
            or retrieval["queries_per_seed"] != QUERIES or len(retrieval["checks"]) != 10
            or failure["complete_runs"] != 30):
        raise ValueError("Final scope incomplete")
    if any(c["tail_parity_mismatches"] or c["queries"] != QUERIES for c in retrieval["checks"]):
        raise ValueError("Incomplete or mismatched retrieval audit")
    with (ROOT / "outputs/failure_analysis/run_summary.csv").open(newline="") as handle:
        runs = list(csv.DictReader(handle))
    zero = [r for r in runs if r["zero_evaluation_success"] == "True"]
    nonzero = [r for r in runs if r["zero_evaluation_success"] == "False"]
    pattern = {"zero_evaluation_runs": len(zero), "nonzero_evaluation_runs": len(nonzero),
               "zero_runs_with_any_saturation": sum(float(r["saturated_nonempty_update"]) > 0 for r in zero),
               "nonzero_runs_with_any_saturation": sum(float(r["saturated_nonempty_update"]) > 0 for r in nonzero)}
    destination = output_directory(ROOT / "outputs/final")
    result = {"retrieval": retrieval, "failure_analysis": failure, "saturation_pattern": pattern,
              "new_training_runs": 0, "interpretation": "descriptive, not causal or confirmatory"}
    write_json(destination / "report.json", result, exclusive=True)
    lines = ["# Post-PAL retrieval and failure diagnostics", "",
             "This report contains no new training runs. All analyses are descriptive.", "",
             "## Retrieval", "",
             "Ten SABER final checkpoints; 1024 fixed HER queries per seed; five random retrieval repeats.",
             "Tail parity and direct full-replay spot checks passed for every checkpoint.", "",
             "| Method | Support rate (seed mean) | Miss rate given full positive (seed mean) |", 
             "| --- | ---: | ---: |"]
    for row in retrieval["aggregate"]:
        miss = row["miss_rate_given_full_positive_seed_mean"]
        miss_text = "NA" if miss is None else f"{miss:.6f}"
        lines.append(f"| {row['method']} | {row['support_rate_seed_mean']:.6f} | {miss_text} |")
    lines.extend(["", "See retrieval/seed_summary.csv and random_repeat_summary.csv for denominators and variation.",
                  "Timing is for a shared CPU reference implementation, not production Numba throughput.", "",
                  "## Failure and saturation", "",
                  f"Zero evaluation success: {len(zero)}/30 runs; nonzero evaluation success: {len(nonzero)}/30.",
                  f"Any saturation among zero/nonzero runs: {pattern['zero_runs_with_any_saturation']}/{len(zero)} and "
                  f"{pattern['nonzero_runs_with_any_saturation']}/{len(nonzero)} respectively.",
                  "These outcome-defined groups do not establish that saturation caused failure.",
                  "First saturation is known only to a logged update window; training and evaluation success are reported separately.",
                  "Saturation selects all eligible batch positions, not necessarily the entire minibatch.", "",
                  "## Limits", "",
                  "- Final-checkpoint search agreement does not establish online learning equivalence.",
                  "- Exhaustive matching uses the same discretized context, not ground truth continuous support.",
                  "- Neither query counts nor random repetitions increase the number of independent training seeds.",
                  "- The original paired AUC intervals are retained unchanged; no new confirmatory tests were selected.",
                  "- Latest/best retention cannot reconstruct all earlier policies and replay buffers.", ""])
    with (destination / "REPORT.md").open("x") as handle:
        handle.write("\n".join(lines))
    finish_artifact(destination, inputs, {"kind": "combined_diagnostic_handoff", "passed": True})
    print(f"Completed handoff: {destination}")


if __name__ == "__main__":
    main()
