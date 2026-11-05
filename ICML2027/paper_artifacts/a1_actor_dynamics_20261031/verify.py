"""Verify published A1 evidence with the Python standard library; no checkpoints.

Run from any directory. --original-experiment optionally compares each archived
input to its local original as well. This command never runs a training step.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("her", "saber", "random", "td_topk")
SEEDS = (101, 102, 103)
RULE = dict(her="uniform", saber="support", random="random", td_topk="td_topk")
CP = [0, 4800, 5000, 6000, 8000, 10000, 20000, 50000, 100000, 200000, 300000, 400000, 500000]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_stream(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(chunk)
    return h.hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return sha_stream(stream)


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def verify(original=None):
    manifest = read(ROOT / "manifest.json")
    for name, expected in manifest["files"].items():
        require(sha(ROOT / name) == expected, "Artifact changed: " + name)
    for name, item in manifest["packed_inputs"].items():
        path = ROOT / name
        if item["encoding"] == "gzip":
            with gzip.open(path, "rb") as stream:
                actual = sha_stream(stream)
        else:
            require(item["encoding"] == "identity", "Unknown encoding")
            actual = sha(path)
        require(actual == item["original_sha256"], "Archive differs from original bytes: " + name)
        if original is not None:
            require(sha(original / item["original_relative_to_experiment"]) == actual,
                    "Local original changed: " + name)
    source = (ROOT / manifest["training_source_relative_to_artifact"]).resolve()
    require(sha(source / "source_manifest.json") == manifest["training_source_manifest_sha256"], "Source manifest changed")
    for name, expected in read(source / "source_manifest.json")["files"].items():
        require(sha(source / name) == expected, "Frozen training source changed: " + name)
    require(sha(ROOT / manifest["exporter_relative_to_artifact"]) == manifest["exporter_sha256"], "Exporter changed")
    cohort = read(ROOT / "cohort/status.json")
    require(cohort["status"] == "complete" and len(cohort["jobs"]) == 12, "Incomplete cohort")
    require({(j["arm"], j["seed"]) for j in cohort["jobs"]} == {(a, s) for a in ARMS for s in SEEDS}, "Wrong cohort cells")
    require(all(j["status"] == "complete" and j["returncode"] == 0 for j in cohort["jobs"]), "Failed cohort cell")
    with (ROOT / "analysis/seed_metrics.csv").open() as stream:
        seed_rows = list(csv.DictReader(stream))
    require(len(seed_rows) == 12, "Wrong summary row count")
    indexed = {(r["arm"], int(r["seed"])): r for r in seed_rows}
    require(len(indexed) == 12, "Duplicate summary row")
    measured = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            path = ROOT / "records" / arm / f"seed_{seed}"
            cfg = read(path / "requested_config.json")
            actual = read(path / "run/config.json")
            actual.pop("thread_hint", None)
            actual.pop("terminal_convention", None)
            require(actual == cfg and cfg["total_steps"] == 500000 and cfg["seeds"]["env_seed"] == seed
                    and cfg["assignment_rule"] == RULE[arm], "Configuration mismatch")
            state = read(path / "status.json")
            audit = read(path / "completion_audit.json")
            require(state["status"] == "complete" and audit["updates"] == 495001
                    and state["source_manifest_sha256"] == manifest["training_source_manifest_sha256"], "Attempt metadata mismatch")
            with gzip.open(path / "run/metrics.csv.gz", "rt") as stream:
                evals = [(int(r["step"]), float(r["eval/env_success_mean"])) for r in csv.DictReader(stream)
                         if r["eval/env_success_mean"] != ""]
            require([s for s, _ in evals] == list(range(10000, 500001, 10000)), "Evaluation grid mismatch")
            require(all(math.isfinite(v) and 0 <= v <= 1 for _, v in evals), "Invalid evaluation value")
            auc = math.fsum((b[0]-a[0])*(a[1]+b[1])/2 for a, b in zip(evals, evals[1:]))/490000
            row = indexed[(arm, seed)]
            require(abs(auc-float(row["auc"])) < 1e-12 and abs(auc-audit["auc"]) < 1e-12, "AUC mismatch")
            require(evals[-1][1] == float(row["final"]) == audit["final"], "Final success mismatch")
            hit = next((s for s, v in evals if v >= .9), None)
            require(hit == audit["hit90"], "Hit90 mismatch")
            zero = all(v == 0 for _, v in evals)
            require(zero == audit["zero_evaluation_run"], "Zero-evaluation mismatch")
            cps = list(jsonl(path / "diagnostics/checkpoints.jsonl.gz"))
            require([r["step"] for r in cps] == CP, "Checkpoint record schedule mismatch")
            require(all(r["total_updates"] == max(0, r["step"]-4999) for r in cps), "Checkpoint update record mismatch")
            probes = list(jsonl(path / "diagnostics/probes.jsonl.gz"))
            expected = [0, 4800] + list(range(5000, 10001, 100)) + list(range(11000, 500001, 1000))
            for kind in ("fixed_original", "fixed_her"):
                rows = [r for r in probes if r["query_source"] == kind]
                require([r["step"] for r in rows] == expected, "Probe schedule mismatch")
                first = next((r["step"] for r in rows if r["tanh_derivative_exact_zero_fraction"] == 1), None)
                require(first == audit["first_observed_all_tanh_zero_"+kind], "Saturation observation mismatch")
            windows = list(jsonl(path / "diagnostics/update_windows.jsonl.gz"))
            next_update = 1
            for r in windows:
                require(r["start_update"] == next_update and r["end_update"]-r["start_update"]+1 == r["n_updates"], "Update window gap")
                next_update = r["end_update"]+1
                if arm == "her":
                    require(r["means"]["critic_coefficient_variance"] == 0
                            and r["means"]["actual_high_weight_fraction"] == 0, "HER was reweighted")
            require(next_update == 495002, "Missing updates")
            require(sum(r["n_transitions"] for r in jsonl(path / "diagnostics/training_windows.jsonl.gz")) == 500000, "Missing transitions")
            require([r["step"] for r in jsonl(path / "diagnostics/evaluation.jsonl.gz")] == [s for s, _ in evals], "Evaluation diagnostic grid mismatch")
            measured[arm].append((auc, evals[-1][1], zero))
    report = read(ROOT / "analysis/report.json")
    require(report["n_seeds"] == 3 and report["confidence_intervals"] is None, "Pilot inference changed")
    for arm, values in measured.items():
        expected = dict(auc_mean=statistics.mean(x[0] for x in values), auc_sd=statistics.stdev(x[0] for x in values),
                        final_mean=statistics.mean(x[1] for x in values), zero_evaluation_runs=sum(x[2] for x in values))
        for key, value in expected.items():
            require(abs(value-report["summary"][arm][key]) < 1e-12, "Aggregate mismatch: " + arm + "/" + key)
    print(json.dumps({"verified": True, "artifact_files": len(manifest["files"]), "runs": 12,
                      "evaluation_checkpoints": 600, "checkpoint_records": 156,
                      "original_inputs_compared": original is not None, "new_training_steps": 0}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-experiment", type=Path)
    args = parser.parse_args()
    verify(args.original_experiment.resolve() if args.original_experiment else None)
