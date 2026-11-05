"""Package completed A1 evidence without modifying the frozen experiment.

Uses only the standard library, performs no training, and refuses to replace an
existing artifact. Raw CSV/JSONL are losslessly compressed with stable gzip headers.
"""
import argparse
import gzip
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "experiments/draweropen_ddpg_actor_dynamics_20261031"
DEFAULT_OUTPUT = HERE / "paper_artifacts/a1_actor_dynamics_20261031"
ARMS = ("her", "saber", "random", "td_topk")
SEEDS = (101, 102, 103)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def publish(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    cohort = source / "outputs/pilot_01"
    status = read(cohort / "status.json")
    source_manifest = read(source / "source_manifest.json")
    source_hash = sha(source / "source_manifest.json")
    require(status["status"] == "complete", "Cohort has not completed")
    require(status["source_manifest_sha256"] == source_hash, "Training source identity differs")
    for name, expected in source_manifest["files"].items():
        require(sha(source / name) == expected, "Frozen source changed: " + name)
    expected_cells = {(a, s) for a in ARMS for s in SEEDS}
    require(len(status["jobs"]) == 12 and
            {(j["arm"], j["seed"]) for j in status["jobs"]} == expected_cells,
            "Unexpected cohort cells")
    for job in status["jobs"]:
        attempt = cohort / "runs" / job["arm"] / f"seed_{job['seed']}"
        state = read(attempt / "status.json")
        audit = read(attempt / "completion_audit.json")
        require(job["returncode"] == 0 and state["status"] == "complete", "Incomplete attempt")
        require(audit["auc_end"] == 500000 and audit["updates"] == 495001
                and audit["scheduled_checkpoints"] == 13, "Unexpected completion audit")
        require(state["source_manifest_sha256"] == source_hash, "Attempt source identity differs")
    require(not (output / "manifest.json").exists(), "Artifact is already sealed")
    output.mkdir(parents=True, exist_ok=True)
    require({p.name for p in output.iterdir()} == {"README.md", "verify.py"},
            "Destination must contain only the publication README and verifier")
    packed = {}

    def copy(relative_source, relative_output, compress=False):
        original = source / relative_source
        destination = output / relative_output
        require(original.is_file(), "Missing input: " + str(original))
        require(not destination.exists(), "Refusing to replace " + str(destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if compress:
            with original.open("rb") as src, destination.open("xb") as dst:
                with gzip.GzipFile(filename="", fileobj=dst, mode="wb", mtime=0) as gz:
                    shutil.copyfileobj(src, gz)
        else:
            shutil.copyfile(original, destination)
        packed[str(relative_output)] = {
            "original_relative_to_experiment": str(relative_source),
            "original_sha256": sha(original),
            "original_bytes": original.stat().st_size,
            "encoding": "gzip" if compress else "identity",
        }

    for name in ("REPORT_ko.md", "report.json", "seed_metrics.csv"):
        copy(Path("outputs/pilot_01/analysis") / name, Path("analysis") / name)
    for name in ("status.json", "plan.json", "launch_receipt.json"):
        copy(Path("outputs/pilot_01") / name, Path("cohort") / name)
    for arm in ARMS:
        for seed in SEEDS:
            src = Path("outputs/pilot_01/runs") / arm / f"seed_{seed}"
            dst = Path("records") / arm / f"seed_{seed}"
            for name in ("status.json", "requested_config.json", "environment.json", "completion_audit.json",
                         "run/config.json", "run/env_snapshot.json", "run/summary.json", "run/best_checkpoint.json",
                         "diagnostics/probe_bank.json", "diagnostics/summary.json", "diagnostics/fixed_probe_bank.npz"):
                copy(src / name, dst / name)
            for name in ("run/metrics.csv", "run/assignment_windows.csv",
                         "diagnostics/checkpoints.jsonl", "diagnostics/evaluation.jsonl", "diagnostics/events.jsonl",
                         "diagnostics/probes.jsonl", "diagnostics/training_windows.jsonl",
                         "diagnostics/update_windows.jsonl", "diagnostics/updates.jsonl"):
                copy(src / name, dst / (name + ".gz"), compress=True)
    for name in ("REPORT.md", "results.json", "gate.json", "source_audit.json", "native_changes.diff"):
        copy(Path("validation") / name, Path("validation") / name)
    copy(Path("validation/runs/check_01/report.json"), Path("validation/smoke_report.json"))
    copy(Path("validation/runs/check_01/unit_tests.log"), Path("validation/unit_tests.txt"))
    manifest = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_source_relative_to_artifact": "../../experiments/draweropen_ddpg_actor_dynamics_20261031",
        "training_source_manifest_sha256": source_hash,
        "exporter_relative_to_artifact": "../../publish_a1_artifact.py",
        "exporter_sha256": sha(Path(__file__)),
        "new_training_runs": 0,
        "cohort_runs": 12,
        "cohort_training_steps": 6000000,
        "packed_inputs": packed,
        "files": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
        "omitted": ["PyTorch checkpoints", "duplicated native run/metrics.jsonl", "console logs", "runtime caches"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(output), "files": len(manifest["files"]),
                      "bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file())}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    publish(args.source, args.output)
