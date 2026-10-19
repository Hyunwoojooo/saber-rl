"""Read-only archive, source, result-summary and link checks; no ML imports."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def check_hashes(root, entries):
    for relative, expected in entries.items():
        path = (root / relative).resolve()
        if root.resolve() not in path.parents or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Missing, escaped or changed file: {relative}")


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"Numeric mismatch: {label}: {actual} != {expected}")


def check_results(folder):
    report = folder / "results/outputs/analysis/report.json"
    if report.exists():
        data = load(report)
        if data.get("problems") or data.get("failures_or_missing"):
            raise ValueError(f"Unexpected incomplete final report: {report}")
        runs = data["runs"] + data.get("reference_runs", [])
        for arm, summary in data["arm_summary"].items():
            metrics = [r["metrics"] for r in runs if r["arm"] == arm]
            if len(metrics) != 10:
                raise ValueError(f"Expected ten paired seeds: {folder.name}/{arm}")
            for metric in ("auc", "fq_auc", "final_success"):
                close(statistics.mean(r[metric] for r in metrics), summary[metric]["mean"], metric)
                close(statistics.stdev(r[metric] for r in metrics), summary[metric]["sd"], metric)
    report = folder / "results/analysis/final.json"
    if report.exists():
        data = load(report)
        if data["status"] != "completed" or len(data["runs"]) != 20:
            raise ValueError("SAC reference cohort is incomplete")
        for a in data["aggregates"]:
            runs = [r for r in data["runs"] if r["task"] == a["task"] and r["method"] == a["method"]]
            if sorted(r["seed"] for r in runs) != [1, 2, 3, 4, 5]:
                raise ValueError("SAC seed set changed")
            for metric in ("auc", "fq_auc", "final"):
                close(statistics.mean(r[metric] for r in runs), a[f"{metric}_mean"], metric)
                close(statistics.stdev(r[metric] for r in runs), a[f"{metric}_sd"], metric)


def check_links(path):
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
        if "://" in target or target.startswith("#"):
            continue
        if not (path.parent / target.split("#", 1)[0]).exists():
            raise ValueError(f"Broken current README link: {path.name}: {target}")


def verify(root=ROOT):
    index = load(root / "experiments/index.json")
    source_count = 0
    for entry in index["experiments"]:
        folder = root / "experiments" / entry["directory"]
        provenance = load(folder / "provenance.json")
        check_hashes(folder / "source", provenance["source_files_sha256"])
        for name in ("source_manifest.json", "recovery_manifest.json"):
            path = folder / "source" / name
            if path.exists():
                check_hashes(folder / "source", load(path)["sha256"])
        path = folder / "source/upstream_manifest.json"
        if path.exists():
            check_hashes(folder / "source", {r["path"]: r["sha256"] for r in load(path)["files"]})
        for report in provenance["reports"]:
            check_hashes(folder, {report["published_path"]: report["published_sha256"]})
        source_count += len(provenance["source_files_sha256"])
        check_results(folder)
        check_links(folder / "README.md")
    check_links(root / "README.md")
    progress_files = list((root / "experiments").glob("*/results/progress.json"))
    if len(progress_files) != 1:
        raise ValueError("Expected exactly one in-progress cohort")
    p = load(progress_files[0])
    if len(p["runs"]) != 20 or sum(p["counts"].values()) != 20:
        raise ValueError("Progress population mismatch")
    for status, n in p["counts"].items():
        if sum(r["status"] == status for r in p["runs"]) != n:
            raise ValueError("Progress count mismatch")
    for r in p["runs"]:
        if r["status"] == "completed" and (r["step"] != 2000000 or r["eval_step"] != 2000000):
            raise ValueError("Partial run represented as completed")
    expected = {"planned": 90, "completed": 70 + p["counts"].get("completed", 0),
                "running": p["counts"].get("running", 0), "pending": p["counts"].get("pending", 0)}
    if index["training_counts"] != expected or sum(expected[k] for k in ("completed", "running", "pending")) != 90:
        raise ValueError("Unique-run totals mismatch")
    inventory = {}
    parsed = 0
    secret_patterns = [r"gh[pousr]_[A-Za-z0-9]{30,}", r"github_pat_[A-Za-z0-9_]{50,}",
                       r"AKIA[0-9A-Z]{16}", r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"]
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.name == "ARCHIVE_MANIFEST.sha256":
            continue
        if path.is_symlink() or path.suffix in {".pt", ".pth", ".pkl", ".log", ".pyc"} or path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError(f"Unexpected archive payload: {path}")
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
            parsed += 1
        text = path.read_text()
        if any(re.search(pattern, text) for pattern in secret_patterns):
            raise ValueError(f"Potential credential: {path}")
        inventory[str(path.relative_to(root))] = digest(path)
    return inventory, {"experiments": len(index["experiments"]), "source_files": source_count,
                       "python_files_parsed": parsed, "files": len(inventory), "training_counts": expected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true", help="Generate the publication manifest after verification")
    args = parser.parse_args()
    inventory, report = verify()
    text = "".join(f"{h}  {p}\n" for p, h in sorted(inventory.items()))
    path = ROOT / "ARCHIVE_MANIFEST.sha256"
    if args.write_manifest:
        path.write_text(text)
    elif not path.exists() or path.read_text() != text:
        raise ValueError("Publication manifest missing or changed")
    print(json.dumps({"passed": True, **report}, indent=2))


if __name__ == "__main__":
    main()
