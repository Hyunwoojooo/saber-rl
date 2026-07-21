#!/usr/bin/env python3
"""Standalone verifier for the FPP DDPG shuffled-policy-query paper artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


sys.dont_write_bytecode = True


RESULT_FILES = (
    "aggregate_results.csv",
    "analysis_results.json",
    "config_parity.json",
    "exact_grid_mean_curves.csv",
    "leave_one_seed_out.csv",
    "paired_differences.csv",
    "paired_inference.csv",
    "per_seed_results.csv",
    "query_diagnostics_per_seed.csv",
    "query_diagnostics_summary.csv",
    "validation.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_digest(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            f"{record['path']}\0{record['sha256']}\0{record['size_bytes']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def frozen_digest(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["packaged_path"])):
        digest.update(
            f"{record['packaged_path']}\0{record['sha256']}\0{record['size_bytes']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("artifact_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_manifest(root: Path) -> dict[str, object]:
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("packaged_files")
    if not isinstance(records, list):
        raise ValueError("artifact_manifest.json: packaged_files is not a list")
    declared_paths: set[str] = set()
    for record in records:
        relative = str(record["path"])
        if relative in declared_paths:
            raise ValueError(f"duplicate manifest path: {relative}")
        declared_paths.add(relative)
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"manifest file missing: {relative}")
        if path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"size mismatch: {relative}")
        if sha256(path) != record["sha256"]:
            raise ValueError(f"SHA256 mismatch: {relative}")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    if actual_paths != declared_paths:
        raise ValueError(
            f"artifact coverage mismatch: missing={sorted(declared_paths - actual_paths)}, "
            f"unexpected={sorted(actual_paths - declared_paths)}"
        )
    if len(records) != int(manifest.get("packaged_file_count_excluding_manifest", -1)):
        raise ValueError("packaged file count mismatch")
    if package_digest(records) != manifest.get("package_content_sha256_excluding_manifest"):
        raise ValueError("aggregate package digest mismatch")
    if manifest.get("method_seed_coverage") != {"main": [1, 2, 3, 4, 5], "shuffled": [1, 2, 3, 4, 5]}:
        raise ValueError("manifest method/seed coverage mismatch")
    if int(manifest.get("production_run_count", -1)) != 10:
        raise ValueError("manifest production run count is not 10")
    protocol = manifest.get("evaluation_protocol", {})
    if protocol.get("primary_success_field") != "eval/env_success_mean":
        raise ValueError("manifest uses the wrong success field")
    if protocol.get("steps") != list(range(10_000, 1_000_001, 10_000)):
        raise ValueError("manifest exact evaluation grid mismatch")
    if manifest.get("paired_bootstrap") != {"resamples": 10_000, "rng_seed": 27_027}:
        raise ValueError("manifest bootstrap protocol mismatch")
    if manifest.get("exact_sign_flip") != {"assignment_count": 32, "two_sided": True}:
        raise ValueError("manifest sign-flip protocol mismatch")
    return manifest


def verify_input_manifest(root: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    inputs = json.loads((root / "input_manifest.json").read_text(encoding="utf-8"))
    if inputs.get("execution_source_commit") != manifest.get("execution_source_commit"):
        raise ValueError("execution source commit mismatch between manifests")
    methods = inputs.get("methods")
    if not isinstance(methods, dict) or set(methods) != {"main", "shuffled"}:
        raise ValueError("input manifest must contain exactly main and shuffled")
    packaged_runs = 0
    for method_key, expected_mode in (("main", "current_policy"), ("shuffled", "shuffled_policy")):
        method = methods[method_key]
        if method.get("expected_support_query_mode") != expected_mode:
            raise ValueError(f"{method_key}: wrong expected query mode")
        runs = method.get("runs")
        if not isinstance(runs, list) or [int(run["seed"]) for run in runs] != [1, 2, 3, 4, 5]:
            raise ValueError(f"{method_key}: input seed coverage mismatch")
        for run in runs:
            packaged_runs += 1
            if int(run.get("evaluation_checkpoint_count", -1)) != 100:
                raise ValueError(f"{method_key} seed {run.get('seed')}: compact grid count mismatch")
            for record in run.get("files", []):
                path = root / record["packaged_path"]
                if not path.is_file():
                    raise FileNotFoundError(path)
                if path.stat().st_size != int(record["packaged_size_bytes"]):
                    raise ValueError(f"packaged input size mismatch: {path}")
                if sha256(path) != record["packaged_sha256"]:
                    raise ValueError(f"packaged input SHA mismatch: {path}")
    if packaged_runs != 10:
        raise ValueError(f"expected 10 packaged runs, got {packaged_runs}")

    source_records = inputs.get("implementation_source")
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("input manifest has no implementation source records")
    for record in source_records:
        path = root / str(record["packaged_path"])
        if not path.is_file() or path.stat().st_size != int(record["size_bytes"]) or sha256(path) != record["sha256"]:
            raise ValueError(f"frozen source record mismatch: {record['packaged_path']}")
    digest = frozen_digest(source_records)
    frozen_line = (root / "frozen_source.sha256").read_text(encoding="utf-8").strip()
    if frozen_line != f"{digest}  code":
        raise ValueError("frozen_source.sha256 mismatch")
    if digest != inputs.get("frozen_source_tree_sha256") or digest != manifest.get("frozen_source_tree_sha256"):
        raise ValueError("frozen source digest mismatch between manifests")
    return inputs


def verify_source_parity(root: Path) -> dict[str, object]:
    parity = json.loads((root / "results" / "source_parity.json").read_text(encoding="utf-8"))
    if parity.get("passed") is not True:
        raise ValueError("packaged source parity report did not pass")
    for record in parity.get("protected_files", []):
        path = root / "code" / str(record["path"])
        if not path.is_file() or sha256(path) != record["intervention_sha256"]:
            raise ValueError(f"protected intervention source mismatch: {record['path']}")
    return parity


def verify_recomputed_analysis(root: Path):
    analyzer = load_module(root / "code" / "analyze_results.py")
    with tempfile.TemporaryDirectory(prefix="verify_ddpg_shuffled_") as tmp:
        regenerated = Path(tmp) / "results"
        payload = analyzer.run_analysis(root / "data" / "main", root / "data" / "shuffled", regenerated)
        for name in RESULT_FILES:
            expected = root / "results" / name
            actual = regenerated / name
            if not expected.is_file() or not actual.is_file():
                raise FileNotFoundError(f"missing result file during replay: {name}")
            if expected.read_bytes() != actual.read_bytes():
                raise ValueError(f"analysis regeneration mismatch: results/{name}")
    if payload.get("validation_passed") is not True:
        raise ValueError("regenerated analysis validation failed")
    inference = payload.get("paired_inference", {})
    if inference.get("bootstrap_resamples") != 10_000 or inference.get("bootstrap_rng_seed") != 27_027:
        raise ValueError("regenerated bootstrap protocol mismatch")
    if inference.get("sign_flip_assignment_count") != 32:
        raise ValueError("regenerated sign-flip assignment count mismatch")
    if not math.isfinite(float(inference.get("mean_auc_difference"))):
        raise ValueError("paired AUC difference is non-finite")
    return payload


def verify_artifact(root: Path) -> dict[str, object]:
    root = root.resolve()
    manifest = verify_manifest(root)
    inputs = verify_input_manifest(root, manifest)
    verify_source_parity(root)
    payload = verify_recomputed_analysis(root)
    readme = (root / "README.md").read_text(encoding="utf-8")
    required_phrases = (
        "shuffled-policy query",
        "not shuffled-weight",
        "not random-support",
        "replay action",
        "actor remains unweighted",
        "does not claim",
    )
    missing = [phrase for phrase in required_phrases if phrase.lower() not in readme.lower()]
    if missing:
        raise ValueError(f"README is missing required interpretation text: {missing}")
    return {
        "status": "PASS",
        "artifact_dir": str(root),
        "packaged_file_count_excluding_manifest": manifest["packaged_file_count_excluding_manifest"],
        "production_run_count": 10,
        "evaluation_points_per_run": 100,
        "execution_source_commit": inputs["execution_source_commit"],
        "bootstrap_assignments": 10_000,
        "sign_flip_assignments": 32,
        "main_minus_shuffled_auc": payload["paired_inference"]["mean_auc_difference"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> int:
    result = verify_artifact(parse_args().artifact_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
