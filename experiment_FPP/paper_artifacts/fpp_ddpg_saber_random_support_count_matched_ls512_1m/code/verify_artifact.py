#!/usr/bin/env python3
"""Standalone verifier for the DDPG batch-wide Random-Support artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

# A verifier must not mutate the supposedly immutable artifact simply by being
# executed from inside it.
sys.dont_write_bytecode = True

from random_support_analysis import (
    ARTIFACT_ID, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, CONTROL_KEY,
    GRID, MAIN_KEY, SCHEMA_VERSION, SEEDS, TOTAL_STEPS, TOTAL_UPDATES,
    AnalysisError, actual_config_differences, analyze, read_json,
    verify_outputs,
)


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ROOT_FILES = {"README.md", "input_manifest.json", "artifact_manifest.json", "SHA256SUMS", "source_commit.txt"}
ROOT_DIRS = {"code", "data", "results", "tests", "technical_gate"}
SEED_MAIN_FILES = {"config.json", "env_snapshot.json", "summary.json", "metrics_selected_10k.csv"}
SEED_CONTROL_FILES = SEED_MAIN_FILES | {"assignment_diagnostics_1000.csv"}
RESULT_FILES = {
    "per_seed_results.csv", "aggregate_results.csv", "paired_differences.csv",
    "paired_inference.csv", "leave_one_seed_out.csv", "mean_learning_curve.csv",
    "assignment_diagnostic_summary.csv", "analysis_results.json", "report.md",
    "validation.json", "completion_validation.json",
}
FORBIDDEN_NAMES = {"metrics.csv", "metrics.jsonl", "run.log", "checkpoint.pt", "best_checkpoint.pt"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".jsonl", ".pyc"}
FINGERPRINT_EXCLUSIONS = {
    "names": ["freeze_manifest.json"],
    "directory_names": ["__pycache__"],
    "suffixes": [".nbc", ".nbi", ".pyc"],
}


class VerificationError(AnalysisError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_oid(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(payload)}\0".encode("ascii") + payload
    ).hexdigest()


def fingerprint_tree(files_sha256: Mapping[str, str]) -> str:
    canonical = json.dumps(
        dict(files_sha256), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        b"random-support-count-matched-source-fingerprint-v1\0" + canonical
    ).hexdigest()


def validate_fingerprint(payload: Any, stage: str) -> dict[str, str]:
    if not isinstance(payload, Mapping):
        raise VerificationError(f"{stage} source fingerprint missing")
    for key, value in {
        "schema_version": 1,
        "algorithm": "sha256",
        "scope": "full_experiment_package",
        "stage": stage,
        "exclusions": FINGERPRINT_EXCLUSIONS,
    }.items():
        if payload.get(key) != value:
            raise VerificationError(f"{stage} source fingerprint {key} mismatch")
    if not isinstance(payload.get("created_utc"), str) or not payload["created_utc"]:
        raise VerificationError(f"{stage} source fingerprint creation time missing")
    files = payload.get("files_sha256")
    if not isinstance(files, Mapping) or not files:
        raise VerificationError(f"{stage} source fingerprint files missing")
    checked: dict[str, str] = {}
    for relative, digest in files.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise VerificationError(f"{stage} fingerprint entry invalid")
        path = Path(relative)
        if (
            path.is_absolute()
            or path.as_posix() != relative
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.name == "freeze_manifest.json"
            or "__pycache__" in path.parts
            or path.suffix in {".pyc", ".nbi", ".nbc"}
            or not HEX64.fullmatch(digest)
        ):
            raise VerificationError(f"{stage} fingerprint entry unsafe: {relative!r}")
        checked[relative] = digest
    if payload.get("file_count") != len(checked):
        raise VerificationError(f"{stage} source fingerprint count mismatch")
    if payload.get("tree_sha256") != fingerprint_tree(checked):
        raise VerificationError(f"{stage} source fingerprint tree mismatch")
    return checked


def files(root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise VerificationError(f"symlink forbidden: {path}")
        if path.is_file():
            output[path.relative_to(root).as_posix()] = path
    return output


def strict_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid/non-finite JSON {path}: {exc}") from exc


def verify_layout(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise VerificationError(f"artifact root invalid: {root}")
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    actual_dirs = {path.name for path in root.iterdir() if path.is_dir()}
    if actual_files != ROOT_FILES or actual_dirs != ROOT_DIRS:
        raise VerificationError(f"root layout mismatch: files={actual_files}, dirs={actual_dirs}")
    for key, expected in ((MAIN_KEY, SEED_MAIN_FILES), (CONTROL_KEY, SEED_CONTROL_FILES)):
        method_root = root / "data" / key
        seed_dirs = {path.name for path in method_root.iterdir() if path.is_dir()}
        if seed_dirs != {f"seed_{seed}" for seed in SEEDS}:
            raise VerificationError(f"{key} seed coverage/layout mismatch")
        for seed in SEEDS:
            seed_root = method_root / f"seed_{seed}"
            actual = {path.name for path in seed_root.iterdir() if path.is_file()}
            if actual != expected:
                raise VerificationError(f"{key} seed {seed} payload mismatch: {actual}")
    if {path.name for path in (root / "results").iterdir() if path.is_file()} != RESULT_FILES:
        raise VerificationError("result file set mismatch")
    if {path.name for path in (root / "technical_gate").iterdir() if path.is_file()} != {"smoke_validation.json", "gate_validation.json", "freeze_manifest.json"}:
        raise VerificationError("technical_gate file set mismatch")
    if not (root / "code/execution_source/config.json").is_file():
        raise VerificationError("frozen execution source config missing")
    all_paths = files(root)
    for relative, path in all_paths.items():
        if path.name in FORBIDDEN_NAMES or path.suffix in FORBIDDEN_SUFFIXES or "__pycache__" in path.parts:
            raise VerificationError(f"forbidden payload: {relative}")


def verify_manifest(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "artifact_manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("artifact_id") != ARTIFACT_ID:
        raise VerificationError("manifest identity mismatch")
    for field in ("execution_source_commit", "canonical_main_source_commit"):
        if not isinstance(manifest.get(field), str) or not HEX40.fullmatch(manifest[field]):
            raise VerificationError(f"manifest {field} invalid")
    if manifest.get("method_seed_coverage") != {MAIN_KEY: list(SEEDS), CONTROL_KEY: list(SEEDS)}:
        raise VerificationError("manifest seed coverage mismatch")
    if manifest.get("evaluation") != {"steps": list(GRID), "checkpoint_count": 100, "auc_normalization_denominator": 990_000}:
        raise VerificationError("manifest evaluation protocol mismatch")
    if manifest.get("bootstrap") != {"resamples": BOOTSTRAP_RESAMPLES, "rng_seed": BOOTSTRAP_SEED} or manifest.get("exact_sign_flip_assignments") != 32:
        raise VerificationError("manifest inference protocol mismatch")
    actual = {name: path for name, path in files(root).items() if name not in {"artifact_manifest.json", "SHA256SUMS"}}
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise VerificationError("manifest files is not a list")
    recorded: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str) or entry["path"] in recorded:
            raise VerificationError(f"invalid/duplicate manifest entry: {entry!r}")
        recorded[entry["path"]] = entry
    if set(recorded) != set(actual):
        raise VerificationError("manifest file set mismatch")
    for name, path in actual.items():
        entry = recorded[name]
        if entry.get("sha256") != sha256(path) or entry.get("size_bytes") != path.stat().st_size:
            raise VerificationError(f"manifest hash/size mismatch: {name}")
    if manifest.get("file_count") != len(actual) or manifest.get("total_size_bytes") != sum(path.stat().st_size for path in actual.values()):
        raise VerificationError("manifest aggregate mismatch")
    source = (root / "source_commit.txt").read_text(encoding="utf-8").strip()
    if source != manifest["execution_source_commit"]:
        raise VerificationError("source_commit.txt mismatch")
    return manifest


def verify_sums(root: Path) -> None:
    recorded: dict[str, str] = {}
    for line_number, line in enumerate((root / "SHA256SUMS").read_text(encoding="utf-8").splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match or match.group(2) in recorded:
            raise VerificationError(f"invalid SHA256SUMS line {line_number}")
        recorded[match.group(2)] = match.group(1)
    actual = {name: path for name, path in files(root).items() if name != "SHA256SUMS"}
    if set(recorded) != set(actual):
        raise VerificationError("SHA256SUMS file set mismatch")
    for name, path in actual.items():
        if recorded[name] != sha256(path):
            raise VerificationError(f"SHA256SUMS mismatch: {name}")


def verify_input_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    payload = read_json(root / "input_manifest.json")
    expected = {
        "schema_version": SCHEMA_VERSION, "artifact_id": ARTIFACT_ID,
        "execution_source_commit": manifest["execution_source_commit"],
        "canonical_main_source_commit": manifest["canonical_main_source_commit"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise VerificationError(f"input manifest {key} mismatch")
    if payload.get("repository") != {"id": "Hyunwoojooo/saber-rl", "branch": "main"}:
        raise VerificationError("repository provenance mismatch")
    methods = payload.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != {MAIN_KEY, CONTROL_KEY}:
        raise VerificationError("input manifest method set mismatch")
    for key, mode in ((MAIN_KEY, "support"), (CONTROL_KEY, "batch_wide_count_matched")):
        method = methods[key]
        if method.get("seeds") != list(SEEDS) or method.get("support_query_mode") != "current_policy" or method.get("support_assignment_mode") != mode:
            raise VerificationError(f"input manifest {key} contract mismatch")
        runs = method.get("runs")
        if not isinstance(runs, list) or [run.get("seed") for run in runs] != list(SEEDS):
            raise VerificationError(f"input manifest {key} run coverage mismatch")
    excluded = payload.get("excluded")
    if not isinstance(excluded, Mapping) or not all(excluded.values()):
        raise VerificationError("input exclusion record incomplete")


def verify_technical_gate(root: Path) -> None:
    smoke = read_json(root / "technical_gate/smoke_validation.json")
    gate = read_json(root / "technical_gate/gate_validation.json")
    freeze = read_json(root / "technical_gate/freeze_manifest.json")
    for payload, kind, steps in (
        (smoke, "smoke", 5_000),
        (gate, "gate", 50_000),
    ):
        if (
            payload.get("kind") != kind
            or payload.get("passed") is not True
            or payload.get("performance_used_for_gate_decision") is not False
            or payload.get("steps") != steps
            or payload.get("seeds") != [1]
        ):
            raise VerificationError(f"{kind} validation protocol mismatch")
        runs = payload.get("runs")
        if not isinstance(runs, list) or len(runs) != 1:
            raise VerificationError(f"{kind} run evidence mismatch")
        record = runs[0]
        if (
            record.get("seed") != 1
            or record.get("steps") != steps
            or record.get("stage") != kind
            or record.get("log", {}).get("resume_exercised") is not True
        ):
            raise VerificationError(f"{kind} seed/step/resume evidence mismatch")
        checkpoint = record.get("checkpoint", {})
        if (
            checkpoint.get("assignment_rng_state_present") is not True
            or checkpoint.get("diagnostic_state_present") is not True
        ):
            raise VerificationError(f"{kind} checkpoint-state evidence missing")
    if freeze.get("production_config_frozen") is not True or freeze.get("performance_used_for_gate_decision") is not False:
        raise VerificationError("freeze contract invalid")
    if freeze.get("smoke_validation") != smoke or freeze.get("gate_validation") != gate:
        raise VerificationError("technical evidence differs from frozen evidence")
    smoke_hashes = validate_fingerprint(smoke.get("source_fingerprint"), "smoke")
    gate_hashes = validate_fingerprint(gate.get("source_fingerprint"), "gate")
    freeze_hashes = validate_fingerprint(freeze.get("source_fingerprint"), "freeze")
    if not (
        smoke_hashes
        == gate_hashes
        == freeze_hashes
        == freeze.get("frozen_files_sha256")
    ):
        raise VerificationError("technical source fingerprint chain mismatch")
    for payload in (smoke, gate):
        diagnostics = payload["runs"][0]["assignment_diagnostics"]
        for field in ("exact_k_mismatch_count", "raw_histogram_mismatch_count", "normalized_histogram_mismatch_count", "invariant_failure_count"):
            if diagnostics.get(field) != 0:
                raise VerificationError(f"technical gate {field} nonzero")


def verify_execution(root: Path, manifest: Mapping[str, Any]) -> None:
    metadata = read_json(root / "code/execution_metadata.json")
    expected = {
        "experiment": "ddpg_saber_random_support_count_matched_ls512",
        "source_commit": manifest["execution_source_commit"],
        "origin_main": manifest["execution_source_commit"],
        "head_matches_origin_main": True, "worktree_clean": True,
        "support_query_mode": "current_policy",
        "support_assignment_mode": "batch_wide_count_matched",
        "assignment_seed_rule": "30000+seed", "steps": TOTAL_STEPS,
        "seeds": list(SEEDS), "weight_scope": "critic_only",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise VerificationError(f"execution metadata {key} mismatch")
    for key in (
        "source_tree_git_oid",
        "config_git_blob_oid",
        "freeze_manifest_git_blob_oid",
    ):
        if not isinstance(metadata.get(key), str) or not HEX40.fullmatch(metadata[key]):
            raise VerificationError(f"execution metadata {key} invalid")
    config = read_json(root / "code/execution_source/config.json")
    for key, value in {
        "run_slug": "ddpg_saber_random_support_count_matched_ls512",
        "support_query_mode": "current_policy",
        "support_assignment_mode": "batch_wide_count_matched",
        "assignment_seed_offset": 30_000,
        "assignment_diagnostic_window_updates": 1_000,
        "seeds": list(SEEDS), "total_steps": TOTAL_STEPS,
    }.items():
        if config.get(key) != value:
            raise VerificationError(f"frozen config {key} mismatch")
    if git_blob_oid(root / "code/execution_source/config.json") != metadata["config_git_blob_oid"]:
        raise VerificationError("packaged config does not match recorded Git blob")
    if git_blob_oid(root / "code/execution_source/freeze_manifest.json") != metadata["freeze_manifest_git_blob_oid"]:
        raise VerificationError("packaged freeze manifest does not match recorded Git blob")
    freeze = read_json(root / "technical_gate/freeze_manifest.json")
    frozen_hashes = freeze.get("frozen_files_sha256")
    if not isinstance(frozen_hashes, Mapping) or not frozen_hashes:
        raise VerificationError("technical freeze has no frozen source hashes")
    required_frozen = {
        "config.json",
        "run.py",
        "source_fingerprint.py",
        "test_source_fingerprint.py",
        "code/src/metacog_lite_v0/weights.py",
        "code/src/metacog_lite_v0/trainer.py",
        "code/src/metacog_lite_v0/numba_support_scan_fpp.py",
        "code/src/test_batch_wide_count_matched.py",
        "code/src/test_trainer_resume_contract.py",
        "code/src/test_resume_output_rollback.py",
        "code/src/weight_parity_fixture.py",
    }
    if not required_frozen <= set(frozen_hashes):
        raise VerificationError(
            f"technical freeze misses required source: {sorted(required_frozen - set(frozen_hashes))}"
        )
    for relative, expected_hash in frozen_hashes.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise VerificationError("invalid technical freeze hash entry")
        if relative.startswith("analysis/"):
            name = Path(relative).name
            frozen_path = root / ("tests" if name.startswith("test_") else "code") / name
        else:
            frozen_path = root / "code/execution_source" / relative
        if not frozen_path.is_file() or sha256(frozen_path) != expected_hash:
            raise VerificationError(f"frozen source hash mismatch: {relative}")


def verify_tests(root: Path, allow_synthetic: bool) -> None:
    evidence = read_json(root / "tests/test_evidence.json")
    status = evidence.get("status")
    if status == "synthetic-test-bypass" and allow_synthetic:
        return
    if status != "passed" or not evidence.get("commands") or not all(item.get("passed") for item in evidence["commands"]):
        raise VerificationError("prepackage tests did not all pass")


def verify_validation(root: Path, bundle: Any, manifest: Mapping[str, Any]) -> None:
    payload = read_json(root / "results/validation.json")
    if payload.get("status") != "passed" or payload.get("artifact_id") != ARTIFACT_ID:
        raise VerificationError("result validation identity/status mismatch")
    if payload.get("execution_source_commit") != manifest["execution_source_commit"] or payload.get("canonical_main_source_commit") != manifest["canonical_main_source_commit"]:
        raise VerificationError("result validation provenance mismatch")
    if payload.get("seed_coverage") != {MAIN_KEY: list(SEEDS), CONTROL_KEY: list(SEEDS)}:
        raise VerificationError("validation seed coverage mismatch")
    if payload.get("completion") != {"steps": TOTAL_STEPS, "updates": TOTAL_UPDATES, "passed": True}:
        raise VerificationError("validation completion mismatch")
    expected_diffs = {
        str(seed): actual_config_differences(
            bundle.runs[MAIN_KEY][seed - 1].config,
            bundle.runs[CONTROL_KEY][seed - 1].config,
        )
        for seed in SEEDS
    }
    if payload.get("config_differences_by_seed") != expected_diffs:
        raise VerificationError("validation config differences mismatch")
    expected_bookkeeping = {
        str(run.seed): {
            "assignment_seed": run.config["seeds"]["assignment_seed"],
            "assignment_diagnostic_window_updates": run.config[
                "assignment_diagnostic_window_updates"
            ],
        }
        for run in bundle.runs[CONTROL_KEY]
    }
    if payload.get("control_bookkeeping_by_seed") != expected_bookkeeping:
        raise VerificationError("validation control bookkeeping mismatch")
    invariant_sources = {
        "exact_k_mismatch": "exact_k_mismatch_count",
        "raw_histogram_mismatch": "raw_histogram_mismatch_count",
        "normalized_histogram_mismatch": "normalized_histogram_mismatch_count",
        "invariant_failure": "invariant_failure_count",
    }
    expected_invariants = {
        output: int(
            sum((run.diagnostics or {})[source] for run in bundle.runs[CONTROL_KEY])
        )
        for output, source in invariant_sources.items()
    }
    expected_invariants["passed"] = all(
        value == 0 for value in expected_invariants.values()
    )
    invariants = payload.get("assignment_invariants", {})
    if invariants != expected_invariants or invariants.get("passed") is not True:
        raise VerificationError("validation assignment invariant mismatch")
    smoke = read_json(root / "technical_gate/smoke_validation.json")
    gate = read_json(root / "technical_gate/gate_validation.json")
    expected_technical = {
        "smoke_passed": smoke["passed"],
        "smoke_resume_exercised": smoke["runs"][0]["log"]["resume_exercised"],
        "gate_passed": gate["passed"],
        "gate_resume_exercised": gate["runs"][0]["log"]["resume_exercised"],
    }
    if payload.get("technical_gate") != expected_technical:
        raise VerificationError("validation technical-gate summary mismatch")
    completion = read_json(root / "results/completion_validation.json")
    if completion.get("kind") != "production" or completion.get("passed") is not True or completion.get("seeds") != list(SEEDS):
        raise VerificationError("packaged production completion evidence invalid")
    production_hashes = validate_fingerprint(
        completion.get("source_fingerprint"), "production"
    )
    freeze = read_json(root / "technical_gate/freeze_manifest.json")
    if production_hashes != freeze.get("frozen_files_sha256"):
        raise VerificationError("production source differs from technical freeze")


def verify_artifact(root: Path, *, allow_synthetic_test_bypass: bool = False) -> dict[str, Any]:
    root = root.resolve()
    verify_layout(root)
    manifest = verify_manifest(root)
    verify_sums(root)
    verify_input_manifest(root, manifest)
    verify_technical_gate(root)
    verify_execution(root, manifest)
    verify_tests(root, allow_synthetic_test_bypass)
    bundle = analyze(root / "data/main", root / "data/control")
    verify_outputs(bundle, root / "results", frozenset({"validation.json", "completion_validation.json"}))
    verify_validation(root, bundle, manifest)
    return {
        "status": "verified", "artifact_id": ARTIFACT_ID,
        "execution_source_commit": manifest["execution_source_commit"],
        "canonical_main_source_commit": manifest["canonical_main_source_commit"],
        "seed_coverage": {MAIN_KEY: list(SEEDS), CONTROL_KEY: list(SEEDS)},
        "exact_grid_points": 100, "assignment_diagnostics": "verified",
        "technical_gate": "verified", "sha256_manifests": "verified",
        "file_count": manifest["file_count"], "total_size_bytes": manifest["total_size_bytes"],
    }


def main() -> int:
    default = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=default)
    args = parser.parse_args()
    try:
        result = verify_artifact(args.artifact_root)
    except (AnalysisError, OSError) as exc:
        print(f"artifact verification failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
