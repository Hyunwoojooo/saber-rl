#!/usr/bin/env python3
"""Build the compact DDPG batch-wide Random-Support artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from random_support_analysis import (
    ARTIFACT_ID, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, CONTROL_KEY,
    GRID, MAIN_KEY, SCHEMA_VERSION, SEEDS, TOTAL_STEPS, TOTAL_UPDATES,
    AnalysisError, RunResult, actual_config_differences, analyze,
    find_repo_root, read_json, write_outputs,
)


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXPERIMENT_REL = Path("experiment_FPP/ddpg_saber_random_support_count_matched_ls512")
CANONICAL_REL = Path("experiment_FPP/b3_critic_only_runtime_opt_v4_ls512")
ARTIFACT_REL = Path("experiment_FPP/paper_artifacts") / ARTIFACT_ID
ANALYSIS_FILES = ("random_support_analysis.py", "analyze_results.py", "build_artifact.py", "verify_artifact.py")
TEST_FILES = ("test_analysis.py", "test_artifact.py")
FINGERPRINT_EXCLUSIONS = {
    "names": ["freeze_manifest.json"],
    "directory_names": ["__pycache__"],
    "suffixes": [".nbc", ".nbi", ".pyc"],
}


class BuildError(AnalysisError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise BuildError(f"{stage} source fingerprint is missing")
    expected = {
        "schema_version": 1,
        "algorithm": "sha256",
        "scope": "full_experiment_package",
        "stage": stage,
        "exclusions": FINGERPRINT_EXCLUSIONS,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise BuildError(f"{stage} source fingerprint {key} mismatch")
    if not isinstance(payload.get("created_utc"), str) or not payload["created_utc"]:
        raise BuildError(f"{stage} source fingerprint creation time missing")
    files = payload.get("files_sha256")
    if not isinstance(files, Mapping) or not files:
        raise BuildError(f"{stage} source fingerprint files missing")
    checked: dict[str, str] = {}
    for relative, digest in files.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise BuildError(f"{stage} source fingerprint entry invalid")
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
            raise BuildError(f"{stage} source fingerprint entry unsafe: {relative!r}")
        checked[relative] = digest
    if payload.get("file_count") != len(checked):
        raise BuildError(f"{stage} source fingerprint file count mismatch")
    if payload.get("tree_sha256") != fingerprint_tree(checked):
        raise BuildError(f"{stage} source fingerprint tree mismatch")
    return checked


def validate_fingerprint_chain(
    freeze: Mapping[str, Any], completion: Mapping[str, Any]
) -> dict[str, str]:
    smoke = freeze.get("smoke_validation")
    gate = freeze.get("gate_validation")
    if not isinstance(smoke, Mapping) or not isinstance(gate, Mapping):
        raise BuildError("freeze manifest lacks technical validation evidence")
    smoke_hashes = validate_fingerprint(smoke.get("source_fingerprint"), "smoke")
    gate_hashes = validate_fingerprint(gate.get("source_fingerprint"), "gate")
    freeze_hashes = validate_fingerprint(freeze.get("source_fingerprint"), "freeze")
    production_hashes = validate_fingerprint(
        completion.get("source_fingerprint"), "production"
    )
    raw_frozen = freeze.get("frozen_files_sha256")
    if not isinstance(raw_frozen, Mapping):
        raise BuildError("freeze manifest frozen file mapping missing")
    if not (
        smoke_hashes
        == gate_hashes
        == freeze_hashes
        == production_hashes
        == dict(raw_frozen)
    ):
        raise BuildError("smoke/gate/freeze/production source fingerprints differ")
    return freeze_hashes


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True)
    if result.returncode:
        raise BuildError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def source_commit_for_path(repo: Path, relative: Path) -> str:
    value = git(repo, "log", "-1", "--format=%H", "--", relative.as_posix())
    if not HEX40.fullmatch(value):
        raise BuildError(f"could not resolve source commit for {relative}")
    return value


def logical(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def sanitize(value: Any, replacements: Sequence[tuple[str, str]], paths: list[str], pointer: str = "$") -> Any:
    if isinstance(value, Mapping):
        return {str(key): sanitize(item, replacements, paths, f"{pointer}/{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, replacements, paths, f"{pointer}/{index}") for index, item in enumerate(value)]
    if isinstance(value, float) and not math.isfinite(value):
        paths.append(pointer)
        return None
    if isinstance(value, str):
        for before, after in replacements:
            if before:
                value = value.replace(before, after)
    return value


def copy_json(source: Path, destination: Path, repo: Path, run_dir: Path | None = None) -> dict[str, Any]:
    payload = read_json(
        source,
        allow_optional_summary_nonfinite=source.name == "summary.json",
    )
    paths: list[str] = []
    replacements = [
        ((run_dir or source.parent).resolve().as_posix(), "${SOURCE_RUN}"),
        (repo.resolve().as_posix(), "${REPO_ROOT}"),
        (Path.home().resolve().as_posix(), "${HOME}"),
        (socket.gethostname(), "${HOST}"),
    ]
    write_json(destination, sanitize(payload, replacements, paths))
    return {"machine_specific_strings_sanitized": True, "nonfinite_json_numbers_replaced_with_null": len(paths), "nonfinite_json_paths": sorted(paths)}


def sanitized_payload(
    payload: Any, repo: Path, run_dir: Path | None = None
) -> Any:
    """Sanitize generated/embedded evidence, not only copied JSON files."""
    replacements = [
        ((run_dir or repo).resolve().as_posix(), "${SOURCE_ROOT}"),
        (repo.resolve().as_posix(), "${REPO_ROOT}"),
        (Path.home().resolve().as_posix(), "${HOME}"),
        (socket.gethostname(), "${HOST}"),
    ]
    paths: list[str] = []
    return sanitize(payload, replacements, paths)


def record(source: Path, repo: Path, role: str, packaged: str, transformations: Mapping[str, Any] | None = None) -> dict[str, Any]:
    output = {"role": role, "source_path": logical(source, repo), "source_sha256": sha256(source), "source_size_bytes": source.stat().st_size, "packaged_path": packaged}
    if transformations is not None:
        output["transformations"] = dict(transformations)
    return output


def copy_metrics(run: RunResult, destination: Path) -> dict[str, Any]:
    fields = list(run.metrics_rows[0])
    if "step" not in fields or not ({"eval/env_success_mean", "eval/success_mean"} & set(fields)):
        raise BuildError(f"{run.metrics_path}: missing compact metric core")
    destination.parent.mkdir(parents=True, exist_ok=True)
    nonfinite = 0
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in run.metrics_rows:
            output = dict(row)
            for key, value in output.items():
                if value not in (None, ""):
                    try:
                        parsed = float(value)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(parsed):
                        output[key] = ""
                        nonfinite += 1
            writer.writerow(output)
    return {"exact_evaluation_rows_selected": len(run.metrics_rows), "output_columns": fields, "optional_nonfinite_cells_blank": nonfinite}


def package_run(run: RunResult, destination: Path, repo: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    files: list[dict[str, Any]] = []
    rel_base = destination.relative_to(destination.parents[2])
    for name in ("config.json", "env_snapshot.json", "summary.json"):
        source = run.run_dir / name
        transform = copy_json(source, destination / name, repo, run.run_dir)
        files.append(record(source, repo, name.removesuffix(".json"), (rel_base / name).as_posix(), transform))
    metrics = destination / "metrics_selected_10k.csv"
    transform = copy_metrics(run, metrics)
    files.append(record(run.metrics_path, repo, "metrics", (rel_base / metrics.name).as_posix(), transform))
    if run.method_key == CONTROL_KEY:
        assert run.diagnostics_path is not None
        target = destination / "assignment_diagnostics_1000.csv"
        shutil.copy2(run.diagnostics_path, target)
        files.append(record(run.diagnostics_path, repo, "assignment_diagnostics", (rel_base / target.name).as_posix(), {"copied_byte_for_byte": True}))
    expected = {"config.json", "env_snapshot.json", "summary.json", "metrics_selected_10k.csv"}
    if run.method_key == CONTROL_KEY:
        expected.add("assignment_diagnostics_1000.csv")
    actual = {path.name for path in destination.iterdir() if path.is_file()}
    if actual != expected:
        raise BuildError(f"{destination}: payload mismatch {actual}")
    return {
        "seed": run.seed,
        "source_run_dir": logical(run.run_dir, repo),
        "packaged_run_dir": rel_base.as_posix(),
        "files": files,
    }


def frozen_source_files(root: Path) -> list[Path]:
    output: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if "analysis" in relative.parts or "__pycache__" in relative.parts or path.suffix in {".pyc", ".pt", ".pth", ".ckpt"}:
            continue
        output.append(path)
    required = {root / "config.json", root / "run.py", root / "freeze_manifest.json", root / "code/src/metacog_lite_v0/weights.py"}
    if not required <= set(output):
        raise BuildError(f"frozen source missing {required - set(output)}")
    return output


def copy_frozen_source(root: Path, destination: Path, repo: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in frozen_source_files(root):
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(record(source, repo, "execution_source", (Path("code/execution_source") / relative).as_posix()))
    return records


def copy_tools(analysis_root: Path, stage: Path, repo: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in ANALYSIS_FILES:
        source = analysis_root / name
        if not source.is_file():
            raise BuildError(f"missing analysis tool {source}")
        target = stage / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(record(source, repo, "analysis_code", f"code/{name}"))
    for name in TEST_FILES:
        source = analysis_root / name
        if not source.is_file():
            raise BuildError(f"missing test {source}")
        target = stage / "tests" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(record(source, repo, "artifact_test", f"tests/{name}"))
    return records


def validate_execution_metadata(control_root: Path, source_commit: str) -> tuple[Path, dict[str, Any]]:
    path = control_root / "_execution/launch_metadata.json"
    payload = read_json(path)
    expected = {
        "experiment": "ddpg_saber_random_support_count_matched_ls512",
        "source_commit": source_commit, "origin_main": source_commit,
        "head_matches_origin_main": True, "worktree_clean": True,
        "support_query_mode": "current_policy",
        "support_assignment_mode": "batch_wide_count_matched",
        "assignment_seed_rule": "30000+seed", "weight_scope": "critic_only",
        "steps": TOTAL_STEPS, "seeds": list(SEEDS),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise BuildError(f"{path}: {key} expected {value!r}, got {payload.get(key)!r}")
    for key in (
        "source_tree_git_oid",
        "config_git_blob_oid",
        "freeze_manifest_git_blob_oid",
    ):
        if not isinstance(payload.get(key), str) or not HEX40.fullmatch(payload[key]):
            raise BuildError(f"{path}: invalid/missing {key}")
    return path, payload


def resolve_source_commit(repo: Path, control_root: Path, supplied: str | None) -> tuple[str, Path, dict[str, Any]]:
    metadata_path = control_root / "_execution/launch_metadata.json"
    metadata = read_json(metadata_path)
    commit = supplied or metadata.get("source_commit")
    if not isinstance(commit, str) or not HEX40.fullmatch(commit):
        raise BuildError("execution source commit must be a full SHA")
    head = git(repo, "rev-parse", "HEAD")
    if head != commit:
        raise BuildError(f"current HEAD {head} differs from execution source {commit}")
    tracked = git(repo, "ls-tree", "-r", "--name-only", commit, "--", EXPERIMENT_REL.as_posix())
    if not tracked:
        raise BuildError(f"experiment source is absent from {commit}")
    dirty = git(repo, "status", "--porcelain", "--", EXPERIMENT_REL.as_posix())
    if dirty:
        raise BuildError(f"experiment source has uncommitted changes:\n{dirty}")
    path, checked = validate_execution_metadata(control_root, commit)
    expected_oids = {
        "source_tree_git_oid": git(
            repo, "rev-parse", f"{commit}:{EXPERIMENT_REL.as_posix()}/code/src"
        ),
        "config_git_blob_oid": git(
            repo, "rev-parse", f"{commit}:{EXPERIMENT_REL.as_posix()}/config.json"
        ),
        "freeze_manifest_git_blob_oid": git(
            repo,
            "rev-parse",
            f"{commit}:{EXPERIMENT_REL.as_posix()}/freeze_manifest.json",
        ),
    }
    for key, value in expected_oids.items():
        if checked.get(key) != value:
            raise BuildError(
                f"execution metadata {key}={checked.get(key)!r} != commit object {value}"
            )
    return commit, path, checked


def verify_freeze_against_commit(
    repo: Path, experiment_root: Path, source_commit: str
) -> dict[str, str]:
    """Prove every frozen byte is tracked in, and identical to, the run commit.

    This check deliberately covers the full freeze manifest rather than a
    hand-picked source list. In particular, it fails closed if an ignored
    analysis file was not force-added before the execution-source commit.
    """
    try:
        relative_root = experiment_root.resolve().relative_to(repo.resolve())
    except ValueError as exc:
        raise BuildError("experiment source must be inside the repository") from exc
    if relative_root != EXPERIMENT_REL:
        raise BuildError(
            f"unexpected experiment source path {relative_root}; expected {EXPERIMENT_REL}"
        )
    freeze_path = experiment_root / "freeze_manifest.json"
    freeze = read_json(freeze_path)
    hashes = freeze.get("frozen_files_sha256")
    if not isinstance(hashes, Mapping) or not hashes:
        raise BuildError(f"{freeze_path}: missing frozen_files_sha256")
    verified: dict[str, str] = {}
    for raw_relative, expected_hash in sorted(hashes.items()):
        if not isinstance(raw_relative, str) or not raw_relative:
            raise BuildError(f"{freeze_path}: invalid frozen path {raw_relative!r}")
        relative = Path(raw_relative)
        if (
            relative.is_absolute()
            or relative.as_posix() != raw_relative
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise BuildError(f"{freeze_path}: unsafe frozen path {raw_relative!r}")
        if not isinstance(expected_hash, str) or not HEX64.fullmatch(expected_hash):
            raise BuildError(
                f"{freeze_path}: invalid SHA256 for frozen path {raw_relative!r}"
            )
        worktree_path = experiment_root / relative
        if not worktree_path.is_file() or sha256(worktree_path) != expected_hash:
            raise BuildError(f"worktree differs from freeze: {raw_relative}")
        object_spec = f"{source_commit}:{(relative_root / relative).as_posix()}"
        exists = subprocess.run(
            ["git", "cat-file", "-e", object_spec],
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if exists.returncode:
            detail = exists.stderr.decode("utf-8", errors="replace").strip()
            raise BuildError(
                f"frozen path is not tracked in execution commit: {raw_relative}"
                + (f" ({detail})" if detail else "")
            )
        blob = subprocess.run(
            ["git", "show", object_spec],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if blob.returncode:
            detail = blob.stderr.decode("utf-8", errors="replace").strip()
            raise BuildError(
                f"cannot read frozen path from execution commit: {raw_relative}"
                + (f" ({detail})" if detail else "")
            )
        committed_hash = hashlib.sha256(blob.stdout).hexdigest()
        if committed_hash != expected_hash:
            raise BuildError(
                f"execution commit bytes differ from freeze: {raw_relative}"
            )
        verified[raw_relative] = committed_hash
    return verified


def prepackage_tests(experiment: Path, analysis_root: Path, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"schema_version": 1, "status": "synthetic-test-bypass", "commands": []}
    paths = [
        analysis_root / "test_analysis.py", analysis_root / "test_artifact.py",
        experiment / "validate_config_parity.py",
        experiment / "test_validation_optional_nan.py",
        experiment / "test_source_fingerprint.py",
        experiment / "code/src/test_batch_wide_count_matched.py",
        experiment / "code/src/test_ddpg_random_support_loss_scope.py",
        experiment / "code/src/test_trainer_resume_contract.py",
        experiment / "code/src/test_resume_output_rollback.py",
        experiment / "code/src/test_support_query_modes.py",
        experiment / "code/src/test_ddpg_query_loss_isolation.py",
        experiment / "code/src/test_current_policy_legacy_parity.py",
        experiment / "code/src/test_b3_runtime_opt_lookup.py",
        experiment / "code/src/test_numba_runtime_opt_fpp.py",
        experiment / "code/src/test_numba_support_scan_fpp.py",
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise BuildError(f"prepackage tests missing: {missing}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(experiment / "code/src") + os.pathsep + env.get("PYTHONPATH", "")
    records: list[dict[str, Any]] = []
    for path in paths:
        result = subprocess.run([sys.executable, "-B", str(path)], cwd=experiment.parents[1], env=env, text=True, capture_output=True)
        item = {"test": path.name, "returncode": result.returncode, "passed": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr}
        records.append(item)
        if result.returncode:
            raise BuildError(f"prepackage test failed: {path}\n{result.stdout}\n{result.stderr}")
    return {"schema_version": 1, "status": "passed", "commands": records}


def artifact_readme(source_commit: str, canonical_commit: str, bundle: Any) -> str:
    inference = bundle.inference_rows[0]
    return f"""# FPP DDPG SABER Random-Support (Batch-Wide Count Matched) (1M)

Compact seeds 1--5 artifact for Main versus Random-Support (Batch-Wide Count Matched).

- execution source: `{source_commit}`
- canonical Main source: `{canonical_commit}`
- factor: `support_assignment_mode: support -> batch_wide_count_matched`
- query remains `current_policy`; actor weights remain one; critic-only scope
- each update samples exactly Main's positive count uniformly without replacement
  from the full minibatch, including Main-positive and context-ineligible positions
- for binary SABER weights this control is distributionally equivalent to uniformly
  shuffled-weight; the two must not be interpreted as independent mechanisms

Primary normalized success AUC uses only exact observed 10k--1M evaluations,
trapezoidal integration, and denominator 990,000. No smoothing, interpolation,
extrapolation, or forward filling is used. Main minus Random-Support mean AUC is
`{inference['mean_delta']:+.6f}` with paired-percentile 95% CI
`[{inference['bootstrap_ci95_low']:.6f}, {inference['bootstrap_ci95_high']:.6f}]`
(10,000 resamples, seed 27027); exact sign-flip enumerates 32 assignments.

`data/control` includes every seed's complete non-overlapping 1,000-update
assignment diagnostics. `technical_gate` preserves validated 5k smoke/resume,
50k gate, and source-freeze evidence. Checkpoints, full metrics, JSONL, raw logs,
and raw smoke/gate directories are excluded.

Verify standalone: `python3 code/verify_artifact.py --artifact-root .`
"""


def all_files(root: Path) -> dict[str, Path]:
    return {path.relative_to(root).as_posix(): path for path in sorted(root.rglob("*")) if path.is_file()}


def write_manifests(root: Path, source_commit: str, canonical_commit: str) -> None:
    excluded = {"artifact_manifest.json", "SHA256SUMS"}
    files = {name: path for name, path in all_files(root).items() if name not in excluded}
    entries = [{"path": name, "sha256": sha256(path), "size_bytes": path.stat().st_size} for name, path in files.items()]
    write_json(root / "artifact_manifest.json", {
        "schema_version": SCHEMA_VERSION, "artifact_id": ARTIFACT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_source_commit": source_commit, "canonical_main_source_commit": canonical_commit,
        "method_seed_coverage": {MAIN_KEY: list(SEEDS), CONTROL_KEY: list(SEEDS)},
        "evaluation": {"steps": list(GRID), "checkpoint_count": len(GRID), "auc_normalization_denominator": 990_000},
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "rng_seed": BOOTSTRAP_SEED},
        "exact_sign_flip_assignments": 32, "manifest_self_excluded": True,
        "sha256sums_self_excluded": True, "file_count": len(entries),
        "total_size_bytes": sum(item["size_bytes"] for item in entries), "files": entries,
    })
    sums = {name: path for name, path in all_files(root).items() if name != "SHA256SUMS"}
    (root / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {name}\n" for name, path in sums.items()), encoding="utf-8", newline="\n")


def build_artifact(
    *, repo: Path, main_root: Path, control_root: Path, artifact_root: Path,
    experiment_root: Path, analysis_root: Path, execution_source_commit: str,
    canonical_source_commit: str, execution_metadata_path: Path,
    execution_metadata: Mapping[str, Any], replace: bool = False,
    run_tests: bool = True, allow_synthetic_verification: bool = False,
) -> dict[str, Any]:
    if not HEX40.fullmatch(execution_source_commit):
        raise BuildError("execution_source_commit must be a full SHA")
    if not HEX40.fullmatch(canonical_source_commit):
        raise BuildError("canonical_source_commit must be a full SHA")
    checked_metadata_path, checked_metadata = validate_execution_metadata(
        control_root, execution_source_commit
    )
    if checked_metadata_path.resolve() != execution_metadata_path.resolve():
        raise BuildError("execution metadata path does not match the registered control root")
    if dict(execution_metadata) != checked_metadata:
        raise BuildError("supplied execution metadata differs from the source file")
    if not allow_synthetic_verification:
        verify_freeze_against_commit(
            repo, experiment_root, execution_source_commit
        )
    bundle = analyze(main_root, control_root)
    if artifact_root.exists() and not replace:
        raise BuildError(f"artifact exists: {artifact_root}; use --replace")
    test_evidence = prepackage_tests(experiment_root, analysis_root, run_tests)
    freeze_path = experiment_root / "freeze_manifest.json"
    freeze = read_json(freeze_path)
    if freeze.get("production_config_frozen") is not True:
        raise BuildError("invalid source freeze manifest")
    completion_path = control_root / "_execution/completion_validation.json"
    completion = read_json(completion_path)
    if completion.get("kind") != "production" or completion.get("passed") is not True:
        raise BuildError("production completion validation did not pass")
    validate_fingerprint_chain(freeze, completion)
    artifact_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{ARTIFACT_ID}.tmp-", dir=artifact_root.parent))
    try:
        methods: dict[str, Any] = {}
        for key in (MAIN_KEY, CONTROL_KEY):
            runs = []
            for run in bundle.runs[key]:
                destination = stage / "data" / key / f"seed_{run.seed}"
                runs.append(package_run(run, destination, repo))
            methods[key] = {"support_query_mode": "current_policy", "support_assignment_mode": "support" if key == MAIN_KEY else "batch_wide_count_matched", "source_commit": canonical_source_commit if key == MAIN_KEY else execution_source_commit, "seeds": list(SEEDS), "runs": runs}
        packaged = analyze(stage / "data/main", stage / "data/control")
        write_outputs(packaged, stage / "results")
        tools = copy_tools(analysis_root, stage, repo)
        frozen = copy_frozen_source(experiment_root, stage / "code/execution_source", repo)
        metadata_transform = copy_json(execution_metadata_path, stage / "code/execution_metadata.json", repo, control_root)
        metadata_record = record(execution_metadata_path, repo, "execution_metadata", "code/execution_metadata.json", metadata_transform)
        write_json(
            stage / "tests/test_evidence.json",
            sanitized_payload(test_evidence, repo, experiment_root),
        )
        write_json(
            stage / "technical_gate/smoke_validation.json",
            sanitized_payload(freeze["smoke_validation"], repo, experiment_root),
        )
        write_json(
            stage / "technical_gate/gate_validation.json",
            sanitized_payload(freeze["gate_validation"], repo, experiment_root),
        )
        copy_json(freeze_path, stage / "technical_gate/freeze_manifest.json", repo, experiment_root)
        copy_json(completion_path, stage / "results/completion_validation.json", repo, control_root)
        (stage / "source_commit.txt").write_text(execution_source_commit + "\n", encoding="utf-8")
        input_manifest = {
            "schema_version": SCHEMA_VERSION, "artifact_id": ARTIFACT_ID,
            "repository": {"id": "Hyunwoojooo/saber-rl", "branch": "main"},
            "execution_source_commit": execution_source_commit,
            "canonical_main_source_commit": canonical_source_commit,
            "methods": methods, "execution_metadata": metadata_record,
            "frozen_execution_source_files": frozen, "tool_and_test_files": tools,
            "technical_gate": {"smoke": logical(freeze_path, repo) + "#smoke_validation", "gate": logical(freeze_path, repo) + "#gate_validation"},
            "excluded": {"checkpoints": True, "full_per_update_metrics": True, "jsonl": True, "raw_logs": True, "raw_smoke_gate_directories": True},
        }
        write_json(stage / "input_manifest.json", input_manifest)
        actual_diffs = {
            str(seed): actual_config_differences(
                packaged.runs[MAIN_KEY][seed - 1].config,
                packaged.runs[CONTROL_KEY][seed - 1].config,
            )
            for seed in SEEDS
        }
        bookkeeping = {
            str(run.seed): {
                "assignment_seed": run.config["seeds"]["assignment_seed"],
                "assignment_diagnostic_window_updates": run.config[
                    "assignment_diagnostic_window_updates"
                ],
            }
            for run in packaged.runs[CONTROL_KEY]
        }
        invariant_fields = {
            "exact_k_mismatch": "exact_k_mismatch_count",
            "raw_histogram_mismatch": "raw_histogram_mismatch_count",
            "normalized_histogram_mismatch": "normalized_histogram_mismatch_count",
            "invariant_failure": "invariant_failure_count",
        }
        invariant_totals = {
            output: int(
                sum(
                    (run.diagnostics or {})[source]
                    for run in packaged.runs[CONTROL_KEY]
                )
            )
            for output, source in invariant_fields.items()
        }
        write_json(stage / "results/validation.json", {
            "schema_version": SCHEMA_VERSION, "artifact_id": ARTIFACT_ID, "status": "passed",
            "execution_source_commit": execution_source_commit,
            "canonical_main_source_commit": canonical_source_commit,
            "seed_coverage": {MAIN_KEY: list(SEEDS), CONTROL_KEY: list(SEEDS)},
            "completion": {"steps": TOTAL_STEPS, "updates": TOTAL_UPDATES, "passed": True},
            "exact_evaluation_grid": {"steps": list(GRID), "points": 100, "no_filling": True, "passed": True},
            "config_differences_by_seed": actual_diffs,
            "control_bookkeeping_by_seed": bookkeeping,
            "assignment_invariants": {
                **invariant_totals,
                "passed": all(value == 0 for value in invariant_totals.values()),
            },
            "technical_gate": {
                "smoke_passed": freeze["smoke_validation"]["passed"],
                "smoke_resume_exercised": freeze["smoke_validation"]["runs"][0]["log"]["resume_exercised"],
                "gate_passed": freeze["gate_validation"]["passed"],
                "gate_resume_exercised": freeze["gate_validation"]["runs"][0]["log"]["resume_exercised"],
            },
            "excluded_large_payloads": input_manifest["excluded"],
        })
        (stage / "README.md").write_text(artifact_readme(execution_source_commit, canonical_source_commit, packaged), encoding="utf-8", newline="\n")
        write_manifests(stage, execution_source_commit, canonical_source_commit)
        sys.path.insert(0, str(analysis_root))
        try:
            from verify_artifact import verify_artifact
            result = verify_artifact(stage, allow_synthetic_test_bypass=allow_synthetic_verification)
        finally:
            sys.path.pop(0)
        if artifact_root.exists():
            shutil.rmtree(artifact_root)
        os.replace(stage, artifact_root)
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    repo = find_repo_root(Path(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, default=repo / "experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512")
    parser.add_argument("--control-root", type=Path, default=repo / "experiment_FPP/runs/ddpg_saber_random_support_count_matched_ls512")
    parser.add_argument("--artifact-root", type=Path, default=repo / ARTIFACT_REL)
    parser.add_argument("--source-commit")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    try:
        commit, metadata_path, metadata = resolve_source_commit(repo, args.control_root, args.source_commit)
        canonical = source_commit_for_path(repo, CANONICAL_REL)
        result = build_artifact(
            repo=repo, main_root=args.main_root, control_root=args.control_root,
            artifact_root=args.artifact_root, experiment_root=repo / EXPERIMENT_REL,
            analysis_root=Path(__file__).resolve().parent,
            execution_source_commit=commit, canonical_source_commit=canonical,
            execution_metadata_path=metadata_path, execution_metadata=metadata,
            replace=args.replace,
        )
    except (AnalysisError, OSError, subprocess.SubprocessError) as exc:
        print(f"artifact build failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
