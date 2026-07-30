#!/usr/bin/env python3
"""Build a compact, validated artifact for DrawerOpen fixed-target diagnostics."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


SEEDS = (6, 7, 8, 9, 10)
EXPECTED_FINAL_STEP = 2_000_000
EXPECTED_FIRST_DIAGNOSTIC_STEP = 5_000
EXPECTED_DIAGNOSTIC_CADENCE = 200
EXPECTED_EVAL_STEPS = tuple(range(10_000, EXPECTED_FINAL_STEP + 1, 10_000))
EXPECTED_TOP10_COUNT = 26
EXECUTION_COMMIT = "22015c0a3743d4a10f72197752022ed0e08ee730"
RUN_STARTED_KST = "2026-07-28T19:52:39+09:00"
RUN_COMPLETED_KST = "2026-07-29T22:39:11+09:00"

SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT_PATH.parents[1]
DRAWER_ROOT = SCRIPT_PATH.parents[3]
REPO_ROOT = SCRIPT_PATH.parents[4]
DEFAULT_INPUT_ROOT = EXPERIMENT_ROOT / "results_2m"
DEFAULT_OUTPUT_ROOT = (
    DRAWER_ROOT
    / "paper_artifacts"
    / "drawer_open_ddpg_saber_fixed_target_diagnostic_s6_10_2m"
)

PATH_KEY_RE = re.compile(r"(?:^|_)(?:root|path|dir)$", re.IGNORECASE)
WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    if value is None or value == "":
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def number(value: Any) -> float:
    return float(value) if finite(value) else float("nan")


def is_path_key(key: str) -> bool:
    normalized = key.strip().lower()
    return normalized in {
        "checkpoint",
        "checkpoint_path",
        "input_root",
        "output_root",
        "resume_path",
        "run_dir",
        "run_path",
        "source_path",
    } or bool(PATH_KEY_RE.search(normalized))


def is_absolute_path_string(value: str) -> bool:
    return (
        value.startswith("~/")
        or Path(value).is_absolute()
        or bool(WINDOWS_ABSOLUTE_RE.match(value))
    )


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): sanitize(item)
            for key, item in value.items()
            if not is_path_key(str(key))
        }
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str) and is_absolute_path_string(value):
        return "<redacted-local-path>"
    return value


def absolute_strings(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            found.extend(absolute_strings(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(absolute_strings(item))
    elif isinstance(value, str) and is_absolute_path_string(value):
        found.append(value)
    return found


def locate_run(input_root: Path, seed: int) -> Path:
    matches: list[Path] = []
    for config_path in input_root.glob("*/config.json"):
        config = read_json(config_path)
        seeds = config.get("seeds", {})
        if int(seeds.get("env_seed", -1)) != seed:
            continue
        if config.get("task_name") != "drawer_open":
            continue
        if config.get("method") != "b3_support_runtime_opt":
            continue
        if config.get("weight_scope") != "critic_only":
            continue
        matches.append(config_path.parent)
    if len(matches) != 1:
        raise AssertionError(
            f"expected one DrawerOpen fixed-target run for seed {seed}, got {matches}"
        )
    return matches[0]


def validate_and_read_metrics(
    run_dir: Path, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics_path = run_dir / "metrics.csv"
    summary = read_json(run_dir / "summary.json")
    required_summary = {
        "total_steps": EXPECTED_FINAL_STEP,
        "fixed_target_residual_diagnostic": True,
        "live_target_residual_diagnostic": True,
        "top10_exact_k": True,
        "diagnostic_snapshot_only": True,
        "resume_supported": False,
    }
    for key, expected in required_summary.items():
        if summary.get(key) != expected:
            raise AssertionError(
                f"seed {seed}: summary[{key!r}]={summary.get(key)!r}, expected {expected!r}"
            )

    diagnostic_steps: set[int] = set()
    eval_rows: list[dict[str, Any]] = []
    required_columns = {
        "step",
        "eval/success_mean",
        "eval/return_mean",
        "diagnostic/fixed_target_from_pre_update",
        "diagnostic/same_minibatch_one_update",
        "diagnostic/snapshot_only_collection",
        "diagnostic/td_error_top10_count",
        "runtime/numba_support_scan_fallback_rate",
        "runtime/audit_psi_A_policy_mismatch_rate",
    }
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required_columns - set(reader.fieldnames or ()))
        if missing:
            raise AssertionError(f"seed {seed}: missing metrics columns: {missing}")
        for row in reader:
            step = int(float(row["step"]))
            if finite(row.get("eval/success_mean")):
                eval_rows.append(
                    {
                        "seed": seed,
                        "step": step,
                        "success": float(row["eval/success_mean"]),
                        "return_mean": number(row.get("eval/return_mean")),
                    }
                )
            if number(row.get("diagnostic/fixed_target_from_pre_update")) != 1.0:
                continue
            diagnostic_steps.add(step)
            exact_values = {
                "diagnostic/same_minibatch_one_update": 1.0,
                "diagnostic/snapshot_only_collection": 1.0,
                "diagnostic/td_error_top10_count": float(EXPECTED_TOP10_COUNT),
                "runtime/numba_support_scan_fallback_rate": 0.0,
                "runtime/audit_psi_A_policy_mismatch_rate": 0.0,
            }
            for key, expected in exact_values.items():
                observed = number(row.get(key))
                if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                    raise AssertionError(
                        f"seed {seed}: {key}={observed} at step {step}, expected {expected}"
                    )

    expected_diagnostic_steps = set(
        range(
            EXPECTED_FIRST_DIAGNOSTIC_STEP,
            EXPECTED_FINAL_STEP + 1,
            EXPECTED_DIAGNOSTIC_CADENCE,
        )
    )
    if diagnostic_steps != expected_diagnostic_steps:
        raise AssertionError(
            f"seed {seed}: diagnostic grid mismatch "
            f"({len(diagnostic_steps)} vs {len(expected_diagnostic_steps)})"
        )
    observed_eval_steps = tuple(row["step"] for row in eval_rows)
    if observed_eval_steps != EXPECTED_EVAL_STEPS:
        raise AssertionError(
            f"seed {seed}: eval grid mismatch "
            f"({len(observed_eval_steps)} vs {len(EXPECTED_EVAL_STEPS)})"
        )

    validation = {
        "seed": seed,
        "total_steps": summary["total_steps"],
        "diagnostic_snapshot_count": len(diagnostic_steps),
        "diagnostic_first_step": min(diagnostic_steps),
        "diagnostic_last_step": max(diagnostic_steps),
        "diagnostic_cadence": EXPECTED_DIAGNOSTIC_CADENCE,
        "eval_row_count": len(eval_rows),
        "exact_top10_count": EXPECTED_TOP10_COUNT,
        "fixed_target_from_pre_update": True,
        "same_minibatch_one_update": True,
        "numba_fallback_rate": 0.0,
        "audit_mismatch_rate": 0.0,
    }
    return eval_rows, validation


def write_eval_curve(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("seed", "step", "success", "return_mean"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def gzip_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_output, compresslevel=9, mtime=0
        ) as compressed:
            shutil.copyfileobj(source_handle, compressed, length=1024 * 1024)


def git_bytes(commit: str, relative_path: Path) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path.as_posix()}"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout


def build_source_manifest() -> dict[str, Any]:
    source_paths: list[Path] = []
    for path in sorted((EXPERIMENT_ROOT / "code").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            source_paths.append(path)
    source_paths.extend(
        [
            EXPERIMENT_ROOT / ".gitignore",
            EXPERIMENT_ROOT / "README.md",
            EXPERIMENT_ROOT / "requirements-runtime.txt",
            EXPERIMENT_ROOT / "run_5seeds_parallel.sh",
            EXPERIMENT_ROOT / "validate_completed_runs.py",
        ]
    )

    files: list[dict[str, Any]] = []
    for path in source_paths:
        relative = path.relative_to(REPO_ROOT)
        current = path.read_bytes()
        committed = git_bytes(EXECUTION_COMMIT, relative)
        if current != committed:
            raise AssertionError(
                f"execution source differs from {EXECUTION_COMMIT}: {relative}"
            )
        files.append(
            {
                "path": relative.as_posix(),
                "size_bytes": len(current),
                "sha256": sha256_bytes(current),
            }
        )
    return {
        "execution_commit": EXECUTION_COMMIT,
        "all_execution_files_match_commit": True,
        "files": files,
    }


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment(output_root: Path) -> dict[str, Any]:
    import torch

    os_release: dict[str, str] = {}
    os_release_path = Path("/etc/os-release")
    if os_release_path.is_file():
        for raw_line in os_release_path.read_text(encoding="utf-8").splitlines():
            if "=" not in raw_line:
                continue
            key, value = raw_line.split("=", 1)
            os_release[key] = value.strip().strip('"')

    driver_version = None
    try:
        driver_version = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass

    gpu: dict[str, Any] = {"count": torch.cuda.device_count()}
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu.update(
            {
                "model": torch.cuda.get_device_name(0),
                "cuda_capability": list(torch.cuda.get_device_capability(0)),
                "cuda_addressable_memory_bytes": properties.total_memory,
                "memory_architecture_note": (
                    "GB10 uses unified memory; this is CUDA-addressable memory, "
                    "not separately reported discrete VRAM."
                ),
            }
        )

    launch_timestamp = datetime.fromisoformat(RUN_STARTED_KST).timestamp()
    site_packages = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    changed_metadata = sorted(
        path.name
        for path in site_packages.glob("*.dist-info")
        if path.stat().st_mtime > launch_timestamp
    )

    pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    (output_root / "provenance" / "pip_freeze.txt").write_text(
        pip_freeze, encoding="utf-8"
    )

    packages = {
        name: package_version(name)
        for name in (
            "gymnasium",
            "gymnasium-robotics",
            "mujoco",
            "metaworld",
            "numpy",
            "scipy",
            "numba",
            "llvmlite",
            "torch",
        )
    }
    return {
        "collection_status": (
            "Post-run collection from the same physical machine and virtual "
            "environment used by the runs."
        ),
        "collection_caveat": (
            "The software snapshot was collected after completion rather than "
            "embedded in each process at launch."
        ),
        "run_started_kst": RUN_STARTED_KST,
        "run_completed_kst": RUN_COMPLETED_KST,
        "site_package_metadata_modified_after_launch": changed_metadata,
        "site_package_metadata_unchanged_after_launch": not changed_metadata,
        "os": {
            "name": os_release.get("PRETTY_NAME"),
            "kernel": platform.release(),
            "architecture": platform.machine(),
        },
        "cpu": {
            "logical_count": os.cpu_count(),
            "platform_processor": platform.processor(),
        },
        "memory": {
            "system_total_bytes": os.sysconf("SC_PAGE_SIZE")
            * os.sysconf("SC_PHYS_PAGES"),
        },
        "gpu": gpu,
        "nvidia_driver": driver_version,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "packages": packages,
    }


def performance_row(seed: int, summary: dict[str, Any]) -> dict[str, Any]:
    raw_hit90 = summary.get("first_hit_0.90")
    hit90 = int(raw_hit90) if finite(raw_hit90) else None
    return {
        "seed": seed,
        "auc": float(summary["eval_AUC_success"]),
        "final_success": float(summary["final_eval_success"]),
        "best_success": float(summary["best_eval_success"]),
        "hit90": hit90,
        "steps_per_sec": float(summary["final_steps_per_sec"]),
        "runtime_hours": float(summary["runtime_total_sec"]) / 3600.0,
    }


def aggregate_performance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    auc = [row["auc"] for row in rows]
    final = [row["final_success"] for row in rows]
    hit90 = [row["hit90"] for row in rows if row["hit90"] is not None]
    return {
        "seed_count": len(rows),
        "seeds": [row["seed"] for row in rows],
        "auc_mean": statistics.mean(auc),
        "auc_sample_sd": statistics.stdev(auc),
        "final_success_mean": statistics.mean(final),
        "final_success_sample_sd": statistics.stdev(final),
        "hit90_coverage": len(hit90),
        "hit90_median_reached_seeds": statistics.median(hit90) if hit90 else None,
        "per_seed": rows,
    }


def write_readme(output_root: Path, aggregate: dict[str, Any]) -> None:
    text = f"""# DrawerOpen DDPG SABER Fixed-Target Diagnostic, Seeds 6-10

This artifact contains the completed 2M-step fixed-target diagnostic runs for
seeds 6-10. It is a separate diagnostic cohort and must not be presented as a
paired HER comparison.

## Scope

- Task: Meta-World `drawer-open-v3`, sparse reward function v2
- Backbone: DDPG + future-goal HER
- Method: SABER Critic-Only
- Seeds: 6-10
- Budget: 2,000,000 environment steps per seed
- Execution source: `{EXECUTION_COMMIT}`
- Completed: {RUN_COMPLETED_KST}

## Packaged data

- `data/seeds/seed_XX/metrics.csv.gz`: complete original metrics CSV,
  deterministically compressed; checkpoints and duplicate JSONL are excluded.
- `data/seeds/seed_XX/config.json`: sanitized run configuration.
- `data/seeds/seed_XX/env_snapshot.json`: sanitized environment definition.
- `data/seeds/seed_XX/summary.json`: sanitized completed-run summary.
- `analysis/eval_curves.csv`: exact 10k-step evaluation grid.
- `analysis/performance_summary.json`: per-seed and aggregate performance.
- `validation/validation_summary.json`: completion and diagnostic-contract checks.
- `provenance/`: execution-source hashes and post-run environment snapshot.

## Performance summary

- Mean AUC: {aggregate["auc_mean"]:.6f} +/- {aggregate["auc_sample_sd"]:.6f}
- Mean final success: {aggregate["final_success_mean"]:.3f} +/- {aggregate["final_success_sample_sd"]:.3f}
- Hit90 coverage: {aggregate["hit90_coverage"]}/{aggregate["seed_count"]}
- Median Hit90 among reached seeds: {int(aggregate["hit90_median_reached_seeds"]):,} steps

All five runs contain exactly 9,976 fixed-target diagnostic snapshots on the
5k-2M/200-step grid, an exact top-10% count of 26 for batch size 256, no Numba
scanner fallback, and no support-audit mismatch.

The software environment was collected immediately after the runs from the
same machine and virtual environment. It was not embedded independently in
each process at launch; this limitation is recorded in `provenance/environment.json`.
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def build_manifest(output_root: Path) -> None:
    files: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_json(
        output_root / "artifact_manifest.json",
        {
            "artifact_id": "drawer_open_ddpg_saber_fixed_target_diagnostic_s6_10_2m",
            "schema_version": 1,
            "execution_commit": EXECUTION_COMMIT,
            "seeds": list(SEEDS),
            "expected_total_steps": EXPECTED_FINAL_STEP,
            "files": files,
            "manifest_self_excluded": True,
        },
    )


def validate_packaged_artifact(output_root: Path) -> None:
    for json_path in output_root.rglob("*.json"):
        leaked = absolute_strings(read_json(json_path))
        if leaked:
            raise AssertionError(f"absolute paths leaked in {json_path}: {leaked[:3]}")
    for path in output_root.rglob("*"):
        if path.is_file() and path.stat().st_size >= 100_000_000:
            raise AssertionError(f"GitHub-incompatible file size: {path}")

    manifest = read_json(output_root / "artifact_manifest.json")
    for item in manifest["files"]:
        path = output_root / item["path"]
        if path.stat().st_size != item["size_bytes"]:
            raise AssertionError(f"size mismatch: {path}")
        if sha256_file(path) != item["sha256"]:
            raise AssertionError(f"SHA256 mismatch: {path}")


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {output_root}")
    output_root.mkdir(parents=True)

    source_manifest = build_source_manifest()
    all_eval_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        run_dir = locate_run(input_root, seed)
        eval_rows, validation = validate_and_read_metrics(run_dir, seed)
        all_eval_rows.extend(eval_rows)
        validation_rows.append(validation)

        destination = output_root / "data" / "seeds" / f"seed_{seed:02d}"
        destination.mkdir(parents=True)
        for name in ("config.json", "env_snapshot.json", "summary.json"):
            write_json(destination / name, sanitize(read_json(run_dir / name)))
        gzip_copy(run_dir / "metrics.csv", destination / "metrics.csv.gz")
        performance_rows.append(performance_row(seed, read_json(run_dir / "summary.json")))

    write_eval_curve(output_root / "analysis" / "eval_curves.csv", all_eval_rows)
    aggregate = aggregate_performance(performance_rows)
    write_json(output_root / "analysis" / "performance_summary.json", aggregate)
    write_json(
        output_root / "validation" / "validation_summary.json",
        {
            "all_runs_validated": True,
            "seed_count": len(validation_rows),
            "runs": validation_rows,
        },
    )
    write_json(
        output_root / "provenance" / "source_manifest.json", source_manifest
    )
    write_json(
        output_root / "provenance" / "environment.json",
        collect_environment(output_root),
    )
    write_readme(output_root, aggregate)
    build_manifest(output_root)
    validate_packaged_artifact(output_root)
    print(f"built and validated artifact: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
