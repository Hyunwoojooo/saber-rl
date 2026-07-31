#!/usr/bin/env python3
"""Build the anonymous, standalone SABER Code & Data Supplement archive."""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import zipfile


REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "code_supplementary/saber_code_and_data"
ZIP_PATH = REPO / "code_supplementary/saber_code_and_data.zip"

SOURCE_SNAPSHOTS = {
    "fpp_ddpg": "experiment_FPP/b3_critic_only_runtime_opt_v4_ls512/code/src",
    "fpp_td3": "experiment_FPP/td3_saber_critic_only_v4_ls512/code/src",
    "fpp_sac": "experiment_FPP/sac_saber_critic_only_v4_ls512/code/src",
    "drawer_open_ddpg": "metaworld_Drawer_Open_DDPG/code/src",
    "drawer_open_td3": "metaworld_Drawer_Open_TD3/B3_v4_512_s1_5/code/src",
    "drawer_close_ddpg": "metaworld_Drawer_Close_DDPG/code/src",
    "drawer_close_td3": "metaworld_Drawer_Close_TD3/B3_v4_512_s1_5/code/src",
    "control_replay_action": "experiment_FPP/ddpg_saber_query_replay_ls512/code/src",
    "control_shuffled_policy": "experiment_FPP/ddpg_saber_query_shuffled_ls512/code/src",
    "control_random_support": (
        "experiment_FPP/ddpg_saber_random_support_count_matched_ls512/code/src"
    ),
    "control_strict_count_matched": (
        "experiment_FPP/ddpg_saber_context_eligible_count_matched_ls512/code/src"
    ),
    "drawer_open_fixed_target_diagnostic": (
        "metaworld_Drawer_Open_DDPG/experiments/"
        "DDPG-HER-B3-Critic-Only-Fixed-Target-Diagnostic-2M/code/src"
    ),
}

SELECTED_ARTIFACTS = {
    "fpp_td3_primary_n10": {
        "source": "experiment_FPP/paper_artifacts/fpp_td3_her_vs_saber_n10_1m",
        "include": (
            "README.md",
            "analysis/**",
            "data/**/metrics_selected_10k.csv",
        ),
    },
    "fpp_inclusive_future_s1_5": {
        "source": (
            "experiment_FPP/paper_artifacts/"
            "fpp_td3_inclusive_future_her_sensitivity_1m"
        ),
        "include": ("**",),
    },
    "fpp_inclusive_future_s6_10": {
        "source": (
            "experiment_FPP/paper_artifacts/"
            "fpp_td3_inclusive_future_her_sensitivity_seeds6_10_1m"
        ),
        "include": ("**",),
    },
    "fpp_high_rr_x4": {
        "source": (
            "experiment_FPP/paper_artifacts/"
            "fpp_ddpg_td3_her_high_rr_x4_1m"
        ),
        "include": ("**",),
    },
    "fpp_actor_critic": {
        "source": (
            "experiment_FPP/paper_artifacts/"
            "fpp_ddpg_saber_actor_critic_v4_ls512_1m"
        ),
        "include": ("**",),
    },
    "fpp_diagnostics": {
        "source": (
            "experiment_FPP/paper_artifacts/"
            "fpp_saber_diagnostic_ddpg_td3_sac_ls512_1m"
        ),
        "include": ("README.md", "METRICS.md", "results/**"),
    },
    "drawer_open_fixed_target_diagnostic_all10": {
        "source": (
            "metaworld_Drawer_Open_DDPG/paper_artifacts/"
            "drawer_open_ddpg_saber_fixed_target_diagnostic_all_10_2m"
        ),
        "include": ("**",),
    },
    "drawer_close_redq": {
        "source": (
            "metaworld_Drawer_Close_REDQ/paper_artifacts/"
            "drawer_close_redq_her_tqb_utd4_2m"
        ),
        "include": ("**",),
    },
}

IGNORED_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ignored(path: Path) -> bool:
    return any(part in IGNORED_NAMES for part in path.parts) or path.suffix in IGNORED_SUFFIXES


def clean_generated() -> None:
    for relative in ("source", "results", "provenance"):
        target = PACKAGE / relative
        if target.exists():
            shutil.rmtree(target)
    for target in (PACKAGE / "MANIFEST.sha256", ZIP_PATH):
        if target.exists():
            target.unlink()
    for cache in sorted(PACKAGE.rglob("__pycache__"), reverse=True):
        if cache.is_dir():
            shutil.rmtree(cache)


def copy_tree(source: Path, target: Path) -> list[dict[str, object]]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    entries: list[dict[str, object]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or ignored(path.relative_to(source)):
            continue
        relative = path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        entries.append(
            {
                "path": destination.relative_to(PACKAGE).as_posix(),
                "source_repository_path": path.relative_to(REPO).as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    return entries


def matches(relative: Path, patterns: tuple[str, ...]) -> bool:
    value = relative.as_posix()
    for pattern in patterns:
        if pattern == "**":
            return True
        if fnmatch.fnmatch(value, pattern):
            return True
        if pattern.endswith("/**") and value.startswith(pattern[:-3] + "/"):
            return True
    return False


def copy_selected(
    source: Path, target: Path, patterns: tuple[str, ...]
) -> list[dict[str, object]]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    entries: list[dict[str, object]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if ignored(relative) or not matches(relative, patterns):
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        entries.append(
            {
                "path": relative.as_posix(),
                "source_repository_path": path.relative_to(REPO).as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    if not entries:
        raise RuntimeError(f"No files selected from {source}")
    (target / "SELECTION_MANIFEST.json").write_text(
        json.dumps(
            {
                "source_repository_path": source.relative_to(REPO).as_posix(),
                "selection_patterns": list(patterns),
                "files": entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return entries


def copy_sources() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for name, relative in SOURCE_SNAPSHOTS.items():
        source = REPO / relative
        target = PACKAGE / "source" / name
        entries = copy_tree(source, target)
        records.append(
            {
                "snapshot": name,
                "repository_path": relative,
                "package_path": f"source/{name}",
                "file_count": len(entries),
                "bytes": sum(int(item["bytes"]) for item in entries),
                "files": entries,
            }
        )
    return records


def copy_results() -> list[dict[str, object]]:
    derived_source = REPO / "paper/supplementary_artifacts"
    derived_target = PACKAGE / "results/derived"
    derived_entries = copy_tree(derived_source, derived_target)
    records: list[dict[str, object]] = [
        {
            "artifact": "derived_supplement_results",
            "repository_path": str(derived_source.relative_to(REPO)),
            "package_path": "results/derived",
            "file_count": len(derived_entries),
            "bytes": sum(int(item["bytes"]) for item in derived_entries),
        }
    ]
    for name, spec in SELECTED_ARTIFACTS.items():
        source = REPO / str(spec["source"])
        target = PACKAGE / "results/selected_artifacts" / name
        entries = copy_selected(source, target, tuple(spec["include"]))
        records.append(
            {
                "artifact": name,
                "repository_path": str(spec["source"]),
                "package_path": f"results/selected_artifacts/{name}",
                "file_count": len(entries),
                "bytes": sum(int(item["bytes"]) for item in entries),
            }
        )
    return records


def write_experiment_catalog() -> None:
    rows: list[dict[str, object]] = []
    for preset_path in sorted((PACKAGE / "presets").glob("*.json")):
        preset = json.loads(preset_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "preset_id": preset["preset_id"],
                "task": preset["task_label"],
                "backbone": preset["backbone"],
                "source_snapshot": preset["source"],
                "entrypoint": preset["entrypoint"],
                "seeds": "1-5",
                "environment_steps_per_seed": preset["paper_steps"],
                "her_ratio": preset["settings"]["her_ratio"],
                "utd": preset["settings"]["updates_per_step"],
                "weight_scope": preset["settings"]["weight_scope"],
            }
        )
    path = PACKAGE / "provenance/experiment_catalog.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_omitted_sources() -> None:
    rows = [
        {
            "source_family": "training_checkpoints",
            "status": "omitted",
            "reason": "Large binary state is not needed to recompute reported tables.",
            "replacement": "Seed-level metrics and cryptographic source bindings.",
        },
        {
            "source_family": "replay_buffers",
            "status": "omitted",
            "reason": "Large transient training state.",
            "replacement": "Exact source snapshots, presets, and contract tests.",
        },
        {
            "source_family": "full_raw_logs",
            "status": "omitted",
            "reason": "Redundant and may contain host-local paths.",
            "replacement": "Sanitized selected metrics and derived result artifacts.",
        },
        {
            "source_family": "repository_git_metadata",
            "status": "omitted",
            "reason": "Anonymous review archive.",
            "replacement": "Source SHA256 inventory and experiment catalog.",
        },
    ]
    path = PACKAGE / "provenance/omitted_large_sources.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_inventory(exclude: set[str] | None = None) -> list[dict[str, object]]:
    excluded = exclude or set()
    inventory = []
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file():
            continue
        if ignored(path.relative_to(PACKAGE)):
            continue
        relative = path.relative_to(PACKAGE).as_posix()
        if relative in excluded:
            continue
        inventory.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return inventory


def write_provenance(
    source_records: list[dict[str, object]],
    artifact_records: list[dict[str, object]],
) -> None:
    provenance = PACKAGE / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    (provenance / "source_snapshot_manifest.json").write_text(
        json.dumps(
            {
                "source_snapshots": source_records,
                "notes": (
                    "Repository-relative paths identify the frozen inputs used by "
                    "the deterministic package builder; no host-local path is retained."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (provenance / "artifact_selection_manifest.json").write_text(
        json.dumps({"artifacts": artifact_records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_experiment_catalog()
    write_omitted_sources()

    inventory = file_inventory({"MANIFEST.sha256", "provenance/package_manifest.json"})
    (provenance / "package_manifest.json").write_text(
        json.dumps(
            {
                "inventory_scope": (
                    "All package files present before package_manifest.json and "
                    "MANIFEST.sha256 are written."
                ),
                "file_count": len(inventory),
                "total_bytes": sum(int(item["bytes"]) for item in inventory),
                "files": inventory,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_top_manifest() -> None:
    entries = file_inventory({"MANIFEST.sha256"})
    (PACKAGE / "MANIFEST.sha256").write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in entries),
        encoding="utf-8",
    )


def make_executable() -> None:
    paths = [PACKAGE / "environment/install.sh"]
    paths.extend((PACKAGE / "reviewer_tools").glob("*.sh"))
    paths.extend((PACKAGE / "reviewer_tools").glob("*.py"))
    for path in paths:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def build_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(
        ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(PACKAGE.rglob("*")):
            if not path.is_file():
                continue
            if ignored(path.relative_to(PACKAGE)):
                continue
            relative = Path(PACKAGE.name) / path.relative_to(PACKAGE)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = path.stat().st_mode & 0o777
            info.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def main() -> int:
    clean_generated()
    source_records = copy_sources()
    artifact_records = copy_results()
    write_provenance(source_records, artifact_records)
    make_executable()
    write_top_manifest()

    checks = [
        [sys.executable, str(PACKAGE / "reviewer_tools/audit_anonymity.py")],
        [sys.executable, str(PACKAGE / "reviewer_tools/launch_experiment.py"), "--list"],
    ]
    for command in checks:
        subprocess.run(command, cwd=PACKAGE, check=True)
    build_zip()
    print(f"Built {ZIP_PATH.relative_to(REPO)} ({ZIP_PATH.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
