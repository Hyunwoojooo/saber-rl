#!/usr/bin/env python3
"""Build the anonymous, standalone SABER Code & Data Supplement archive."""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
from pathlib import Path
import re
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

TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
}
IGNORED_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
SELECTED_IGNORED_NAMES = {
    "MANIFEST.sha256",
    "SELECTION_MANIFEST.json",
    "SHA256SUMS",
    "artifact_manifest.json",
    "source_commit.txt",
    "source_inputs_manifest.json",
    "source_manifest.json",
}
HIGH_RR_STALE_LAUNCHER = (
    "source/td3/run_fpp_td3_her_high_rr_x4_resume_after_reboot.sh"
)
GIT_SHA1_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{40}(?![0-9a-fA-F])")

KNOWN_ANONYMOUS_REFERENCES = {
    "05c47923312c56ae84ea4d4fdaae4a3dab63ac8f": (
        "snapshot_fpp_actor_critic_results_v1"
    ),
    "2d52b29666233616c7b1e2029aa8090e7d74bd66": (
        "snapshot_fpp_actor_critic_source_v1"
    ),
    "22015c0a3743d4a10f72197752022ed0e08ee730": (
        "snapshot_drawer_open_fixed_target_v1"
    ),
    "e27438c1ea288fdb7bd6c75c49283d347fee77d9": (
        "snapshot_fpp_ddpg_high_rr_v1"
    ),
    "783a34cfeddfddf03bc5a5d18700f39e92e507dd": (
        "snapshot_fpp_td3_high_rr_v1"
    ),
}

PROVENANCE_TERM_REPLACEMENTS = (
    ("Hyunwoojooo/metacog-rl", "anonymous_repository/source_a"),
    ("Hyunwoojooo/saber-rl", "anonymous_repository/source_b"),
    ("canonical_critic_source_commit", "canonical_critic_source_snapshot_id"),
    ("first_retained_results_commit", "first_retained_results_snapshot_id"),
    ("data-introduction commit", "data snapshot"),
    ("execution_source_commit", "execution_source_snapshot_id"),
    ("execution_commit", "execution_snapshot_id"),
    ("source_commit", "source_snapshot_id"),
    ("commit_sha", "source_snapshot_id"),
    ("repo_head", "repository_snapshot_id"),
    ("origin_main", "canonical_snapshot_id"),
    ("canonical critic source commit", "canonical critic source snapshot"),
    ("first retained results commit", "first retained results snapshot"),
    ("execution source commit", "execution source snapshot"),
    ("execution commit", "execution snapshot"),
    ("source commit", "source snapshot"),
    ("results commit", "results snapshot"),
    ("same commit", "same anonymous snapshot"),
    ("no execution SHA", "no execution snapshot ID"),
    ("committed extension artifact", "verified extension artifact"),
    ("Git blob SHA-1", "retained source-file identity"),
    ("Git blobs", "retained source files"),
    ("Git blob", "retained source file"),
)

ARTIFACT_INDEX_PATHS = {
    "primary_seed_results": ("results/derived/primary_seed_results", "complete"),
    "mechanism_controls_ddpg": (
        "results/derived/mechanism_controls_ddpg",
        "complete",
    ),
    "td_error_weighting_controls": (
        "results/derived/technical_supplement/secondary_reference_summary.csv",
        "complete_compact_summary",
    ),
    "phase_seed_results": ("results/derived/phase_seed_results", "complete"),
    "action_bin_sensitivity": (
        "results/derived/action_bin_sensitivity",
        "complete",
    ),
    "actor_scope_ddpg": ("results/derived/actor_scope_ddpg", "complete"),
    "technical_supplement": (
        "results/derived/technical_supplement",
        "complete",
    ),
    "fpp_ddpg_actor_critic": (
        "results/selected_artifacts/fpp_actor_critic",
        "complete",
    ),
    "fpp_diagnostic": (
        "results/selected_artifacts/fpp_diagnostics",
        "complete",
    ),
    "drawerclose_diagnostic": (
        "results/derived/technical_supplement/diagnostic_cohort_summary.csv",
        "complete_compact_summary",
    ),
    "draweropen_fixed_target": (
        "results/selected_artifacts/drawer_open_fixed_target_diagnostic_all10",
        "complete",
    ),
    "fpp_td3_runtime": (
        "results/derived/technical_supplement/runtime_summary.csv",
        "complete_compact_summary",
    ),
    "execution_environment_audit": (
        "results/derived/technical_supplement/execution_environment_summary.csv",
        "complete_compact_summary",
    ),
    "fpp_td3_n10": (
        "results/selected_artifacts/fpp_td3_primary_n10",
        "complete",
    ),
    "fpp_td3_inclusive": (
        "results/selected_artifacts/fpp_inclusive_future_s1_5",
        "complete",
    ),
    "fpp_td3_inclusive_extension": (
        "results/selected_artifacts/fpp_inclusive_future_s6_10",
        "complete",
    ),
    "fpp_aer_noher": (
        "results/derived/technical_supplement/secondary_reference_summary.csv",
        "complete_compact_summary",
    ),
    "draweropen_ddpg_aer_noher": (
        "results/derived/technical_supplement/secondary_reference_summary.csv",
        "complete_compact_summary",
    ),
    "draweropen_td3_aer_noher": (
        "results/derived/technical_supplement/secondary_reference_summary.csv",
        "complete_compact_summary",
    ),
    "drawerclose_ddpg_aer_noher": (
        "results/derived/technical_supplement/secondary_reference_summary.csv",
        "complete_compact_summary",
    ),
    "drawerclose_td3_aer_noher": (
        "results/derived/technical_supplement/secondary_reference_summary.csv",
        "complete_compact_summary",
    ),
    "fpp_ddpg_td3_high_rr_x4": (
        "results/selected_artifacts/fpp_high_rr_x4",
        "complete_with_documented_stale_metadata",
    ),
    "drawerclose_redq": (
        "results/selected_artifacts/drawer_close_redq",
        "complete",
    ),
    "update_budget_references": (
        "results/derived/update_budget_references",
        "complete",
    ),
}


class AnonymousReferenceRegistry:
    """Replace public Git object identifiers with deterministic package IDs."""

    def __init__(self) -> None:
        self.mapping = dict(KNOWN_ANONYMOUS_REFERENCES)

    def replacement(self, match: re.Match[str]) -> str:
        value = match.group(0).lower()
        if value not in self.mapping:
            self.mapping[value] = f"snapshot_reference_{len(self.mapping) + 1:03d}"
        return self.mapping[value]

    def sanitize(self, text: str) -> str:
        text = GIT_SHA1_RE.sub(self.replacement, text)
        for source, replacement in PROVENANCE_TERM_REPLACEMENTS:
            text = re.sub(re.escape(source), replacement, text, flags=re.IGNORECASE)
        return text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ignored(path: Path) -> bool:
    return any(part in IGNORED_NAMES for part in path.parts) or path.suffix in IGNORED_SUFFIXES


def copy_sanitized(
    source: Path,
    destination: Path,
    registry: AnonymousReferenceRegistry,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.suffix.lower() not in TEXT_SUFFIXES:
        return
    raw = destination.read_bytes()
    if b"\x00" in raw:
        return
    text = raw.decode("utf-8", errors="strict")
    sanitized = registry.sanitize(text)
    if sanitized != text:
        destination.write_text(sanitized, encoding="utf-8")


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


def copy_tree(
    source: Path,
    target: Path,
    registry: AnonymousReferenceRegistry,
) -> list[dict[str, object]]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    entries: list[dict[str, object]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or ignored(path.relative_to(source)):
            continue
        relative = path.relative_to(source)
        destination = target / relative
        copy_sanitized(path, destination, registry)
        entries.append(
            {
                "path": destination.relative_to(PACKAGE).as_posix(),
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
    artifact_name: str,
    source: Path,
    target: Path,
    patterns: tuple[str, ...],
    registry: AnonymousReferenceRegistry,
) -> list[dict[str, object]]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    entries: list[dict[str, object]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if (
            ignored(relative)
            or not matches(relative, patterns)
            or relative.name in SELECTED_IGNORED_NAMES
            or relative.parts[0:1] == ("code",)
            or (
                artifact_name == "fpp_high_rr_x4"
                and relative.as_posix() == HIGH_RR_STALE_LAUNCHER
            )
        ):
            continue
        destination = target / relative
        copy_sanitized(path, destination, registry)
        entries.append(
            {
                "path": relative.as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    if not entries:
        raise RuntimeError(f"No files selected from {source}")
    (target / "SELECTION_MANIFEST.json").write_text(
        json.dumps(
            {
                "artifact_snapshot_id": f"artifact_{artifact_name}_v1",
                "package_path": target.relative_to(PACKAGE).as_posix(),
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


def copy_sources(registry: AnonymousReferenceRegistry) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for name, relative in SOURCE_SNAPSHOTS.items():
        source = REPO / relative
        target = PACKAGE / "source" / name
        entries = copy_tree(source, target, registry)
        records.append(
            {
                "source_snapshot_id": f"snapshot_{name}_v1",
                "package_path": f"source/{name}",
                "file_count": len(entries),
                "bytes": sum(int(item["bytes"]) for item in entries),
                "files": entries,
            }
        )
    return records


def copy_results(registry: AnonymousReferenceRegistry) -> list[dict[str, object]]:
    derived_source = REPO / "paper/supplementary_artifacts"
    derived_target = PACKAGE / "results/derived"
    derived_entries = copy_tree(derived_source, derived_target, registry)
    records: list[dict[str, object]] = [
        {
            "artifact": "derived_supplement_results",
            "package_path": "results/derived",
            "file_count": len(derived_entries),
            "bytes": sum(int(item["bytes"]) for item in derived_entries),
        }
    ]
    for name, spec in SELECTED_ARTIFACTS.items():
        source = REPO / str(spec["source"])
        target = PACKAGE / "results/selected_artifacts" / name
        entries = copy_selected(
            name,
            source,
            target,
            tuple(spec["include"]),
            registry,
        )
        records.append(
            {
                "artifact": name,
                "package_path": f"results/selected_artifacts/{name}",
                "file_count": len(entries),
                "bytes": sum(int(item["bytes"]) for item in entries),
            }
        )
    return records


def rewrite_execution_host_mapping() -> None:
    path = PACKAGE / "results/derived/technical_supplement/execution_host_mapping.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    for row in rows:
        row["source_provenance"] = (
            "anonymous source snapshot; package SHA256 inventory retained"
        )
        if row["variant"] == "HER High-RR x4 UTD4":
            row["environment_exactness"] = (
                "matched production environment; compact result snapshot retained"
            )
            row["notes"] = (
                "fresh uninterrupted run from step 0; recorded resume fields and "
                "runtime-counter resets are stale logging metadata; no checkpoint "
                "continuation"
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rewrite_artifact_index() -> None:
    path = PACKAGE / "results/derived/technical_supplement/artifact_index.csv"
    rows = [
        {
            "artifact_key": key,
            "package_relative_path": package_path,
            "status": status,
        }
        for key, (package_path, status) in ARTIFACT_INDEX_PATHS.items()
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_high_rr_correction() -> None:
    root = PACKAGE / "results/selected_artifacts/fpp_high_rr_x4"
    provenance_path = root / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    correction = provenance.setdefault("execution_metadata_correction", {})
    correction.update(
        {
            "checkpoint_continuation": False,
            "fresh_run_scope": "all DDPG and TD3 seeds began from step 0",
            "recorded_resume_metadata_stale": True,
            "resume_launcher_used_for_production": False,
            "runtime_counter_resets_are_process_restarts": False,
        }
    )
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "PROVENANCE_CORRECTION.md").write_text(
        "# High-RR Execution-Metadata Boundary\n\n"
        "All DDPG and TD3 High-RR runs began from scratch at environment step 0 "
        "and completed without checkpoint continuation. Some retained TD3 "
        "configuration fields and runtime-counter segments use resume-oriented "
        "labels; these are stale logging metadata and are not execution history.\n\n"
        "- `checkpoint_continuation`: `false`\n"
        "- `recorded_resume_metadata_stale`: `true`\n"
        "- `resume_launcher_used_for_production`: `false`\n\n"
        "The unused resume launcher is intentionally excluded from this reviewer "
        "package. Original retained per-seed configurations are preserved so the "
        "metadata boundary remains auditable. Package integrity is established by "
        "`SELECTION_MANIFEST.json` and the top-level `MANIFEST.sha256`.\n",
        encoding="utf-8",
    )
    readme = root / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = text.replace(
        "\nVerify from this directory with `python3 "
        "code/verify_fpp_high_rr_compact_artifact.py --artifact-root .`.\n",
        "\nVerify this selection through `python reviewer_tools/verify_package.py` "
        "from the package root.\n",
    )
    readme.write_text(text, encoding="utf-8")


def refresh_selection_manifests() -> None:
    root = PACKAGE / "results/selected_artifacts"
    for artifact_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest = artifact_dir / "SELECTION_MANIFEST.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        files = []
        for path in sorted(artifact_dir.rglob("*")):
            if not path.is_file() or path == manifest or ignored(path.relative_to(artifact_dir)):
                continue
            files.append(
                {
                    "bytes": path.stat().st_size,
                    "path": path.relative_to(artifact_dir).as_posix(),
                    "sha256": sha256(path),
                }
            )
        payload["files"] = files
        manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def refresh_derived_manifests() -> None:
    derived_root = PACKAGE / "results/derived"
    for artifact_dir in sorted(path for path in derived_root.iterdir() if path.is_dir()):
        manifest = artifact_dir / "artifact_manifest.json"
        if not manifest.is_file():
            continue
        files = []
        for path in sorted(artifact_dir.rglob("*")):
            if not path.is_file() or path == manifest or ignored(path.relative_to(artifact_dir)):
                continue
            files.append(
                {
                    "bytes": path.stat().st_size,
                    "path": path.relative_to(PACKAGE).as_posix(),
                    "sha256": sha256(path),
                }
            )
        manifest.write_text(
            json.dumps(
                {
                    "files": files,
                    "root": artifact_dir.relative_to(PACKAGE).as_posix(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def normalize_copied_results() -> None:
    rewrite_execution_host_mapping()
    rewrite_artifact_index()
    write_high_rr_correction()
    refresh_selection_manifests()
    refresh_derived_manifests()


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
        {
            "source_family": "secondary_baseline_training_sources",
            "status": "partially_included",
            "reason": (
                "The archive prioritizes SABER, matched HER, and mechanism-control "
                "source; not every secondary AER, HGR, PER, or REDQ launcher is bundled."
            ),
            "replacement": "Compact seed-level evidence and analysis inputs.",
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
                    "Anonymous snapshot IDs and per-file SHA256 values bind every "
                    "frozen source input; no public Git or host-local identity is retained."
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
    registry = AnonymousReferenceRegistry()
    source_records = copy_sources(registry)
    artifact_records = copy_results(registry)
    normalize_copied_results()
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
