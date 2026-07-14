#!/usr/bin/env python3
"""Build a deterministic double-blind review archive from the canonical artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Mapping


DEFAULT_SOURCE = Path(
    "experiment_FPP/paper_artifacts/"
    "fpp_td3_saber_context_eligible_count_matched"
)
DEFAULT_OUTPUT = Path(
    "output/review_artifacts/"
    "fpp_td3_saber_context_eligible_count_matched"
)
MANIFEST_NAME = "artifact_manifest.json"
LOCAL_JUNK_PATTERNS = ("__pycache__", "*.pyc", ".DS_Store")
REDACTED_COMMIT = "<REDACTED_COMMIT>"
COMMIT_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])")
COMMIT_BYTES_RE = re.compile(rb"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])")

TEXT_REPLACEMENTS = (
    ("<HOME>/Hyunwoo", "<SOURCE_HOME>"),
    ("/Users/joo", "<SOURCE_HOME>"),
    ('b"Hyunwoo" + b"jooo"', 'b"AUTHOR" + b"_HANDLE"'),
    ("Hyunwoojooo", "<AUTHOR_HANDLE>"),
    ("Hyunwoo", "<AUTHOR_NAME>"),
    ("jooo", "<AUTHOR_HANDLE_SUFFIX>"),
    ("mimiclab", "<REDACTED_LAB>"),
)

FORBIDDEN_BYTES = (
    b"hyunwoo",
    b"hyunwoojooo",
    b"jooo",
    b"/users/joo",
    b"<home>/hyunwoo",
    b"mimiclab",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def is_local_junk_path(relative: Path) -> bool:
    return (
        "__pycache__" in relative.parts
        or relative.suffix == ".pyc"
        or relative.name == ".DS_Store"
    )


def package_file_paths(package_root: Path) -> list[Path]:
    return sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file()
        and not is_local_junk_path(path.relative_to(package_root))
    )


def package_files(package_root: Path) -> list[dict[str, object]]:
    records = []
    for path in package_file_paths(package_root):
        relative = path.relative_to(package_root).as_posix()
        if relative == MANIFEST_NAME:
            continue
        records.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def content_digest(records: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_manifest(package_root: Path) -> None:
    manifest_path = package_root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("packaged_files")
    if not isinstance(expected, list):
        raise ValueError(f"{manifest_path}: packaged_files is missing")

    actual = package_files(package_root)
    if actual != expected:
        expected_by_path = {str(item["path"]): item for item in expected}
        actual_by_path = {str(item["path"]): item for item in actual}
        missing = sorted(set(expected_by_path) - set(actual_by_path))
        extra = sorted(set(actual_by_path) - set(expected_by_path))
        changed = sorted(
            path
            for path in set(expected_by_path) & set(actual_by_path)
            if expected_by_path[path] != actual_by_path[path]
        )
        raise ValueError(
            "artifact manifest mismatch: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )

    if manifest.get("packaged_file_count_excluding_manifest") != len(actual):
        raise ValueError("artifact manifest file count is inconsistent")
    if manifest.get("package_content_sha256_excluding_manifest") != content_digest(actual):
        raise ValueError("artifact manifest content digest is inconsistent")


def verify_structured_redaction(package_root: Path) -> None:
    input_manifest = json.loads(
        (package_root / "input_manifest.json").read_text(encoding="utf-8")
    )
    commit_values = {
        "input.repository.commit": input_manifest.get("repository", {}).get("commit"),
        "input.implementation_source.repository_parent_commit": input_manifest.get(
            "implementation_source", {}
        ).get("repository_parent_commit"),
        "input.implementation_source.source_commit": input_manifest.get(
            "implementation_source", {}
        ).get("source_commit"),
    }
    for method, payload in input_manifest.get("methods", {}).items():
        commit_values[f"input.methods.{method}.source_commit"] = payload.get(
            "source_commit"
        )

    freeze_manifest = json.loads(
        (package_root / "code/freeze_manifest.json").read_text(encoding="utf-8")
    )
    commit_values["freeze.repository_commit_at_freeze"] = freeze_manifest.get(
        "repository_commit_at_freeze"
    )

    artifact_manifest = json.loads(
        (package_root / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    commit_values["artifact.repository.commit"] = artifact_manifest.get(
        "repository", {}
    ).get("commit")

    remaining = {key: value for key, value in commit_values.items() if value is not None}
    if remaining:
        raise ValueError(f"structured commit fields were not redacted: {remaining}")


def scan_anonymity(package_root: Path) -> None:
    violations: list[str] = []
    for path in package_file_paths(package_root):
        relative = path.relative_to(package_root).as_posix()
        data = path.read_bytes()
        lowered = data.lower()
        tokens = [
            token.decode("ascii", errors="replace")
            for token in FORBIDDEN_BYTES
            if token in lowered
        ]
        if COMMIT_BYTES_RE.search(lowered):
            tokens.append("raw-40-hex-commit")
        if tokens:
            violations.append(f"{relative}: {', '.join(tokens)}")
    if violations:
        raise ValueError("double-blind scan failed:\n" + "\n".join(violations))


def sanitize_text_files(package_root: Path) -> int:
    redacted_commits = 0
    for path in package_file_paths(package_root):
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        sanitized = original
        for source, replacement in TEXT_REPLACEMENTS:
            sanitized = sanitized.replace(source, replacement)
        sanitized, count = COMMIT_RE.subn(REDACTED_COMMIT, sanitized)
        redacted_commits += count
        if sanitized != original:
            path.write_text(sanitized, encoding="utf-8")
    return redacted_commits


def redact_structured_provenance(package_root: Path) -> None:
    freeze_path = package_root / "code/freeze_manifest.json"
    freeze_manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_manifest["repository_commit_at_freeze"] = None
    freeze_manifest["repository_provenance_note"] = (
        "Repository commit identifier withheld for double-blind review."
    )
    write_json(freeze_path, freeze_manifest)

    input_path = package_root / "input_manifest.json"
    input_manifest = json.loads(input_path.read_text(encoding="utf-8"))
    input_manifest.setdefault("repository", {})["commit"] = None
    input_manifest["repository"]["provenance_note"] = (
        "Repository commit identifier withheld for double-blind review."
    )

    methods = input_manifest.get("methods", {})
    for method, payload in methods.items():
        had_commit = payload.get("source_commit") is not None
        payload["source_commit"] = None
        if had_commit:
            payload["provenance_note"] = (
                "Source commit identifier withheld for double-blind review; "
                "exact input SHA256 records are retained."
            )

    implementation = input_manifest.setdefault("implementation_source", {})
    implementation["repository_parent_commit"] = None
    implementation["source_commit"] = None
    implementation["provenance_note"] = (
        "Repository commit identifiers withheld for double-blind review."
    )
    implementation["file_hash_scope"] = "packaged post-redaction bytes"
    for record in implementation.get("files", []):
        packaged_path = record.get("packaged_path")
        if not isinstance(packaged_path, str):
            continue
        packaged_file = package_root / packaged_path
        if packaged_file.is_file():
            record["sha256"] = sha256(packaged_file)

    transform = input_manifest.setdefault("transform", {})
    transform["json"] = (
        "Machine-specific paths, source-user namespaces, URLs, and Git commit "
        "identifiers sanitized; numeric experiment values unchanged."
    )
    input_manifest["double_blind_review"] = {
        "source_user_namespaces_redacted": True,
        "source_git_commit_ids_withheld": True,
        "numeric_experiment_values_changed": False,
    }
    write_json(input_path, input_manifest)

    artifact_path = package_root / MANIFEST_NAME
    artifact_manifest = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact_manifest.setdefault("repository", {})["commit"] = None
    write_json(artifact_path, artifact_manifest)


def normalize_packaged_run_paths(package_root: Path) -> None:
    main_root = package_root / "data/main"
    for run_dir in sorted(path for path in main_root.iterdir() if path.is_dir()):
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["output_root"] = "data/main"
        write_json(config_path, config)

        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["run_dir"] = f"data/main/{run_dir.name}"
        write_json(summary_path, summary)


def changed_payload_file_count(source: Path, package_root: Path) -> int:
    changed = 0
    for path in package_file_paths(package_root):
        relative = path.relative_to(package_root)
        if relative.as_posix() == MANIFEST_NAME:
            continue
        source_path = source / relative
        if not source_path.is_file() or sha256(source_path) != sha256(path):
            changed += 1
    return changed


def append_review_note(package_root: Path) -> None:
    readme = package_root / "README.md"
    marker = "## Double-Blind Review Packaging"
    text = readme.read_text(encoding="utf-8").rstrip()
    text = re.sub(
        r"- Source repository commit at packaging:.*",
        "- Source repository commit at packaging: withheld for double-blind review",
        text,
    )
    if marker not in text:
        text += (
            "\n\n"
            f"{marker}\n\n"
            "This review copy redacts source-user namespaces and Git commit IDs. "
            "Experiment values and input SHA256 records are unchanged. Per-file "
            "package hashes and the aggregate package digest were regenerated after "
            "redaction. Public repository provenance can be restored in a "
            "non-anonymous release.\n"
        )
    readme.write_text(text + "\n", encoding="utf-8")


def regenerate_manifest(
    package_root: Path,
    canonical_manifest: Mapping[str, object],
    canonical_manifest_sha256: str,
    redaction_stats: Mapping[str, int],
) -> None:
    manifest_path = package_root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repository = manifest.get("repository")
    if isinstance(repository, dict):
        repository["commit"] = None
    manifest["review_package"] = {
        "double_blind": True,
        "source_user_namespaces_redacted": True,
        "source_git_commit_ids_redacted": True,
        "numeric_experiment_values_changed": False,
        "canonical_artifact_manifest_sha256": canonical_manifest_sha256,
        "canonical_package_content_sha256_excluding_manifest": canonical_manifest.get(
            "package_content_sha256_excluding_manifest"
        ),
        **dict(redaction_stats),
    }
    records = package_files(package_root)
    manifest["packaged_file_count_excluding_manifest"] = len(records)
    manifest["package_content_sha256_excluding_manifest"] = content_digest(records)
    manifest["packaged_files"] = records
    manifest["self_hash_note"] = (
        "artifact_manifest.json is excluded to avoid a recursive self-hash."
    )
    write_json(manifest_path, manifest)


def write_deterministic_zip(package_root: Path, archive_path: Path) -> None:
    root_name = package_root.name
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in package_file_paths(package_root):
            relative = path.relative_to(package_root).as_posix()
            info = zipfile.ZipInfo(f"{root_name}/{relative}")
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            mode = stat.S_IMODE(path.stat().st_mode)
            info.external_attr = (mode & 0xFFFF) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build_review_package(source: Path, output: Path) -> Path:
    source = source.resolve()
    output = output.resolve()
    archive_path = output.with_suffix(".zip")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists() or archive_path.exists():
        raise FileExistsError(
            f"refusing to replace generated output: {output} or {archive_path}"
        )
    for path in source.rglob("*"):
        if is_local_junk_path(path.relative_to(source)):
            continue
        if path.is_symlink():
            raise ValueError(f"source artifact contains a symlink: {path}")

    verify_manifest(source)
    canonical_manifest_path = source / MANIFEST_NAME
    canonical_manifest = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
    canonical_manifest_sha256 = sha256(canonical_manifest_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".context_matched_review_", dir=output.parent) as temp:
        staged = Path(temp) / output.name
        shutil.copytree(
            source,
            staged,
            ignore=shutil.ignore_patterns(*LOCAL_JUNK_PATTERNS),
        )
        redaction_stats = sanitize_text_files(staged)
        normalize_packaged_run_paths(staged)
        redact_structured_provenance(staged)
        append_review_note(staged)
        redaction_stats = {
            "changed_payload_files_excluding_manifest": changed_payload_file_count(
                source, staged
            ),
            "redacted_git_commit_ids": redaction_stats,
        }
        regenerate_manifest(
            staged,
            canonical_manifest,
            canonical_manifest_sha256,
            redaction_stats,
        )
        scan_anonymity(staged)
        verify_structured_redaction(staged)
        verify_manifest(staged)

        staged_archive = Path(temp) / archive_path.name
        write_deterministic_zip(staged, staged_archive)
        shutil.move(staged, output)
        shutil.move(staged_archive, archive_path)

    return archive_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a sanitized review ZIP from the canonical FPP TD3 artifact."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-dir", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if args.verify_dir is not None:
        target = args.verify_dir
        target = target if target.is_absolute() else repo_root / target
        verify_manifest(target)
        scan_anonymity(target)
        verify_structured_redaction(target)
        print(f"verified {target}")
        return 0

    source = args.source_dir
    source = source if source.is_absolute() else repo_root / source
    output = args.output_dir
    output = output if output.is_absolute() else repo_root / output
    archive = build_review_package(source, output)
    print(f"packaged {output}")
    print(f"archive {archive}")
    print(f"archive_sha256 {sha256(archive)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
