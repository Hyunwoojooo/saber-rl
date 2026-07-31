#!/usr/bin/env python3
"""Fail on author identity, public Git provenance, or common secret patterns."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".csv", ".json", ".md", ".py", ".sh", ".txt", ".toml", ".yaml", ".yml"
}
PATTERNS = {
    "absolute_home_path": re.compile(
        r"(?:/home/[A-Za-z0-9._-]+/|/Users/[A-Za-z0-9._-]+/|"
        r"[A-Za-z]:\\\\Users\\\\[A-Za-z0-9._-]+\\\\)"
    ),
    "email_address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "github_repository_url": re.compile(r"https?://(?:www\.)?github\.com/", re.I),
    "public_git_sha1": re.compile(
        r"(?<![0-9a-fA-F])[0-9a-fA-F]{40}(?![0-9a-fA-F])"
    ),
    "commit_provenance_key": re.compile(
        r"\b(?:source|execution|first_retained_results|canonical_critic_source|"
        r"repo(?:sitory)?_head|origin_main)[_-]?commit(?:[_-]?(?:sha|id))?\b|"
        r"\bcommit[_-](?:sha|id)\b",
        re.I,
    ),
    "short_git_reference": re.compile(
        r"\b(?:commit|git\s+(?:blob|object)|repo(?:sitory)?\s+head|origin\s+main|"
        r"source\s+prefix|checkout\s+prefix|base\s+prefix)\b[^,\n]{0,40}"
        r"\b[0-9a-f]{7,40}\b",
        re.I,
    ),
    "api_key_assignment": re.compile(
        r"(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"][^'\"]+",
        re.I,
    ),
}
IDENTITY_TOKEN_SHA256 = {
    "5896e28f61ab55c60fe467231da25299abb9fdffd4e8be1289b67a57d840c9a5",
    "b1159946984245a1174304228978d0f156a882f9622d65862318c706f5eac337",
    "3dfd13311a7da5a9b2011ee966dcdaa130a769dc532abda089d65fd4f02544dd",
    "5a1be198ac4924ce0b4ad77935c879275b65482791ce04520658eeb52380e059",
    "6e73085e56d5b3bd0ddfd32f2dd55178dec61ed22d00ccd62aff829bb65e2e4b",
}
IDENTITY_TOKEN_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9._-]{2,}\b")
SKIP = {"MANIFEST.sha256", "audit_anonymity.py"}
FORBIDDEN_FILENAMES = {"source_commit.txt"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in SKIP:
            continue
        if path.name.lower() in FORBIDDEN_FILENAMES:
            findings.append(
                f"{path.relative_to(root)}: filename exposes commit provenance"
            )
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in IDENTITY_TOKEN_RE.finditer(text):
            digest = hashlib.sha256(match.group(0).lower().encode()).hexdigest()
            if digest in IDENTITY_TOKEN_SHA256:
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{path.relative_to(root)}:{line}: identity_token")
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{path.relative_to(root)}:{line}: {label}")
    if findings:
        print("\n".join(findings))
        return 1
    print(f"Anonymity audit passed for {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
