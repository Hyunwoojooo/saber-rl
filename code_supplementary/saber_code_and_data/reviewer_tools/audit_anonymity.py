#!/usr/bin/env python3
"""Fail on identity-bearing paths, repository URLs, or common secret patterns."""

from __future__ import annotations

import argparse
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
    "api_key_assignment": re.compile(
        r"(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"][^'\"]+",
        re.I,
    ),
}
SKIP = {"MANIFEST.sha256", "audit_anonymity.py"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in SKIP or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
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
