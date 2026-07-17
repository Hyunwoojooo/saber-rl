#!/usr/bin/env python3
"""Regenerate the outer bundle SHA256SUMS inventory."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path, nargs="?", default=Path("."))
    root = parser.parse_args().artifact_root.resolve()
    output = root / "SHA256SUMS"
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path != output
    )
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("bundle must not contain symlinks")
    output.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files),
        encoding="utf-8",
        newline="\n",
    )
    print(f"[bundle-checksums-written] {len(files)} files -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
