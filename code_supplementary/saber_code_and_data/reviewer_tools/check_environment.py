#!/usr/bin/env python3
"""Report the reproduction environment and optionally enforce exact versions."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys


EXPECTED = {
    "numpy": "2.3.5",
    "scipy": "1.18.0",
    "numba": "0.66.0",
    "llvmlite": "0.48.0",
    "gymnasium": "1.3.0",
    "gymnasium-robotics": "1.4.2",
    "mujoco": "3.3.0",
    "metaworld": "3.1.1",
    "torch": "2.12.1",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    installed: dict[str, str | None] = {}
    mismatches: list[str] = []
    for name, expected in EXPECTED.items():
        try:
            value = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            value = None
        installed[name] = value
        if value != expected:
            mismatches.append(f"{name}: expected {expected}, found {value}")

    payload: dict[str, object] = {
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "packages": installed,
        "exact_version_mismatches": mismatches,
    }
    try:
        import torch

        payload["torch_build"] = torch.__version__
        payload["cuda_build"] = torch.version.cuda
        payload["cuda_available"] = torch.cuda.is_available()
        payload["cudnn"] = (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        )
    except Exception as exc:  # pragma: no cover - environment-specific report
        payload["torch_import_error"] = repr(exc)

    print(json.dumps(payload, indent=2, sort_keys=True))
    if sys.version_info[:2] != (3, 12):
        mismatches.append(f"python: expected 3.12.x, found {platform.python_version()}")
    return 1 if args.strict and mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
