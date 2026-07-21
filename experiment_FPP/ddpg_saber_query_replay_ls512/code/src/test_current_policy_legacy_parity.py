#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def run_fixture(source_root: Path, fixture_root: Path) -> dict[str, object]:
    env = os.environ.copy()
    paths = [str(source_root), str(fixture_root)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    result = subprocess.run(
        [sys.executable, "-c", "from weight_parity_fixture import emit; emit()"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(result.stdout)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    fixture_root = Path(__file__).resolve().parent
    experiment_root = fixture_root.parents[2]
    baseline_root = experiment_root / "b3_critic_only_runtime_opt_v4_ls512" / "code" / "src"
    if not baseline_root.is_dir():
        raise FileNotFoundError(f"baseline source root missing: {baseline_root}")

    # Query support must not alter DDPG optimization, replay, or network semantics.
    for relative in (
        "metacog_lite_v0/agent.py",
        "metacog_lite_v0/replay.py",
        "metacog_lite_v0/networks.py",
    ):
        current_path = fixture_root / relative
        baseline_path = baseline_root / relative
        if sha256(current_path) != sha256(baseline_path):
            raise AssertionError(f"canonical DDPG component changed unexpectedly: {relative}")

    legacy = run_fixture(baseline_root, fixture_root)
    current = run_fixture(fixture_root, fixture_root)
    if current != legacy:
        raise AssertionError(
            "current_policy support/weights/counts or final weight RNG state differ from canonical DDPG"
        )
    scanner = "Numba predicate" if os.environ.get("PARITY_NUMBA", "0") == "1" else "Python reference"
    print(f"test_current_policy_legacy_parity.py passed ({scanner})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
