#!/usr/bin/env python3
from __future__ import annotations

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


def main() -> int:
    fixture_root = Path(__file__).resolve().parent
    experiment_root = fixture_root.parents[2]
    baseline_root = experiment_root / "b3_critic_only_runtime_opt_v4_ls512" / "code" / "src"
    if not baseline_root.is_dir():
        raise FileNotFoundError(f"baseline source root missing: {baseline_root}")

    legacy = run_fixture(baseline_root, fixture_root)
    current = run_fixture(fixture_root, fixture_root)
    if current != legacy:
        raise AssertionError("current_policy weight arrays or weight_rng state differ from the legacy baseline")
    scanner = "Numba predicate" if os.environ.get("PARITY_NUMBA", "0") == "1" else "Python reference"
    print(f"test_current_policy_legacy_parity.py passed ({scanner})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
