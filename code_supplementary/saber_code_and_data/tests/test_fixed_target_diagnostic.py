from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source/drawer_open_fixed_target_diagnostic"
MODULE = "test_drawer_open_fixed_target_diagnostic.DrawerOpenFixedTargetDiagnosticContractTest"
METHODS = (
    "test_metrics_use_fixed_target_and_preserve_live_target_audit",
    "test_top10_uses_exact_stable_k_when_td_errors_tie",
    "test_supplied_pre_update_target_is_used_without_mutation",
    "test_instrumentation_does_not_change_parameter_update",
)


class FixedTargetDiagnosticTest(unittest.TestCase):
    def test_fixed_target_contract_suite(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SOURCE)
        command = [
            sys.executable,
            "-m",
            "unittest",
            *(f"{MODULE}.{method}" for method in METHODS),
        ]
        result = subprocess.run(
            command,
            cwd=SOURCE,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
