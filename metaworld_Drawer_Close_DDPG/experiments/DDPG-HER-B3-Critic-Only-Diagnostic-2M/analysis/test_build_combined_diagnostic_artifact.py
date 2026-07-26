from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import build_combined_diagnostic_artifact as combined


class CombinedDiagnosticArtifactBuilderTest(unittest.TestCase):
    def test_all_ten_build_is_summary_only_and_deterministic(self) -> None:
        experiment_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "combined"
            argv = [
                "build_combined_diagnostic_artifact.py",
                "--experiment-root",
                str(experiment_root),
                "--output-root",
                str(output_root),
            ]
            self._run_main(argv)
            first_manifest = (output_root / "artifact_manifest.json").read_bytes()
            self._run_main(argv)
            second_manifest = (output_root / "artifact_manifest.json").read_bytes()
            self.assertEqual(first_manifest, second_manifest)

            self.assertFalse((output_root / "data").exists())
            self.assertFalse(
                list(output_root.rglob("diagnostic_snapshots.csv.gz"))
            )

            validation = json.loads(
                (output_root / "validation" / "validation_summary.json").read_text()
            )
            self.assertTrue(validation["all_checks_passed"])
            self.assertEqual(validation["observed_seeds"], list(range(1, 11)))
            self.assertEqual(validation["phase_window_rows"], 30)
            self.assertEqual(
                validation["paper_summary_direction_counts"],
                {
                    "one_update_reduction_positive_greater": 10,
                    "pre_update_td_positive_greater": 10,
                },
            )

            summary = json.loads(
                (output_root / "results" / "paper_summary.json").read_text()
            )
            self.assertAlmostEqual(
                summary["support_positive_rate"]["mean"],
                0.2982370971581796,
            )
            self.assertAlmostEqual(
                summary["top10_td_error_support_enrichment"]["mean"],
                1.5682066172529907,
            )
            self.assertAlmostEqual(
                summary["fixed_target_one_update_residual_reduction"][
                    "positive_to_negative_ratio"
                ],
                3.4760770003146657,
            )

            provenance = json.loads(
                (output_root / "provenance.json").read_text()
            )
            self.assertEqual(len(provenance["source_artifacts"]), 2)
            self.assertEqual(set(provenance["seed_sources"]), {
                str(seed) for seed in range(1, 11)
            })

    @staticmethod
    def _run_main(argv: list[str]) -> None:
        with mock.patch.object(sys, "argv", argv):
            with contextlib.redirect_stdout(io.StringIO()):
                result = combined.main()
        if result != 0:
            raise AssertionError(f"combined artifact main returned {result}")


if __name__ == "__main__":
    unittest.main()
