from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import build_diagnostic_artifact as artifact


class DiagnosticArtifactBuilderTest(unittest.TestCase):
    def test_default_snapshot_grid_has_9976_rows(self) -> None:
        steps = artifact.expected_snapshot_steps(
            5_000,
            2_000_000,
            200,
        )
        self.assertEqual(len(steps), 9_976)
        self.assertEqual(steps[0], 5_000)
        self.assertEqual(steps[-1], 2_000_000)

    def test_sanitization_removes_local_paths(self) -> None:
        payload = {
            "output_root": "/home/researcher/results",
            "nested": {
                "run_dir": "/tmp/run",
                "python": "/home/researcher/.venv/bin/python",
                "task": "drawer_close",
            },
        }
        sanitized = artifact.sanitize_json_value(payload)

        self.assertNotIn("output_root", sanitized)
        self.assertNotIn("run_dir", sanitized["nested"])
        self.assertEqual(
            sanitized["nested"]["python"],
            "<redacted-local-path>",
        )
        self.assertEqual(sanitized["nested"]["task"], "drawer_close")

    def test_end_to_end_build_is_deterministic_and_rejects_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            input_root = temp_root / "input"
            output_root = temp_root / "output"
            run_dir = input_root / "seed_01_run"
            run_dir.mkdir(parents=True)
            self._write_run(run_dir)

            argv = [
                "build_diagnostic_artifact.py",
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--seeds",
                "1",
                "--expected-final-step",
                "600",
                "--expected-first-diagnostic-step",
                "200",
                "--expected-diagnostic-cadence",
                "200",
                "--expected-snapshot-count",
                "3",
                "--expected-batch-size",
                "32",
            ]
            self._run_main(argv)
            first_manifest = (
                output_root / "artifact_manifest.json"
            ).read_bytes()
            self._run_main(argv)
            second_manifest = (
                output_root / "artifact_manifest.json"
            ).read_bytes()
            self.assertEqual(first_manifest, second_manifest)

            manifest = json.loads(first_manifest)
            manifest_paths = {
                row["path"] for row in manifest["files"]
            }
            self.assertNotIn("artifact_manifest.json", manifest_paths)

            sanitized_config = artifact.read_json(
                output_root
                / "data"
                / "seeds"
                / "seed_01"
                / "config.json"
            )
            self.assertNotIn("output_root", sanitized_config)
            self.assertEqual(
                sanitized_config["python"],
                "<redacted-local-path>",
            )

            metrics_path = run_dir / "metrics.csv"
            rows = artifact.read_csv(metrics_path)
            rows = [row for row in rows if int(row["step"]) != 400]
            artifact.write_csv(
                metrics_path,
                rows,
                list(rows[0].keys()),
            )
            with self.assertRaisesRegex(
                ValueError,
                "diagnostic grid incomplete",
            ):
                self._run_main(argv)

    @staticmethod
    def _run_main(argv: list[str]) -> None:
        with mock.patch.object(sys, "argv", argv):
            with contextlib.redirect_stdout(io.StringIO()):
                result = artifact.main()
        if result != 0:
            raise AssertionError(f"artifact main returned {result}")

    @staticmethod
    def _write_run(run_dir: Path) -> None:
        config = {
            "task_name": "drawer_close",
            "method": "b3_support_runtime_opt",
            "weight_scope": "critic_only",
            "total_steps": 600,
            "batch_size": 32,
            "output_root": "/home/researcher/results",
            "python": "/home/researcher/.venv/bin/python",
            "seeds": {"env_seed": 1},
        }
        summary = {
            "total_steps": 600,
            "run_dir": "/home/researcher/results/seed_01_run",
        }
        env_snapshot = {
            "cwd": "/home/researcher/project",
            "platform": "test",
        }
        for name, payload in (
            ("config.json", config),
            ("summary.json", summary),
            ("env_snapshot.json", env_snapshot),
        ):
            (run_dir / name).write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
        (run_dir / "metrics.jsonl").write_text(
            "{}\n",
            encoding="utf-8",
        )
        (run_dir / "checkpoint.pt").write_bytes(b"fixture")

        sources = [
            source for source, _ in artifact.diagnostic_columns()
        ]
        rows = []
        for episode, step in enumerate(
            (200, 400, 600),
            start=1,
        ):
            row = {source: "0.1" for source in sources}
            row["step"] = str(step)
            row["episode"] = str(episode)
            row["diagnostic/fixed_target_from_pre_update"] = "1"
            row["diagnostic/same_minibatch_one_update"] = "1"
            row["diagnostic/support_positive_rate"] = "0.25"
            row["diagnostic/td_error_top10_count"] = "4"
            row["diagnostic/td_error_top10_fraction"] = "0.125"
            row["diagnostic/td_error_top10_support_rate"] = "0.5"
            row["diagnostic/td_error_top10_enrichment_ratio"] = "2"
            row["diagnostic/td_error_support_pos_mean"] = "0.4"
            row["diagnostic/td_error_support_neg_mean"] = "0.2"
            row["diagnostic/td_error_pre_mean"] = "0.25"
            row["diagnostic/td_error_post_mean"] = "0.20"
            row[
                "diagnostic/residual_reduction_support_pos_mean"
            ] = "0.08"
            row[
                "diagnostic/residual_reduction_support_neg_mean"
            ] = "0.04"
            row[
                "diagnostic/live_target_td_error_post_mean"
            ] = "0.21"
            row[
                "diagnostic/live_target_residual_reduction_mean"
            ] = "0.04"
            for phase in artifact.PHASES:
                row[
                    f"diagnostic/phase_{phase}_batch_rate"
                ] = "0.25"
                row[
                    f"diagnostic/phase_{phase}_support_positive_rate"
                ] = "0.25"
                row[
                    f"diagnostic/phase_{phase}_td_error_mean"
                ] = "0.25"
                row[
                    f"diagnostic/phase_{phase}_residual_reduction_mean"
                ] = "0.05"
            rows.append(row)

        with (run_dir / "metrics.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=sources)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
