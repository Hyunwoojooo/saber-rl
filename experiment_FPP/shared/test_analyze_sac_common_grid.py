from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiment_FPP.shared import analyze_sac_common_grid as analysis


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "step",
        "train/episode_success",
        analysis.PRIMARY_SUCCESS_FIELD,
        analysis.FALLBACK_SUCCESS_FIELD,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _per_seed_row(
    method_key: str,
    method: str,
    seed: int,
    auc: float,
    hit90: int | None,
) -> dict[str, object]:
    return {
        "method_key": method_key,
        "method": method,
        "seed": seed,
        "common_grid_auc": auc,
        "common_grid_final_success": auc + 0.1,
        "common_grid_hit90": hit90,
        "native_auc": auc + 0.05,
        "native_minus_common_auc": 0.05,
    }


class CommonGridAnalysisTests(unittest.TestCase):
    def test_known_synthetic_trapezoidal_auc(self) -> None:
        auc = analysis.normalized_trapezoidal_auc(
            steps=(50_000, 100_000, 150_000),
            successes=(0.0, 1.0, 1.0),
        )
        self.assertAlmostEqual(auc, 0.75)

    def test_duplicate_train_and_eval_rows_are_handled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp) / "metrics_selected_10k.csv"
            _write_metrics(
                metrics,
                [
                    {"step": 50_000, "train/episode_success": 0.1},
                    {
                        "step": 50_000,
                        analysis.PRIMARY_SUCCESS_FIELD: 0.25,
                        analysis.FALLBACK_SUCCESS_FIELD: 0.90,
                    },
                    {
                        "step": 50_000,
                        analysis.PRIMARY_SUCCESS_FIELD: 0.25,
                        analysis.FALLBACK_SUCCESS_FIELD: 0.80,
                    },
                    {
                        "step": 100_000,
                        analysis.PRIMARY_SUCCESS_FIELD: "nan",
                        analysis.FALLBACK_SUCCESS_FIELD: 0.75,
                    },
                ],
            )

            grid = analysis.read_evaluation_grid(
                metrics, required_steps=(50_000, 100_000)
            )

            self.assertEqual(grid.values, {50_000: 0.25, 100_000: 0.75})
            self.assertEqual(grid.duplicate_counts[50_000], 2)
            self.assertEqual(
                grid.source_fields[50_000], analysis.PRIMARY_SUCCESS_FIELD
            )
            self.assertEqual(
                grid.source_fields[100_000], analysis.FALLBACK_SUCCESS_FIELD
            )

    def test_missing_common_checkpoint_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp) / "metrics_selected_10k.csv"
            _write_metrics(
                metrics,
                [
                    {
                        "step": 50_000,
                        analysis.PRIMARY_SUCCESS_FIELD: 0.25,
                    }
                ],
            )
            with self.assertRaisesRegex(
                analysis.GridValidationError,
                "missing required common-grid checkpoints: 100000",
            ):
                analysis.read_evaluation_grid(
                    metrics, required_steps=(50_000, 100_000)
                )

    def test_hit90_and_reached_seed_count(self) -> None:
        hits = (100_000, None, 300_000, 200_000, None)
        rows = [
            _per_seed_row("her", "HER", seed, 0.1 * seed, hits[seed - 1])
            for seed in analysis.REQUIRED_SEEDS
        ]

        aggregate = analysis.aggregate_rows(rows)

        self.assertEqual(len(aggregate), 1)
        self.assertEqual(
            aggregate[0]["common_grid_hit90_reached_seed_count"], 3
        )
        self.assertEqual(
            aggregate[0]["common_grid_hit90_median_reached"], 200_000
        )
        self.assertEqual(aggregate[0]["seeds"], "1;2;3;4;5")

    def test_seed_pairing_is_exact(self) -> None:
        rows = []
        for seed in analysis.REQUIRED_SEEDS:
            rows.append(
                _per_seed_row("saber", "HER+SABER", seed, seed / 10.0, 100_000)
            )
            rows.append(_per_seed_row("her", "HER", seed, seed / 100.0, 200_000))
        comparison = analysis.ComparisonSpec("SABER - HER", "saber", "her")

        paired = analysis.build_paired_rows(
            list(reversed(rows)), comparisons=(comparison,)
        )

        self.assertEqual([row["seed"] for row in paired], [1, 2, 3, 4, 5])
        for row in paired:
            seed = int(row["seed"])
            self.assertAlmostEqual(
                float(row["common_grid_auc_difference"]), seed * 0.09
            )

        incomplete = [
            row
            for row in rows
            if not (row["method_key"] == "her" and row["seed"] == 3)
        ]
        with self.assertRaisesRegex(
            analysis.ArtifactValidationError, "missing her/seed=3"
        ):
            analysis.build_paired_rows(incomplete, comparisons=(comparison,))

    def test_bootstrap_is_deterministic(self) -> None:
        differences = (0.05, -0.01, 0.03, 0.08, 0.02)

        first = analysis.paired_percentile_bootstrap(differences)
        second = analysis.paired_percentile_bootstrap(differences)

        self.assertEqual(first, second)
        self.assertEqual(first["resamples"], 10_000)
        self.assertEqual(first["rng_seed"], 27_027)
        self.assertLessEqual(first["ci_lower_95"], first["paired_difference_mean"])
        self.assertGreaterEqual(first["ci_upper_95"], first["paired_difference_mean"])

    def test_input_artifacts_are_not_modified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "synthetic_s1_1000000"
            metrics = run_dir / "metrics_selected_10k.csv"
            summary = run_dir / "summary.json"
            config = run_dir / "config.json"
            _write_metrics(
                metrics,
                [
                    {
                        "step": step,
                        analysis.PRIMARY_SUCCESS_FIELD: index / 20.0,
                    }
                    for index, step in enumerate(analysis.COMMON_GRID_STEPS, start=1)
                ],
            )
            summary.write_text(
                json.dumps({"eval_AUC_success": 0.42}) + "\n",
                encoding="utf-8",
            )
            config.write_text(
                json.dumps({"seeds": {"env_seed": 1}}) + "\n",
                encoding="utf-8",
            )
            inputs = (metrics, summary, config)
            before = {
                path: (path.read_bytes(), path.stat().st_mtime_ns) for path in inputs
            }
            artifact = analysis.RunArtifact(
                condition_key="synthetic",
                method="Synthetic",
                seed=1,
                run_dir=run_dir,
                metrics_path=metrics,
                summary_path=summary,
            )

            row = analysis.analyze_run(artifact)
            aggregate = analysis.aggregate_rows([row])
            output_dir = root / "report"
            analysis.write_outputs(
                output_dir=output_dir,
                specs=(analysis.ConditionSpec("synthetic", "Synthetic", run_dir.parent),),
                per_seed=(row,),
                aggregate=aggregate,
                paired=(),
                bootstrap=(),
                repo_root=root,
                analysis_command="synthetic-test",
            )

            for path in inputs:
                self.assertEqual(path.read_bytes(), before[path][0])
                self.assertEqual(path.stat().st_mtime_ns, before[path][1])
            self.assertTrue((output_dir / "grid_validation.json").is_file())
            self.assertTrue((output_dir / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
