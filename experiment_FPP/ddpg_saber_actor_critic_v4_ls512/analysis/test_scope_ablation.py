#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_artifact import build_artifact  # noqa: E402
from scope_ablation import (  # noqa: E402
    ACTOR_KEY,
    CRITIC_KEY,
    EXPECTED_CONFIG,
    GRID,
    MissingPairedRunsError,
    AnalysisError,
    analyze_scope_ablation,
    exact_two_sided_sign_flip_p,
    normalized_trapezoid_auc,
    paired_bootstrap_ci,
)
from verify_artifact import ArtifactVerificationError, verify_artifact  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def base_config(seed: int, scope: str, run_name: str, output_root: Path) -> dict[str, object]:
    return {
        **EXPECTED_CONFIG,
        "smoke": False,
        "weight_scope": scope,
        "run_name": run_name,
        "output_root": str(output_root),
        "seeds": {
            "env_seed": seed,
            "torch_seed": seed,
            "numpy_seed": seed,
            "action_noise_seed": seed,
            "eval_seed": 10_000 + seed,
            "weight_seed": 20_000 + seed,
        },
    }


def env_snapshot() -> dict[str, object]:
    environment = {
        "env_id": "FetchPickAndPlace-v4",
        "entry_point": "gymnasium_robotics.envs.fetch.pick_and_place:MujocoFetchPickAndPlaceEnv",
        "obs_dim": 25,
        "goal_dim": 3,
        "state_dim": 28,
        "action_dim": 4,
        "action_low": [-1.0, -1.0, -1.0, -1.0],
        "action_high": [1.0, 1.0, 1.0, 1.0],
        "max_episode_steps": 50,
        "distance_threshold": 0.05,
        "spec_kwargs": {"reward_type": "sparse"},
    }
    return {
        "train_env": dict(environment),
        "eval_env": dict(environment),
        "task_spec": {
            "task": "FetchPickAndPlace",
            "epsilon_g": 0.05,
            "d_reach": 0.08,
            "d_goal_near": 0.05,
            "z_lift_min": 0.48,
            "grip_close_thresh": 0.04,
        },
    }


def write_execution_metadata(actor_root: Path, commit: str) -> None:
    write_json(
        actor_root / "_execution/launch_metadata.json",
        {
            "schema_version": 1,
            "experiment": "ddpg_saber_actor_critic_v4_ls512",
            "timestamp_utc": "2026-07-20T12:34:56Z",
            "repo_head": commit,
            "origin_main": commit,
            "head_matches_origin_main": True,
            "worktree_clean": True,
            "steps": 1_000_000,
            "seeds": [1, 2, 3, 4, 5],
            "max_parallel": 2,
        },
    )


def write_run(root: Path, scope: str, seed: int, curve: list[float], *, raw: bool) -> Path:
    slug = "ddpg_saber_actor_critic_v4_ls512" if scope == ACTOR_KEY else "b3_critic_only_runtime_opt_v4_ls512"
    run_name = f"fetch_pick_and_place_{slug}_s{seed}_1000000_ls512"
    run_dir = root / run_name
    run_dir.mkdir(parents=True)
    config = base_config(seed, scope, run_name, root)
    auc = normalized_trapezoid_auc(curve)
    hit = next((step for step, value in zip(GRID, curve) if value >= 0.9), None)
    summary = {
        "method": "b3_support_runtime_opt",
        "total_steps": 1_000_000,
        "total_updates": 995_001,
        "final_eval_success": curve[-1],
        "best_eval_success": max(curve),
        "eval_AUC_success": auc,
        "first_hit_0.90": "" if hit is None else str(hit),
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "env_snapshot.json", env_snapshot())
    metrics = run_dir / ("metrics.csv" if raw else "metrics_selected_10k.csv")
    fields = [
        "step",
        "eval/return_mean",
        "eval/success_mean",
        "eval/goal_success_mean",
        "eval/env_success_mean",
        "eval/best_success",
        "eval/AUC_success",
        "eval/first_hit_0.90",
    ]
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, (step, value) in enumerate(zip(GRID, curve)):
            if raw:
                writer.writerow({"step": step})  # non-evaluation training row
            prefix_auc = normalized_trapezoid_auc(curve) if step == GRID[-1] else ""
            writer.writerow(
                {
                    "step": step,
                    "eval/return_mean": -50.0 * (1.0 - value),
                    "eval/success_mean": value,
                    "eval/goal_success_mean": value,
                    "eval/env_success_mean": value,
                    "eval/best_success": max(curve[: index + 1]),
                    "eval/AUC_success": prefix_auc,
                    "eval/first_hit_0.90": "" if hit is None else hit,
                }
            )
    return run_dir


class SyntheticFixture:
    def __init__(self, root: Path) -> None:
        self.repo = root
        (root / "version_v0_lite").mkdir(parents=True)
        self.actor = root / "experiment_FPP/runs/ddpg_saber_actor_critic_v4_ls512"
        self.critic = root / "experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512"
        deltas = (0.04, 0.02, 0.00, -0.01, 0.03)
        for seed, delta in zip(range(1, 6), deltas):
            critic_curve = [min(0.94, 0.05 + index * 0.009 + seed * 0.002) for index in range(100)]
            actor_curve = [min(1.0, max(0.0, value + delta)) for value in critic_curve]
            write_run(self.critic, CRITIC_KEY, seed, critic_curve, raw=False)
            write_run(self.actor, ACTOR_KEY, seed, actor_curve, raw=True)
        write_execution_metadata(self.actor, "a" * 40)
        self.source = root / "experiment_FPP/ddpg_saber_actor_critic_v4_ls512"
        write_json(
            self.source / "config.json",
            {
                "trainer_method": "b3_support_runtime_opt",
                "weight_scope": "actor_critic",
                "weighting_mode": "batch_norm",
                "local_search_max": 512,
                "total_steps": 1_000_000,
                "seeds": [1, 2, 3, 4, 5],
            },
        )
        (self.source / "run.py").write_text("print('synthetic runner')\n", encoding="utf-8")
        trainer = self.source / "code/src/train_fetch_pick_and_place_lite_v0.py"
        trainer.parent.mkdir(parents=True)
        trainer.write_text("# synthetic frozen trainer\n", encoding="utf-8")
        (self.source / "run_smoke_20k_seed1.sh").write_text(
            "#!/usr/bin/env bash\n# executable smoke launcher, not a result payload\n",
            encoding="utf-8",
        )


class ScopeAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = SyntheticFixture(Path(self.temp.name))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_exact_grid_auc_and_paired_statistics(self) -> None:
        self.assertAlmostEqual(normalized_trapezoid_auc([0.4] * 100), 0.4)
        bundle = analyze_scope_ablation(self.fixture.actor, self.fixture.critic)
        self.assertEqual(len(bundle.per_seed_rows), 10)
        self.assertEqual(len(bundle.paired_rows), 5)
        auc_stats = next(row for row in bundle.paired_stat_rows if row["metric"] == "success_auc")
        self.assertEqual(auc_stats["n_pairs"], 5)
        self.assertEqual(auc_stats["positive_count"], 3)
        self.assertEqual(auc_stats["negative_count"], 1)
        self.assertEqual(auc_stats["zero_count"], 1)
        self.assertEqual(auc_stats["bootstrap_resamples"], 10_000)

    def test_bootstrap_is_fixed_and_sign_flip_is_exact(self) -> None:
        values = [0.1] * 5
        self.assertEqual(paired_bootstrap_ci(values), paired_bootstrap_ci(values))
        self.assertEqual(paired_bootstrap_ci(values), (0.1, 0.1))
        self.assertAlmostEqual(exact_two_sided_sign_flip_p(values), 2.0 / 32.0)

    def test_missing_pair_fails_explicitly(self) -> None:
        shutil.rmtree(next(self.fixture.actor.glob("*_s5_1000000_ls512")))
        with self.assertRaisesRegex(MissingPairedRunsError, r"actor_critic seeds=\[5\]"):
            analyze_scope_ablation(self.fixture.actor, self.fixture.critic)

    def test_config_parity_rejects_a_second_factor(self) -> None:
        run = next(self.fixture.actor.glob("*_s1_1000000_ls512"))
        config = json.loads((run / "config.json").read_text())
        config["extra_factor"] = 0.95
        write_json(run / "config.json", config)
        with self.assertRaisesRegex(AnalysisError, "config parity failed"):
            analyze_scope_ablation(self.fixture.actor, self.fixture.critic)

    def test_missing_grid_point_is_not_interpolated(self) -> None:
        run = next(self.fixture.actor.glob("*_s1_1000000_ls512"))
        path = run / "metrics.csv"
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        fields = list(rows[0])
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(row for row in rows if not (row["step"] == "500000" and row["eval/success_mean"]))
        with self.assertRaisesRegex(AnalysisError, "no interpolation"):
            analyze_scope_ablation(self.fixture.actor, self.fixture.critic)

    def test_compact_artifact_build_verify_and_tamper_detection(self) -> None:
        artifact = self.fixture.repo / "experiment_FPP/paper_artifacts/fpp_ddpg_saber_actor_critic_v4_ls512_1m"
        result = build_artifact(
            repo=self.fixture.repo,
            actor_root=self.fixture.actor,
            critic_root=self.fixture.critic,
            artifact_root=artifact,
            source_commit="a" * 40,
            experiment_source_root=self.fixture.source,
            analysis_source_root=HERE,
        )
        self.assertEqual(result["validation_gates"], "verified")
        self.assertTrue((artifact / "validation.json").is_file())
        self.assertTrue((artifact / "execution_metadata.json").is_file())
        self.assertTrue((artifact / "source/config.json").is_file())
        self.assertTrue((artifact / "source/run_smoke_20k_seed1.sh").is_file())
        mean_curve = artifact / "analysis/mean_learning_curve.csv"
        with mean_curve.open(encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 200)
        self.assertFalse(any(path.suffix == ".pt" for path in artifact.rglob("*")))
        self.assertEqual(verify_artifact(artifact)["frozen_source"], "verified")

        analysis_json = artifact / "analysis/analysis_results.json"
        analysis_json.write_text(analysis_json.read_text() + " ", encoding="utf-8")
        with self.assertRaisesRegex(ArtifactVerificationError, "SHA256"):
            verify_artifact(artifact)

    def test_forbidden_checkpoint_is_rejected(self) -> None:
        artifact = self.fixture.repo / "experiment_FPP/paper_artifacts/fpp_ddpg_saber_actor_critic_v4_ls512_1m"
        write_execution_metadata(self.fixture.actor, "b" * 40)
        build_artifact(
            repo=self.fixture.repo,
            actor_root=self.fixture.actor,
            critic_root=self.fixture.critic,
            artifact_root=artifact,
            source_commit="b" * 40,
            experiment_source_root=self.fixture.source,
            analysis_source_root=HERE,
        )
        (artifact / "checkpoint.pt").write_bytes(b"forbidden")
        with self.assertRaisesRegex(ArtifactVerificationError, "checkpoint"):
            verify_artifact(artifact)

    def test_wrong_total_updates_is_rejected_from_summary(self) -> None:
        run = next(self.fixture.actor.glob("*_s1_1000000_ls512"))
        summary = json.loads((run / "summary.json").read_text())
        summary["total_updates"] = 995_000
        write_json(run / "summary.json", summary)
        with self.assertRaisesRegex(AnalysisError, "total_updates=995000"):
            analyze_scope_ablation(self.fixture.actor, self.fixture.critic)

    def test_environment_identity_is_fail_closed(self) -> None:
        run = next(self.fixture.actor.glob("*_s1_1000000_ls512"))
        snapshot = json.loads((run / "env_snapshot.json").read_text())
        snapshot["train_env"]["action_dim"] = 3
        write_json(run / "env_snapshot.json", snapshot)
        with self.assertRaisesRegex(AnalysisError, "env_snapshot.train_env.action_dim"):
            analyze_scope_ablation(self.fixture.actor, self.fixture.critic)

    def test_execution_source_commit_mismatch_fails_build(self) -> None:
        artifact = self.fixture.repo / "experiment_FPP/paper_artifacts/fpp_ddpg_saber_actor_critic_v4_ls512_1m"
        with self.assertRaisesRegex(AnalysisError, "repo_head expected"):
            build_artifact(
                repo=self.fixture.repo,
                actor_root=self.fixture.actor,
                critic_root=self.fixture.critic,
                artifact_root=artifact,
                source_commit="c" * 40,
                experiment_source_root=self.fixture.source,
                analysis_source_root=HERE,
            )


if __name__ == "__main__":
    unittest.main()
