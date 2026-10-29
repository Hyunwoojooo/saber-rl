#!/usr/bin/env python3
from __future__ import annotations
import csv
import json
import tempfile
from pathlib import Path
import analyze_results as analysis
import build_artifact
import validate_completion
import verify_artifact

def test_compact_seed_directory_discovery() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for seed in analysis.SEEDS:
            run = root / f"seed_{seed}"
            run.mkdir()
            (run / "summary.json").write_text("{}\n", encoding="utf-8")
            (run / "metrics_selected_10k.csv").write_text(
                "step,eval/env_success_mean\n", encoding="utf-8"
            )
        discovered = analysis.discover_runs(root)
        assert list(discovered) == list(analysis.SEEDS)
        assert [discovered[seed].name for seed in analysis.SEEDS] == [
            f"seed_{seed}" for seed in analysis.SEEDS
        ]

def test_canonical_run_name_seed_discovery_is_preserved() -> None:
    run = Path("drawer_open_DDPG-HER-B3-Replay-Action-Query-2M_s4_2m_cuda")
    assert analysis.seed_from_run(run) == 4

def test_standalone_verifier_defaults_to_current_artifact_directory() -> None:
    assert verify_artifact.DEFAULT_ARTIFACT == Path(".")


def _write_post_warmup_row(path: Path, *, critic: str) -> None:
    fields = [
        "step", "config/support_query_mode", "config/weight_scope",
        *validate_completion.REQUIRED_UPDATE_FIELDS,
    ]
    # Preserve first occurrence because loss/critic is already in REQUIRED_UPDATE_FIELDS.
    fields = list(dict.fromkeys(fields))
    row = {field: "1.0" for field in fields}
    row.update(
        {
            "step": "6000",
            "config/support_query_mode": "replay_action",
            "config/weight_scope": "critic_only",
            "train/replay_size": "6000",
            "loss/critic": critic,
            "query/query_bin_matches_current_policy_rate": "0.5",
            "query/query_bin_matches_replay_action_rate": "1.0",
            "query/context_count_eligible_rate": "0.5",
            "query/support_positive_rate": "0.5",
            "query/high_weight_count": "8",
            "query/raw_weight_mean": "1.5",
            "query/normalized_weight_mean": "1.0",
        }
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def _must_raise_value_error(function, *args) -> None:
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_nonfinite_post_warmup_loss_cannot_be_skipped() -> None:
    with tempfile.TemporaryDirectory() as temp:
        metrics = Path(temp) / "metrics.csv"
        _write_post_warmup_row(metrics, critic="")
        _must_raise_value_error(
            validate_completion.validate_metrics, metrics, 5_000
        )
        _must_raise_value_error(build_artifact.query_summary, metrics, 5_000)


def test_analysis_json_csv_numeric_tamper_is_rejected() -> None:
    cases = (
        ([{"method_key": "replay", "seed": 1, "auc": 0.5}], [{"method_key": "replay", "seed": "1", "auc": "0.6"}], ("method_key", "seed"), "per_seed_results"),
        ([{"method_key": "replay", "auc_mean": 0.5}], [{"method_key": "replay", "auc_mean": "0.6"}], ("method_key",), "aggregate_results"),
        ([{"contrast_key": "main_minus_replay", "mean_auc_difference": 0.1}], [{"contrast_key": "main_minus_replay", "mean_auc_difference": "0.2"}], ("contrast_key",), "paired_comparisons"),
        ([{"step": 10_000, "replay_mean_success": 0.3}], [{"step": "10000", "replay_mean_success": "0.4"}], ("step",), "exact_grid_mean_curves"),
    )
    for json_rows, tampered_csv, key_fields, label in cases:
        _must_raise_value_error(
            verify_artifact.crosscheck_json_csv_rows,
            json_rows, tampered_csv, key_fields, label,
        )

def _readme_statistics_fixture():
    aggregate = {}
    lines = []
    for index, method in enumerate(verify_artifact.METHODS, start=1):
        row = {
            "method": f"Method {method}",
            "auc_mean": str(0.1 * index),
            "auc_sample_sd": str(0.01 * index),
            "final_success_mean": str(0.2 * index),
            "final_success_sample_sd": str(0.02 * index),
        }
        aggregate[method] = row
        lines.append(
            f"| {row['method']} | {float(row['auc_mean']):.6f} ± "
            f"{float(row['auc_sample_sd']):.6f} | "
            f"{float(row['final_success_mean']):.6f} ± "
            f"{float(row['final_success_sample_sd']):.6f} |"
        )
    primary = {
        "mean_auc_difference": "0.123",
        "bootstrap_ci95_lower": "-0.010",
        "bootstrap_ci95_upper": "0.250",
        "exact_two_sided_sign_flip_pvalue": "0.125",
    }
    lines.append(
        "Primary Main − Replay AUC difference: 0.123000; "
        "exhaustive paired-bootstrap 95% CI [-0.010000, 0.250000]; "
        "exact two-sided sign-flip p=0.125000."
    )
    return "\n".join(lines), aggregate, {"main_minus_replay": primary}


def test_readme_final_statistic_tamper_is_rejected() -> None:
    readme, aggregate, paired = _readme_statistics_fixture()
    verify_artifact.verify_readme_statistics(readme, aggregate, paired)
    # Alter only replay Final mean; AUC and every primary value remain intact.
    tampered = readme.replace("0.600000 ± 0.060000 |", "0.601000 ± 0.060000 |")
    assert tampered != readme
    _must_raise_value_error(
        verify_artifact.verify_readme_statistics,
        tampered,
        aggregate,
        paired,
    )



def test_summary_exact_grid_tamper_is_rejected() -> None:
    expected = {"auc": 0.4, "final_success": 0.8, "best_success": 1.0, "hit90": 700_000}
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "summary.json"
        summary = {"eval_AUC_success": 0.4, "final_eval_env_success": 0.8, "best_eval_env_success": 1.0, "first_hit_0.90": 700_000}
        path.write_text(json.dumps(summary), encoding="utf-8")
        verify_artifact.verify_summary_record(path, expected)
        summary["eval_AUC_success"] = 0.41
        path.write_text(json.dumps(summary), encoding="utf-8")
        _must_raise_value_error(verify_artifact.verify_summary_record, path, expected)


def test_comparator_upstream_provenance_tamper_is_rejected() -> None:
    packaging_commit = "8" * 40
    upstream_rows = []
    for method in ("her", "main"):
        for seed in verify_artifact.SEEDS:
            for suffix in ("metrics", "summary"):
                upstream_rows.append({"role": f"{method}.seed_{seed}.{suffix}", "source_path": f"original/{method}/seed_{seed}/{suffix}", "size_bytes": 100 + seed, "sha256": ("a" if suffix == "metrics" else "b") * 64})
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp); upstream_path = root / "provenance/comparator_input_manifest.json"; upstream_path.parent.mkdir(parents=True)
        upstream = {"source_commit": packaging_commit, "input_file_count": len(upstream_rows), "inputs": upstream_rows}
        upstream_path.write_text(json.dumps(upstream, sort_keys=True), encoding="utf-8"); upstream_sha = verify_artifact.sha256(upstream_path)
        local_rows = [{"role": row["role"], "packaging_artifact_source_commit": packaging_commit, "original_training_source_commit": None, "original_training_source_commit_status": "unknown_not_recorded", "upstream_input_provenance": dict(row)} for row in upstream_rows]
        local_rows.append({"role": "provenance.comparator_input_manifest", "sha256": upstream_sha, "packaging_artifact_source_commit": packaging_commit})
        input_manifest = {"comparator": {"artifact_git_commit": "93ae5c03633726f8e7783a157cac8516348f088f", "packaging_artifact_source_commit": packaging_commit, "original_training_source_commit": None, "original_training_source_commit_status": "unknown_not_recorded", "upstream_input_manifest_path": "provenance/comparator_input_manifest.json", "upstream_input_manifest_sha256": upstream_sha, "arms_reused": ["her", "main"], "retrained": False, "original_comparator_config_env_available": False}, "inputs": local_rows}
        manifest = {"comparator_packaging_source_commit": packaging_commit}
        verify_artifact.verify_comparator_provenance(root, manifest, input_manifest)
        tampered = json.loads(json.dumps(input_manifest)); tampered["inputs"][0]["upstream_input_provenance"]["sha256"] = "f" * 64
        _must_raise_value_error(verify_artifact.verify_comparator_provenance, root, manifest, tampered)


def test_head_bound_freeze_metadata_tamper_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source = root / "source_snapshot/code.py"
        launcher = root / (
            "source_snapshot/root_launcher/"
            "DDPG-HER-B3-Replay-Action-Query-2M.sh"
        )
        source.parent.mkdir(parents=True)
        launcher.parent.mkdir(parents=True)
        source.write_text("source\n", encoding="utf-8")
        launcher.write_text("#!/bin/sh\n", encoding="utf-8")
        source_sha = verify_artifact.sha256(source)
        launcher_sha = verify_artifact.sha256(launcher)
        launcher_repo_path = (
            "metaworld_Drawer_Open_DDPG/"
            "DDPG-HER-B3-Replay-Action-Query-2M.sh"
        )
        freeze = {
            "schema_version": 2,
            "head_binding_verified": True,
            "recorded_commit_was_current_head_at_write": True,
            "files": [
                {
                    "path": "code.py",
                    "repo_path": (
                        "metaworld_Drawer_Open_DDPG/replay_action_query_2m/"
                        "code.py"
                    ),
                    "role": "experiment_source",
                    "size": source.stat().st_size,
                    "sha256": source_sha,
                    "head_sha256": source_sha,
                    "git_blob_oid": "c" * 40,
                    "head_bound": True,
                }
            ],
            "entrypoints": [
                {
                    "path": launcher_repo_path,
                    "repo_path": launcher_repo_path,
                    "role": "root_launcher",
                    "size": launcher.stat().st_size,
                    "sha256": launcher_sha,
                    "head_sha256": launcher_sha,
                    "git_blob_oid": "d" * 40,
                    "head_bound": True,
                }
            ],
        }
        (root / "frozen_source.sha256").write_text(
            f"{source_sha}  source_snapshot/code.py\n"
            f"{launcher_sha}  source_snapshot/root_launcher/{launcher.name}\n",
            encoding="utf-8",
        )
        verify_artifact.verify_frozen_snapshot(root, freeze)
        tampered = json.loads(json.dumps(freeze))
        tampered["entrypoints"][0]["head_sha256"] = "f" * 64
        _must_raise_value_error(
            verify_artifact.verify_frozen_snapshot, root, tampered
        )

def main() -> int:
    tests = (
        test_compact_seed_directory_discovery,
        test_canonical_run_name_seed_discovery_is_preserved,
        test_standalone_verifier_defaults_to_current_artifact_directory,
        test_nonfinite_post_warmup_loss_cannot_be_skipped,
        test_analysis_json_csv_numeric_tamper_is_rejected,
        test_readme_final_statistic_tamper_is_rejected,
        test_summary_exact_grid_tamper_is_rejected,
        test_comparator_upstream_provenance_tamper_is_rejected,
        test_head_bound_freeze_metadata_tamper_is_rejected,
    )
    for test in tests:
        test()
    print(f"{len(tests)} focused replay-query analysis/artifact tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
