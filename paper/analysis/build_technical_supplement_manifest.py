#!/usr/bin/env python3
"""Bind Technical Supplement summaries to their canonical source artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper" / "supplementary_artifacts" / "technical_supplement"

CANONICAL_SOURCES = (
    "paper/supplementary_artifacts/primary_seed_results/artifact_manifest.json",
    "paper/supplementary_artifacts/action_bin_sensitivity/artifact_manifest.json",
    "paper/supplementary_artifacts/actor_scope_ddpg/artifact_manifest.json",
    "experiment_FPP/paper_artifacts/"
    "fpp_saber_diagnostic_ddpg_td3_sac_ls512_1m/artifact_manifest.json",
    "metaworld_Drawer_Close_DDPG/experiments/"
    "DDPG-HER-B3-Critic-Only-Diagnostic-2M/diagnostic_artifact_all_10/"
    "artifact_manifest.json",
    "metaworld_Drawer_Open_DDPG/paper_artifacts/"
    "drawer_open_ddpg_saber_fixed_target_diagnostic_all_10_2m/"
    "artifact_manifest.json",
    "experiment_FPP/paper_artifacts/"
    "fpp_td3_her_vs_saber_runtime_profile_100k/artifact_manifest.json",
    "experiment_FPP/paper_artifacts/"
    "fpp_td3_her_vs_saber_n10_1m/artifact_manifest.json",
    "experiment_FPP/paper_artifacts/"
    "fpp_td3_inclusive_future_her_sensitivity_1m/artifact_manifest.json",
    "experiment_FPP/paper_artifacts/"
    "fpp_ddpg_aer_noher_seeds1_5_1m/results/method_summary.csv",
    "experiment_FPP/paper_artifacts/"
    "fpp_td3_aer_noher_seeds1_5_1m/analysis/method_summary.csv",
    "experiment_FPP/paper_artifacts/"
    "fpp_sac_aer_noher_seeds1_5_1m/results/method_summary.csv",
    "metaworld_Drawer_Open_DDPG_AER/paper_artifacts/"
    "drawer_open_ddpg_aer_noher_2m/artifact_manifest.json",
    "metaworld_Drawer_Open_TD3_AER/paper_artifacts/"
    "drawer_open_td3_aer_noher_2m/artifact_manifest.json",
    "metaworld_Drawer_Close_DDPG_AER/paper_artifacts/"
    "drawer_close_ddpg_aer_noher_2m/artifact_manifest.json",
    "experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/"
    "common_grid_50k/artifact_manifest.json",
    "metaworld_Drawer_Close_REDQ/paper_artifacts/"
    "drawer_close_redq_her_tqb_utd4_2m/artifact_manifest.json",
    "experiment_FPP/paper_artifacts/"
    "fpp_sac_her_per_style_loss_weighting_ls512_1m/artifact_manifest.json",
)

PENDING = (
    "experiment_FPP/runs/ddpg_saber_actor_critic_v4_ls512/"
    "fetch_pick_and_place_ddpg_saber_actor_critic_v4_ls512_s{seed}_"
    "1000000_ls512/metrics_selected_10k.csv"
)


def digest(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def td_error_sources() -> list[str]:
    sources: list[str] = []
    for seed in range(1, 6):
        sources.append(
            "experiment_FPP/runs/per_style_loss_weighting_ls512/"
            f"fetch_pick_and_place_per_style_loss_weighting_ls512_s{seed}_"
            "1000000_ls512/metrics_selected_10k.csv"
        )
        sources.append(
            "metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-PER-Style-2M/"
            f"drawer_open_DDPG-HER-PER-Style-2M_s{seed}_2m_cuda/metrics.csv"
        )
        sources.append(
            "metaworld_Drawer_Close_DDPG/results_2m/DDPG-HER-PER-Style-2M/"
            f"drawer_close_DDPG-HER-PER-Style-2M_s{seed}_2m_cuda/metrics.csv"
        )
    return sources


def drawerclose_td3_aer_sources() -> list[str]:
    root = ROOT / "metaworld_Drawer_Close_TD3_AER" / "results_2m"
    run_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if len(run_dirs) != 15:
        raise RuntimeError(f"expected 15 DrawerClose TD3 AER runs, found {len(run_dirs)}")

    filenames = (
        "config.json",
        "env_snapshot.json",
        "final_eval_fixed.json",
        "metrics.csv",
        "summary.json",
    )
    sources = ["metaworld_Drawer_Close_TD3_AER/README.md"]
    for run_dir in run_dirs:
        for filename in filenames:
            sources.append(
                str((run_dir / filename).relative_to(ROOT))
            )
    return sources


def main() -> None:
    source_paths = [
        ROOT / path
        for path in (
            *CANONICAL_SOURCES,
            *td_error_sources(),
            *drawerclose_td3_aer_sources(),
        )
    ]
    missing = [str(path.relative_to(ROOT)) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing canonical sources: {missing}")

    source_manifest = {
        "scope": "Technical Supplement derived summaries",
        "sources": [digest(path) for path in sorted(source_paths)],
        "pending": [PENDING.format(seed=seed) for seed in range(1, 6)],
    }
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    files = sorted(
        path for path in OUTPUT.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    artifact_manifest = {
        "root": str(OUTPUT.relative_to(ROOT)),
        "files": [digest(path) for path in files],
    }
    (OUTPUT / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
