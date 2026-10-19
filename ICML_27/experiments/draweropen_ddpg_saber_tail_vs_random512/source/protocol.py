"""Frozen online retrieval comparison, with audited historical tail references."""
from base_protocol import ROOT, make_config as base_config, evaluation_seed

ARMS = ("tail", "random512")
SEEDS = tuple(range(1, 11))
BOOTSTRAP_SEED = 20261004
BOOTSTRAP_RESAMPLES = 10_000
ANALYSIS_VERSION = "retrieval-v1"
ORIGIN = ROOT.parent / "2026-09-25_draweropen_ddpg_assignment"
ENV_FIELDS = ("python", "packages", "torch_cuda_build", "cudnn", "gpu", "threads",
              "host", "platform", "architecture", "nvidia_driver")
SCAN_MODES = {"tail": "deterministic_tail_v1", "random512": "uniform_random_update_v1"}


def make_config(arm, seed, output_root, profile="production", device="cuda"):
    if arm not in ARMS:
        raise ValueError("Unknown retrieval arm")
    cfg = base_config("support", seed, output_root, profile, device)
    cfg.support_scan_mode = SCAN_MODES[arm]
    cfg.normalize()
    return cfg
