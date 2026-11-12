"""Fixed before any measurement.

Design: ICML2027/experment_design/04_first_order_usefulness/README_ko.md (early phase, set "a1")
and PROTOCOL_late_ko.md in this folder (late phase, sets "mature" and "ext2m").
The measurement settings below are shared by every set so that results stay comparable.
"""
from pathlib import Path

from .core import A1

# ---------------------------------------------------------------- measurement settings (all sets)
N_BATCHES = 64                 # first-order batches per checkpoint
BATCH_SEED0 = 900_000          # batch b uses seed BATCH_SEED0 + b
KSTEP_SEED0 = 910_000          # K-step batch k uses seed KSTEP_SEED0 + k
K_RECORD = (1, 10, 100)
R1_EPISODES = 20               # reset seeds 50_000 + e
R2_EPISODES = 20               # reset seeds 60_000 + e
R2_QUERIES = 1024
# FetchPickAndPlace episodes are 50 steps (DrawerOpen: 200), so more episodes give the same query counts
FPP_R1_EPISODES = 80           # 4,000 policy queries
FPP_R2_EPISODES = 32           # 32 queries per behaviour episode
MASK_RNG_SEED = 920_000        # Random / TD tie draws in the first-order analysis
BRANCH_RNG_SEED = 930_000      # K-step Random / TD tie draws
H4_MIN_AGREEMENT = 0.80
# A checkpoint whose policy-query RMS residual sqrt(R0) is below this is numerically null on that
# bank: float32 resolves Q ~ -50 only to ~4e-6, so differences between weightings there are
# rounding noise. Such checkpoints are listed but not counted as + or -. (Every early-phase
# checkpoint is far above it: smallest sqrt(R0) there is 0.04.)
FLOOR_RMS = 1e-3

# ---------------------------------------------------------------- set "a1": early phase (10k-500k)
RUNS = A1 / "outputs" / "pilot_01" / "runs"
TRAINED_ARMS = ("saber", "her", "random", "td_topk")      # arm that produced the checkpoint
PRIMARY_ARMS = ("saber", "her")
SEEDS = (101, 102, 103)
STAGES = (10_000, 20_000, 50_000, 100_000, 200_000, 300_000, 400_000, 500_000)
# pre-registered "consistent": same sign in every seed for at least this many of the 8 stages
CONSISTENT_MIN_STAGES = 6


def checkpoint_path(arm: str, seed: int, stage: int) -> Path:
    return RUNS / arm / f"seed_{seed}" / "scheduled_checkpoints" / f"step_{stage:07d}.pt"


# ---------------------------------------------------------------- late phase (existing 2M runs, no new training)
LOCAL = Path.home() / "ICML2027"
MATURE = LOCAL / "draweropen_ddpg_mature_policy_pilot_20261014" / "outputs" / "production"
EXT = LOCAL / "draweropen_ddpg_assignment_extension_20261020" / "outputs" / "production"

MATURE_ARMS = ("saber", "her")
MATURE_SEED = 1
MATURE_STAGES = (500_000, 1_000_000, 1_500_000, 2_000_000)
EXT_ARMS = {"saber": "support", "random": "random", "td_topk": "td_topk"}     # our label -> run folder
EXT_SEEDS = tuple(range(11, 21))
EXT_STAGE = 2_000_000
# pre-registered for ext2m: one run (arm, seed) is one unit; "consistent +/-" when the exact
# one-sided sign test over the counted runs gives p <= LATE_ALPHA (e.g. 15 of 20, 9 of 10).
# Runs that are numerically null or where the two weightings coincide (p = 0 or 1 in every
# batch) are ties and are not counted.
LATE_ALPHA = 0.05
# run outcome strata from the checkpoint's own last evaluation (20 episodes)
SUCCESS_MIN, FAILURE_MAX = 0.9, 0.1


def mature_path(arm: str, stage: int) -> Path:
    return (MATURE / f"{arm}_s01_2m_01" / "training" / f"draweropen_ddpg_{arm}_s01_2m_mature_pilot"
            / f"checkpoint_step_{stage}.pt")


def ext_path(arm: str, seed: int) -> Path:
    return EXT / EXT_ARMS[arm] / f"seed_{seed}" / "attempt_01" / "run" / "checkpoint.pt"


def checkpoints(set_name: str) -> list[tuple[str, int, int, Path]]:
    """(trained arm, seed, stage, path) for every checkpoint of a set, in a fixed order."""
    if set_name == "a1":
        return [(a, s, st, checkpoint_path(a, s, st)) for a in TRAINED_ARMS for s in SEEDS for st in STAGES]
    if set_name == "mature":
        return [(a, MATURE_SEED, st, mature_path(a, st)) for a in MATURE_ARMS for st in MATURE_STAGES]
    if set_name == "ext2m":
        return [(a, s, EXT_STAGE, ext_path(a, s)) for a in EXT_ARMS for s in EXT_SEEDS]
    if set_name in FPP_SETS:
        info = FPP_SETS[set_name]
        return [(a, s, st, fpp_path(set_name, a, s, st)) for a in info["arms"] for s in info["seeds"]
                for st in FPP_STAGES]
    raise ValueError(f"unknown checkpoint set {set_name!r}")


# ---------------------------------------------------------------- FetchPickAndPlace TD3 (existing 1M runs)
REPO = A1.parents[2]
FPP = REPO / "experiment_FPP"
FPP_STAGES_AVAILABLE = tuple(range(50_000, 1_000_001, 50_000))
# measured stages (fixed 2026-11-10 before any FPP measurement): denser early, whole run covered
FPP_STAGES = (50_000, 100_000, 150_000, 200_000, 300_000, 400_000, 500_000, 750_000, 1_000_000)
# pre-registered "consistent": same sign in every seed for at least this many of the 9 stages
FPP_CONSISTENT_MIN_STAGES = 7
RERUN = REPO / "ICML2027" / "experiments" / "fpp_td3_primary_checkpoint_rerun_20261110" / "outputs" / "runs"
FPP_SETS = {
    # paired HER / SABER runs under the inclusive future-goal convention (supplement table,
    # seeds 6-10 half: mean AUC gain +0.011); the only FPP HER/SABER pairs that kept checkpoints
    "fpp_incl": dict(
        code_src=FPP / "td3_inclusive_future_sensitivity_ls512" / "code" / "src",
        runs=FPP / "runs" / "td3_inclusive_future_sensitivity_ls512_seeds6_10",
        arms={"her": "fetch_pick_and_place_td3_her_inclusive_future_ls512_s{seed}_1000000_ls512",
              "saber": "fetch_pick_and_place_td3_saber_inclusive_future_v4_ls512_s{seed}_1000000_ls512"},
        seeds=(6, 7, 8, 9, 10), her_inclusive=True),
    # primary (exclusive-future) HER / SABER, seeds 1-5, retrained on 2026-11-10 only to keep
    # checkpoints; HER and SABER code differ only in config.as_dict, so one package serves both
    "fpp_primary": dict(
        code_src=FPP / "td3_saber_critic_only_v4_ls512" / "code" / "src",
        runs=RERUN,
        arms={"her": "td3_her_baseline_ls512/fetch_pick_and_place_td3_her_baseline_ls512_s{seed}_1000000_ls512",
              "saber": "td3_saber_critic_only_v4_ls512/fetch_pick_and_place_td3_saber_critic_only_v4_ls512_s{seed}_1000000_ls512"},
        seeds=(1, 2, 3, 4, 5), her_inclusive=False),
}


def fpp_path(set_name: str, arm: str, seed: int, stage: int) -> Path:
    info = FPP_SETS[set_name]
    return info["runs"] / info["arms"][arm].format(seed=seed) / f"checkpoint_step_{stage}.pt"


def set_info(set_name: str) -> dict:
    if set_name in FPP_SETS:
        return dict(family="fpp_td3", code_src=FPP_SETS[set_name]["code_src"],
                    her_inclusive=FPP_SETS[set_name]["her_inclusive"])
    return dict(family="drawer_ddpg", code_src=None, her_inclusive=False)


def bank_settings(family: str) -> tuple[int, int, int]:
    """(R1 episodes, R2 episodes, R2 queries); read at call time so validate.py can shrink them."""
    if family == "fpp_td3":
        return FPP_R1_EPISODES, FPP_R2_EPISODES, R2_QUERIES
    return R1_EPISODES, R2_EPISODES, R2_QUERIES


SETS = ("a1", "mature", "ext2m") + tuple(FPP_SETS)
