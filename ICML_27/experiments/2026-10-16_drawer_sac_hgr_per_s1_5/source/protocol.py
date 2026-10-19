"""HGR/PER extension of the frozen Drawer SAC pipeline; reference arms are audit-only."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code/src"))
from metacog_lite_v0.config import LiteConfig, SeedConfig

SPEC = json.loads((ROOT / "protocol.json").read_text())
SCHEDULE = json.loads((ROOT / "schedule_spec.json").read_text())
PAIR_DIFFERENCES = {"method", "weight_scope", "lambda_support", "run_name",
                    "numba_support_scan", "support_cache_enabled", "local_scan_cache_enabled"}


def make_config(task, method, seed, device="cuda", smoke=False, output_root=None):
    if task not in SPEC["tasks"] or method not in {"her", "saber", *SPEC["methods"]} or seed not in SPEC["seeds"]:
        raise ValueError(f"Unknown task/method or seed outside the declared cohort {SPEC['seeds']}")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Select cpu or cuda explicitly; no silent device fallback")
    cfg = LiteConfig(**SPEC["common"])
    cfg.task_name = task
    cfg.metaworld_env_name = task.replace("_", "-") + "-v3"
    cfg.env_id = "Meta-World/MT1"
    cfg.method = {"her": "sac", "saber": "sac_saber", "hgr": "sac_hgr", "per": "sac_her_per_full"}[method]
    cfg.lambda_support = 1.0 if method == "saber" else 0.0
    cfg.weight_scope = "none" if method == "her" else "critic_only"
    cfg.support_cache_enabled = cfg.local_scan_cache_enabled = method == "saber"
    cfg.numba_support_scan = method == "saber"
    cfg.per_beta_steps = SPEC["common"]["total_steps"]
    cfg.hgr_beta_anneal_updates = (SPEC["common"]["total_steps"] - SPEC["common"]["warmup_steps"] + 1)
    for key, value in SPEC["runtime"].items():
        setattr(cfg, key, value)
    cfg.device = device
    cfg.seeds = SeedConfig(env_seed=seed, torch_seed=seed, numpy_seed=seed,
                          action_noise_seed=seed, weight_seed=20000 + seed, eval_seed=10000 + seed)
    cfg.run_name = f"{task}_sac_{method}_s{seed}_2m"
    cfg.output_root = str(Path(output_root).resolve() if output_root else ROOT / "runs")
    cfg.experiment_id = SPEC["experiment_id"]
    cfg.validation_only = bool(smoke)
    if smoke:
        if device != "cpu":
            raise ValueError("The short smoke preset is CPU-only")
        cfg.run_name = f"{task}_sac_{method}_s{seed}_smoke"
        cfg.output_root = str(Path(output_root).resolve() if output_root else ROOT / "validation/smoke")
        cfg.total_steps = 600
        cfg.start_steps = cfg.warmup_steps = 256
        cfg.buffer_capacity = 4096
        cfg.eval_freq = cfg.checkpoint_every_steps = cfg.log_every_steps = 200
        cfg.eval_episodes = 2
    cfg.normalize()
    return cfg


def config_pairs():
    for task in SPEC["tasks"]:
        for seed in SPEC["seeds"]:
            yield make_config(task, "her", seed), make_config(task, "saber", seed)


def make_gpu_validation_config(task, method, seed, profile, output_root):
    if profile not in {"gpu_smoke", "stability"}:
        raise ValueError("Unknown GPU validation profile")
    if seed != SCHEDULE["validation_seed"]:
        raise ValueError("GPU validation seed is fixed in advance")
    cfg = make_config(task, method, seed, "cuda")
    cfg.validation_only = True
    cfg.validation_profile = profile
    cfg.run_name = f"{task}_sac_{method}_s{seed}_{profile}"
    cfg.output_root = str(Path(output_root).resolve())
    cfg.total_steps = SCHEDULE[f"{profile}_steps"]
    if profile == "gpu_smoke":
        cfg.start_steps = cfg.warmup_steps = 256
        cfg.buffer_capacity = 4096
        cfg.eval_freq = cfg.checkpoint_every_steps = cfg.log_every_steps = 200
        cfg.eval_episodes = 2
    # Stability retains the production warmup, network, batch, replay capacity,
    # evaluation schedule and checkpoint cadence; only the total horizon differs.
    cfg.normalize()
    return cfg


def validate_parity():
    pairs = []
    for her, saber in config_pairs():
        a, b = her.as_dict(), saber.as_dict()
        differences = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
        if differences != PAIR_DIFFERENCES:
            raise AssertionError(f"Unexpected paired config differences: {differences}")
        pairs.append({"task": her.task_name, "seed": her.seeds.env_seed,
                      "differences": sorted(differences)})
    return pairs


if __name__ == "__main__":
    runs = len(SPEC["tasks"]) * len(SPEC["methods"]) * len(SPEC["seeds"])
    print(json.dumps({"pairs": validate_parity(), "runs": runs,
                      "environment_steps": runs * SPEC["common"]["total_steps"]}, indent=2))
