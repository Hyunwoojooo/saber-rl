"""A1 exploratory pilot; none of these functions starts training."""
import copy
import json
import random
from bootstrap import ROOT
from metacog_lite_v0.config import LiteConfig, SeedConfig

ARMS = ("her", "saber", "random", "td_topk")
SEEDS = (101, 102, 103)
RULE = dict(her="uniform", saber="support", random="random", td_topk="td_topk")
PROFILES = {"pilot": 500_000, "smoke": 10_200}
CHECKPOINTS = (0, 4800, 5000, 6000, 8000, 10000, 20000, 50000,
               100000, 200000, 300000, 400000, 500000)
PROBE_SIZE = 32
INDIVIDUAL_GRADIENT_SAMPLES = 8
ORDER_SEED = 20261031


def make_config(arm, seed, output, profile="pilot"):
    if arm not in ARMS or seed not in SEEDS or profile not in PROFILES:
        raise ValueError("Arm, seed or profile outside the declared pilot")
    data = json.loads((ROOT / "config_template.json").read_text())
    resolved_env = data.pop("resolved_env_id")  # as_dict-derived metadata, not a constructor field
    data.update(total_steps=PROFILES[profile], device="cpu", output_root=str(output),
                run_name="run", assignment_rule=RULE[arm], save_step_checkpoints=False)
    data["seeds"] = SeedConfig(env_seed=seed, torch_seed=seed, numpy_seed=seed,
        action_noise_seed=seed, weight_seed=20000+seed, assignment_seed=30000+seed,
        eval_seed=10000+seed)
    cfg = LiteConfig(**data)
    requested = copy.deepcopy(cfg.as_dict())
    if requested["resolved_env_id"] != resolved_env:
        raise RuntimeError("Resolved environment changed from the executed template")
    cfg.normalize()
    if requested != cfg.as_dict():
        raise RuntimeError("normalize changed the explicit experimental configuration")
    if cfg.start_steps != 10000 or cfg.warmup_steps != 5000:
        raise RuntimeError("Smoke must preserve the random-prefix and learning-start rules")
    return cfg


def checkpoints(total):
    return sorted({s for s in CHECKPOINTS if s <= total} | {total})


def probe_due(step, total):
    return (step in checkpoints(total) or
            (5000 <= step <= 10000 and step % 100 == 0) or
            (step > 10000 and step % 1000 == 0))


def schedule():
    # Three rows of a randomized cyclic Latin square: each arm occupies
    # each launch position at most once. Seeds form complete execution blocks.
    rng = random.Random(ORDER_SEED)
    labels = list(ARMS)
    rng.shuffle(labels)
    offsets = list(range(4))
    rng.shuffle(offsets)
    rows = []
    for seed, offset in zip(SEEDS, offsets):
        for position in range(4):
            rows.append(dict(seed=seed, arm=labels[(position+offset) % 4],
                             block=seed, position=position+1))
    return rows

