"""CPU-only, read-only checkpoint audit. Never takes environment/training steps."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

from common import (ROOT, ASSIGNMENT, ASSIGNMENT_SHA, SEEDS, QUERIES, RANDOM_REPEATS,
                    QUERY_SEED, RETRIEVAL_SEED, SCAN_LIMIT, TOTAL_STEPS, TOTAL_UPDATES,
                    completed_attempt, digest, read_json, write_json, write_csv,
                    output_directory, finish_artifact, verify_source)


def restore(seed):
    import numpy as np
    import torch
    from metacog_lite_v0.config import LiteConfig, SeedConfig
    from metacog_lite_v0.agent import DDPGAgent
    from metacog_lite_v0.replay import HERReplayBuffer
    from metacog_lite_v0.task_drawer_open import MetaWorldDrawerOpenTaskSpec

    attempt = completed_attempt(ASSIGNMENT, "support", seed, ASSIGNMENT_SHA)
    run = attempt / "run"
    paths = [attempt / "status.json", run / "config.json", run / "env_snapshot.json",
             run / "summary.json", run / "checkpoint.pt"]
    inputs = {str(p): digest(p) for p in paths}
    saved = read_json(run / "config.json")
    names = {field.name for field in dataclasses.fields(LiteConfig)}
    unknown = set(saved) - names - {"resolved_env_id", "terminal_convention", "assignment_log_enabled"}
    if unknown:
        raise ValueError(f"Unrecognized saved configuration fields: {unknown}")
    values = {k: v for k, v in saved.items() if k in names}
    values["seeds"] = SeedConfig(**values["seeds"])
    cfg = LiteConfig(**values)
    required = {"task_name": "drawer_open", "method": "b3_support_runtime_opt",
                "assignment_experiment": True, "assignment_rule": "support",
                "support_assignment_mode": "support", "support_scan_mode": "deterministic_tail_v1",
                "support_query_mode": "current_policy", "policy_support_mode": "bin",
                "N_local_min": 4, "M_Q_min": 4, "M_B": 16, "local_search_max": 512,
                "action_bins_per_dim": 5, "lambda_support": 1.0, "her_ratio": .8,
                "weight_scope": "critic_only", "weighting_mode": "batch_norm"}
    if any(getattr(cfg, k) != v for k, v in required.items()):
        raise ValueError("Unexpected reference training/support definition")
    # Only a trusted, local checkpoint from this frozen run is deserialized.
    payload = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    if (payload.get("global_step") != TOTAL_STEPS or payload.get("total_updates") != TOTAL_UPDATES
            or payload.get("assignment_rule") != "support"
            or payload.get("checkpoint_continuation_permitted") is not False
            or json.loads(json.dumps(payload["config"])) != saved):
        raise ValueError("Final checkpoint identity/configuration mismatch")
    rp = payload["replay"]
    if "action_index" not in rp or rp["size"] != cfg.buffer_capacity:
        raise ValueError("Complete retained replay and index are required")
    replay = HERReplayBuffer(capacity=rp["capacity"], obs_dim=rp["obs_dim"],
                             goal_dim=rp["goal_dim"], action_dim=rp["action_dim"])
    replay.load_state_dict(rp)
    snapshot = read_json(run / "env_snapshot.json")
    env = snapshot["train_env"]
    task = MetaWorldDrawerOpenTaskSpec(
        action_low=np.asarray(env["action_low"], dtype=np.float32),
        action_high=np.asarray(env["action_high"], dtype=np.float32),
        epsilon_g=env["distance_threshold"], d_reach=cfg.drawer_d_reach,
        d_goal_near=cfg.drawer_d_goal_near, d_opening_mid=cfg.drawer_d_opening_mid,
        hand_handle_bins=tuple(cfg.drawer_hand_handle_bins), handle_goal_bins=tuple(cfg.drawer_handle_goal_bins),
        handle_y_bins=tuple(cfg.drawer_handle_y_bins), gripper_bins=tuple(cfg.drawer_gripper_bins),
        horizon_bins=tuple(cfg.drawer_horizon_bins))
    if json.loads(json.dumps(task.metadata())) != snapshot["task_spec"]:
        raise ValueError("Task reconstruction differs from the saved task specification")
    for array in (replay.observation, replay.desired_goal, replay.action_exec):
        if not np.isfinite(array[:replay.size]).all():
            raise ValueError("Non-finite retained replay")
    expected_keys = task.coarse_hash_batch(replay.observation[:replay.size], replay.h[:replay.size])
    if not np.array_equal(expected_keys, replay.action_index._key_by_slot[:replay.size]):
        raise ValueError("Coarse index is incomplete or inconsistent with replay")
    expected_bins = task.action_bin_batch(replay.action_exec[:replay.size], cfg.action_bins_per_dim)
    if not np.array_equal(expected_bins, replay.action_bin[:replay.size]):
        raise ValueError("Retained action bins differ from the frozen definition")
    for key in np.unique(expected_keys):
        # The full key contains the coarse key's components. Validate the entire
        # index before using exhaustive bucket matching as a full-replay oracle.
        actual = replay.action_index.candidates(int(key), replay.size)
        if not np.array_equal(np.sort(actual), np.flatnonzero(expected_keys == key)):
            raise ValueError("Index membership differs from full replay")
    cfg.device = "cpu"
    agent = DDPGAgent(state_dim=env["state_dim"], action_dim=env["action_dim"],
                      hidden_dim=cfg.hidden_dim, actor_lr=cfg.actor_lr, critic_lr=cfg.critic_lr,
                      gamma=cfg.gamma, tau=cfg.tau, action_low=task.action_low,
                      action_high=task.action_high, device=torch.device("cpu"),
                      l2_action_penalty=cfg.l2_action_penalty)
    agent.load_state_dict(payload["agent"])
    for module in (agent.actor, agent.actor_target, agent.critic, agent.critic_target):
        module.eval()
    return cfg, task, replay, agent, inputs


def run_seed(seed, count, destination):
    import numpy as np
    import torch
    from metacog_lite_v0.weights import LiteWeightComputer, GapStats
    from retrieval_core import choose_candidates, scan_reference, summarize_method

    start = time.perf_counter()
    cfg, task, replay, agent, inputs = restore(seed)
    initial_updates = agent.total_it
    initial_actor = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    sample_rng = np.random.default_rng(QUERY_SEED + seed)
    gap = GapStats(cfg.gap_window, cfg.N_gap_min, cfg.gap_q_low, cfg.gap_q_high, cfg.eps_G)
    computer = LiteWeightComputer(cfg, task, gap)
    if not computer._numba_support_scan_enabled:
        raise RuntimeError("Frozen optimized scanner unavailable; do not silently change parity reference")
    rows, queries, samples, expected = [], [], [], []
    parity_seconds = 0.0
    for block in range(count // cfg.batch_size):
        batch = replay.sample(cfg.batch_size, cfg.her_ratio, sample_rng)
        x = np.concatenate([batch.observation, batch.desired_goal_train], axis=1).astype(np.float32)
        with torch.no_grad():
            action = agent.policy_actions(torch.from_numpy(x)).cpu().numpy().astype(np.float32)
        if not np.isfinite(action).all():
            raise ValueError("Non-finite policy action")
        before = time.perf_counter()
        baseline = computer.compute(batch, replay, agent, np.random.default_rng(QUERY_SEED + seed + block),
                                    torch.device("cpu"), assignment_rng=np.random.default_rng(seed))
        parity_seconds += time.perf_counter() - before
        if not computer._numba_support_scan_enabled:
            raise RuntimeError("Optimized reference fell back during parity check")
        policy_bins = task.action_bin_batch(action, cfg.action_bins_per_dim)
        full_keys = task.full_context_hash_batch(batch.observation, batch.desired_goal_train, batch.h)
        coarse_keys = task.coarse_hash_batch(batch.observation, batch.h)
        raw = 1 + baseline.support_mask.astype(np.float32) * cfg.lambda_support
        if (not np.array_equal(baseline.w_p_raw, raw)
                or not np.allclose(baseline.omega_critic, raw / raw.mean(), atol=2e-6)
                or not np.array_equal(baseline.omega_actor, np.ones(cfg.batch_size))):
            raise ValueError("Frozen mean-one critic-only weight definition did not reproduce")
        for i in range(cfg.batch_size):
            index = block * cfg.batch_size + i
            queries.append({"seed": seed, "query": index, "batch": block, "slot": int(batch.indices[i]),
                            "horizon": int(batch.h[i]), "full_key": int(full_keys[i]),
                            "coarse_key": int(coarse_keys[i]), "policy_action_bin": int(policy_bins[i]),
                            "goal_x": float(batch.desired_goal_train[i, 0]),
                            "goal_y": float(batch.desired_goal_train[i, 1]),
                            "goal_z": float(batch.desired_goal_train[i, 2]),
                            "realized_batch_her_ratio": batch.her_relabel_ratio,
                            "phase": int(task.phase_batch(batch.observation[i:i+1],
                                                         batch.desired_goal_train[i:i+1])[0])})
            samples.append((batch.desired_goal_train[i].copy(), full_keys[i], coarse_keys[i], policy_bins[i]))
            expected.append((bool(baseline.context_eligible_mask[i]), bool(baseline.support_mask[i])))
    setup_seconds = time.perf_counter() - start
    options = [("tail", 0), ("full", 0)] + [("random", r) for r in range(RANDOM_REPEATS)]
    for index, (goal, full_key, coarse_key, policy_bin) in enumerate(samples):
        # Rotate order without changing either HER queries or random candidate sets.
        order = options[index % len(options):] + options[:index % len(options)]
        outcomes = {}
        for mode, repeat in order:
            rng = np.random.default_rng(np.random.SeedSequence([RETRIEVAL_SEED, seed, index, repeat]))
            t0 = time.perf_counter()
            candidates = replay.action_index.candidates(int(coarse_key), replay.size)
            selected = choose_candidates(candidates, mode, SCAN_LIMIT, rng)
            result = scan_reference(task, replay, selected, goal, int(full_key), int(policy_bin), cfg)
            elapsed = time.perf_counter() - t0
            row = {"seed": seed, "query": index, "method": mode, "repeat": repeat,
                   "coarse_count": len(candidates), "scanned_count": len(selected),
                   "truncated": mode != "full" and len(candidates) > SCAN_LIMIT,
                   **result, "reference_lookup_seconds": elapsed}
            rows.append(row)
            outcomes[mode, repeat] = result
        if (outcomes["tail", 0]["eligible"], outcomes["tail", 0]["support"]) != expected[index]:
            raise ValueError(f"Tail parity mismatch for seed {seed}, query {index}")
        if index < 16:
            direct = scan_reference(task, replay, np.arange(replay.size), goal, int(full_key), int(policy_bin), cfg)
            if direct != outcomes["full", 0]:
                raise ValueError("Exhaustive bucket result differs from direct full-replay matching")
        if (index + 1) % 256 == 0:
            print(f"retrieval seed {seed}: {index+1}/{count} queries", flush=True)
    if agent.total_it != initial_updates or any(not torch.equal(v, initial_actor[k])
                                               for k, v in agent.actor.state_dict().items()):
        raise ValueError("Read-only diagnostic changed the actor/update state")
    full = {r["query"]: r for r in rows if r["method"] == "full"}
    tail = {r["query"]: r for r in rows if r["method"] == "tail"}
    summaries, repeats = [], []
    for mode in ("tail", "random", "full"):
        chosen = [r for r in rows if r["method"] == mode]
        summaries.append({"seed": seed, "method": mode, **summarize_method(chosen, full, tail)})
        for repeat in sorted({r["repeat"] for r in chosen}):
            repeats.append({"seed": seed, "method": mode, "repeat": repeat,
                            **summarize_method([r for r in chosen if r["repeat"] == repeat], full, tail)})
    write_csv(destination / f"queries_seed_{seed:02d}.csv", queries)
    write_csv(destination / f"measurements_seed_{seed:02d}.csv", rows)
    return summaries, repeats, inputs, {"seed": seed, "queries": count,
            "tail_parity_mismatches": 0, "direct_full_replay_crosschecks": min(count, 16),
            "setup_including_jit_seconds": setup_seconds,
            "frozen_optimized_weight_compute_seconds": parity_seconds,
            "total_seconds": time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("pilot", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # Explicit CPU isolation before importing Torch, NumPy, or Numba.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(ROOT / "upstream/code/src"))
    import torch
    import numpy as np
    from freeze import verify
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    verify()
    verify_source(ASSIGNMENT, ASSIGNMENT_SHA)
    seeds = (1,) if args.profile == "pilot" else SEEDS
    count = 256 if args.profile == "pilot" else QUERIES
    destination = output_directory(args.output_dir)
    summaries, repeats, validations = [], [], []
    inputs = {}
    for seed in seeds:
        torch.manual_seed(QUERY_SEED + seed)
        a, b, hashes, checks = run_seed(seed, count, destination)
        summaries.extend(a)
        repeats.extend(b)
        inputs.update(hashes)
        validations.append(checks)
    aggregate = []
    for mode in ("tail", "random", "full"):
        group = [r for r in summaries if r["method"] == mode]
        row = {"method": mode, "n_independent_training_seeds": len(group)}
        for key in ("support_rate", "eligible_rate", "support_disagreement_vs_full",
                    "miss_rate_given_full_positive", "eligibility_disagreement_vs_full",
                    "support_disagreement_vs_tail", "reference_cpu_lookup_mean_ms"):
            vals = [r[key] for r in group if r[key] is not None]
            row[key + "_valid_seeds"] = len(vals)
            row[key + "_seed_mean"] = statistics.mean(vals) if vals else None
            row[key + "_seed_sd"] = statistics.stdev(vals) if len(vals) > 1 else None
        aggregate.append(row)
    write_csv(destination / "seed_summary.csv", summaries)
    write_csv(destination / "random_repeat_summary.csv", repeats)
    write_csv(destination / "aggregate_summary.csv", aggregate)
    notes = [
        "Final checkpoints of all ten SABER runs, independent of success; pilot is not an additional seed.",
        "Queries are freshly drawn from retained replay with the frozen strict-future HER rule, not logged historical queries.",
        "Actor inference uses the restored policy on CPU. All retrieval methods use identical policy bins and queries.",
        "Full means exhaustive matching in the complete coarse bucket after validating the entire replay index.",
        "The first 16 queries per seed additionally check direct full-replay matching.",
        "Full matching is a reference for the same discretized definition, not ground truth continuous support.",
        "Tail means maintained index order, not temporal recency. Random chooses candidates, not weighted minibatch positions.",
        "Random repetitions and queries are nested measurements, not additional independent training seeds.",
        "Zero full positives gives an undefined conditional miss rate, retained as null, not zero.",
        "Lookup timing uses a common exhaustive NumPy reference kernel, includes candidate enumeration/selection, excludes actor inference and index maintenance.",
        "Reference timing does not benchmark the optimized Numba/cached production implementation or algorithm-level wall-clock efficiency.",
        "No environment steps, optimizer steps, policy changes, production resumption, or online-performance conclusions.",
        "Only final checkpoints are used; this does not establish retrieval behavior throughout training.",
    ]
    write_json(destination / "report.json", {"profile": args.profile, "seeds": list(seeds),
               "queries_per_seed": count, "random_repeats": RANDOM_REPEATS, "checks": validations,
               "aggregate": aggregate, "notes": notes,
               "environment": {"python": platform.python_version(), "platform": platform.platform(),
                               "torch": torch.__version__, "numpy": np.__version__,
                               "device": "cpu", "threads": torch.get_num_threads()}}, exclusive=True)
    finish_artifact(destination, inputs, {"kind": "checkpoint_retrieval_audit", "profile": args.profile})


if __name__ == "__main__":
    main()
