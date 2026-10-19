"""Read-only final-checkpoint retrieval audit in both learned state distributions."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch

from protocol import ROOT, SEEDS, ARMS, make_config
from analyze import accepted_attempt, validate_run, read_json
from audit_reuse import file_hash, verify_catalog
from freeze import verify
from train import write_json
from retrieval_core import choose_candidates, scan_reference
from metacog_lite_v0.agent import DDPGAgent
from metacog_lite_v0.replay import HERReplayBuffer
from metacog_lite_v0.task_drawer_open import MetaWorldDrawerOpenTaskSpec
from metacog_lite_v0.weights import LiteWeightComputer, GapStats


def write_csv(path, rows):
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(attempt, arm, seed, destination, profile="production", count=1024):
    status = read_json(attempt / "status.json")
    historical = status["arm"] == "support"
    validate_run(attempt, profile, historical=historical)
    cfg = make_config(arm, seed, attempt, profile, "cpu")
    run = attempt / "run"
    paths = [attempt / "status.json", run / "config.json", run / "env_snapshot.json", run / "checkpoint.pt"]
    inputs = {str(p): file_hash(p) for p in paths}
    payload = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    if (json.loads(json.dumps(payload["config"])) != read_json(run / "config.json")
            or payload["global_step"] != cfg.total_steps
            or payload["total_updates"] != cfg.total_steps - cfg.warmup_steps + 1
            or payload["checkpoint_continuation_permitted"] is not False):
        raise ValueError("Checkpoint/config/budget mismatch")
    if arm == "random512":
        state = payload["retrieval_state"]
        if state["mode"] != cfg.support_scan_mode or state["completed_queries"] != payload["total_updates"]:
            raise ValueError("Missing/inconsistent retrieval RNG checkpoint")
    rp = payload["replay"]
    replay = HERReplayBuffer(rp["capacity"], rp["obs_dim"], rp["goal_dim"], rp["action_dim"])
    replay.load_state_dict(rp)
    snapshot = read_json(run / "env_snapshot.json")
    env = snapshot["train_env"]
    task = MetaWorldDrawerOpenTaskSpec(
        action_low=np.array(env["action_low"], np.float32), action_high=np.array(env["action_high"], np.float32),
        epsilon_g=env["distance_threshold"], d_reach=cfg.drawer_d_reach,
        d_goal_near=cfg.drawer_d_goal_near, d_opening_mid=cfg.drawer_d_opening_mid,
        hand_handle_bins=tuple(cfg.drawer_hand_handle_bins), handle_goal_bins=tuple(cfg.drawer_handle_goal_bins),
        handle_y_bins=tuple(cfg.drawer_handle_y_bins), gripper_bins=tuple(cfg.drawer_gripper_bins),
        horizon_bins=tuple(cfg.drawer_horizon_bins))
    if json.loads(json.dumps(task.metadata())) != snapshot["task_spec"]:
        raise ValueError("Task reconstruction mismatch")
    keys = task.coarse_hash_batch(replay.observation[:replay.size], replay.h[:replay.size])
    for key in np.unique(keys):
        np.testing.assert_array_equal(np.sort(replay.action_index.candidates(int(key), replay.size)),
                                      np.flatnonzero(keys == key))
    np.testing.assert_array_equal(replay.action_bin[:replay.size],
                                  task.action_bin_batch(replay.action_exec[:replay.size], cfg.action_bins_per_dim))
    agent = DDPGAgent(state_dim=env["state_dim"], action_dim=env["action_dim"], hidden_dim=cfg.hidden_dim,
                      actor_lr=cfg.actor_lr, critic_lr=cfg.critic_lr, gamma=cfg.gamma, tau=cfg.tau,
                      action_low=task.action_low, action_high=task.action_high,
                      device=torch.device("cpu"), l2_action_penalty=cfg.l2_action_penalty)
    agent.load_state_dict(payload["agent"])
    actor_before = copy.deepcopy(agent.actor.state_dict())
    tail_cfg = copy.deepcopy(cfg)
    tail_cfg.support_scan_mode = "deterministic_tail_v1"
    tail = LiteWeightComputer(tail_cfg, task, GapStats(cfg.gap_window, cfg.N_gap_min,
                                                     cfg.gap_q_low, cfg.gap_q_high, cfg.eps_G))
    sample_rng = np.random.default_rng(202610040 + seed)
    random_rngs = [np.random.default_rng(np.random.SeedSequence([202610041, seed, r])) for r in range(5)]
    rows, query_rows = [], []
    if count % cfg.batch_size:
        raise ValueError("Query count must be whole fixed-size minibatches")
    for block in range(count // cfg.batch_size):
        batch = replay.sample(cfg.batch_size, cfg.her_ratio, sample_rng)
        x = np.concatenate([batch.observation, batch.desired_goal_train], axis=1).astype(np.float32)
        with torch.no_grad():
            actions = agent.policy_actions(torch.from_numpy(x)).cpu().numpy()
        if not np.isfinite(actions).all():
            raise ValueError("Non-finite policy query")
        expected = tail.compute(batch, replay, agent, np.random.default_rng(0), torch.device("cpu"),
                                assignment_rng=np.random.default_rng(0))
        if not tail._numba_support_scan_enabled:
            raise ValueError("Optimized parity reference silently fell back")
        coarse = task.coarse_hash_batch(batch.observation, batch.h)
        full = task.full_context_hash_batch(batch.observation, batch.desired_goal_train, batch.h)
        bins = task.action_bin_batch(actions, cfg.action_bins_per_dim)
        subsets = {}
        for key in np.unique(coarse):
            candidates = replay.action_index.candidates(int(key), replay.size)
            subsets[int(key), "tail", 0] = choose_candidates(candidates, "tail", 512)
            subsets[int(key), "full", 0] = candidates
            for repeat in range(5):
                subsets[int(key), "random", repeat] = choose_candidates(candidates, "random", 512, random_rngs[repeat])
        for i in range(cfg.batch_size):
            query = block * cfg.batch_size + i
            query_rows.append({"arm": arm, "seed": seed, "query": query, "slot": int(batch.indices[i]),
                               "coarse_key": int(coarse[i]), "full_key": int(full[i]),
                               "policy_bin": int(bins[i]), "horizon": int(batch.h[i]),
                               **{f"goal_{j}": float(g) for j, g in enumerate(batch.desired_goal_train[i])}})
            results = {}
            for method, repeat in [("tail", 0), ("full", 0)] + [("random", r) for r in range(5)]:
                selected = subsets[int(coarse[i]), method, repeat]
                result = scan_reference(task, replay, selected, batch.desired_goal_train[i], int(full[i]), int(bins[i]), cfg)
                rows.append({"arm": arm, "seed": seed, "query": query, "method": method, "repeat": repeat,
                             "selected_candidates": len(selected), **result})
                results[method, repeat] = result
            if (results["tail", 0]["support"] != bool(expected.support_mask[i]) or
                    results["tail", 0]["eligible"] != bool(expected.context_eligible_mask[i])):
                raise ValueError("Tail optimized/reference mask mismatch")
            if query < 16:
                direct = scan_reference(task, replay, np.arange(replay.size), batch.desired_goal_train[i],
                                        int(full[i]), int(bins[i]), cfg)
                if direct != results["full", 0]:
                    raise ValueError("Full replay and full coarse bucket differ")
        print(f"{arm} seed{seed}: {len(query_rows)}/{count} queries", flush=True)
    for key, value in agent.actor.state_dict().items():
        torch.testing.assert_close(value, actor_before[key], rtol=0, atol=0)
    if agent.total_it != payload["agent"]["total_it"]:
        raise ValueError("Read-only audit performed an update")
    for path, value in inputs.items():
        if file_hash(path) != value:
            raise ValueError("Input file changed during audit")
    write_csv(destination / f"{arm}_seed_{seed:02d}_queries.csv", query_rows)
    write_csv(destination / f"{arm}_seed_{seed:02d}_measurements.csv", rows)
    full_rows = {r["query"]: r for r in rows if r["method"] == "full"}
    summaries = []
    for method in ("tail", "random", "full"):
        group = [r for r in rows if r["method"] == method]
        positives = sum(full_rows[r["query"]]["support"] for r in group)
        misses = sum(full_rows[r["query"]]["support"] and not r["support"] for r in group)
        if any(r["support"] and not full_rows[r["query"]]["support"] for r in group):
            raise ValueError("Subset support exceeds exhaustive reference")
        summaries.append({"training_arm": arm, "seed": seed, "method": method, "queries": count,
                          "measurements": len(group), "support_rate": sum(r["support"] for r in group)/len(group),
                          "eligible_rate": sum(r["eligible"] for r in group)/len(group),
                          "full_positive_measurements": positives, "missed_full_positives": misses,
                          "miss_rate_given_full_positive": misses/positives if positives else None})
    return summaries, inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-attempt", type=Path)
    args = ap.parse_args()
    snapshot = verify()
    torch.set_num_threads(1)
    destination = ROOT / "outputs" / ("retrieval_pilot" if args.pilot_attempt else "final_retrieval")
    destination.mkdir(parents=True, exist_ok=False)
    if args.pilot_attempt:
        jobs = [(args.pilot_attempt, "random512", 1, "smoke", 256)]
    else:
        catalog = verify_catalog(ROOT / "validation/reuse_catalog.json")
        jobs = [(Path(catalog["runs"][str(seed)]["attempt"]) if arm == "tail" else
                 accepted_attempt(ROOT / "outputs/production/random512" / f"seed_{seed:02d}"),
                 arm, seed, "production", 1024) for seed in SEEDS for arm in ARMS]
    summary, inputs = [], {}
    for job in jobs:
        result, hashes = audit(job[0], job[1], job[2], destination, job[3], job[4])
        summary += result
        inputs.update(hashes)
    write_csv(destination / "seed_summary.csv", summary)
    write_json(destination / "report.json", {
        "complete": True, "training_checkpoints": len(jobs), "seed_summary": summary,
        "tail_parity_mismatches": 0, "direct_full_replay_checks": 16 * len(jobs),
        "notes": ["Final checkpoints only; no causal learning or equivalence conclusion.",
                  "Full matching uses the frozen discrete context, not continuous support truth.",
                  "Random subsets are shared per bucket per query batch and repeated five times.",
                  "No timing superiority claim; query repeats are not independent training seeds."]})
    write_json(destination / "manifest.json", {"source_manifest_sha256": snapshot, "inputs_sha256": inputs,
        "outputs_sha256": {p.name: file_hash(p) for p in destination.iterdir() if p.is_file()}})


if __name__ == "__main__":
    main()
