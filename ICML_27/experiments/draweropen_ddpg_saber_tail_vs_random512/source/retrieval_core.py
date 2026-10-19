"""Common exact-matching reference kernel for all three retrieval rules."""
from __future__ import annotations

import numpy as np


def choose_candidates(candidates, mode, limit, rng=None):
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.ndim != 1 or limit <= 0:
        raise ValueError("Invalid candidates or scan limit")
    if mode not in {"tail", "random", "full"}:
        raise ValueError(mode)
    if mode == "full" or len(candidates) <= limit:
        return candidates.copy()
    if mode == "tail":
        return candidates[-limit:].copy()
    if rng is None:
        raise ValueError("Random retrieval needs an independent RNG")
    return rng.choice(candidates, size=limit, replace=False).astype(np.int64)


def scan_reference(task, replay, selected, goal, full_key, policy_bin, cfg):
    selected = np.asarray(selected, dtype=np.int64)
    if not len(selected):
        return {"local_count": 0, "eligible": False, "occupancy": False, "support": False}
    goals = np.broadcast_to(np.asarray(goal, dtype=np.float32), (len(selected), len(goal)))
    # Every candidate is re-contextualized under the QUERY'S HER goal.
    keys = task.full_context_hash_batch(replay.observation[selected], goals, replay.h[selected])
    local = selected[keys == full_key]
    bins = replay.action_bin[local]
    if np.any(bins < 0):
        raise ValueError("Missing retained action bins")
    count = len(local)
    eligible = count >= cfg.N_local_min and min(count, cfg.M_B) >= cfg.M_Q_min
    occupancy = bool(np.any(bins == policy_bin))
    return {"local_count": count, "eligible": bool(eligible), "occupancy": occupancy,
            "support": bool(eligible and occupancy)}


def summarize_method(rows, full, tail):
    if not rows:
        raise ValueError("No retrieval measurements")
    n = len(rows)
    full_positive = sum(bool(full[r["query"]]["support"]) for r in rows)
    full_eligible = sum(bool(full[r["query"]]["eligible"]) for r in rows)
    missed = sum(bool(full[r["query"]]["support"]) and not r["support"] for r in rows)
    eligibility_missed = sum(bool(full[r["query"]]["eligible"]) and not r["eligible"] for r in rows)
    if any(r["support"] and not full[r["query"]]["support"] for r in rows):
        raise ValueError("Subset retrieval produced support absent in the exhaustive reference")
    if any(r["eligible"] and not full[r["query"]]["eligible"] for r in rows):
        raise ValueError("Subset retrieval eligibility exceeds exhaustive reference")
    missed_by_population = sum(bool(full[r["query"]]["support"]) and not r["eligible"] for r in rows)
    missed_by_membership = sum(bool(full[r["query"]]["support"]) and r["eligible"]
                               and not r["occupancy"] for r in rows)
    if missed != missed_by_population + missed_by_membership:
        raise ValueError("Support-miss decomposition is inconsistent")
    times = np.asarray([r["reference_lookup_seconds"] for r in rows])
    return {"query_measurements": n, "unique_queries": len({r["query"] for r in rows}),
            "support_rate": sum(r["support"] for r in rows) / n,
            "eligible_rate": sum(r["eligible"] for r in rows) / n,
            "support_disagreement_vs_full": missed / n,
            "full_positive_measurements": full_positive,
            "missed_full_positives": missed,
            "miss_rate_given_full_positive": missed / full_positive if full_positive else None,
            "eligibility_disagreement_vs_full": eligibility_missed / n,
            "miss_rate_given_full_eligible": eligibility_missed / full_eligible if full_eligible else None,
            "population_related_support_misses": missed_by_population,
            "occupancy_related_support_misses": missed_by_membership,
            "support_disagreement_vs_tail": sum(r["support"] != tail[r["query"]]["support"] for r in rows) / n,
            "tail_negative_method_positive_rate": sum(r["support"] and not tail[r["query"]]["support"] for r in rows) / n,
            "tail_positive_method_negative_rate": sum(not r["support"] and tail[r["query"]]["support"] for r in rows) / n,
            "mean_coarse_candidates": float(np.mean([r["coarse_count"] for r in rows])),
            "mean_scanned_candidates": float(np.mean([r["scanned_count"] for r in rows])),
            "mean_local_count": float(np.mean([r["local_count"] for r in rows])),
            "truncated_fraction": sum(r["truncated"] for r in rows) / n,
            "reference_cpu_lookup_mean_ms": float(times.mean() * 1000),
            "reference_cpu_lookup_median_ms": float(np.median(times) * 1000),
            "reference_cpu_lookup_p95_ms": float(np.quantile(times, .95) * 1000)}
