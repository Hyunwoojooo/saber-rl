from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

RUN_SPECS = [
    (
        "FPP-DDPG-SABER",
        REPO_ROOT / "experiment_FPP" / "runs" / "b3_critic_only_runtime_opt_v4_ls512",
        "fetch_pick_and_place_b3_critic_only_runtime_opt_v4_ls512_s{seed}_1000000_ls512",
    ),
    (
        "FPP-TD3-SABER",
        REPO_ROOT / "experiment_FPP" / "runs" / "td3_saber_critic_only_v4_ls512",
        "fetch_pick_and_place_td3_saber_critic_only_v4_ls512_s{seed}_1000000_ls512",
    ),
    (
        "FPP-SAC-SABER",
        REPO_ROOT / "experiment_FPP" / "runs" / "sac_saber_critic_only_v4_ls512",
        "fetch_pick_and_place_sac_saber_critic_only_v4_ls512_s{seed}_1000000_ls512",
    ),
]

SEEDS = (1, 2, 3, 4, 5)


def num(value: object) -> float:
    if value in (None, ""):
        return math.nan
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def finite(values: Iterable[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def mean_or_nan(values: Iterable[float]) -> float:
    vals = finite(values)
    return mean(vals) if vals else math.nan


def sd_or_zero(values: Iterable[float]) -> float:
    vals = finite(values)
    return stdev(vals) if len(vals) > 1 else 0.0


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return math.nan
    mx = mean(x for x, _ in pairs)
    my = mean(y for _, y in pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0.0 or vy <= 0.0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def ranks(values: list[float]) -> list[float]:
    indexed = sorted((v, i) for i, v in enumerate(values))
    out = [math.nan] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][0] == indexed[start][0]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        for _, original_idx in indexed[start:end]:
            out[original_idx] = rank
        start = end
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return math.nan
    return pearson(ranks([x for x, _ in pairs]), ranks([y for _, y in pairs]))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dedupe_step_rows(rows: list[dict[str, str]]) -> list[dict[str, float]]:
    by_step: dict[int, dict[str, float]] = {}
    for row in rows:
        step = int(num(row.get("step")))
        support_rate = num(row.get("lite/psi_q_rank_rate"))
        td_mean = num(row.get("loss/td_error_abs_mean"))
        td_p90 = num(row.get("loss/td_error_abs_p90"))
        success = num(row.get("eval/success_mean"))
        if not (math.isfinite(support_rate) and math.isfinite(td_mean)):
            continue
        by_step[step] = {
            "step": float(step),
            "support_rate": support_rate,
            "td_error_abs_mean": td_mean,
            "td_error_abs_p90": td_p90,
            "eval_success": success,
        }
    return [by_step[step] for step in sorted(by_step)]


def analyze_run(method: str, seed: int, path: Path) -> dict[str, object]:
    rows_raw = read_csv(path)
    rows = dedupe_step_rows(rows_raw)
    support = [float(row["support_rate"]) for row in rows]
    td_mean = [float(row["td_error_abs_mean"]) for row in rows]
    td_p90 = [float(row["td_error_abs_p90"]) for row in rows]
    eval_success = [float(row["eval_success"]) for row in rows]

    return {
        "method": method,
        "seed": seed,
        "path": str(path),
        "n_raw_rows": len(rows_raw),
        "n_step_rows": len(rows),
        "first_step": int(rows[0]["step"]) if rows else None,
        "last_step": int(rows[-1]["step"]) if rows else None,
        "support_rate_mean": mean_or_nan(support),
        "support_rate_first": support[0] if support else math.nan,
        "support_rate_last": support[-1] if support else math.nan,
        "td_error_abs_mean_mean": mean_or_nan(td_mean),
        "td_error_abs_mean_first": td_mean[0] if td_mean else math.nan,
        "td_error_abs_mean_last": td_mean[-1] if td_mean else math.nan,
        "support_td_mean_pearson": pearson(support, td_mean),
        "support_td_mean_spearman": spearman(support, td_mean),
        "support_td_p90_pearson": pearson(support, td_p90),
        "support_td_p90_spearman": spearman(support, td_p90),
        "support_eval_success_pearson": pearson(support, eval_success),
        "artifact_level": "checkpoint_aggregate",
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    methods = sorted({str(row["method"]) for row in rows})
    fields = [
        "n_step_rows",
        "support_rate_mean",
        "support_rate_first",
        "support_rate_last",
        "td_error_abs_mean_mean",
        "td_error_abs_mean_first",
        "td_error_abs_mean_last",
        "support_td_mean_pearson",
        "support_td_mean_spearman",
        "support_td_p90_pearson",
        "support_td_p90_spearman",
        "support_eval_success_pearson",
    ]
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        record: dict[str, object] = {"method": method, "n": len(group)}
        for field in fields:
            vals = [num(row.get(field)) for row in group]
            record[f"{field}_mean"] = mean_or_nan(vals)
            record[f"{field}_sd"] = sd_or_zero(vals)
        out.append(record)
    return out


def clean_json(value: object) -> object:
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return value


def csv_value(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fields})


def fmt(value: object, digits: int = 3) -> str:
    v = num(value)
    if not math.isfinite(v):
        return "n/a"
    return f"{v:.{digits}f}"


def write_markdown(path: Path, per_seed: list[dict[str, object]], summary: list[dict[str, object]]) -> None:
    lines = [
        "# FPP Support vs TD-error Existing-Artifact Diagnostic",
        "",
        f"Generated: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        "",
        "## Scope",
        "",
        "This diagnostic uses existing `metrics_selected_10k.csv` files from FPP SABER runs.",
        "It is a checkpoint-level aggregate diagnostic, not a sample-level diagnostic.",
        "",
        "Available:",
        "",
        "- checkpoint-level `lite/psi_q_rank_rate`",
        "- checkpoint-level `loss/td_error_abs_mean` and `loss/td_error_abs_p90`",
        "- checkpoint-level support coverage and success curves",
        "",
        "Not available in current artifacts:",
        "",
        "- per-sample `psi_i`",
        "- per-sample `|delta_i|`",
        "- top-k support vs TD-error overlap",
        "- support-positive vs support-negative residual distributions",
        "- before/after update residual reduction by support group",
        "",
        "## Aggregate Correlations",
        "",
        "| Method | n | support rate | TD abs mean | corr(support, TD mean) | Spearman | corr(support, TD p90) | corr(support, eval success) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| "
            f"{row['method']} | "
            f"{row['n']} | "
            f"{fmt(row['support_rate_mean_mean'])} +/- {fmt(row['support_rate_mean_sd'])} | "
            f"{fmt(row['td_error_abs_mean_mean_mean'])} +/- {fmt(row['td_error_abs_mean_mean_sd'])} | "
            f"{fmt(row['support_td_mean_pearson_mean'])} +/- {fmt(row['support_td_mean_pearson_sd'])} | "
            f"{fmt(row['support_td_mean_spearman_mean'])} +/- {fmt(row['support_td_mean_spearman_sd'])} | "
            f"{fmt(row['support_td_p90_pearson_mean'])} +/- {fmt(row['support_td_p90_pearson_sd'])} | "
            f"{fmt(row['support_eval_success_pearson_mean'])} +/- {fmt(row['support_eval_success_pearson_sd'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-seed Correlations",
            "",
            "| Method | Seed | rows | support mean | TD mean | Pearson | Spearman | TD p90 Pearson |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(per_seed, key=lambda item: (str(item["method"]), int(item["seed"]))):
        lines.append(
            "| "
            f"{row['method']} | "
            f"{row['seed']} | "
            f"{row['n_step_rows']} | "
            f"{fmt(row['support_rate_mean'])} | "
            f"{fmt(row['td_error_abs_mean_mean'])} | "
            f"{fmt(row['support_td_mean_pearson'])} | "
            f"{fmt(row['support_td_mean_spearman'])} | "
            f"{fmt(row['support_td_p90_pearson'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Existing artifacts do not close the reviewer-critical sample-level claim.",
            "- The aggregate checkpoint correlations are useful as a quick sanity check: support coverage and mean TD-error are not the same logged quantity.",
            "- Because both support rate and TD-error drift over training, these correlations are confounded by learning progress and should not be presented as the main mechanism evidence.",
            "- To close the mechanism claim, an instrumented rerun should log per-sample `psi_i`, `|delta_i|`, weights, phase, support count, and residual before/after the critic update for a small diagnostic stream.",
            "",
            "## Recommended Next Diagnostic",
            "",
            "Run an instrumented FPP DDPG SABER job for seeds 1--3 or 1--5 that writes a small sampled diagnostic table every 10k or 20k steps. The diagnostic table should contain per-sample rows, capped to a small random subset per checkpoint to avoid large artifacts.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    per_seed: list[dict[str, object]] = []
    for method, root, pattern in RUN_SPECS:
        for seed in SEEDS:
            path = root / pattern.format(seed=seed) / "metrics_selected_10k.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            per_seed.append(analyze_run(method, seed, path))

    summary = summarize(per_seed)
    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "artifact_level": "checkpoint_aggregate",
        "per_seed": per_seed,
        "summary": summary,
        "missing_for_claim_closure": [
            "per-sample psi_i",
            "per-sample abs TD-error",
            "top-k overlap",
            "support-positive vs support-negative residual distributions",
            "before/after update residual reduction by support group",
        ],
    }
    write_csv(OUT_DIR / "fpp_support_td_error_existing_per_seed.csv", per_seed)
    write_csv(OUT_DIR / "fpp_support_td_error_existing_summary.csv", summary)
    (OUT_DIR / "fpp_support_td_error_existing_summary.json").write_text(
        json.dumps(clean_json(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    write_markdown(OUT_DIR / "fpp_support_td_error_existing_report.md", per_seed, summary)
    print(OUT_DIR / "fpp_support_td_error_existing_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
