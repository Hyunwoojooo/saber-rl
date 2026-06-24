from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent


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


def variance_or_nan(values: Iterable[float]) -> float:
    vals = finite(values)
    if not vals:
        return math.nan
    mu = mean(vals)
    return mean((v - mu) ** 2 for v in vals)


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


def csv_value(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def clean_json(value: object) -> object:
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return value


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fields})


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(clean_json(payload), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def fmt(value: object, digits: int = 3) -> str:
    v = num(value)
    if not math.isfinite(v):
        return "n/a"
    return f"{v:.{digits}f}"


def find_diagnostic_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        path = item.expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.is_file():
            if path.name == "diagnostic_samples.csv":
                files.append(path)
            continue
        if path.is_dir():
            files.extend(sorted(path.rglob("diagnostic_samples.csv")))
    return sorted(dict.fromkeys(files))


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = str(path.parent.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path.parent)
            rows.append(
                {
                    "source": source,
                    "path": str(path),
                    "step": int(num(row.get("step"))),
                    "update": int(num(row.get("update"))),
                    "method": str(row.get("method", "")),
                    "seed": int(num(row.get("env_seed"))),
                    "sample_rank": int(num(row.get("sample_rank"))),
                    "psi": num(row.get("psi_q_rank")),
                    "td_before": num(row.get("td_abs_before")),
                    "td_after": num(row.get("td_abs_after")),
                    "td_reduction": num(row.get("td_abs_reduction")),
                    "target": num(row.get("target")),
                    "q_before": num(row.get("q_before")),
                    "q_after": num(row.get("q_after")),
                    "w_raw": num(row.get("w_raw")),
                    "omega_critic": num(row.get("omega_critic")),
                    "local_pool_size": num(row.get("local_pool_size")),
                    "support_candidate_count": num(row.get("support_candidate_count")),
                    "her_relabel": num(row.get("her_relabel")),
                    "policy_action_bin": num(row.get("policy_action_bin")),
                    "replay_action_bin": num(row.get("replay_action_bin")),
                }
            )
    return rows


def topk_support_overlap(psi: list[float], td: list[float], k: int) -> float:
    pairs = [(float(td_i), float(psi_i)) for psi_i, td_i in zip(psi, td) if math.isfinite(td_i)]
    if k <= 0 or not pairs:
        return math.nan
    k = min(k, len(pairs))
    top = sorted(pairs, key=lambda item: item[0], reverse=True)[:k]
    return float(mean(1.0 if psi_i > 0.5 else 0.0 for _, psi_i in top))


def group_summary(rows: list[dict[str, object]], label: str, group_rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "source": rows[0]["source"] if rows else "",
        "method": rows[0]["method"] if rows else "",
        "seed": rows[0]["seed"] if rows else "",
        "group": label,
        "n": len(group_rows),
        "td_before_mean": mean_or_nan(num(row.get("td_before")) for row in group_rows),
        "td_after_mean": mean_or_nan(num(row.get("td_after")) for row in group_rows),
        "td_reduction_mean": mean_or_nan(num(row.get("td_reduction")) for row in group_rows),
        "target_mean": mean_or_nan(num(row.get("target")) for row in group_rows),
        "target_var": variance_or_nan(num(row.get("target")) for row in group_rows),
        "q_before_mean": mean_or_nan(num(row.get("q_before")) for row in group_rows),
        "q_before_var": variance_or_nan(num(row.get("q_before")) for row in group_rows),
        "q_after_mean": mean_or_nan(num(row.get("q_after")) for row in group_rows),
        "q_after_var": variance_or_nan(num(row.get("q_after")) for row in group_rows),
        "local_pool_size_mean": mean_or_nan(num(row.get("local_pool_size")) for row in group_rows),
        "support_candidate_count_mean": mean_or_nan(num(row.get("support_candidate_count")) for row in group_rows),
    }


def analyze_step(rows: list[dict[str, object]]) -> dict[str, object]:
    psi = [num(row.get("psi")) for row in rows]
    td_before = [num(row.get("td_before")) for row in rows]
    td_reduction = [num(row.get("td_reduction")) for row in rows]
    support_rows = [row for row in rows if num(row.get("psi")) > 0.5]
    unsupported_rows = [row for row in rows if num(row.get("psi")) <= 0.5]
    k_support = len(support_rows)
    k_top10 = max(1, int(math.ceil(0.10 * len(rows)))) if rows else 0
    return {
        "source": rows[0]["source"],
        "path": rows[0]["path"],
        "method": rows[0]["method"],
        "seed": rows[0]["seed"],
        "step": rows[0]["step"],
        "update": rows[0]["update"],
        "n": len(rows),
        "support_rate": mean_or_nan(1.0 if v > 0.5 else 0.0 for v in psi),
        "her_relabel_rate": mean_or_nan(num(row.get("her_relabel")) for row in rows),
        "corr_psi_td_before_pearson": pearson(psi, td_before),
        "corr_psi_td_before_spearman": spearman(psi, td_before),
        "corr_psi_td_reduction_pearson": pearson(psi, td_reduction),
        "top_td_same_k_support_rate": topk_support_overlap(psi, td_before, k_support),
        "top_td_10pct_support_rate": topk_support_overlap(psi, td_before, k_top10),
        "support_n": k_support,
        "unsupported_n": len(unsupported_rows),
        "td_before_support_mean": mean_or_nan(num(row.get("td_before")) for row in support_rows),
        "td_before_unsupported_mean": mean_or_nan(num(row.get("td_before")) for row in unsupported_rows),
        "td_after_support_mean": mean_or_nan(num(row.get("td_after")) for row in support_rows),
        "td_after_unsupported_mean": mean_or_nan(num(row.get("td_after")) for row in unsupported_rows),
        "td_reduction_support_mean": mean_or_nan(num(row.get("td_reduction")) for row in support_rows),
        "td_reduction_unsupported_mean": mean_or_nan(num(row.get("td_reduction")) for row in unsupported_rows),
        "target_var_support": variance_or_nan(num(row.get("target")) for row in support_rows),
        "target_var_unsupported": variance_or_nan(num(row.get("target")) for row in unsupported_rows),
        "q_before_var_support": variance_or_nan(num(row.get("q_before")) for row in support_rows),
        "q_before_var_unsupported": variance_or_nan(num(row.get("q_before")) for row in unsupported_rows),
    }


def summarize(per_step: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in per_step:
        groups[(str(row["source"]), str(row["method"]), int(row["seed"]))].append(row)
    fields = [
        "support_rate",
        "her_relabel_rate",
        "corr_psi_td_before_pearson",
        "corr_psi_td_before_spearman",
        "corr_psi_td_reduction_pearson",
        "top_td_same_k_support_rate",
        "top_td_10pct_support_rate",
        "td_before_support_mean",
        "td_before_unsupported_mean",
        "td_after_support_mean",
        "td_after_unsupported_mean",
        "td_reduction_support_mean",
        "td_reduction_unsupported_mean",
        "target_var_support",
        "target_var_unsupported",
    ]
    for (source, method, seed), rows in sorted(groups.items()):
        record: dict[str, object] = {
            "source": source,
            "method": method,
            "seed": seed,
            "n_steps": len(rows),
            "first_step": min(int(row["step"]) for row in rows),
            "last_step": max(int(row["step"]) for row in rows),
            "n_samples_total": int(sum(int(row["n"]) for row in rows)),
        }
        for field in fields:
            vals = [num(row.get(field)) for row in rows]
            record[f"{field}_mean"] = mean_or_nan(vals)
            record[f"{field}_sd"] = sd_or_zero(vals)
        out.append(record)
    return out


def write_markdown(path: Path, files: list[Path], summary: list[dict[str, object]]) -> None:
    lines = [
        "# SABER Sample-level Support Diagnostic",
        "",
        f"Generated: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        "",
        "## Scope",
        "",
        "This report summarizes `diagnostic_samples.csv` files produced by the optional FPP diagnostic logger.",
        "The key comparison is sample-level support indicator versus fixed-target TD residual behavior.",
        "",
        "## Inputs",
        "",
    ]
    if files:
        for file in files:
            rel = file.relative_to(REPO_ROOT) if file.is_relative_to(REPO_ROOT) else file
            lines.append(f"- `{rel}`")
    else:
        lines.append("- No `diagnostic_samples.csv` files found.")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Source | Seed | steps | samples | support rate | corr(psi, TD) | Spearman | top-k overlap | TD reduction support | TD reduction unsupported | target var support | target var unsupported |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary:
        lines.append(
            "| "
            f"{row['source']} | "
            f"{row['seed']} | "
            f"{row['n_steps']} | "
            f"{row['n_samples_total']} | "
            f"{fmt(row.get('support_rate_mean'))} | "
            f"{fmt(row.get('corr_psi_td_before_pearson_mean'))} | "
            f"{fmt(row.get('corr_psi_td_before_spearman_mean'))} | "
            f"{fmt(row.get('top_td_same_k_support_rate_mean'))} | "
            f"{fmt(row.get('td_reduction_support_mean_mean'))} | "
            f"{fmt(row.get('td_reduction_unsupported_mean_mean'))} | "
            f"{fmt(row.get('target_var_support_mean'))} | "
            f"{fmt(row.get('target_var_unsupported_mean'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Low `corr(psi, TD)` and low top-k overlap support the claim that SABER is not merely TD-error prioritization.",
            "- Larger support-positive residual reduction supports the critic-fitting reliability interpretation.",
            "- Similar or worse residual reduction for support-positive samples weakens the structured-support mechanism claim.",
            "- Target variance by support group is diagnostic evidence only; it should not be written as a theorem.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze SABER sample-level diagnostic logs.")
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        default=[REPO_ROOT / "experiment_FPP" / "runs"],
        help="Run directories, roots, or diagnostic_samples.csv files.",
    )
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    files = find_diagnostic_files(args.input)
    all_rows: list[dict[str, object]] = []
    for file in files:
        all_rows.extend(read_rows(file))

    by_step: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        by_step[(str(row["source"]), int(row["seed"]), int(row["step"]))].append(row)
    per_step = [analyze_step(rows) for _, rows in sorted(by_step.items()) if rows]
    summary = summarize(per_step)

    group_rows: list[dict[str, object]] = []
    by_source: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        by_source[(str(row["source"]), int(row["seed"]))].append(row)
    for _, rows in sorted(by_source.items()):
        support_rows = [row for row in rows if num(row.get("psi")) > 0.5]
        unsupported_rows = [row for row in rows if num(row.get("psi")) <= 0.5]
        group_rows.append(group_summary(rows, "support_positive", support_rows))
        group_rows.append(group_summary(rows, "support_negative", unsupported_rows))

    out_dir = args.out_dir
    write_csv(out_dir / "sample_diagnostic_per_step.csv", per_step)
    write_csv(out_dir / "sample_diagnostic_group_stats.csv", group_rows)
    write_csv(out_dir / "sample_diagnostic_summary.csv", summary)
    write_json(
        out_dir / "sample_diagnostic_summary.json",
        {
            "inputs": [str(file) for file in files],
            "n_rows": len(all_rows),
            "n_step_groups": len(per_step),
            "summary": summary,
        },
    )
    write_markdown(out_dir / "sample_diagnostic_report.md", files, summary)
    print(out_dir / "sample_diagnostic_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
