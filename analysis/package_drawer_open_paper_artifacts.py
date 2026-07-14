#!/usr/bin/env python3
"""Build compact, blind-safe DrawerOpen paper artifacts from raw run folders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Iterable, Sequence


TOTAL_STEPS = 2_000_000
EVAL_FREQ = 10_000
EVAL_GRID = tuple(range(EVAL_FREQ, TOTAL_STEPS + 1, EVAL_FREQ))
HGR_SEEDS = tuple(range(1, 6))
SABER_SEEDS = tuple(range(1, 11))
EXTENSION_SEEDS = tuple(range(6, 11))

DIAGNOSTIC_FIELDS = (
    "diagnostic/support_positive_rate",
    "diagnostic/td_error_support_pos_mean",
    "diagnostic/td_error_support_neg_mean",
    "diagnostic/support_to_td_error_corr",
    "diagnostic/residual_reduction_support_pos_mean",
    "diagnostic/residual_reduction_support_neg_mean",
    "diagnostic/residual_reduction_support_pos_rate",
    "diagnostic/residual_reduction_support_neg_rate",
    "diagnostic/support_to_residual_reduction_corr",
)

PER_SEED_FIELDS = (
    "method_key",
    "paper_label",
    "cohort",
    "seed",
    "run_name",
    "final_success",
    "best_success",
    "success_auc",
    "best_final_drop",
    "hit90_reached",
    "first_hit_0.90",
    "runtime_total_sec",
    "runtime_hours",
    "steps_per_sec",
    "numba_support_scan_enabled",
)

AGGREGATE_FIELDS = (
    "method_key",
    "paper_label",
    "cohort",
    "seeds",
    "n_seeds",
    "final_success_mean",
    "final_success_sample_std",
    "best_success_mean",
    "best_success_sample_std",
    "success_auc_mean",
    "success_auc_sample_std",
    "best_final_drop_mean",
    "best_final_drop_sample_std",
    "hit90_count",
    "hit90_total",
    "hit90_median_step_among_hits",
    "eval_grid_start",
    "eval_grid_end",
    "eval_grid_frequency",
    "eval_grid_points",
)


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite(value: object, *, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {context}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value for {context}: {value!r}")
    return result


def _optional_finite(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metrics_path(run_dir: Path) -> Path:
    for name in ("metrics.csv", "metrics_selected_10k.csv"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No metrics CSV in {run_dir}")


def _extract_eval_and_diagnostic_rows(
    path: Path,
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    fields, rows = _read_csv(path)
    if "step" not in fields or "eval/env_success_mean" not in fields:
        raise ValueError(f"Missing evaluation columns: {path}")

    eval_by_step: dict[int, dict[str, str]] = {}
    diagnostic_by_step: dict[int, dict[str, str]] = {}
    grid = set(EVAL_GRID)
    for line, row in enumerate(rows, start=2):
        step = int(_finite(row.get("step"), context=f"{path}:{line}:step"))
        if step not in grid:
            continue
        eval_value = _optional_finite(row.get("eval/env_success_mean"))
        if eval_value is not None:
            if step in eval_by_step:
                raise ValueError(f"Duplicate evaluation at step {step}: {path}")
            eval_by_step[step] = row
        elif any(_optional_finite(row.get(field)) is not None for field in DIAGNOSTIC_FIELDS):
            if step in diagnostic_by_step:
                raise ValueError(f"Duplicate diagnostic row at step {step}: {path}")
            diagnostic_by_step[step] = row

    missing = [step for step in EVAL_GRID if step not in eval_by_step]
    if missing:
        raise ValueError(f"Missing exact-grid evaluations in {path}: {missing[:5]}")
    return fields, [eval_by_step[step] for step in EVAL_GRID], [
        diagnostic_by_step[step] for step in EVAL_GRID if step in diagnostic_by_step
    ]


def _auc(values: Sequence[float]) -> float:
    if len(values) != len(EVAL_GRID):
        raise ValueError(f"Expected {len(EVAL_GRID)} values, got {len(values)}")
    area = sum(
        0.5 * (values[index] + values[index + 1])
        * (EVAL_GRID[index + 1] - EVAL_GRID[index])
        for index in range(len(values) - 1)
    )
    return area / float(EVAL_GRID[-1] - EVAL_GRID[0])


def _sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _aggregate(
    method_key: str,
    paper_label: str,
    cohort: str,
    rows: Sequence[dict[str, object]],
    seeds: Sequence[int],
) -> dict[str, object]:
    selected = [row for row in rows if int(row["seed"]) in seeds]
    if len(selected) != len(seeds):
        raise ValueError(f"Missing seeds for {cohort}: expected {seeds}")
    finals = [float(row["final_success"]) for row in selected]
    bests = [float(row["best_success"]) for row in selected]
    aucs = [float(row["success_auc"]) for row in selected]
    drops = [float(row["best_final_drop"]) for row in selected]
    hits = [
        int(row["first_hit_0.90"])
        for row in selected
        if bool(row["hit90_reached"])
    ]
    return {
        "method_key": method_key,
        "paper_label": paper_label,
        "cohort": cohort,
        "seeds": ",".join(str(seed) for seed in seeds),
        "n_seeds": len(seeds),
        "final_success_mean": statistics.fmean(finals),
        "final_success_sample_std": _sample_std(finals),
        "best_success_mean": statistics.fmean(bests),
        "best_success_sample_std": _sample_std(bests),
        "success_auc_mean": statistics.fmean(aucs),
        "success_auc_sample_std": _sample_std(aucs),
        "best_final_drop_mean": statistics.fmean(drops),
        "best_final_drop_sample_std": _sample_std(drops),
        "hit90_count": len(hits),
        "hit90_total": len(seeds),
        "hit90_median_step_among_hits": statistics.median(hits) if hits else "",
        "eval_grid_start": EVAL_GRID[0],
        "eval_grid_end": EVAL_GRID[-1],
        "eval_grid_frequency": EVAL_FREQ,
        "eval_grid_points": len(EVAL_GRID),
    }


def _copy_if_present(source: Path, target: Path) -> None:
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _run_result(
    run_dir: Path,
    seed: int,
    method_key: str,
    paper_label: str,
    cohort: str,
    target_data: Path,
    include_diagnostic: bool,
) -> tuple[dict[str, object], dict[str, float] | None]:
    summary = _read_json(run_dir / "summary.json")
    fields, eval_rows, diagnostic_rows = _extract_eval_and_diagnostic_rows(_metrics_path(run_dir))
    values = [_finite(row["eval/env_success_mean"], context=f"seed {seed} eval") for row in eval_rows]
    final_success = values[-1]
    best_success = max(values)
    success_auc = _auc(values)
    native_auc = _finite(summary["eval_AUC_success"], context=f"seed {seed} native AUC")
    if abs(success_auc - native_auc) > 1e-12:
        raise ValueError(f"AUC mismatch for seed {seed}: {success_auc} != {native_auc}")
    first_hit = next((step for step, value in zip(EVAL_GRID, values) if value >= 0.9), None)

    run_target = target_data / run_dir.name
    run_target.mkdir(parents=True, exist_ok=True)
    _write_csv(run_target / "metrics_eval_10k.csv", fields, eval_rows)
    _copy_if_present(run_dir / "summary.json", run_target / "summary.json")
    _copy_if_present(run_dir / "config.json", run_target / "config.json")
    _copy_if_present(run_dir / "env_snapshot.json", run_target / "env_snapshot.json")

    diagnostic_means: dict[str, float] | None = None
    if include_diagnostic:
        if len(diagnostic_rows) != len(EVAL_GRID):
            raise ValueError(
                f"Expected {len(EVAL_GRID)} diagnostic rows for seed {seed}, got {len(diagnostic_rows)}"
            )
        _write_csv(run_target / "diagnostic_train_10k.csv", fields, diagnostic_rows)
        diagnostic_means = {}
        for field in DIAGNOSTIC_FIELDS:
            finite_values = [
                value
                for row in diagnostic_rows
                if (value := _optional_finite(row.get(field))) is not None
            ]
            if not finite_values:
                raise ValueError(f"No finite {field} values for seed {seed}")
            diagnostic_means[field] = statistics.fmean(finite_values)

    runtime_sec = _optional_finite(summary.get("runtime_total_sec")) or 0.0
    final_lite = summary.get("final_lite_metrics")
    numba_enabled = ""
    if isinstance(final_lite, dict):
        value = _optional_finite(final_lite.get("runtime/numba_support_scan_enabled"))
        numba_enabled = "" if value is None else value
    row = {
        "method_key": method_key,
        "paper_label": paper_label,
        "cohort": cohort,
        "seed": seed,
        "run_name": run_dir.name,
        "final_success": final_success,
        "best_success": best_success,
        "success_auc": success_auc,
        "best_final_drop": best_success - final_success,
        "hit90_reached": first_hit is not None,
        "first_hit_0.90": "" if first_hit is None else first_hit,
        "runtime_total_sec": runtime_sec,
        "runtime_hours": runtime_sec / 3600.0,
        "steps_per_sec": _optional_finite(summary.get("final_steps_per_sec")) or 0.0,
        "numba_support_scan_enabled": numba_enabled,
    }
    return row, diagnostic_means


def _write_manifest(root: Path, artifact_id: str) -> None:
    payload = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        data = path.read_bytes()
        payload.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {
        "artifact_id": artifact_id,
        "blind_safe": True,
        "payload_file_count": len(payload),
        "payload": payload,
    }
    (root / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _copy_verifier(root: Path, verifier: Path) -> None:
    target = root / "code" / "verify_artifact.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(verifier, target)


def _build_hgr(source: Path, output: Path, verifier: Path) -> None:
    artifact_id = "DRAWEROPEN-DDPG-HGR-SAMPLER-ONLY-2M"
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in HGR_SEEDS:
        run_dir = source / f"drawer_open_ddpg_hgr_tdpriority_s{seed}_2m_opt_cuda"
        row, _ = _run_result(
            run_dir,
            seed,
            "drawer_open_ddpg_hgr_sampler_only",
            "HGR-style sampler-only",
            "seeds_1_5",
            output / "data",
            False,
        )
        rows.append(row)
    aggregates = [
        _aggregate(
            "drawer_open_ddpg_hgr_sampler_only",
            "HGR-style sampler-only",
            "seeds_1_5",
            rows,
            HGR_SEEDS,
        )
    ]
    _write_csv(output / "per_seed_performance.csv", PER_SEED_FIELDS, rows)
    _write_csv(output / "aggregate_performance.csv", AGGREGATE_FIELDS, aggregates)
    readme = f"""# DrawerOpen DDPG HGR-style sampler-only 2M artifact

Blind artifact ID: `{artifact_id}`

This package contains five completed 2M-step DrawerOpen DDPG+HER runs. The method changes episode and future-goal sampling using TD-error ranking while leaving critic and actor loss weights equal to one. It omits canonical HGR importance correction and must be cited as an HGR-style sampler-only ablation.

## Recomputed result

- Final success: `{aggregates[0]['final_success_mean']:.3f} +/- {aggregates[0]['final_success_sample_std']:.3f}`
- Success AUC: `{aggregates[0]['success_auc_mean']:.3f} +/- {aggregates[0]['success_auc_sample_std']:.3f}`
- Hit90: `{aggregates[0]['hit90_count']}/{aggregates[0]['hit90_total']}`, median `{int(float(aggregates[0]['hit90_median_step_among_hits'])) // 1000}k` among reached seeds
- Evaluation grid: exact 10k through 2M, 200 observed points per seed

## Scope

This result is a sampling stress control, not a faithful or canonical HGR comparison. Large checkpoints, raw high-frequency metrics, and logs are excluded. Seed summaries, configs, environment snapshots, and exact-grid evaluation curves are included.

## Verification

```bash
python3 code/verify_artifact.py
```
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    _copy_verifier(output, verifier)
    _write_manifest(output, artifact_id)


def _build_saber(source: Path, output: Path, verifier: Path) -> None:
    artifact_id = "DRAWEROPEN-DDPG-SABER-SEED-EXTENSION-2M"
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for seed in SABER_SEEDS:
        run_dir = source / f"drawer_open_DDPG-HER-B3-Critic-Only-2M_s{seed}_2m_cuda"
        cohort = "original_seeds_1_5" if seed <= 5 else "extension_seeds_6_10"
        row, diagnostic = _run_result(
            run_dir,
            seed,
            "drawer_open_ddpg_saber",
            "HER+SABER",
            cohort,
            output / "data",
            seed in EXTENSION_SEEDS,
        )
        rows.append(row)
        if diagnostic is not None:
            diagnostics.append(
                {
                    "seed": seed,
                    **diagnostic,
                    "td_error_pos_minus_neg": (
                        diagnostic["diagnostic/td_error_support_pos_mean"]
                        - diagnostic["diagnostic/td_error_support_neg_mean"]
                    ),
                    "residual_reduction_pos_minus_neg": (
                        diagnostic["diagnostic/residual_reduction_support_pos_mean"]
                        - diagnostic["diagnostic/residual_reduction_support_neg_mean"]
                    ),
                }
            )
    aggregates = [
        _aggregate("drawer_open_ddpg_saber", "HER+SABER", "original_seeds_1_5", rows, range(1, 6)),
        _aggregate("drawer_open_ddpg_saber", "HER+SABER", "extension_seeds_6_10", rows, range(6, 11)),
        _aggregate("drawer_open_ddpg_saber", "HER+SABER", "pooled_seeds_1_10", rows, range(1, 11)),
    ]
    _write_csv(output / "per_seed_performance.csv", PER_SEED_FIELDS, rows)
    _write_csv(output / "aggregate_performance.csv", AGGREGATE_FIELDS, aggregates)

    diagnostic_fields = (
        "seed",
        *DIAGNOSTIC_FIELDS,
        "td_error_pos_minus_neg",
        "residual_reduction_pos_minus_neg",
    )
    _write_csv(output / "diagnostic_per_seed.csv", diagnostic_fields, diagnostics)
    diagnostic_aggregate = []
    for field in diagnostic_fields[1:]:
        values = [float(row[field]) for row in diagnostics]
        diagnostic_aggregate.append(
            {
                "metric": field,
                "n_seeds": len(values),
                "mean_of_seed_means": statistics.fmean(values),
                "sample_std_of_seed_means": _sample_std(values),
                "positive_seed_count": sum(value > 0.0 for value in values),
            }
        )
    _write_csv(
        output / "diagnostic_aggregate.csv",
        ("metric", "n_seeds", "mean_of_seed_means", "sample_std_of_seed_means", "positive_seed_count"),
        diagnostic_aggregate,
    )

    original, extension, pooled = aggregates
    readme = f"""# DrawerOpen DDPG SABER seed-extension artifact

Blind artifact ID: `{artifact_id}`

This package separates the original seeds 1--5 result from an independent diagnostic extension on seeds 6--10. It also reports a descriptive pooled seeds 1--10 summary. The extension does not add matched HER or control seeds, so it must not be used to construct a paired 10-seed method comparison.

## Recomputed performance

| Cohort | Final success | Success AUC | Hit90 |
| --- | ---: | ---: | ---: |
| Original seeds 1--5 | {original['final_success_mean']:.3f} +/- {original['final_success_sample_std']:.3f} | {original['success_auc_mean']:.3f} +/- {original['success_auc_sample_std']:.3f} | {original['hit90_count']}/{original['hit90_total']} |
| Extension seeds 6--10 | {extension['final_success_mean']:.3f} +/- {extension['final_success_sample_std']:.3f} | {extension['success_auc_mean']:.3f} +/- {extension['success_auc_sample_std']:.3f} | {extension['hit90_count']}/{extension['hit90_total']} |
| Pooled seeds 1--10 | {pooled['final_success_mean']:.3f} +/- {pooled['final_success_sample_std']:.3f} | {pooled['success_auc_mean']:.3f} +/- {pooled['success_auc_sample_std']:.3f} | {pooled['hit90_count']}/{pooled['hit90_total']} |

All performance rows use the exact observed 10k--2M grid with 200 evaluations per seed. The original paired five-seed comparisons in the paper remain the confirmatory method comparisons.

## Diagnostic scope

Seeds 6--10 add a no-gradient post-update TD-error pass. It does not change replay sampling, loss weights, or optimizer steps. Diagnostic summaries use the training row at each exact 10k checkpoint. The added pass and a Python rather than Numba support scanner make runtime incomparable with seeds 1--5; runtime is therefore not pooled or used for efficiency claims.

The diagnostic is observational. It can describe TD-error enrichment and immediate residual correction but cannot establish lower-noise targets or a causal mechanism.

## Verification

```bash
python3 code/verify_artifact.py
```
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    _copy_verifier(output, verifier)
    _write_manifest(output, artifact_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hgr-source", type=Path, required=True)
    parser.add_argument("--saber-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--verifier",
        type=Path,
        default=Path(__file__).with_name("verify_drawer_open_paper_artifact.py"),
    )
    args = parser.parse_args()
    if not args.verifier.is_file():
        raise FileNotFoundError(args.verifier)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _build_hgr(
        args.hgr_source,
        args.output_root / "drawer_open_ddpg_hgr_sampler_only_2m",
        args.verifier,
    )
    _build_saber(
        args.saber_source,
        args.output_root / "drawer_open_ddpg_saber_seed_extension_2m",
        args.verifier,
    )
    print(f"Built DrawerOpen artifacts under {args.output_root}")


if __name__ == "__main__":
    main()
