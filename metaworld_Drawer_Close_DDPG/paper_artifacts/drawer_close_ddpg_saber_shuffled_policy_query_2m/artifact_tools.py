#!/usr/bin/env python3
"""Build and verify the compact DrawerClose shuffled-policy-query artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SEEDS = (1, 2, 3, 4, 5)
EVAL_STEPS = tuple(range(10_000, 2_000_001, 10_000))
SUCCESS_FIELD = "eval/env_success_mean"
METHODS = (
    ("her", "DrawerClose DDPG HER"),
    ("main", "DrawerClose DDPG HER + SABER Main"),
    ("shuffled", "DrawerClose DDPG Shuffled-Policy Query"),
)
CONTRASTS = (
    ("main_minus_shuffled", "SABER Main - Shuffled-Policy Query", "main", "shuffled"),
    ("shuffled_minus_her", "Shuffled-Policy Query - HER", "shuffled", "her"),
    ("main_minus_her", "SABER Main - HER", "main", "her"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sanitize(value: Any, replacements: Sequence[tuple[str, str]]) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): sanitize(item, replacements) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item, replacements) for item in value]
    if isinstance(value, str):
        result = value
        for old, new in replacements:
            result = result.replace(old, new)
        home_prefix = "/" + "home" + "/"
        result = re.sub(re.escape(home_prefix) + r"[^/\s]+", "<home>", result)
        return result
    return value


def load_curve(path: Path) -> tuple[list[int], list[float], list[dict[str, str]], list[str]]:
    expected = set(EVAL_STEPS)
    rows_by_step: dict[int, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if SUCCESS_FIELD not in fields:
            raise ValueError(f"{path}: missing {SUCCESS_FIELD}")
        for row in reader:
            raw_success = row.get(SUCCESS_FIELD)
            if raw_success in ("", None):
                continue
            step_value = finite_float(row.get("step"))
            success = finite_float(raw_success)
            if step_value is None or success is None:
                raise ValueError(f"{path}: non-finite evaluation row")
            step = int(step_value)
            if step not in expected:
                raise ValueError(f"{path}: unexpected evaluation step {step}")
            if step in rows_by_step:
                raise ValueError(f"{path}: duplicate evaluation step {step}")
            if not 0.0 <= success <= 1.0:
                raise ValueError(f"{path}: success outside [0,1] at {step}")
            rows_by_step[step] = row
    if set(rows_by_step) != expected:
        missing = sorted(expected - set(rows_by_step))
        raise ValueError(f"{path}: incomplete exact grid; missing={missing}")
    rows = [rows_by_step[step] for step in EVAL_STEPS]
    return list(EVAL_STEPS), [float(row[SUCCESS_FIELD]) for row in rows], rows, fields


def write_rows(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or [])
    if not fieldnames:
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def normalized_auc(steps: Sequence[int], values: Sequence[float]) -> float:
    area = math.fsum(
        (right - left) * (y_left + y_right) / 2.0
        for left, right, y_left, y_right in zip(steps, steps[1:], values, values[1:])
    )
    return area / float(steps[-1] - steps[0])


def first_hit90(steps: Sequence[int], values: Sequence[float]) -> int | None:
    return next((int(step) for step, value in zip(steps, values) if value >= 0.9), None)


def mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def sample_sd(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def exhaustive_bootstrap(differences: Sequence[float]) -> tuple[float, float]:
    samples = [
        mean([differences[index] for index in indices])
        for indices in itertools.product(range(len(differences)), repeat=len(differences))
    ]
    return percentile(samples, 0.025), percentile(samples, 0.975)


def sign_flip_p(differences: Sequence[float]) -> float:
    observed = abs(mean(differences))
    statistics_ = [
        abs(mean([sign * value for sign, value in zip(signs, differences)]))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return sum(value >= observed - 1e-15 for value in statistics_) / len(statistics_)


def analyze(data_root: Path) -> dict[str, object]:
    curves: dict[tuple[str, int], list[float]] = {}
    per_seed: list[dict[str, object]] = []
    for method_key, label in METHODS:
        for seed in SEEDS:
            run = data_root / method_key / f"seed_{seed}"
            metrics = run / "metrics_selected_10k.csv"
            summary = read_json(run / "summary.json")
            steps, values, _, _ = load_curve(metrics)
            auc = normalized_auc(steps, values)
            final = values[-1]
            best = max(values)
            native_auc = finite_float(summary.get("eval_AUC_success"))
            native_final = finite_float(summary.get("final_eval_env_success"))
            if native_auc is not None and not math.isclose(native_auc, auc, abs_tol=1e-12):
                raise ValueError(f"{method_key} seed {seed}: summary AUC mismatch")
            if native_final is not None and not math.isclose(native_final, final, abs_tol=1e-12):
                raise ValueError(f"{method_key} seed {seed}: summary final mismatch")
            curves[(method_key, seed)] = values
            per_seed.append(
                {
                    "method_key": method_key,
                    "method": label,
                    "seed": seed,
                    "evaluation_points": len(values),
                    "auc": auc,
                    "final_success": final,
                    "best_success": best,
                    "best_to_final_drop": best - final,
                    "hit90": first_hit90(steps, values),
                }
            )
    aggregates: list[dict[str, object]] = []
    by_key_seed = {(str(row["method_key"]), int(row["seed"])): row for row in per_seed}
    for method_key, label in METHODS:
        rows = [by_key_seed[(method_key, seed)] for seed in SEEDS]
        aucs = [float(row["auc"]) for row in rows]
        finals = [float(row["final_success"]) for row in rows]
        bests = [float(row["best_success"]) for row in rows]
        drops = [float(row["best_to_final_drop"]) for row in rows]
        hits = [int(row["hit90"]) for row in rows if row["hit90"] is not None]
        aggregates.append(
            {
                "method_key": method_key,
                "method": label,
                "seed_count": len(rows),
                "auc_mean": mean(aucs),
                "auc_sample_sd": sample_sd(aucs),
                "final_success_mean": mean(finals),
                "final_success_sample_sd": sample_sd(finals),
                "best_success_mean": mean(bests),
                "best_success_sample_sd": sample_sd(bests),
                "best_to_final_drop_mean": mean(drops),
                "best_to_final_drop_sample_sd": sample_sd(drops),
                "hit90_seed_count": len(hits),
                "hit90_median_reached": None if not hits else statistics.median(hits),
                "hit90_mean_reached": None if not hits else mean(hits),
            }
        )
    paired: list[dict[str, object]] = []
    for key, label, left, right in CONTRASTS:
        differences = [
            float(by_key_seed[(left, seed)]["auc"]) - float(by_key_seed[(right, seed)]["auc"])
            for seed in SEEDS
        ]
        low, high = exhaustive_bootstrap(differences)
        leave_one_out = [
            mean([value for index, value in enumerate(differences) if index != omitted])
            for omitted in range(len(differences))
        ]
        row: dict[str, object] = {
            "contrast_key": key,
            "contrast": label,
            "left_method": left,
            "right_method": right,
            "paired_seed_count": len(SEEDS),
            "mean_auc_difference": mean(differences),
            "sample_sd_auc_difference": sample_sd(differences),
            "bootstrap_ci95_lower": low,
            "bootstrap_ci95_upper": high,
            "bootstrap_resample_count": len(SEEDS) ** len(SEEDS),
            "exact_two_sided_sign_flip_pvalue": sign_flip_p(differences),
            "sign_flip_assignment_count": 2 ** len(SEEDS),
            "positive_seed_count": sum(value > 1e-12 for value in differences),
            "zero_seed_count": sum(abs(value) <= 1e-12 for value in differences),
            "negative_seed_count": sum(value < -1e-12 for value in differences),
            "leave_one_seed_out_mean_min": min(leave_one_out),
            "leave_one_seed_out_mean_max": max(leave_one_out),
        }
        for seed, difference in zip(SEEDS, differences):
            row[f"seed_{seed}_auc_difference"] = difference
        paired.append(row)
    mean_curves: list[dict[str, object]] = []
    for index, step in enumerate(EVAL_STEPS):
        row: dict[str, object] = {"step": step}
        for method_key, label in METHODS:
            values = [curves[(method_key, seed)][index] for seed in SEEDS]
            row[f"{method_key}_mean_success"] = mean(values)
            row[f"{method_key}_sample_sd_success"] = sample_sd(values)
            row[f"{method_key}_method"] = label
        mean_curves.append(row)
    return {
        "schema_version": 1,
        "seeds": list(SEEDS),
        "evaluation_steps": list(EVAL_STEPS),
        "evaluation_field": SUCCESS_FIELD,
        "auc_convention": "observed-only trapezoid normalized by (2,000,000 - 10,000)",
        "smoothing": False,
        "interpolation": False,
        "forward_fill": False,
        "bootstrap": "ordered exhaustive paired bootstrap, 5^5=3125 index sequences",
        "per_seed": per_seed,
        "aggregate": aggregates,
        "paired_comparisons": paired,
        "mean_curves": mean_curves,
    }


def discover_raw_runs(raw_root: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for run in sorted(raw_root.iterdir()):
        if not run.is_dir() or not (run / "summary.json").is_file():
            continue
        match = re.search(r"_s(\d+)_", run.name)
        if match and int(match.group(1)) in SEEDS:
            found[int(match.group(1))] = run
    if set(found) != set(SEEDS):
        raise ValueError(f"raw result coverage mismatch: {sorted(found)}")
    return found


def input_record(repo_root: Path, path: Path, role: str) -> dict[str, object]:
    resolved = path.resolve()
    try:
        source = resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        source = f"<external>/{resolved.name}"
    return {
        "role": role,
        "source_path": source,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def copy_eval_rows(source: Path, destination: Path) -> None:
    _, _, rows, fields = load_curve(source)
    write_rows(destination, rows, fields)


def write_analysis(output: Path, result: Mapping[str, object]) -> None:
    write_json(output / "analysis_results.json", result)
    write_rows(output / "per_seed_results.csv", result["per_seed"])  # type: ignore[arg-type]
    write_rows(output / "aggregate_results.csv", result["aggregate"])  # type: ignore[arg-type]
    write_rows(output / "paired_comparisons.csv", result["paired_comparisons"])  # type: ignore[arg-type]
    write_rows(output / "exact_grid_mean_curves.csv", result["mean_curves"])  # type: ignore[arg-type]


def packaged_files(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]


def readme(result: Mapping[str, object], source_commit: str) -> str:
    aggregate = {str(row["method_key"]): row for row in result["aggregate"]}  # type: ignore[index]
    paired = {
        str(row["contrast_key"]): row for row in result["paired_comparisons"]  # type: ignore[index]
    }
    primary = paired["main_minus_shuffled"]
    lines = [
        "# DrawerClose DDPG Shuffled-Policy Query, 2M",
        "",
        "Compact paper artifact for paired seeds 1–5. Shuffled-Policy Query applies a "
        "within-minibatch derangement to current-policy actions, preserving the action "
        "multiset while disrupting sample–query alignment. All other SABER settings "
        "remain critic-only and unchanged.",
        "",
        "AUC is the normalized trapezoidal area over the exact observed 10k–2M success "
        "grid. No smoothing, interpolation, forward filling, or extrapolation is used.",
        "",
        "| Method | AUC mean ± sample SD | Final mean ± sample SD | Hit90 median (coverage) |",
        "|---|---:|---:|---:|",
    ]
    for key in ("her", "main", "shuffled"):
        row = aggregate[key]
        hit = row["hit90_median_reached"]
        hit_text = "—" if hit is None else f"{int(float(hit))/1000:.0f}k"
        lines.append(
            f"| {row['method']} | {float(row['auc_mean']):.6f} ± "
            f"{float(row['auc_sample_sd']):.6f} | "
            f"{float(row['final_success_mean']):.6f} ± "
            f"{float(row['final_success_sample_sd']):.6f} | "
            f"{hit_text} ({int(row['hit90_seed_count'])}/5) |"
        )
    lines.extend(
        [
            "",
            "Main − Shuffled AUC difference: "
            f"{float(primary['mean_auc_difference']):.6f}; exhaustive paired-bootstrap "
            f"95% CI [{float(primary['bootstrap_ci95_lower']):.6f}, "
            f"{float(primary['bootstrap_ci95_upper']):.6f}]; Main is higher in "
            f"{int(primary['positive_seed_count'])}/5 seeds.",
            "",
            "The interval includes zero, so this cohort does not independently establish "
            "a sample-alignment effect. It does show a higher Main mean and earlier median "
            "Hit90, while the shuffled control remains substantially above HER.",
            "",
            f"Training source commit: `{source_commit}`. HER and Main comparator curves are "
            "copied from the verified strict context-eligible count-matched artifact and "
            "were not rerun for this comparison.",
            "",
            "Verify from this directory:",
            "",
            "    python artifact_tools.py verify --artifact-dir .",
            "",
            "Checkpoints, complete training-row metrics, JSONL files, and logs are excluded.",
        ]
    )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> None:
    repo = args.repo_root.resolve()
    raw = args.raw_root.resolve()
    comparator = args.comparator_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    runs = discover_raw_runs(raw)
    stage = Path(tempfile.mkdtemp(prefix=".shuffled-artifact-", dir=output.parent))
    replacements = (
        (repo.as_posix(), "<repo>"),
        (raw.as_posix(), "<raw-results>"),
    )
    inputs: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    try:
        for method in ("her", "main"):
            for seed in SEEDS:
                source = comparator / "data" / method / f"seed_{seed}"
                target = stage / "data" / method / f"seed_{seed}"
                copy_eval_rows(source / "metrics_selected_10k.csv", target / "metrics_selected_10k.csv")
                write_json(target / "summary.json", sanitize(read_json(source / "summary.json"), replacements))
                inputs.extend(
                    (
                        input_record(repo, source / "metrics_selected_10k.csv", f"{method}.seed_{seed}.metrics"),
                        input_record(repo, source / "summary.json", f"{method}.seed_{seed}.summary"),
                    )
                )
        for seed, run in runs.items():
            target = stage / "data" / "shuffled" / f"seed_{seed}"
            config = read_json(run / "config.json")
            summary = read_json(run / "summary.json")
            seeds = config.get("seeds", {})
            checks = {
                "total_steps": summary.get("total_steps") == 2_000_000,
                "total_updates": summary.get("total_updates") == 1_995_001,
                "support_query_mode": summary.get("support_query_mode") == "shuffled_policy"
                and config.get("support_query_mode") == "shuffled_policy",
                "critic_only": config.get("weight_scope") == "critic_only",
                "her_ratio": math.isclose(float(config.get("her_ratio", -1)), 0.8),
                "utd_one": int(config.get("updates_per_step", -1)) == 1,
                "eval_protocol": int(config.get("eval_freq", -1)) == 10_000
                and int(config.get("eval_episodes", -1)) == 20,
                "seed_identity": isinstance(seeds, dict)
                and int(seeds.get("env_seed", -1)) == seed
                and int(seeds.get("torch_seed", -1)) == seed,
            }
            if not all(checks.values()):
                raise ValueError(f"seed {seed}: completion/config checks failed: {checks}")
            copy_eval_rows(run / "metrics.csv", target / "metrics_selected_10k.csv")
            write_json(target / "summary.json", sanitize(summary, replacements))
            write_json(target / "config.json", sanitize(config, replacements))
            write_json(target / "env_snapshot.json", sanitize(read_json(run / "env_snapshot.json"), replacements))
            validation.append({"seed": seed, "run_name": run.name, "checks": checks})
            for name in ("metrics.csv", "summary.json", "config.json", "env_snapshot.json"):
                inputs.append(input_record(repo, run / name, f"shuffled.seed_{seed}.{name}"))
        result = analyze(stage / "data")
        write_analysis(stage, result)
        write_json(
            stage / "completion_validation.json",
            {
                "schema_version": 1,
                "status": "passed",
                "seed_coverage": list(SEEDS),
                "evaluation_points_per_seed": len(EVAL_STEPS),
                "records": validation,
            },
        )
        write_json(
            stage / "input_manifest.json",
            {
                "schema_version": 1,
                "training_source_commit": args.source_commit,
                "comparator_artifact": comparator.relative_to(repo).as_posix(),
                "inputs": inputs,
                "excluded": ["checkpoints", "metrics.jsonl", "run logs", "non-evaluation training rows"],
            },
        )
        shutil.copyfile(Path(__file__), stage / "artifact_tools.py")
        (stage / "source_commit.txt").write_text(args.source_commit + "\n", encoding="utf-8")
        (stage / "README.md").write_text(readme(result, args.source_commit), encoding="utf-8")
        records = packaged_files(stage)
        write_json(
            stage / "artifact_manifest.json",
            {
                "schema_version": 1,
                "artifact_id": "drawer_close_ddpg_saber_shuffled_policy_query_2m",
                "training_source_commit": args.source_commit,
                "method_seed_coverage": {key: list(SEEDS) for key, _ in METHODS},
                "evaluation_points_per_seed": len(EVAL_STEPS),
                "file_count": len(records),
                "files": records,
            },
        )
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify(artifact: Path) -> None:
    artifact = artifact.resolve()
    manifest = read_json(artifact / "artifact_manifest.json")
    expected = {
        str(row["path"]): (int(row["size_bytes"]), str(row["sha256"]))
        for row in manifest.get("files", [])
    }
    actual = {
        path.relative_to(artifact).as_posix(): (path.stat().st_size, sha256(path))
        for path in sorted(artifact.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    if actual != expected:
        raise ValueError("artifact manifest mismatch")
    forbidden = ("/" + "home" + "/", "mimiclab-" + "gigabyte-gen4")
    for path in artifact.rglob("*"):
        if not path.is_file() or path.suffix in {".pt", ".png", ".jpg", ".jpeg", ".pdf"}:
            continue
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in forbidden):
            raise ValueError(f"{path}: absolute/private path leaked")
    recomputed = analyze(artifact / "data")
    packaged = read_json(artifact / "analysis_results.json")
    if recomputed != packaged:
        raise ValueError("recomputed analysis differs from packaged analysis")
    validation = read_json(artifact / "completion_validation.json")
    if validation.get("status") != "passed" or validation.get("seed_coverage") != list(SEEDS):
        raise ValueError("completion validation is not passed")
    print(
        json.dumps(
            {
                "status": "passed",
                "artifact": artifact.as_posix(),
                "files": len(actual),
                "seeds": list(SEEDS),
                "evaluation_points_per_seed": len(EVAL_STEPS),
            },
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--repo-root", type=Path, required=True)
    build_parser.add_argument("--raw-root", type=Path, required=True)
    build_parser.add_argument("--comparator-root", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--source-commit", required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--artifact-dir", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "build":
        build(args)
        verify(args.output)
    else:
        verify(args.artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
