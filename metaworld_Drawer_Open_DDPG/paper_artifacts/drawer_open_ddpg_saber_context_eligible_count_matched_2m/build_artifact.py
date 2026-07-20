#!/usr/bin/env python3
"""Build the non-overwriting compact DrawerOpen strict-control paper artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import analyze_results as analysis
import validate_completion as completion


ARTIFACT_ID = "drawer_open_ddpg_saber_context_eligible_count_matched_2m"
EXPERIMENT = "DDPG-HER-B3-Strict-Context-Eligible-Count-Matched-2M"
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_HER_ROOT = REPO_ROOT / "metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B0-2M"
DEFAULT_MAIN_ROOT = REPO_ROOT / "metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B3-Critic-Only-2M"
DEFAULT_STRICT_ROOT = REPO_ROOT / f"metaworld_Drawer_Open_DDPG/results_2m/{EXPERIMENT}"
DEFAULT_OUTPUT = REPO_ROOT / f"metaworld_Drawer_Open_DDPG/paper_artifacts/{ARTIFACT_ID}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def clean_json(value: Any, replacements: Sequence[tuple[str, str]]) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): clean_json(item, replacements) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item, replacements) for item in value]
    if isinstance(value, str):
        result = value
        for old, new in replacements:
            result = result.replace(old, new)
        return result
    return value


def record_input(repo_root: Path, path: Path, role: str) -> dict[str, object]:
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


def copy_selected_metrics(source: Path, destination: Path) -> None:
    expected = set(analysis.EVAL_STEPS)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if analysis.PRIMARY_SUCCESS_FIELD not in fieldnames:
            raise ValueError(f"{source}: missing primary evaluation field")
        rows = []
        seen: set[int] = set()
        for row in reader:
            if row.get(analysis.PRIMARY_SUCCESS_FIELD, "") in ("", None):
                continue
            step_value = analysis.finite_float(row.get("step"))
            success = analysis.finite_float(row.get(analysis.PRIMARY_SUCCESS_FIELD))
            if step_value is None or success is None:
                raise ValueError(f"{source}: non-finite evaluation row")
            step = int(step_value)
            if step not in expected or step in seen:
                raise ValueError(f"{source}: unexpected or duplicate evaluation step {step}")
            seen.add(step)
            rows.append(row)
    if seen != expected or [int(float(row["step"])) for row in rows] != list(analysis.EVAL_STEPS):
        raise ValueError(f"{source}: exact observed-only evaluation grid mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def packaged_files(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "artifact_manifest.json":
            continue
        rows.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    return rows


def content_digest(records: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in records:
        digest.update(str(row["path"]).encode())
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def source_files_from_freeze(freeze_path: Path) -> list[dict[str, object]]:
    payload = analysis.read_json(freeze_path)
    rows = payload.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{freeze_path}: empty source freeze")
    return [dict(row) for row in rows if isinstance(row, dict)]


def readme_text(result: Mapping[str, object], source_commit: str) -> str:
    aggregate = {str(row["method_key"]): row for row in result["aggregate"]}  # type: ignore[index]
    primary = next(
        row for row in result["paired_comparisons"]  # type: ignore[index]
        if row["contrast_key"] == "main_minus_strict"
    )
    delta = float(primary["mean_auc_difference"])
    if delta > analysis.ZERO_TOLERANCE:
        interpretation = (
            "SABER Main exceeded the strict control on mean paired AUC. This is evidence "
            "that exact action-bin membership may contribute beyond context eligibility "
            "and positive count; five seeds do not establish a universal causal effect."
        )
    elif delta < -analysis.ZERO_TOLERANCE:
        interpretation = (
            "The strict control exceeded SABER Main on mean paired AUC, indicating that "
            "exact membership may not be the optimal assignment on this task and should "
            "be reported as a task-dependent limitation."
        )
    else:
        interpretation = (
            "Mean paired AUC was equal at recorded precision; the gain may be explained "
            "by populated context or generic critic-loss redistribution rather than exact membership."
        )
    lines = [
        f"# {EXPERIMENT}",
        "",
        "Compact, independently verifiable artifact for seeds 1–5. Primary AUC uses only the",
        "observed 10k–2M grid, trapezoidal integration, and no smoothing, interpolation,",
        "forward-fill, or extrapolation.",
        "",
        "| Method | AUC mean ± sample SD | Final mean ± sample SD |",
        "|---|---:|---:|",
    ]
    for key in ("her", "main", "strict"):
        row = aggregate[key]
        lines.append(
            f"| {row['method']} | {float(row['auc_mean']):.6f} ± "
            f"{float(row['auc_sample_sd']):.6f} | {float(row['final_success_mean']):.6f} ± "
            f"{float(row['final_success_sample_sd']):.6f} |"
        )
    lines += [
        "",
        "Primary paired contrast (SABER Main − Strict Control): "
        f"{delta:.6f}, exhaustive paired-bootstrap 95% CI "
        f"[{float(primary['bootstrap_ci95_lower']):.6f}, "
        f"{float(primary['bootstrap_ci95_upper']):.6f}], exact two-sided sign-flip "
        f"p={float(primary['exact_two_sided_sign_flip_pvalue']):.6f}.",
        "",
        interpretation,
        "",
        f"Frozen source commit: `{source_commit}`.",
        "",
        "Verify from this directory with:",
        "",
        "```bash",
        "python verify_artifact.py --artifact-dir .",
        "```",
        "",
        "Checkpoints, full raw metrics, JSONL, logs, correctness, smoke, and pilot outputs",
        "are intentionally excluded.",
    ]
    return "\n".join(lines) + "\n"


def build(
    repo_root: Path,
    her_root: Path,
    main_root: Path,
    strict_root: Path,
    output_dir: Path,
) -> None:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    gate = completion.validate_all(strict_root.resolve(), repo_root)
    methods = (
        analysis.MethodSpec("her", "DrawerOpen DDPG HER", her_root.resolve()),
        analysis.MethodSpec("main", "DrawerOpen DDPG HER + SABER Main", main_root.resolve()),
        analysis.MethodSpec(
            "strict",
            "DrawerOpen DDPG Strict Context-Eligible Count-Matched",
            strict_root.resolve(),
        ),
    )
    result = analysis.analyze(methods)
    runs = {method.key: analysis.discover_runs(method.root) for method in methods}
    replacements = ((repo_root.as_posix(), "."),)
    stage = Path(tempfile.mkdtemp(prefix=f".{ARTIFACT_ID}.", dir=output_dir.parent))
    inputs: list[dict[str, object]] = []
    try:
        packaged_result = clean_json(result, replacements)
        for row in packaged_result["per_seed"]:
            key = str(row["method_key"])
            seed = int(row["seed"])
            row["run_dir"] = f"data/{key}/seed_{seed}"
            row["metrics_path"] = f"data/{key}/seed_{seed}/metrics_selected_10k.csv"
        analysis.write_csv(stage / "per_seed_results.csv", packaged_result["per_seed"])
        analysis.write_csv(stage / "aggregate_results.csv", packaged_result["aggregate"])
        analysis.write_csv(stage / "paired_comparisons.csv", packaged_result["paired_comparisons"])
        analysis.write_csv(stage / "exact_grid_mean_curves.csv", packaged_result["mean_curves"])
        write_json(stage / "analysis_results.json", packaged_result)

        gate_records = {int(row["seed"]): row for row in gate["records"]}
        for method in methods:
            for seed, run_dir in runs[method.key].items():
                target = stage / "data" / method.key / f"seed_{seed}"
                metrics = analysis.metrics_path_for(run_dir)
                copy_selected_metrics(metrics, target / "metrics_selected_10k.csv")
                inputs.append(record_input(repo_root, metrics, f"{method.key}.seed_{seed}.metrics"))
                summary_path = run_dir / "summary.json"
                write_json(
                    target / "summary.json",
                    clean_json(analysis.read_json(summary_path), replacements),
                )
                inputs.append(record_input(repo_root, summary_path, f"{method.key}.seed_{seed}.summary"))
                if method.key == "strict":
                    config_path = run_dir / "config.json"
                    env_path = run_dir / "env_snapshot.json"
                    config = analysis.read_json(config_path)
                    trace = run_dir / f"strict_diagnostics_{int(config['assignment_diagnostic_window_updates'])}.csv"
                    shutil.copyfile(config_path, target / "config.json")
                    shutil.copyfile(env_path, target / "env_snapshot.json")
                    shutil.copyfile(trace, target / "strict_diagnostic_trace.csv")
                    write_json(
                        target / "completion_validation.json",
                        clean_json(gate_records[seed], replacements),
                    )
                    for source, role in (
                        (config_path, "config"),
                        (env_path, "env_snapshot"),
                        (trace, "strict_diagnostic_trace"),
                        (run_dir / "checkpoint.pt", "completion_checkpoint_excluded"),
                    ):
                        inputs.append(record_input(repo_root, source, f"strict.seed_{seed}.{role}"))

        experiment_root = HERE
        provenance_names = (
            "config_parity.json",
            "checkpoint_resume_audit.json",
            "source_parity.json",
            "freeze_manifest.json",
        )
        for name in provenance_names:
            source = experiment_root / name
            if not source.is_file():
                raise FileNotFoundError(f"required provenance file missing: {source}")
            shutil.copyfile(source, stage / "provenance" / name)
            inputs.append(record_input(repo_root, source, f"provenance.{name}"))

        freeze_path = experiment_root / "freeze_manifest.json"
        frozen_rows = source_files_from_freeze(freeze_path)
        frozen_lines = []
        for row in frozen_rows:
            relative = str(row["path"])
            source = experiment_root / relative
            destination = stage / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if destination.stat().st_size != int(row["size"]) or sha256(destination) != row["sha256"]:
                raise ValueError(f"frozen source copy mismatch: {relative}")
            frozen_lines.append(f"{row['sha256']}  source_snapshot/{relative}")
        (stage / "frozen_source.sha256").write_text("\n".join(frozen_lines) + "\n", encoding="utf-8")

        for name in ("build_artifact.py", "verify_artifact.py"):
            source = experiment_root / name
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copyfile(source, stage / name)

        write_json(stage / "completion_validation.json", clean_json(gate, replacements))
        input_manifest = {
            "schema_version": 1,
            "artifact_id": ARTIFACT_ID,
            "source_commit": gate["source_commit"],
            "input_file_count": len(inputs),
            "inputs": inputs,
            "transform": {
                "metrics": "exact observed evaluation rows at 10k..2M; all columns retained",
                "json": "machine-local repository path replaced by '.'; NaN/Inf replaced by null",
                "checkpoints_copied": False,
                "raw_metrics_copied": False,
                "jsonl_copied": False,
                "logs_copied": False,
                "smoke_or_pilot_copied": False,
            },
        }
        write_json(stage / "input_manifest.json", input_manifest)
        (stage / "README.md").write_text(
            readme_text(packaged_result, str(gate["source_commit"])),
            encoding="utf-8",
        )
        records = packaged_files(stage)
        manifest = {
            "schema_version": 1,
            "artifact_id": ARTIFACT_ID,
            "experiment": EXPERIMENT,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": gate["source_commit"],
            "method_seed_coverage": {key: list(analysis.SEEDS) for key in ("her", "main", "strict")},
            "evaluation": {
                "field": analysis.PRIMARY_SUCCESS_FIELD,
                "steps": list(analysis.EVAL_STEPS),
                "points_per_seed": len(analysis.EVAL_STEPS),
                "episodes_per_point": completion.EVAL_EPISODES,
                "auc": "observed-only normalized trapezoid",
                "smoothing": False,
                "interpolation": False,
                "forward_fill": False,
            },
            "bootstrap": {
                "method": "ordered exhaustive paired bootstrap",
                "resamples": analysis.BOOTSTRAP_RESAMPLE_COUNT,
                "confidence_level": 0.95,
            },
            "exact_sign_flip": {"two_sided": True, "assignments": analysis.SIGN_FLIP_COUNT},
            "completion_passed": gate["passed"],
            "config_parity_passed": True,
            "source_parity_passed": True,
            "packaged_file_count_excluding_manifest": len(records),
            "package_content_sha256_excluding_manifest": content_digest(records),
            "packaged_files": records,
            "self_hash_note": "artifact_manifest.json is excluded to avoid recursive self-hash",
        }
        write_json(stage / "artifact_manifest.json", manifest)
        stage.rename(output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--her-root", type=Path, default=DEFAULT_HER_ROOT)
    parser.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    parser.add_argument("--strict-root", type=Path, default=DEFAULT_STRICT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build(args.repo_root, args.her_root, args.main_root, args.strict_root, args.output_dir)
    print(json.dumps({"status": "built", "artifact": args.output_dir.resolve().as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
