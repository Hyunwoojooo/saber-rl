#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import analyze_context_count_only_results as analysis


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ARTIFACT = HERE / "paper_artifacts/drawer_open_ddpg_context_count_only_2m"
CONTEXT_ROOT = HERE / "results_2m/DDPG-HER-B3-Context-Count-Only-2M"
HER_ROOT = HERE / "results_2m/DDPG-HER-B0-2M"
MAIN_ROOT = HERE / "results_2m/DDPG-HER-B3-Critic-Only-2M"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def selected_rows(source: Path, evaluation_only: bool) -> tuple[list[str], list[dict[str, str]]]:
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            if evaluation_only:
                if row.get("eval/env_success_mean", "") in {"", "nan", "NaN"}:
                    continue
            else:
                if row.get("query/context_count_eligible_rate", "") in {"", "nan", "NaN"}:
                    continue
            rows.append(row)
    return fields, rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    runs = analysis.discover_runs(CONTEXT_ROOT)
    with tempfile.TemporaryDirectory(prefix="context-artifact-", dir=output.parent) as temp:
        stage = Path(temp) / output.name
        stage.mkdir(parents=True)
        methods = (
            analysis.MethodSpec("her", "DrawerOpen DDPG HER", HER_ROOT),
            analysis.MethodSpec("main", "DrawerOpen DDPG HER + SABER Main", MAIN_ROOT),
            analysis.MethodSpec("context", "Context eligibility only (count-unmatched descriptive control)", CONTEXT_ROOT),
        )
        analysis.analyze(methods, stage)
        for seed, run_dir in runs.items():
            seed_dir = stage / f"seed_{seed}"
            seed_dir.mkdir()
            for name in ("config.json", "env_snapshot.json", "summary.json"):
                shutil.copy2(run_dir / name, seed_dir / name)
            fields, eval_rows = selected_rows(run_dir / "metrics.csv", True)
            if [int(float(row["step"])) for row in eval_rows] != list(analysis.EVAL_STEPS):
                raise AssertionError(f"seed {seed}: exact grid mismatch")
            write_csv(seed_dir / "metrics_selected_10k.csv", fields, eval_rows)
            fields, diagnostic_rows = selected_rows(run_dir / "metrics.csv", False)
            write_csv(seed_dir / "context_diagnostics.csv", fields, diagnostic_rows)
        for name in (
            "analyze_context_count_only_results.py",
            "test_analyze_context_count_only_results.py",
            "validate_context_count_only_run.py",
            "validate_context_count_only_config_parity.py",
            "test_context_count_only_config_parity.py",
            "context_count_only_source_manifest.json",
            "DDPG-HER-B3-Context-Count-Only-2M.sh",
            "build_context_count_only_artifact.py",
            "verify_context_count_only_artifact.py",
        ):
            shutil.copy2(HERE / name, stage / name)
        (stage / "README.md").write_text(
            "# DrawerOpen DDPG context-count-only 2M\n\n"
            "Context eligibility only (count-unmatched descriptive control). "
            "All curves use the exact observed 10k grid without smoothing, interpolation, or forward fill. "
            "Main minus context is descriptive because positive counts and weight histograms differ.\n"
        )
        inputs = []
        for seed, run_dir in runs.items():
            for name in ("config.json", "env_snapshot.json", "summary.json", "metrics.csv"):
                path = run_dir / name
                inputs.append({"seed": seed, "path": str(path.relative_to(REPO)), "size": path.stat().st_size, "sha256": digest(path)})
        write_json(stage / "input_manifest.json", {"files": inputs})
        files = []
        for path in sorted(stage.rglob("*")):
            if path.is_file() and path.name != "artifact_manifest.json":
                files.append({"path": path.relative_to(stage).as_posix(), "size": path.stat().st_size, "sha256": digest(path)})
        write_json(stage / "artifact_manifest.json", {"schema_version": 1, "files": files})
        output.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(output)
    print(json.dumps({"artifact": str(output), "size_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
