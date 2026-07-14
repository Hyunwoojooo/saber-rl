from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DISPLAY_COMPONENTS = (
    "Interaction",
    "Replay + HER",
    "Batch preparation",
    "Support and weighting",
    "Critic update",
    "Actor + target update",
    "Eval/logging amortized",
)


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("profile_schema_version") is None:
        raise ValueError(f"Not a runtime profile summary: {path}")
    return payload


def component_ms(summary: dict[str, Any], name: str) -> float:
    return float(summary.get("components", {}).get(name, {}).get("ms_per_env_step", 0.0))


def validate_matched_pair(her: dict[str, Any], saber: dict[str, Any]) -> None:
    if her.get("method") != "td3_her":
        raise ValueError(f"HER summary has unexpected method: {her.get('method')!r}")
    if saber.get("method") != "td3_saber_runtime_opt":
        raise ValueError(f"SABER summary has unexpected method: {saber.get('method')!r}")
    for field in (
        "seed",
        "total_steps",
        "profile_start_step",
        "profile_end_step",
        "profile_sample_every_updates",
        "profiled_env_steps",
    ):
        if her.get(field) != saber.get(field):
            raise ValueError(
                f"Profile pair is not matched for {field}: HER={her.get(field)!r}, "
                f"SABER={saber.get(field)!r}"
            )
    her_device = her.get("hardware_metadata", {}).get("cuda_device_name", her.get("device"))
    saber_device = saber.get("hardware_metadata", {}).get("cuda_device_name", saber.get("device"))
    if her_device != saber_device:
        raise ValueError(f"Profile pair used different devices: HER={her_device!r}, SABER={saber_device!r}")


def build_comparison(her: dict[str, Any], saber: dict[str, Any]) -> dict[str, Any]:
    validate_matched_pair(her, saber)
    her_sps = float(her["steps_per_sec_excluding_eval"])
    saber_sps = float(saber["steps_per_sec_excluding_eval"])
    her_training = float(her["training_excluding_eval_sec"])
    saber_training = float(saber["training_excluding_eval_sec"])
    rows = []
    for component in DISPLAY_COMPONENTS:
        her_ms = component_ms(her, component)
        saber_ms = component_ms(saber, component)
        rows.append(
            {
                "component": component,
                "her_ms_per_env_step": her_ms,
                "saber_ms_per_env_step": saber_ms,
                "delta_ms_per_env_step": saber_ms - her_ms,
                "saber_training_share": float(
                    saber.get("components", {}).get(component, {}).get("training_share", 0.0)
                ),
            }
        )
    her_total_ms = 1_000.0 * her_training / max(1, int(her["profiled_env_steps"]))
    saber_total_ms = 1_000.0 * saber_training / max(1, int(saber["profiled_env_steps"]))
    rows.append(
        {
            "component": "Total training loop",
            "her_ms_per_env_step": her_total_ms,
            "saber_ms_per_env_step": saber_total_ms,
            "delta_ms_per_env_step": saber_total_ms - her_total_ms,
            "saber_training_share": 1.0,
        }
    )
    return {
        "comparison_schema_version": "1.0",
        "her_method": her.get("method"),
        "saber_method": saber.get("method"),
        "seed": her.get("seed"),
        "profile_start_step": her.get("profile_start_step"),
        "profile_end_step": her.get("profile_end_step"),
        "her_steps_per_sec_excluding_eval": her_sps,
        "saber_steps_per_sec_excluding_eval": saber_sps,
        "throughput_slowdown": 1.0 - saber_sps / max(1e-12, her_sps),
        "her_training_sec": her_training,
        "saber_training_sec": saber_training,
        "relative_training_overhead": (saber_training - her_training) / max(1e-12, her_training),
        "component_delta_ms_per_step": {
            row["component"]: row["delta_ms_per_env_step"] for row in rows
        },
        "rows": rows,
        "hardware_specific": True,
        "interpretation_limit": (
            "Implementation profile on one hardware/software stack; not a wall-clock superiority claim "
            "and not a basis for linear 1M-step extrapolation."
        ),
    }


def write_outputs(comparison: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runtime_profile_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = (
        "component",
        "her_ms_per_env_step",
        "saber_ms_per_env_step",
        "delta_ms_per_env_step",
        "saber_training_share",
    )
    with (output_dir / "runtime_profile_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison["rows"])

    lines = [
        "# FPP TD3 HER vs HER+SABER Runtime Profile",
        "",
        f"- Throughput slowdown: `{100.0 * comparison['throughput_slowdown']:.2f}%`",
        f"- Relative training overhead: `{100.0 * comparison['relative_training_overhead']:.2f}%`",
        "",
        "| Component | HER ms/env-step | SABER ms/env-step | Delta | SABER training share |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison["rows"]:
        lines.append(
            "| {component} | {her_ms_per_env_step:.4f} | {saber_ms_per_env_step:.4f} | "
            "{delta_ms_per_env_step:+.4f} | {share:.2f}% |".format(
                **row,
                share=100.0 * float(row["saber_training_share"]),
            )
        )
    lines.extend(
        [
            "",
            "This is a hardware-specific implementation profile. It does not establish wall-clock superiority, ",
            "and the 100k measurements must not be linearly extrapolated to a 1M-step runtime claim.",
            "",
        ]
    )
    (output_dir / "runtime_profile_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare matched TD3 HER and HER+SABER runtime profiles.")
    parser.add_argument("--her-summary", type=Path, required=True)
    parser.add_argument("--saber-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison = build_comparison(load_summary(args.her_summary), load_summary(args.saber_summary))
    write_outputs(comparison, args.output_dir)
    print(args.output_dir / "runtime_profile_comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
