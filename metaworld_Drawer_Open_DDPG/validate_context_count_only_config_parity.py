#!/usr/bin/env python3
"""Statically compare the Context-count-only launcher with DrawerOpen SABER Main."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MAIN_LAUNCHER = ROOT / "DDPG-HER-B3-Critic-Only-2M.sh"
CONTROL_LAUNCHER = ROOT / "DDPG-HER-B3-Context-Count-Only-2M.sh"
CONTROL_SLUG = "DDPG-HER-B3-Context-Count-Only-2M"
ALLOWED_OPTION_DIFFERENCES = {"--support_query_mode"}

# These defaults are part of the backward-compatible Main contract. They are
# made explicit in the control launcher so its frozen command is self-describing.
IMPLICIT_MAIN_OPTIONS: dict[str, str] = {
    "--action_bins_per_dim": "5",
    "--support_query_mode": "current_policy",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _function_body(source: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^{re.escape(name)}\(\)\s*\{{\n(?P<body>.*?)^\}}\s*$",
        source,
    )
    if match is None:
        raise ValueError(f"could not find shell function: {name}")
    return match.group("body")


def _options_from_function(source: str, name: str) -> dict[str, str | bool]:
    options: dict[str, str | bool] = {}
    for raw_line in _function_body(source, name).splitlines():
        line = raw_line.strip()
        if line.endswith("\\"):
            line = line[:-1].rstrip()
        if not line.startswith("--"):
            continue
        tokens = shlex.split(line)
        if not tokens or not tokens[0].startswith("--"):
            continue
        option = tokens[0]
        if option in options:
            raise ValueError(f"duplicate option in {name}: {option}")
        options[option] = True if len(tokens) == 1 else " ".join(tokens[1:])
    return options


def _run_variant_tokens(source: str) -> list[str]:
    lines = source.splitlines()
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped.startswith("run_variant") or stripped.startswith("run_variant()"):
            continue
        logical_lines = []
        cursor = index
        while cursor < len(lines):
            part = lines[cursor].strip()
            continued = part.endswith("\\")
            if continued:
                part = part[:-1].rstrip()
            logical_lines.append(part)
            if not continued:
                break
            cursor += 1
        tokens = shlex.split(" ".join(logical_lines))
        if len(tokens) != 5 or tokens[0] != "run_variant":
            raise ValueError(f"unexpected run_variant invocation: {tokens}")
        return tokens
    raise ValueError("run_variant invocation not found")


def _slug_assignment(source: str) -> str | None:
    match = re.search(r'^SLUG="(?P<slug>[^"]+)"$', source, re.MULTILINE)
    return None if match is None else match.group("slug")


def launcher_options(source: str, *, main: bool) -> dict[str, str | bool]:
    options = {
        **_options_from_function(source, "common_args"),
        **_options_from_function(source, "b3_args"),
    }
    invocation = _run_variant_tokens(source)
    options["--method"] = invocation[3]
    options["--weight_scope"] = invocation[4]
    if main:
        for option, value in IMPLICIT_MAIN_OPTIONS.items():
            options.setdefault(option, value)
    return options


def _option_differences(
    main: dict[str, str | bool], control: dict[str, str | bool]
) -> list[dict[str, Any]]:
    differences = []
    for option in sorted(set(main) | set(control)):
        main_value = main.get(option, "<missing>")
        control_value = control.get(option, "<missing>")
        if main_value != control_value:
            differences.append(
                {
                    "option": option,
                    "main": main_value,
                    "control": control_value,
                    "allowed": option in ALLOWED_OPTION_DIFFERENCES,
                }
            )
    return differences


def _source_checks(source: str) -> dict[str, bool]:
    required_snippets = {
        "default_seeds_1_5": 'SEEDS="${SEEDS:-1 2 3 4 5}"',
        "default_steps_2m": 'STEPS="${STEPS:-2000000}"',
        "default_step_tag_2m": 'STEP_TAG="${STEP_TAG:-2m}"',
        "cuda_override": 'CUDA_DEVICE="${CUDA_DEVICE:-0}"',
        "output_root_override": 'OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results_2m}"',
        "log_root_override": 'LOG_ROOT="${LOG_ROOT:-$ROOT/logs_2m}"',
        "context_slug": f'SLUG="{CONTROL_SLUG}"',
        "configured_seed_iteration": 'for seed in "${SEED_ARRAY[@]}"; do',
        "context_query_mode": "--support_query_mode context_count_only",
        "critic_only_scope": '"critic_only"',
        "main_method": '"b3_support_runtime_opt"',
        "run_name_contract": 'local run_name="drawer_open_${slug}_s${seed}_${STEP_TAG}_cuda"',
        "output_contract": 'local output_dir="$OUTPUT_ROOT/$slug"',
        "descriptive_label": (
            "Context eligibility only (count-unmatched descriptive control)"
        ),
    }
    checks = {
        name: snippet in source for name, snippet in required_snippets.items()
    }
    forbidden_options = (
        "--random_fixed_count_k",
        "--per_alpha",
        "--eps_PER",
        "--U_PER_max",
        "--support_assignment_mode",
    )
    checks["no_other_assignment_control"] = not any(
        option in source for option in forbidden_options
    )
    return checks


def build_report() -> dict[str, Any]:
    main_source = MAIN_LAUNCHER.read_text(encoding="utf-8")
    control_source = CONTROL_LAUNCHER.read_text(encoding="utf-8")
    main_options = launcher_options(main_source, main=True)
    control_options = launcher_options(control_source, main=False)
    differences = _option_differences(main_options, control_options)
    nonallowed = [row for row in differences if not row["allowed"]]
    source_checks = _source_checks(control_source)
    control_invocation = _run_variant_tokens(control_source)
    control_slug = (
        _slug_assignment(control_source)
        if control_invocation[2] == "$SLUG"
        else control_invocation[2]
    )
    identity_checks = {
        "slug": control_slug == CONTROL_SLUG,
        "method": control_options.get("--method") == "b3_support_runtime_opt",
        "weight_scope": control_options.get("--weight_scope") == "critic_only",
        "query_mode": (
            control_options.get("--support_query_mode") == "context_count_only"
        ),
        "main_query_mode": (
            main_options.get("--support_query_mode") == "current_policy"
        ),
        "only_allowed_option_difference": (
            {row["option"] for row in differences}
            == ALLOWED_OPTION_DIFFERENCES
        ),
    }
    return {
        "schema_version": 1,
        "description": (
            "Context eligibility only (count-unmatched descriptive control)"
        ),
        "main_launcher": str(MAIN_LAUNCHER.relative_to(ROOT.parent)),
        "main_launcher_sha256": _sha256(MAIN_LAUNCHER),
        "control_launcher": str(CONTROL_LAUNCHER.relative_to(ROOT.parent)),
        "control_launcher_sha256": _sha256(CONTROL_LAUNCHER),
        "allowed_option_differences": sorted(ALLOWED_OPTION_DIFFERENCES),
        "option_differences": differences,
        "nonallowed_option_differences": nonallowed,
        "source_checks": source_checks,
        "identity_checks": identity_checks,
        "passed": (
            not nonallowed
            and all(source_checks.values())
            and all(identity_checks.values())
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON report path (nothing is written when omitted)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report()
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["passed"]:
        raise RuntimeError("Context-count-only/Main launcher parity failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
