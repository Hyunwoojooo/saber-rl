#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path


FOLDER = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("query_ablation_run", FOLDER / "run.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load run.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)

MODES = ("current_policy", "replay_action", "shuffled_policy", "context_count_only")
ALLOWED_MODE_DIFFERENCES = {"--run_name", "--output_root", "--support_query_mode"}


def command_options(command: list[str]) -> dict[str, str | bool]:
    options: dict[str, str | bool] = {}
    index = 2
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            raise AssertionError(f"unexpected command token: {token}")
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            options[token] = command[index + 1]
            index += 2
        else:
            options[token] = True
            index += 1
    return options


def build(mode: str, seed: int, steps: int | None = None, smoke: bool = False) -> dict[str, str | bool]:
    command = RUN.build_command(
        mode=mode,
        seed=seed,
        steps=steps,
        device="cuda",
        output_root=None,
        smoke=smoke,
        validate_only=False,
    )
    return command_options(command)


def main() -> int:
    cfg = RUN.load_config(FOLDER)
    for seed in cfg["seeds"]:
        by_mode = {mode: build(mode, seed) for mode in MODES}
        reference = by_mode["current_policy"]
        for mode, options in by_mode.items():
            keys = set(reference) | set(options)
            differences = {key for key in keys if reference.get(key) != options.get(key)}
            assert differences <= ALLOWED_MODE_DIFFERENCES, (mode, sorted(differences))
            assert options["--support_query_mode"] == mode
            assert options["--env_seed"] == str(seed)
            assert options["--torch_seed"] == str(seed)
            assert options["--numpy_seed"] == str(seed)
            assert options["--action_noise_seed"] == str(seed)
            assert options["--weight_seed"] == str(20_000 + seed)
            assert options["--eval_seed"] == str(10_000 + seed)
            assert options["--total_steps"] == "1000000"
            assert options["--batch_size"] == "256"
            assert options["--eval_freq"] == "10000"
            assert options["--eval_episodes"] == "20"
            assert options["--local_search_max"] == "512"
            expected_root = str(RUN.repo_root_from(FOLDER) / cfg["mode_output_roots"][mode])
            assert options["--output_root"] == expected_root

    smoke = build("replay_action", 1, smoke=True)
    assert smoke["--total_steps"] == "5000"
    assert smoke["--smoke"] is True
    pilot = build("replay_action", 1, steps=50_000)
    assert pilot["--total_steps"] == "50000"
    assert "--smoke" not in pilot

    print("test_run_config_parity.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
