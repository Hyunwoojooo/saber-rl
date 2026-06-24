from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from experiment_registry import EXPERIMENTS, load_folder_config, repo_root_from_shared, run_name


def workspace_root_from_shared() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_trainer(repo_root: Path, workspace_root: Path) -> Path:
    candidates = [
        repo_root / "version_v0_lite" / "src" / "train_fetch_pick_and_place_lite_v0.py",
        workspace_root
        / "experiment_FPP"
        / "b3_critic_only_runtime_opt_v4_ls512"
        / "code"
        / "src"
        / "train_fetch_pick_and_place_lite_v0.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find FPP trainer. Searched:\n{searched}")


def resolve_python(repo_root: Path, workspace_root: Path) -> Path:
    candidates = [
        repo_root / ".venv" / "bin" / "python",
        workspace_root / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def build_command(experiment: str, seed: int, steps: int | None, device: str, output_root: str | None, smoke: bool) -> list[str]:
    if experiment not in EXPERIMENTS:
        known = ", ".join(EXPERIMENTS)
        raise SystemExit(f"Unknown experiment {experiment!r}. Known: {known}")
    spec = EXPERIMENTS[experiment]
    folder_cfg = load_folder_config(experiment)
    repo_root = repo_root_from_shared()
    workspace_root = workspace_root_from_shared()
    trainer = resolve_trainer(repo_root, workspace_root)
    python_bin = resolve_python(repo_root, workspace_root)
    n_steps = int(spec.total_steps if steps is None else steps)
    out_root = output_root or str(workspace_root / "experiment_FPP" / "runs" / experiment)
    cmd = [
        str(python_bin),
        str(trainer),
        "--task_name", "fetch_pick_and_place",
        "--run_name", run_name(spec, seed, n_steps),
        "--output_root", out_root,
        "--method", spec.trainer_method,
        "--reward_type", spec.reward_type,
        "--total_steps", str(n_steps),
        "--batch_size", str(spec.batch_size),
        "--her_ratio", str(spec.her_ratio),
        "--updates_per_step", str(spec.updates_per_step),
        "--eval_freq", str(spec.eval_freq),
        "--eval_episodes", str(spec.eval_episodes),
        "--M_B", str(spec.M_B),
        "--M_Q_min", str(spec.M_Q_min),
        "--N_local_min", str(spec.N_local_min),
        "--N_gap_min", str(spec.N_gap_min),
        "--local_search_max", str(spec.local_search_max),
        "--lambda_support", str(spec.lambda_support),
        "--lambda_TD", str(spec.lambda_TD),
        "--U_TD_max", str(spec.U_TD_max),
        "--per_alpha", str(spec.per_alpha),
        "--eps_PER", str(spec.eps_PER),
        "--U_PER_max", str(spec.U_PER_max),
        "--weighting_mode", folder_cfg.get("weighting_mode", "batch_norm"),
        "--weight_scope", spec.weight_scope,
        "--policy_support_mode", folder_cfg.get("policy_support_mode", "bin"),
        "--device", device,
        "--env_seed", str(seed),
        "--torch_seed", str(seed),
        "--numpy_seed", str(seed),
        "--action_noise_seed", str(seed),
        "--weight_seed", str(20000 + int(seed)),
        "--eval_seed", str(10000 + int(seed)),
    ]
    if smoke:
        cmd.append("--smoke")
    return cmd


def main(default_experiment: str | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one SABER FetchPickAndPlace experiment folder.")
    parser.add_argument("--experiment", default=default_experiment, choices=sorted(EXPERIMENTS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.experiment is None:
        parser.error("--experiment is required when not launched from an experiment folder")
    cmd = build_command(args.experiment, args.seed, args.steps, args.device, args.output_root, args.smoke)
    printable = " ".join(shlex.quote(part) for part in cmd)
    if args.dry_run:
        print(printable)
        return 0
    print(printable, flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
