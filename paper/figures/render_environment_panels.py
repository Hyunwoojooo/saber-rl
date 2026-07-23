#!/usr/bin/env python3
"""Render reproducible environment panels for the SABER paper."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class CameraPreset:
    lookat: tuple[float, float, float]
    distance: float
    azimuth: float
    elevation: float


@dataclass(frozen=True)
class TaskPreset:
    key: str
    output_name: str
    environment: str
    seed: int
    capture_step: int
    camera: CameraPreset


TASK_PRESETS = {
    "drawer-open": TaskPreset(
        key="drawer-open",
        output_name="drawer_open_v3_render.png",
        environment="Meta-World/MT1:drawer-open-v3",
        seed=1,
        capture_step=90,
        camera=CameraPreset(
            lookat=(-0.04, 0.68, 0.22),
            distance=1.50,
            azimuth=140.0,
            elevation=-23.0,
        ),
    ),
    "drawer-close": TaskPreset(
        key="drawer-close",
        output_name="drawer_close_v3_render.png",
        environment="Meta-World/MT1:drawer-close-v3",
        seed=1,
        capture_step=82,
        camera=CameraPreset(
            lookat=(-0.04, 0.76, 0.22),
            distance=1.50,
            azimuth=140.0,
            elevation=-23.0,
        ),
    ),
    "fpp": TaskPreset(
        key="fpp",
        output_name="fetch_pick_and_place_v4_render.png",
        environment="FetchPickAndPlace-v4",
        seed=3,
        capture_step=0,
        camera=CameraPreset(
            lookat=(1.32, 0.75, 0.58),
            distance=1.70,
            azimuth=115.0,
            elevation=-22.0,
        ),
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("all", *TASK_PRESETS),
        default="all",
        help="Panel to render. The default regenerates all paper panels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--capture-step", type=int)
    parser.add_argument("--lookat", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--distance", type=float)
    parser.add_argument("--azimuth", type=float)
    parser.add_argument("--elevation", type=float)
    return parser.parse_args(argv)


def _make_metaworld_env(task: str, seed: int, width: int, height: int) -> Any:
    import gymnasium as gym
    import metaworld

    gym.register_envs(metaworld)
    return gym.make(
        "Meta-World/MT1",
        env_name=f"{task}-v3",
        seed=seed,
        render_mode="rgb_array",
        width=width,
        height=height,
        max_episode_steps=200,
        terminate_on_success=False,
        reward_function_version="v2",
        disable_env_checker=True,
    )


def _make_fpp_env(width: int, height: int) -> Any:
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
    return gym.make(
        "FetchPickAndPlace-v4",
        render_mode="rgb_array",
        width=width,
        height=height,
        max_episode_steps=50,
    )


def _rollout_metaworld(
    env: Any, task: str, observation: Any, capture_step: int
) -> tuple[Any, dict[str, object]]:
    from metaworld.policies import (
        SawyerDrawerCloseV3Policy,
        SawyerDrawerOpenV3Policy,
    )

    policy = (
        SawyerDrawerOpenV3Policy()
        if task == "drawer-open"
        else SawyerDrawerCloseV3Policy()
    )
    info: dict[str, object] = {}
    for _ in range(capture_step):
        action = policy.get_action(observation)
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            raise RuntimeError(f"episode ended before capture step {capture_step}")
    return observation, info


def _configure_camera(env: Any, camera: CameraPreset) -> None:
    import mujoco

    env.render()
    viewer = env.unwrapped.mujoco_renderer.viewer
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.fixedcamid = -1
    viewer.cam.lookat[:] = camera.lookat
    viewer.cam.distance = camera.distance
    viewer.cam.azimuth = camera.azimuth
    viewer.cam.elevation = camera.elevation


def _camera_from_args(preset: CameraPreset, args: argparse.Namespace) -> CameraPreset:
    return CameraPreset(
        lookat=tuple(args.lookat) if args.lookat is not None else preset.lookat,
        distance=args.distance if args.distance is not None else preset.distance,
        azimuth=args.azimuth if args.azimuth is not None else preset.azimuth,
        elevation=args.elevation if args.elevation is not None else preset.elevation,
    )


def render_task(preset: TaskPreset, args: argparse.Namespace) -> dict[str, object]:
    import numpy as np
    from PIL import Image

    seed = args.seed if args.seed is not None else preset.seed
    capture_step = (
        args.capture_step if args.capture_step is not None else preset.capture_step
    )
    camera = _camera_from_args(preset.camera, args)

    if preset.key == "fpp":
        env = _make_fpp_env(args.width, args.height)
    else:
        env = _make_metaworld_env(preset.key, seed, args.width, args.height)

    try:
        observation, info = env.reset(seed=seed)
        if preset.key != "fpp":
            observation, info = _rollout_metaworld(
                env, preset.key, observation, capture_step
            )

        _configure_camera(env, camera)
        frame = np.asarray(env.render())
        expected_shape = (args.height, args.width, 3)
        if frame.shape != expected_shape or frame.dtype != np.uint8:
            raise RuntimeError(
                f"unexpected render format: shape={frame.shape}, dtype={frame.dtype}"
            )

        output = args.output_dir / preset.output_name
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(output, format="PNG", optimize=False)

        success_key = "is_success" if preset.key == "fpp" else "success"
        metadata: dict[str, object] = {
            "output": str(output.resolve()),
            "environment": preset.environment,
            "seed": seed,
            "capture_step": capture_step,
            "camera": {
                "lookat": camera.lookat,
                "distance": camera.distance,
                "azimuth": camera.azimuth,
                "elevation": camera.elevation,
            },
            "width": args.width,
            "height": args.height,
            "env_success": float(info.get(success_key, 0.0)),
            "illustrative_only": True,
        }
        print(json.dumps(metadata, sort_keys=True))
        return metadata
    finally:
        env.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    selected = (
        TASK_PRESETS.values() if args.task == "all" else [TASK_PRESETS[args.task]]
    )
    for preset in selected:
        render_task(preset, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
