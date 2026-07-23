#!/usr/bin/env python3
"""Render a reproducible Meta-World DrawerOpen environment panel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("drawer_open_v3_render.png"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--capture-step", type=int, default=90)
    parser.add_argument("--camera-name", default="corner4")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    import gymnasium as gym
    import metaworld
    import numpy as np
    from metaworld.policies import SawyerDrawerOpenV3Policy
    from PIL import Image

    gym.register_envs(metaworld)
    env = gym.make(
        "Meta-World/MT1",
        env_name="drawer-open-v3",
        seed=args.seed,
        render_mode="rgb_array",
        camera_name=args.camera_name,
        width=args.width,
        height=args.height,
        max_episode_steps=200,
        terminate_on_success=False,
        reward_function_version="v2",
        disable_env_checker=True,
    )

    try:
        observation, _ = env.reset(seed=args.seed)
        policy = SawyerDrawerOpenV3Policy()
        info: dict[str, object] = {}
        for _ in range(args.capture_step):
            action = policy.get_action(observation)
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                raise RuntimeError(
                    f"episode ended before capture step {args.capture_step}"
                )

        frame = np.asarray(env.render())
        expected_shape = (args.height, args.width, 3)
        if frame.shape != expected_shape or frame.dtype != np.uint8:
            raise RuntimeError(
                f"unexpected render format: shape={frame.shape}, dtype={frame.dtype}"
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(args.output, format="PNG", optimize=False)
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "environment": "Meta-World/MT1:drawer-open-v3",
                    "seed": args.seed,
                    "capture_step": args.capture_step,
                    "camera_name": args.camera_name,
                    "width": args.width,
                    "height": args.height,
                    "env_success": float(info.get("success", 0.0)),
                    "illustrative_only": True,
                },
                sort_keys=True,
            )
        )
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
