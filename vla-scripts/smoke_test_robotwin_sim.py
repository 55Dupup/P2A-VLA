#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_ROBOTWIN_PATH = Path("/path/to/RoboTwin")
REQUIRED_CAMERAS = ("head_camera", "left_camera", "right_camera")


def validate_observation(observation: dict[str, Any]) -> dict[str, Any]:
    camera_shapes: dict[str, list[int]] = {}
    camera_observations = observation.get("observation", {})
    for camera_name in REQUIRED_CAMERAS:
        if camera_name not in camera_observations:
            raise ValueError(f"Missing RoboTwin camera observation: {camera_name}")
        rgb = np.asarray(camera_observations[camera_name].get("rgb"))
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.size == 0:
            raise ValueError(f"Invalid {camera_name} RGB shape: {rgb.shape}")
        camera_shapes[camera_name] = list(rgb.shape)

    joint_action = np.asarray(observation.get("joint_action", {}).get("vector"))
    if joint_action.shape != (14,):
        raise ValueError(f"Expected RoboTwin 14D joint action, got {joint_action.shape}")
    if not np.isfinite(joint_action).all():
        raise ValueError("RoboTwin joint action contains non-finite values")
    return {"camera_shapes": camera_shapes, "joint_action_shape": list(joint_action.shape)}


def load_task_args(robotwin_path: Path, task_name: str, task_config: str) -> dict[str, Any]:
    with (robotwin_path / "task_config" / f"{task_config}.yml").open(encoding="utf-8") as config_file:
        args = yaml.safe_load(config_file)
    with (robotwin_path / "task_config" / "_embodiment_config.yml").open(
        encoding="utf-8"
    ) as config_file:
        embodiment_types = yaml.safe_load(config_file)
    with (robotwin_path / "task_config" / "_camera_config.yml").open(encoding="utf-8") as config_file:
        camera_types = yaml.safe_load(config_file)

    embodiment = args["embodiment"]
    if len(embodiment) != 1:
        raise ValueError(f"Smoke test expects a shared dual-arm embodiment, got {embodiment}")
    robot_file = embodiment_types[embodiment[0]]["file_path"]
    with Path(robot_file, "config.yml").open(encoding="utf-8") as config_file:
        embodiment_config = yaml.safe_load(config_file)

    head_camera = camera_types[args["camera"]["head_camera_type"]]
    args.update(
        task_name=task_name,
        task_config=task_config,
        left_robot_file=robot_file,
        right_robot_file=robot_file,
        left_embodiment_config=embodiment_config,
        right_embodiment_config=embodiment_config,
        dual_arm_embodied=True,
        head_camera_h=head_camera["h"],
        head_camera_w=head_camera["w"],
        eval_mode=True,
        eval_video_log=False,
        collect_data=False,
        save_data=False,
        render_freq=0,
    )
    return args


def run_smoke_test(
    robotwin_path: Path,
    task_name: str,
    task_config: str,
    seed: int,
    max_seed_attempts: int,
) -> dict[str, Any]:
    robotwin_path = robotwin_path.resolve()
    os.chdir(robotwin_path)
    sys.path.insert(0, str(robotwin_path))
    task_module = importlib.import_module(f"envs.{task_name}")
    unstable_error = importlib.import_module("envs.utils.create_actor").UnStableError
    args = load_task_args(robotwin_path, task_name, task_config)
    for current_seed in range(seed, seed + max_seed_attempts):
        task = getattr(task_module, task_name)()
        try:
            task.setup_demo(now_ep_num=0, seed=current_seed, is_test=True, **args)
            result = validate_observation(task.get_obs())
        except unstable_error:
            continue
        finally:
            if hasattr(task, "scene"):
                task.close_env()
        result.update(task_name=task_name, task_config=task_config, seed=current_seed)
        return result
    raise RuntimeError(
        f"No stable RoboTwin reset for {task_name} in seeds "
        f"[{seed}, {seed + max_seed_attempts - 1}]"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one real RoboTwin three-camera simulator reset")
    parser.add_argument("--robotwin-path", type=Path, default=DEFAULT_ROBOTWIN_PATH)
    parser.add_argument("--task-name", default="adjust_bottle")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--max-seed-attempts", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run_smoke_test(
                args.robotwin_path,
                args.task_name,
                args.task_config,
                args.seed,
                args.max_seed_attempts,
            )
        )
    )


if __name__ == "__main__":
    main()
