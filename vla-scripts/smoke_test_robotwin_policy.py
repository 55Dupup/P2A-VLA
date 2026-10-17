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


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROBOTWIN_PATH = Path("/path/to/RoboTwin")


def validate_robotwin_action(action: Any) -> np.ndarray:
    action_array = np.asarray(action, dtype=np.float32)
    if action_array.shape != (14,):
        raise ValueError(f"Expected one 14D RoboTwin action, got {action_array.shape}")
    if not np.isfinite(action_array).all():
        raise ValueError("RoboTwin policy action contains non-finite values")
    return action_array


def run_policy_smoke(
    *,
    robotwin_path: Path,
    checkpoint_path: Path,
    task_name: str,
    task_config: str,
    instruction: str,
    seed: int,
    max_seed_attempts: int,
) -> dict[str, Any]:
    robotwin_path = robotwin_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    os.chdir(robotwin_path)
    for path in (str(ROOT), str(SCRIPT_DIR), str(robotwin_path)):
        if path not in sys.path:
            sys.path.insert(0, path)

    simulator_smoke = importlib.import_module("smoke_test_robotwin_sim")
    task_module = importlib.import_module(f"envs.{task_name}")
    unstable_error = importlib.import_module("envs.utils.create_actor").UnStableError
    policy_module = importlib.import_module(
        "experiments.robot.robotwin.model2robotwin_interface"
    )
    policy = policy_module.get_model(
        {
            "policy_ckpt_path": str(checkpoint_path),
            "unnorm_key": "robotwin_lerobot_unified",
            "dataset_name": "robotwin_lerobot_unified",
            "num_images_in_input": 3,
            "use_proprio": False,
            "actions_are_normalized": False,
            "gripper_convention": "width_0_1",
            "use_l1_regression": True,
            "use_pro_version": True,
            "use_minivlm": True,
            "num_open_loop_steps": 16,
            "n_action_steps": 16,
        }
    )

    task_args = simulator_smoke.load_task_args(robotwin_path, task_name, task_config)
    for current_seed in range(seed, seed + max_seed_attempts):
        task = getattr(task_module, task_name)()
        try:
            task.setup_demo(now_ep_num=0, seed=current_seed, is_test=True, **task_args)
            observation = task.get_obs()
            simulator_smoke.validate_observation(observation)
            action = validate_robotwin_action(policy.step(observation, instruction))
            task.take_action(action)
            simulator_smoke.validate_observation(task.get_obs())
        except unstable_error:
            continue
        finally:
            if hasattr(task, "scene"):
                task.close_env()
        return {
            "task_name": task_name,
            "task_config": task_config,
            "seed": current_seed,
            "action_shape": list(action.shape),
            "action_min": float(action.min()),
            "action_max": float(action.max()),
            "num_images_in_input": 3,
            "use_proprio": False,
        }
    raise RuntimeError(
        f"No stable RoboTwin reset for {task_name} in seeds "
        f"[{seed}, {seed + max_seed_attempts - 1}]"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one checkpoint inference and action in a real RoboTwin environment"
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--robotwin-path", type=Path, default=DEFAULT_ROBOTWIN_PATH)
    parser.add_argument("--task-name", default="adjust_bottle")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument(
        "--instruction",
        default="Pick up the bottle from the table and keep it upright.",
    )
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--max-seed-attempts", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run_policy_smoke(
                robotwin_path=args.robotwin_path,
                checkpoint_path=args.checkpoint_path,
                task_name=args.task_name,
                task_config=args.task_config,
                instruction=args.instruction,
                seed=args.seed,
                max_seed_attempts=args.max_seed_attempts,
            )
        )
    )


if __name__ == "__main__":
    main()
