from __future__ import annotations

import argparse
import os
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.robot.robotwin.protocols import load_robotwin_manifest, resolve_robotwin_protocol
from experiments.robot.robotwin.tasks import ROBOTWIN_TASKS, eval_task_configs_for_protocol


OFFICIAL_ROBOTWIN_EVAL_EPISODES = 100


@dataclass
class RoboTwinEvalConfig:
    robotwin_path: Path
    manifest_path: Path
    checkpoint_path: Path
    output_dir: Path
    protocol: str = "auto"
    task_name: str | None = None
    task_config: str | None = None
    episodes: int | None = None
    unnorm_key: str | None = None
    num_images_in_input: int = 3
    use_proprio: bool = False
    center_crop: bool = True
    deploy_policy_template: Path = Path("experiments/robot/robotwin/deploy_policy.yml")
    policy_name: str = "model2robotwin_interface"
    ckpt_setting: str = "vla_adapter_robotwin"
    seed: int = 0
    robotwin_python: str = "python"
    dry_run: bool = False


def validate_robotwin_eval_config(cfg: RoboTwinEvalConfig) -> None:
    eval_policy = cfg.robotwin_path / "script" / "eval_policy.py"
    if not eval_policy.exists():
        raise FileNotFoundError(f"Missing RoboTwin eval_policy.py: {eval_policy}")
    if not cfg.manifest_path.exists():
        raise FileNotFoundError(f"Missing RoboTwin manifest: {cfg.manifest_path}")
    if not cfg.checkpoint_path.exists():
        raise FileNotFoundError(f"Missing VLA-Adapter checkpoint: {cfg.checkpoint_path}")
    if not cfg.deploy_policy_template.exists():
        raise FileNotFoundError(f"Missing RoboTwin deploy policy template: {cfg.deploy_policy_template}")


def write_eval_summary(path: Path, cfg: RoboTwinEvalConfig, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": {k: str(v) for k, v in asdict(cfg).items()}, "results": results}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def planned_eval_jobs(cfg: RoboTwinEvalConfig) -> list[dict]:
    manifest = load_robotwin_manifest(cfg.manifest_path)
    profile = resolve_robotwin_protocol(cfg.protocol, manifest)
    episodes = cfg.episodes if cfg.episodes is not None else profile.eval_episodes
    if episodes != OFFICIAL_ROBOTWIN_EVAL_EPISODES:
        raise ValueError(
            "RoboTwin script/eval_policy.py hardcodes 100 episodes per task and exposes no CLI override; "
            f"--episodes must be {OFFICIAL_ROBOTWIN_EVAL_EPISODES}, got {episodes}"
        )
    unnorm_key = (cfg.unnorm_key or manifest.dataset_name or profile.dataset_name).strip()
    if not unnorm_key:
        raise ValueError("RoboTwin evaluation requires a non-empty checkpoint statistics key")
    task_names = (cfg.task_name,) if cfg.task_name else ROBOTWIN_TASKS
    task_configs = (cfg.task_config,) if cfg.task_config else eval_task_configs_for_protocol(
        profile.name, manifest.trained_task_config
    )
    return [
        {
            "task_name": task_name,
            "task_config": task_config,
            "episodes": episodes,
            "protocol": profile.name,
            "unnorm_key": unnorm_key,
        }
        for task_name in task_names
        for task_config in task_configs
    ]


def build_robotwin_eval_command(cfg: RoboTwinEvalConfig, job: dict) -> list[str]:
    eval_policy_path = (cfg.robotwin_path / "script" / "eval_policy.py").expanduser().resolve()
    deploy_policy_path = cfg.deploy_policy_template.expanduser().resolve()
    checkpoint_path = cfg.checkpoint_path.expanduser().resolve()
    return [
        cfg.robotwin_python,
        str(eval_policy_path),
        "--config",
        str(deploy_policy_path),
        "--overrides",
        "--policy_ckpt_path",
        str(checkpoint_path),
        "--unnorm_key",
        job["unnorm_key"],
        "--dataset_name",
        job["unnorm_key"],
        "--task_name",
        job["task_name"],
        "--task_config",
        job["task_config"],
        "--ckpt_setting",
        cfg.ckpt_setting,
        "--seed",
        str(cfg.seed),
        "--policy_name",
        cfg.policy_name,
        "--num_images_in_input",
        str(cfg.num_images_in_input),
        "--use_proprio",
        str(cfg.use_proprio),
        "--center_crop",
        str(cfg.center_crop),
        "--actions_are_normalized",
        "False",
        "--gripper_convention",
        "width_0_1",
    ]


def build_robotwin_eval_env(cfg: RoboTwinEvalConfig) -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[3]
    adapter_dir = repo_root / "experiments" / "robot" / "robotwin"
    pythonpath_entries = [str(adapter_dir), str(repo_root), str(cfg.robotwin_path)]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def run_robotwin_eval(cfg: RoboTwinEvalConfig, jobs: list[dict]) -> list[dict]:
    results = []
    env = build_robotwin_eval_env(cfg)
    for job in jobs:
        command = build_robotwin_eval_command(cfg, job)
        completed = subprocess.run(command, cwd=cfg.robotwin_path, env=env, check=False)
        result = {**job, "returncode": completed.returncode, "command": command}
        results.append(result)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)
    return results


def parse_args() -> RoboTwinEvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate a VLA-Adapter RoboTwin checkpoint")
    parser.add_argument("--robotwin-path", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", default="auto")
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--task-config", choices=["demo_clean", "demo_randomized"], default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--num-images-in-input", type=int, default=3)
    parser.add_argument("--use-proprio", action="store_true")
    parser.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deploy-policy-template", type=Path, default=Path("experiments/robot/robotwin/deploy_policy.yml"))
    parser.add_argument("--policy-name", default="model2robotwin_interface")
    parser.add_argument("--ckpt-setting", default="vla_adapter_robotwin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robotwin-python", default="python")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return RoboTwinEvalConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    jobs = planned_eval_jobs(cfg)
    write_eval_summary(cfg.output_dir / "planned_eval_jobs.json", cfg, jobs)
    if cfg.dry_run:
        print(f"Prepared {len(jobs)} RoboTwin eval jobs")
        return

    validate_robotwin_eval_config(cfg)
    results = run_robotwin_eval(cfg, jobs)
    write_eval_summary(cfg.output_dir / "summary.json", cfg, results)
    print(f"Completed {len(results)} RoboTwin eval jobs")


if __name__ == "__main__":
    main()
