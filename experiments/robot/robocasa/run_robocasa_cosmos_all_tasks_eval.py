import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import draccus
import numpy as np

from experiments.robot.robot_utils import DATE_TIME, set_seed_everywhere
from experiments.robot.robocasa.run_robocasa_cosmos_eval import (
    TASK_MAX_STEPS,
    RoboCasaCosmosEvalConfig,
    _build_policy,
    _validate_config,
    run_task,
)


logger = logging.getLogger(__name__)


@dataclass
class RoboCasaCosmosAllTasksEvalConfig(RoboCasaCosmosEvalConfig):
    task_names: str = ",".join(TASK_MAX_STEPS)
    continue_on_error: bool = True


def _write_summary(path: Path, cfg: RoboCasaCosmosAllTasksEvalConfig, results: dict) -> None:
    completed = [result for result in results.values() if "error" not in result]
    macro_success_rate = float(np.mean([result["success_rate"] for result in completed])) if completed else 0.0
    total_successes = sum(sum(result["episode_successes"]) for result in completed)
    total_episodes = sum(result["n_episodes"] for result in completed)
    payload = {
        "checkpoint": cfg.pretrained_checkpoint,
        "num_requested_tasks": len([name for name in cfg.task_names.split(",") if name.strip()]),
        "num_completed_tasks": len(completed),
        "macro_success_rate": macro_success_rate,
        "micro_success_rate": float(total_successes / total_episodes) if total_episodes else 0.0,
        "total_successes": int(total_successes),
        "total_episodes": int(total_episodes),
        "config": asdict(cfg),
        "tasks": results,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@draccus.wrap()
def run_robocasa_cosmos_all_tasks_eval(cfg: RoboCasaCosmosAllTasksEvalConfig) -> float:
    _validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    task_names = [name.strip() for name in cfg.task_names.split(",") if name.strip()]
    unknown_tasks = [name for name in task_names if name not in TASK_MAX_STEPS]
    if unknown_tasks:
        raise ValueError(f"Unknown RoboCasa tasks: {unknown_tasks}")

    output_dir = Path(cfg.local_log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"EVAL-robocasa-cosmos-all-tasks-{DATE_TIME}.json"

    logger.info("Loading policy once for %d RoboCasa tasks", len(task_names))
    policy = _build_policy(cfg)
    results = {}

    for task_index, task_name in enumerate(task_names, start=1):
        logger.info("Evaluating task %d/%d: %s", task_index, len(task_names), task_name)
        # Match separate official task invocations by resetting process-level RNG state per task.
        set_seed_everywhere(cfg.seed)
        cfg.task_name = task_name
        try:
            successes, lengths, video_paths = run_task(cfg, policy)
            results[task_name] = {
                "n_episodes": len(successes),
                "success_rate": float(np.mean(successes)) if successes else 0.0,
                "episode_successes": [bool(value) for value in successes],
                "episode_lengths": [int(value) for value in lengths],
                "video_paths": video_paths,
            }
        except Exception as exc:
            logger.exception("Task %s failed", task_name)
            results[task_name] = {"error": f"{type(exc).__name__}: {exc}"}
            _write_summary(summary_path, cfg, results)
            if not cfg.continue_on_error:
                raise
        else:
            _write_summary(summary_path, cfg, results)

    logger.info("Saved all-task RoboCasa summary to %s", summary_path)
    completed = [result for result in results.values() if "error" not in result]
    return float(np.mean([result["success_rate"] for result in completed])) if completed else 0.0


if __name__ == "__main__":
    run_robocasa_cosmos_all_tasks_eval()
