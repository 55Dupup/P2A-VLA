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
class RoboCasaCosmosResolutionDiagnosticConfig(RoboCasaCosmosEvalConfig):
    resolutions: str = "128,224,480"
    task_names: str = "PnPCounterToCab,PnPMicrowaveToCounter,OpenDrawer"
    num_trials_per_task: int = 2
    save_action_trace: bool = True


@draccus.wrap()
def run_resolution_diagnostic(cfg: RoboCasaCosmosResolutionDiagnosticConfig) -> None:
    _validate_config(cfg)
    resolutions = [int(value.strip()) for value in cfg.resolutions.split(",") if value.strip()]
    task_names = [value.strip() for value in cfg.task_names.split(",") if value.strip()]
    unknown_tasks = [name for name in task_names if name not in TASK_MAX_STEPS]
    if unknown_tasks:
        raise ValueError(f"Unknown RoboCasa tasks: {unknown_tasks}")

    root_dir = Path(cfg.local_log_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    summary_path = root_dir / f"resolution-diagnostic-{DATE_TIME}.json"

    set_seed_everywhere(cfg.seed)
    logger.info("Loading policy once for render-resolution diagnostic")
    policy = _build_policy(cfg)
    results = {}

    for resolution in resolutions:
        resolution_results = {}
        cfg.env_img_res = resolution
        cfg.local_log_dir = str(root_dir / f"render_{resolution}")
        for task_name in task_names:
            logger.info("Evaluating resolution=%d task=%s", resolution, task_name)
            set_seed_everywhere(cfg.seed)
            cfg.task_name = task_name
            successes, lengths, video_paths = run_task(cfg, policy)
            resolution_results[task_name] = {
                "success_rate": float(np.mean(successes)) if successes else 0.0,
                "episode_successes": [bool(value) for value in successes],
                "episode_lengths": [int(value) for value in lengths],
                "video_paths": video_paths,
            }
            results[str(resolution)] = resolution_results
            summary_path.write_text(
                json.dumps({"config": asdict(cfg), "results": results}, indent=2), encoding="utf-8"
            )

    logger.info("Saved resolution diagnostic to %s", summary_path)


if __name__ == "__main__":
    run_resolution_diagnostic()
