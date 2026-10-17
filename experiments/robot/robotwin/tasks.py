from __future__ import annotations

from prismatic.vla.datasets.rlds.oxe.robotwin_registry import ROBOTWIN_TASKS


def eval_task_configs_for_protocol(protocol_name: str, trained_task_config: str | None = None) -> tuple[str, ...]:
    if protocol_name in {
        "robotwin_lerobot_unified",
        "robotwin_leaderboard_clean50",
        "robotwin_mixed_clean_randomized",
    }:
        return ("demo_clean", "demo_randomized")
    if protocol_name == "robotwin_generated_single_setting":
        if trained_task_config not in {"demo_clean", "demo_randomized"}:
            raise ValueError("robotwin_generated_single_setting requires trained_task_config")
        return (trained_task_config,)
    raise ValueError(f"Unknown RoboTwin protocol: {protocol_name}")
