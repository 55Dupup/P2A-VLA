"""Protocol profiles and manifest parsing for RoboTwin evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


DEFAULT_CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")
DEFAULT_EVAL_TASK_CONFIGS = ("demo_clean", "demo_randomized")
GENERATED_TASK_CONFIGS = frozenset(DEFAULT_EVAL_TASK_CONFIGS)


class ProtocolMismatchError(ValueError):
    """Raised when a requested RoboTwin protocol conflicts with a manifest."""


@dataclass(frozen=True)
class RoboTwinManifest:
    dataset_name: str
    source: str
    profile: str | None
    task_count: int | None
    trajectory_counts: Mapping[str, int]
    camera_keys: tuple[str, ...]
    action_dim: int | None
    state_dim: int | None
    action_order: str | None
    robotwin_action_order: str | None
    action_normalization: str | None
    actions_are_normalized: bool | None
    gripper_convention: str | None
    trained_task_config: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "trajectory_counts",
            MappingProxyType(dict(self.trajectory_counts)),
        )


@dataclass(frozen=True)
class RoboTwinProtocolProfile:
    name: str
    dataset_name: str
    camera_names: tuple[str, ...]
    default_num_images: int
    default_use_proprio: bool
    eval_task_configs: tuple[str, ...]
    cross_setting_task_configs: tuple[str, ...]
    eval_episodes: int
    description: str


PROFILE_DEFAULTS: Mapping[str, RoboTwinProtocolProfile] = {
    "robotwin_lerobot_unified": RoboTwinProtocolProfile(
        name="robotwin_lerobot_unified",
        dataset_name="robotwin_lerobot_unified",
        camera_names=DEFAULT_CAMERA_NAMES,
        default_num_images=3,
        default_use_proprio=False,
        eval_task_configs=DEFAULT_EVAL_TASK_CONFIGS,
        cross_setting_task_configs=(),
        eval_episodes=100,
        description="Unified RoboTwin LeRobot dataset evaluated on clean and randomized settings.",
    ),
    "robotwin_leaderboard_clean50": RoboTwinProtocolProfile(
        name="robotwin_leaderboard_clean50",
        dataset_name="robotwin_demo_clean_50",
        camera_names=DEFAULT_CAMERA_NAMES,
        default_num_images=3,
        default_use_proprio=False,
        eval_task_configs=DEFAULT_EVAL_TASK_CONFIGS,
        cross_setting_task_configs=(),
        eval_episodes=100,
        description="Clean 50-task RoboTwin leaderboard dataset with clean and randomized evaluation.",
    ),
    "robotwin_mixed_clean_randomized": RoboTwinProtocolProfile(
        name="robotwin_mixed_clean_randomized",
        dataset_name="robotwin_mixed_clean_randomized",
        camera_names=DEFAULT_CAMERA_NAMES,
        default_num_images=3,
        default_use_proprio=False,
        eval_task_configs=DEFAULT_EVAL_TASK_CONFIGS,
        cross_setting_task_configs=(),
        eval_episodes=100,
        description="Mixed clean/randomized RoboTwin dataset evaluated on both settings.",
    ),
    "robotwin_generated_single_setting": RoboTwinProtocolProfile(
        name="robotwin_generated_single_setting",
        dataset_name="",
        camera_names=DEFAULT_CAMERA_NAMES,
        default_num_images=3,
        default_use_proprio=False,
        eval_task_configs=(),
        cross_setting_task_configs=(),
        eval_episodes=100,
        description="Generated RoboTwin data from one task setting, with same-setting first evaluation.",
    ),
}


def load_robotwin_manifest(path: str | Path) -> RoboTwinManifest:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolMismatchError(
            f"Failed to parse RoboTwin manifest {manifest_path}: invalid JSON ({exc.msg})"
        ) from exc

    try:
        if not isinstance(payload, dict):
            raise ProtocolMismatchError("RoboTwin manifest must be a JSON object")

        return RoboTwinManifest(
            dataset_name=_optional_str(payload, "dataset_name") or "",
            source=_optional_str(payload, "source") or "",
            profile=_optional_str(payload, "profile"),
            task_count=_optional_int(payload, "task_count"),
            trajectory_counts=_trajectory_counts(payload.get("trajectory_counts", {})),
            camera_keys=tuple(
                _string_sequence(payload.get("camera_keys", DEFAULT_CAMERA_NAMES), "camera_keys")
            ),
            action_dim=_optional_int(payload, "action_dim"),
            state_dim=_optional_int(payload, "state_dim"),
            action_order=_optional_str(payload, "action_order"),
            robotwin_action_order=_optional_str(payload, "robotwin_action_order"),
            action_normalization=_optional_str(payload, "action_normalization"),
            actions_are_normalized=_optional_bool(payload, "actions_are_normalized"),
            gripper_convention=_optional_str(payload, "gripper_convention"),
            trained_task_config=_optional_str(payload, "trained_task_config"),
        )
    except ProtocolMismatchError as exc:
        raise ProtocolMismatchError(f"Invalid RoboTwin manifest {manifest_path}: {exc}") from exc


def resolve_robotwin_protocol(
    requested: str | None,
    manifest: RoboTwinManifest | None = None,
) -> RoboTwinProtocolProfile:
    requested_profile = (requested or "auto").strip()
    if requested_profile == "auto":
        if manifest is None or not manifest.profile:
            raise ProtocolMismatchError("ROBOTWIN_PROTOCOL=auto requires manifest profile")
        requested_profile = manifest.profile
    elif manifest is not None and manifest.profile and manifest.profile != requested_profile:
        raise ProtocolMismatchError(
            f"Requested RoboTwin profile {requested_profile!r} does not match manifest profile {manifest.profile!r}"
        )

    try:
        profile = PROFILE_DEFAULTS[requested_profile]
    except KeyError as exc:
        known_profiles = ", ".join(sorted(PROFILE_DEFAULTS))
        raise ProtocolMismatchError(
            f"Unknown RoboTwin protocol profile {requested_profile!r}; known profiles: {known_profiles}"
        ) from exc

    if profile.name == "robotwin_generated_single_setting":
        return _resolve_generated_single_setting(profile, manifest)

    return profile


def _resolve_generated_single_setting(
    profile: RoboTwinProtocolProfile,
    manifest: RoboTwinManifest | None,
) -> RoboTwinProtocolProfile:
    if manifest is None or not manifest.trained_task_config:
        raise ProtocolMismatchError(
            "robotwin_generated_single_setting requires manifest trained_task_config"
        )
    if manifest.trained_task_config not in GENERATED_TASK_CONFIGS:
        valid_configs = ", ".join(sorted(GENERATED_TASK_CONFIGS))
        raise ProtocolMismatchError(
            "robotwin_generated_single_setting trained_task_config must be one of "
            f"{valid_configs}; got {manifest.trained_task_config!r}"
        )

    cross_setting = tuple(
        task_config
        for task_config in DEFAULT_EVAL_TASK_CONFIGS
        if task_config != manifest.trained_task_config
    )
    return replace(
        profile,
        dataset_name=manifest.dataset_name,
        eval_task_configs=(manifest.trained_task_config,),
        cross_setting_task_configs=cross_setting,
    )


def _optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProtocolMismatchError(f"RoboTwin manifest field {key!r} must be a string")
    return value


def _optional_int(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    return _coerce_int(value, key)


def _optional_bool(payload: Mapping[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ProtocolMismatchError(f"RoboTwin manifest field {key!r} must be a boolean")
    return value


def _string_sequence(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ProtocolMismatchError(f"RoboTwin manifest field {key!r} must be a list of strings")
    if not value:
        raise ProtocolMismatchError(f"RoboTwin manifest field {key!r} must contain at least one string")
    if not all(isinstance(item, str) for item in value):
        raise ProtocolMismatchError(f"RoboTwin manifest field {key!r} must contain only strings")
    return tuple(value)


def _trajectory_counts(value: Any) -> Mapping[str, int]:
    if not isinstance(value, dict):
        raise ProtocolMismatchError("RoboTwin manifest field 'trajectory_counts' must be an object")

    counts: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str):
            raise ProtocolMismatchError("RoboTwin manifest trajectory_counts keys must be strings")
        counts[key] = _coerce_int(count, "trajectory_counts")
    return MappingProxyType(dict(counts))


def _coerce_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolMismatchError(f"RoboTwin manifest field {key!r} must be an integer")
    return value
