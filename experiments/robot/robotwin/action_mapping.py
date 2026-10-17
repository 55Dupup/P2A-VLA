from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class GripperConvention(str, Enum):
    WIDTH_0_1 = "width_0_1"
    BINARY_OPEN_CLOSE = "binary_open_close"


@dataclass(frozen=True)
class RobotwinActionMetadata:
    action_min: np.ndarray
    action_max: np.ndarray
    gripper_convention: GripperConvention


def _as_14d(action: np.ndarray) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32)
    if array.shape[-1] != 14:
        raise ValueError(f"RoboTwin action must be 14D, got shape {array.shape}")
    return array


def reorder_model_to_robotwin(model_action: np.ndarray) -> np.ndarray:
    action = _as_14d(model_action)
    return np.concatenate([action[..., 0:6], action[..., 12:13], action[..., 6:12], action[..., 13:14]], axis=-1)


def unnormalize_action(normalized_action: np.ndarray, metadata: RobotwinActionMetadata) -> np.ndarray:
    action = np.asarray(normalized_action, dtype=np.float32)
    action_min = np.asarray(metadata.action_min, dtype=np.float32)
    action_max = np.asarray(metadata.action_max, dtype=np.float32)
    if action_min.shape[-1] != action.shape[-1] or action_max.shape[-1] != action.shape[-1]:
        raise ValueError(
            f"RoboTwin action bounds must match action shape, got {action_min.shape}, {action_max.shape}, {action.shape}"
        )
    return 0.5 * (action + 1.0) * (action_max - action_min) + action_min


def _convert_grippers(model_order_action: np.ndarray, convention: GripperConvention | str) -> np.ndarray:
    action = np.array(model_order_action, dtype=np.float32, copy=True)
    resolved_convention = GripperConvention(convention)
    if resolved_convention == GripperConvention.WIDTH_0_1:
        return action
    if resolved_convention == GripperConvention.BINARY_OPEN_CLOSE:
        action[..., 12] = np.where(action[..., 12] > 0.0, 1.0, 0.0)
        action[..., 13] = np.where(action[..., 13] > 0.0, 1.0, 0.0)
        return action
    raise ValueError(f"Unsupported RoboTwin gripper convention: {convention}")


def convert_model_action_to_robotwin(
    model_action: np.ndarray,
    metadata: RobotwinActionMetadata,
    *,
    actions_are_normalized: bool,
) -> np.ndarray:
    action = _as_14d(model_action)
    if actions_are_normalized:
        action = unnormalize_action(action, metadata)
    action = _convert_grippers(action, metadata.gripper_convention)
    return reorder_model_to_robotwin(action)
