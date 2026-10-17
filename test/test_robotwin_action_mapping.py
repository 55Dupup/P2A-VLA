import numpy as np
import pytest

from experiments.robot.robotwin.action_mapping import (
    GripperConvention,
    RobotwinActionMetadata,
    convert_model_action_to_robotwin,
    reorder_model_to_robotwin,
    unnormalize_action,
)


def test_reorder_model_to_robotwin_uses_non_symmetric_positions():
    model_action = np.array([10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 30, 40], dtype=np.float32)
    robotwin_action = reorder_model_to_robotwin(model_action)
    assert robotwin_action.tolist() == [10, 11, 12, 13, 14, 15, 30, 20, 21, 22, 23, 24, 25, 40]


def test_unnormalize_bounds_round_trip():
    metadata = RobotwinActionMetadata(
        action_min=np.array([-2.0, -4.0], dtype=np.float32),
        action_max=np.array([2.0, 4.0], dtype=np.float32),
        gripper_convention=GripperConvention.WIDTH_0_1,
    )
    actual = unnormalize_action(np.array([-1.0, 1.0], dtype=np.float32), metadata)
    np.testing.assert_allclose(actual, np.array([-2.0, 4.0], dtype=np.float32))


def test_convert_model_action_to_robotwin_unnormalizes_then_reorders():
    metadata = RobotwinActionMetadata(
        action_min=np.zeros(14, dtype=np.float32),
        action_max=np.arange(14, dtype=np.float32) + 10.0,
        gripper_convention=GripperConvention.WIDTH_0_1,
    )
    normalized = np.ones(14, dtype=np.float32)
    robotwin_action = convert_model_action_to_robotwin(normalized, metadata, actions_are_normalized=True)
    assert robotwin_action.tolist() == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 22.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 23.0]


def test_gripper_binary_open_close_to_width():
    metadata = RobotwinActionMetadata(
        action_min=np.full(14, -1.0, dtype=np.float32),
        action_max=np.full(14, 1.0, dtype=np.float32),
        gripper_convention=GripperConvention.BINARY_OPEN_CLOSE,
    )
    action = np.zeros(14, dtype=np.float32)
    action[12] = -0.5
    action[13] = 0.5
    converted = convert_model_action_to_robotwin(action, metadata, actions_are_normalized=False)
    assert converted[6] == 0.0
    assert converted[13] == 1.0


def test_action_shape_must_be_14d():
    with pytest.raises(ValueError, match="14"):
        reorder_model_to_robotwin(np.zeros(13, dtype=np.float32))
