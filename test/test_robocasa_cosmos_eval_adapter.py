from types import SimpleNamespace

import numpy as np

from experiments.robot.robocasa.model2robocasa_interface import (
    ObservationAdapterConfig,
    VLAAdapterRoboCasaPolicy,
)
from experiments.robot.robocasa.run_robocasa_cosmos_eval import append_fixed_mobile_base_action


def make_policy():
    policy = VLAAdapterRoboCasaPolicy.__new__(VLAAdapterRoboCasaPolicy)
    policy.cfg = SimpleNamespace(num_images_in_input=3, use_proprio=True, proprio_dim=9)
    policy.adapter_cfg = ObservationAdapterConfig(
        image_obs_key="video.left_image",
        instruction_obs_key="annotation.human.coarse_action",
        state_obs_keys=("state",),
        extra_image_obs_keys=("video.right_image", "video.wrist_image"),
        flip_images=True,
    )
    return policy


def test_cosmos_policy_observation_uses_left_right_wrist_order_and_flips_images():
    left = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    right = np.full((4, 4, 3), 20, dtype=np.uint8)
    wrist = np.full((4, 4, 3), 30, dtype=np.uint8)
    observations = {
        "video.left_image": left[None],
        "video.right_image": right[None],
        "video.wrist_image": wrist[None],
        "state": np.arange(9, dtype=np.float32)[None],
    }

    policy_obs = make_policy()._build_policy_observation(observations, env_idx=0)

    np.testing.assert_array_equal(policy_obs["full_image"], np.flipud(left))
    np.testing.assert_array_equal(policy_obs["wrist_image"], np.flipud(right))
    np.testing.assert_array_equal(policy_obs["wrist_image_1"], np.flipud(wrist))
    np.testing.assert_array_equal(policy_obs["state"], np.arange(9, dtype=np.float32))


def test_append_fixed_mobile_base_action_matches_cosmos_eval_behavior():
    action = np.arange(7, dtype=np.float32)

    env_action = append_fixed_mobile_base_action(action, env_action_dim=12)

    np.testing.assert_array_equal(env_action[:7], action)
    np.testing.assert_array_equal(env_action[7:], np.array([0, 0, 0, 0, -1], dtype=np.float32))


def test_append_fixed_mobile_base_action_leaves_12d_actions_unchanged():
    action = np.arange(12, dtype=np.float32)

    env_action = append_fixed_mobile_base_action(action, env_action_dim=12)

    np.testing.assert_array_equal(env_action, action)
