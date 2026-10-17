from types import SimpleNamespace

import numpy as np

from experiments.robot.robocasa.model2robocasa_interface import (
    ObservationAdapterConfig,
    VLAAdapterRoboCasaPolicy,
)


def make_policy(use_proprio=True):
    policy = VLAAdapterRoboCasaPolicy.__new__(VLAAdapterRoboCasaPolicy)
    policy.cfg = SimpleNamespace(num_images_in_input=3, use_proprio=use_proprio, proprio_dim=9)
    policy.adapter_cfg = ObservationAdapterConfig(
        image_obs_key="video.right_image",
        instruction_obs_key="annotation.human.coarse_action",
        state_obs_keys=("state",),
        extra_image_obs_keys=("video.left_image", "video.wrist_image"),
    )
    return policy


def test_merged_panda_policy_observation_uses_configured_three_views_and_state():
    observations = {
        "video.right_image": np.full((1, 1, 4, 4, 3), 10, dtype=np.uint8),
        "video.left_image": np.full((1, 1, 4, 4, 3), 20, dtype=np.uint8),
        "video.wrist_image": np.full((1, 1, 4, 4, 3), 30, dtype=np.uint8),
        "state": np.arange(9, dtype=np.float32).reshape(1, 1, 9),
    }

    policy_obs = make_policy()._build_policy_observation(observations, env_idx=0)

    assert policy_obs["full_image"].shape == (4, 4, 3)
    assert policy_obs["full_image"][0, 0, 0] == 10
    assert policy_obs["wrist_image"][0, 0, 0] == 20
    assert policy_obs["wrist_image_1"][0, 0, 0] == 30
    np.testing.assert_array_equal(policy_obs["state"], np.arange(9, dtype=np.float32))


def test_merged_panda_policy_observation_omits_state_when_proprio_disabled():
    observations = {
        "video.right_image": np.full((1, 1, 4, 4, 3), 10, dtype=np.uint8),
        "video.left_image": np.full((1, 1, 4, 4, 3), 20, dtype=np.uint8),
        "video.wrist_image": np.full((1, 1, 4, 4, 3), 30, dtype=np.uint8),
        "state": np.arange(9, dtype=np.float32).reshape(1, 1, 9),
    }

    policy_obs = make_policy(use_proprio=False)._build_policy_observation(observations, env_idx=0)

    assert policy_obs["full_image"].shape == (4, 4, 3)
    assert policy_obs["wrist_image"][0, 0, 0] == 20
    assert policy_obs["wrist_image_1"][0, 0, 0] == 30
    assert "state" not in policy_obs


def test_merged_panda_modality_config_includes_extra_image_keys():
    modality_config = make_policy().get_modality_config()

    assert modality_config["video"].modality_keys == [
        "video.right_image",
        "video.left_image",
        "video.wrist_image",
    ]
    assert modality_config["state"].modality_keys == ["state"]


def test_merged_panda_modality_config_omits_state_keys_when_proprio_disabled():
    modality_config = make_policy(use_proprio=False).get_modality_config()

    assert modality_config["video"].modality_keys == [
        "video.right_image",
        "video.left_image",
        "video.wrist_image",
    ]
    assert modality_config["state"].modality_keys == []
