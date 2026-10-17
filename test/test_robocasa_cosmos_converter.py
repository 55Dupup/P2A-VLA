import numpy as np
import pytest

from vla_scripts_import import load_script_module


converter = load_script_module(
    "convert_robocasa_cosmos_to_rlds",
    "vla-scripts/convert_robocasa_cosmos_to_rlds.py",
)


def test_extract_cosmos_action_uses_first_7_dims_from_raw_12d_action():
    raw_action = np.arange(12, dtype=np.float32)

    action = converter.extract_cosmos_action(raw_action)

    np.testing.assert_array_equal(action, np.arange(7, dtype=np.float32))


def test_extract_cosmos_action_accepts_already_7d_action():
    raw_action = np.arange(7, dtype=np.float32)

    action = converter.extract_cosmos_action(raw_action)

    np.testing.assert_array_equal(action, raw_action)


def test_validate_episode_arrays_requires_three_aligned_views_state_and_action():
    episode = converter.CosmosEpisodeArrays(
        left_images=np.zeros((2, 224, 224, 3), dtype=np.uint8),
        right_images=np.zeros((2, 224, 224, 3), dtype=np.uint8),
        wrist_images=np.zeros((2, 224, 224, 3), dtype=np.uint8),
        states=np.zeros((2, 9), dtype=np.float32),
        actions=np.zeros((2, 12), dtype=np.float32),
        language="turn off the microwave",
    )

    converter.validate_episode_arrays(episode)


def test_validate_episode_arrays_rejects_wrong_proprio_dim():
    episode = converter.CosmosEpisodeArrays(
        left_images=np.zeros((2, 224, 224, 3), dtype=np.uint8),
        right_images=np.zeros((2, 224, 224, 3), dtype=np.uint8),
        wrist_images=np.zeros((2, 224, 224, 3), dtype=np.uint8),
        states=np.zeros((2, 8), dtype=np.float32),
        actions=np.zeros((2, 12), dtype=np.float32),
        language="turn off the microwave",
    )

    with pytest.raises(ValueError, match="states must be shaped"):
        converter.validate_episode_arrays(episode)
