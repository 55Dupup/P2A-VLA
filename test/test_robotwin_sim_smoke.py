import numpy as np
import pytest

from vla_scripts_import import load_script_module


smoke = load_script_module("smoke_test_robotwin_sim", "vla-scripts/smoke_test_robotwin_sim.py")


def test_validate_observation_accepts_three_rgb_views_and_14d_action():
    observation = {
        "observation": {
            camera: {"rgb": np.zeros((480, 640, 3), dtype=np.uint8)}
            for camera in smoke.REQUIRED_CAMERAS
        },
        "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
    }

    result = smoke.validate_observation(observation)

    assert result["camera_shapes"]["head_camera"] == [480, 640, 3]
    assert result["joint_action_shape"] == [14]


def test_validate_observation_rejects_missing_wrist_view():
    observation = {
        "observation": {"head_camera": {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}},
        "joint_action": {"vector": np.zeros(14)},
    }

    with pytest.raises(ValueError, match="left_camera"):
        smoke.validate_observation(observation)


def test_parse_args_defaults_to_official_style_seed_retry(monkeypatch):
    monkeypatch.setattr(smoke.sys, "argv", ["smoke_test_robotwin_sim.py"])

    args = smoke.parse_args()

    assert args.seed == 100000
    assert args.max_seed_attempts == 20
