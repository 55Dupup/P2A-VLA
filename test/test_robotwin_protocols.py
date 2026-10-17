import json
from pathlib import Path

import pytest

from experiments.robot.robotwin.protocols import (
    ProtocolMismatchError,
    load_robotwin_manifest,
    resolve_robotwin_protocol,
)


def write_manifest(path: Path, **overrides):
    payload = {
        "dataset_name": "robotwin_lerobot_unified",
        "source": "lerobot/robotwin_unified",
        "profile": "robotwin_lerobot_unified",
        "task_count": 50,
        "trajectory_counts": {"demo_clean": 2500, "demo_randomized": 2500},
        "camera_keys": ["head_camera", "left_camera", "right_camera"],
        "action_dim": 14,
        "state_dim": 14,
        "action_order": "model_left6_right6_left_gripper_right_gripper",
        "robotwin_action_order": "left6_left_gripper_right6_right_gripper",
        "action_normalization": "bounds",
        "actions_are_normalized": False,
        "gripper_convention": "width_0_1",
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_auto_resolves_lerobot_unified_manifest(tmp_path):
    manifest_path = tmp_path / "robotwin_manifest.json"
    write_manifest(manifest_path)
    manifest = load_robotwin_manifest(manifest_path)
    assert manifest.trajectory_counts["demo_clean"] == 2500
    assert dict(manifest.trajectory_counts) == {"demo_clean": 2500, "demo_randomized": 2500}
    profile = resolve_robotwin_protocol("auto", manifest)
    assert profile.name == "robotwin_lerobot_unified"
    assert profile.camera_names == ("head_camera", "left_camera", "right_camera")
    assert profile.default_num_images == 3
    assert profile.default_use_proprio is False
    assert profile.eval_task_configs == ("demo_clean", "demo_randomized")
    assert profile.eval_episodes == 100


def test_trajectory_counts_are_immutable(tmp_path):
    manifest_path = tmp_path / "robotwin_manifest.json"
    write_manifest(manifest_path)
    manifest = load_robotwin_manifest(manifest_path)
    with pytest.raises(TypeError):
        manifest.trajectory_counts["demo_clean"] = 1


def test_auto_requires_manifest_profile(tmp_path):
    manifest_path = tmp_path / "robotwin_manifest.json"
    write_manifest(manifest_path, profile=None)
    manifest = load_robotwin_manifest(manifest_path)
    with pytest.raises(ProtocolMismatchError, match="profile"):
        resolve_robotwin_protocol("auto", manifest)


def test_requested_profile_must_match_manifest(tmp_path):
    manifest_path = tmp_path / "robotwin_manifest.json"
    write_manifest(manifest_path, profile="robotwin_mixed_clean_randomized")
    manifest = load_robotwin_manifest(manifest_path)
    with pytest.raises(ProtocolMismatchError, match="robotwin_leaderboard_clean50"):
        resolve_robotwin_protocol("robotwin_leaderboard_clean50", manifest)


def test_generated_single_setting_uses_trained_task_config(tmp_path):
    manifest_path = tmp_path / "robotwin_manifest.json"
    write_manifest(
        manifest_path,
        dataset_name="robotwin_generated_demo_randomized",
        source="local_hdf5",
        profile="robotwin_generated_single_setting",
        trained_task_config="demo_randomized",
        trajectory_counts={"demo_randomized": 500},
    )
    manifest = load_robotwin_manifest(manifest_path)
    profile = resolve_robotwin_protocol("auto", manifest)
    assert profile.name == "robotwin_generated_single_setting"
    assert profile.eval_task_configs == ("demo_randomized",)
    assert profile.cross_setting_task_configs == ("demo_clean",)


def test_invalid_manifest_json_reports_path(tmp_path):
    manifest_path = tmp_path / "robotwin_manifest.json"
    manifest_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ProtocolMismatchError, match=str(manifest_path)) as excinfo:
        load_robotwin_manifest(manifest_path)
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


def test_empty_camera_keys_rejected(tmp_path):
    manifest_path = tmp_path / "robotwin_manifest.json"
    write_manifest(manifest_path, camera_keys=[])
    with pytest.raises(ProtocolMismatchError, match="camera_keys"):
        load_robotwin_manifest(manifest_path)


def test_malformed_trajectory_count_rejected(tmp_path):
    manifest_path = tmp_path / "robotwin_manifest.json"
    write_manifest(manifest_path, trajectory_counts={"demo_clean": "many"})
    with pytest.raises(ProtocolMismatchError, match="trajectory_counts") as excinfo:
        load_robotwin_manifest(manifest_path)
    assert isinstance(excinfo.value.__cause__, ProtocolMismatchError)
    assert "trajectory_counts" in str(excinfo.value.__cause__)


@pytest.mark.parametrize(
    ("overrides", "field_name"),
    [
        ({"task_count": "50"}, "task_count"),
        ({"action_dim": 14.0}, "action_dim"),
        ({"trajectory_counts": {"demo_clean": True}}, "trajectory_counts"),
    ],
)
def test_integer_fields_reject_non_int_values(tmp_path, overrides, field_name):
    manifest_path = tmp_path / "robotwin_manifest.json"
    write_manifest(manifest_path, **overrides)
    with pytest.raises(ProtocolMismatchError, match=field_name):
        load_robotwin_manifest(manifest_path)


def test_generated_single_setting_requires_trained_task_config(tmp_path):
    manifest_path = tmp_path / "robotwin_manifest.json"
    write_manifest(
        manifest_path,
        profile="robotwin_generated_single_setting",
        trained_task_config=None,
    )
    manifest = load_robotwin_manifest(manifest_path)
    with pytest.raises(ProtocolMismatchError, match="trained_task_config"):
        resolve_robotwin_protocol("auto", manifest)
