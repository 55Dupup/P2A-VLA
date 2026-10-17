import json
from argparse import Namespace
from pathlib import Path

from vla_scripts_import import load_script_module


verify = load_script_module("verify_robotwin_setup", "vla-scripts/verify_robotwin_setup.py")


def write_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
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
                "action_bounds": {"min": [-1.0] * 14, "max": [1.0] * 14},
                "actions_are_normalized": False,
                "gripper_convention": "width_0_1",
                "trained_task_config": None,
            }
        ),
        encoding="utf-8",
    )


def test_collect_setup_checks_reports_ready_paths(tmp_path):
    robotwin_path = tmp_path / "RoboTwin"
    (robotwin_path / "script").mkdir(parents=True)
    (robotwin_path / "script" / "eval_policy.py").write_text("", encoding="utf-8")
    data_root = tmp_path / "robotwin_rlds"
    data_root.mkdir()
    manifest = data_root / "robotwin_manifest.json"
    write_manifest(manifest)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    converter = tmp_path / "lerobot2rlds.py"
    converter.write_text("", encoding="utf-8")
    conda_prefix = tmp_path / "env"
    conda_prefix.mkdir()

    checks = verify.collect_setup_checks(
        robotwin_path=robotwin_path,
        data_root=data_root,
        manifest_path=manifest,
        checkpoint_path=checkpoint,
        conda_prefix=conda_prefix,
        converter_path=converter,
        require_checkpoint=True,
    )

    assert all(check.ok for check in checks)
    assert {check.name for check in checks} == {
        "conda_env",
        "robotwin_eval_policy",
        "rlds_data_root",
        "robotwin_manifest",
        "vla_checkpoint",
        "lerobot_converter",
        "lerobot_source",
        "manifest_contract",
    }


def test_collect_setup_checks_reports_missing_checkpoint_and_manifest_contract(tmp_path):
    manifest = tmp_path / "robotwin_manifest.json"
    manifest.write_text(json.dumps({"camera_keys": ["head_camera"], "action_dim": 7}), encoding="utf-8")

    checks = verify.collect_setup_checks(
        robotwin_path=tmp_path / "missing_robotwin",
        data_root=tmp_path / "missing_data",
        manifest_path=manifest,
        checkpoint_path=tmp_path / "missing_checkpoint",
        conda_prefix=tmp_path / "missing_env",
        converter_path=tmp_path / "missing_converter.py",
        require_checkpoint=True,
    )

    failures = {check.name: check.message for check in checks if not check.ok}
    assert "vla_checkpoint" in failures
    assert "robotwin_manifest" not in failures
    assert "manifest_contract" in failures
    assert "three cameras" in failures["manifest_contract"]
    assert verify.exit_code_for_checks(checks) == 1


def test_collect_setup_checks_reports_non_lerobot_source_dir(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    checks = verify.collect_setup_checks(
        robotwin_path=tmp_path / "missing_robotwin",
        data_root=tmp_path / "missing_data",
        manifest_path=tmp_path / "missing_manifest.json",
        checkpoint_path=None,
        conda_prefix=tmp_path / "missing_env",
        converter_path=tmp_path / "missing_converter.py",
        source_dir=source_dir,
        require_checkpoint=False,
    )

    failures = {check.name: check.message for check in checks if not check.ok}
    assert "lerobot_source" in failures
    assert "meta" in failures["lerobot_source"]


def test_parse_args_skips_lerobot_source_check_by_default(monkeypatch):
    monkeypatch.setattr("sys.argv", ["verify_robotwin_setup.py"])
    args = verify.parse_args()

    assert args.source_dir is None
    assert args.require_simulator is False
    assert args.require_assets is False


def test_check_simulator_dependencies_uses_cloned_python(tmp_path):
    python_path = tmp_path / "env" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Namespace(returncode=0, stdout="", stderr="")

    check = verify.check_simulator_dependencies(tmp_path / "env", run_fn=fake_run)

    assert check.ok
    assert calls[0][0] == [str(python_path), "-c", verify.SIMULATOR_IMPORTS]
    assert calls[0][1]["check"] is False


def test_check_simulator_assets_requires_clean_and_randomized_assets(tmp_path):
    robotwin_path = tmp_path / "RoboTwin"
    (robotwin_path / "assets" / "embodiments" / "aloha-agilex").mkdir(parents=True)
    (robotwin_path / "assets" / "embodiments" / "aloha-agilex" / "config.yml").write_text("", encoding="utf-8")
    (robotwin_path / "assets" / "embodiments" / "aloha-agilex" / "curobo_left.yml").write_text(
        "", encoding="utf-8"
    )
    (robotwin_path / "assets" / "embodiments" / "aloha-agilex" / "curobo_right.yml").write_text(
        "", encoding="utf-8"
    )
    (robotwin_path / "assets" / "objects" / "001_bottle").mkdir(parents=True)

    missing = verify.check_simulator_assets(robotwin_path)
    assert not missing.ok
    assert "background_texture" in missing.message

    (robotwin_path / "assets" / "background_texture" / "seen").mkdir(parents=True)
    (robotwin_path / "assets" / "background_texture" / "unseen").mkdir(parents=True)
    assert verify.check_simulator_assets(robotwin_path).ok
