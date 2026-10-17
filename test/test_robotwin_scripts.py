from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def read_script(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_setup_robotwin_env_uses_data_conda_prefix_and_refuses_base_env():
    script = read_script("vla-scripts/setup_robotwin_env.sh")
    assert "/path/to/conda_envs/vla-adapter-robotwin" in script
    assert "create --prefix" in script
    assert "--clone vla-adapter" in script
    assert "Refusing to install RoboTwin dependencies into vla-adapter" in script
    assert "HF_ENDPOINT" in script
    assert "http://localhost:7895" in script
    assert '"sapien==3.0.0b1"' in script
    assert '"mplib==0.2.1"' in script
    assert '"transforms3d==0.4.2"' in script
    assert '"open3d==0.18.0"' in script
    assert '"toppra==0.6.8"' in script
    assert 'CUROBO_VERSION="${CUROBO_VERSION:-v0.7.8}"' in script
    assert "d64c4b005459db10c5dd867d8b30a87d5bda9bdb" in script
    assert '"warp-lang==1.12.0"' in script
    assert '"setuptools==69.5.1"' in script
    assert "--no-build-isolation" in script
    assert 'TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"' in script
    assert "script/requirements.txt" not in script


def test_finetune_robotwin_defaults_to_auto_three_views_no_proprio():
    script = read_script("vla-scripts/finetune_robotwin.sh")
    assert 'ROBOTWIN_PROTOCOL="${ROBOTWIN_PROTOCOL:-auto}"' in script
    assert 'ROBOTWIN_NUM_IMAGES="${ROBOTWIN_NUM_IMAGES:-3}"' in script
    assert 'ROBOTWIN_USE_PROPRIO="${ROBOTWIN_USE_PROPRIO:-False}"' in script
    assert 'ROBOTWIN_IMAGE_AUG="${ROBOTWIN_IMAGE_AUG:-True}"' in script
    assert '--num_images_in_input "${ROBOTWIN_NUM_IMAGES}"' in script
    assert '--use_proprio "${ROBOTWIN_USE_PROPRIO}"' in script
    assert '--image_aug "${ROBOTWIN_IMAGE_AUG}"' in script
    assert "robotwin_manifest.json" in script
    assert "resolve_robotwin_protocol" in script
    assert "ROBOT_PLATFORM == 'ROBOTWIN'" in script
    assert 'ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-${ROBOTWIN_CONDA_PREFIX}/bin/python}"' in script
    assert 'ROBOTWIN_TORCHRUN="${ROBOTWIN_TORCHRUN:-${ROBOTWIN_CONDA_PREFIX}/bin/torchrun}"' in script
    assert '"${ROBOTWIN_PYTHON}" vla-scripts/verify_robotwin_setup.py' in script
    assert '"${ROBOTWIN_PYTHON}" - <<PY' in script
    assert '"${ROBOTWIN_PYTHON}" -c' in script
    assert '"${ROBOTWIN_TORCHRUN}" --standalone' in script
    assert "convert_robotwin_lerobot_to_rlds.py" in script
    assert "prism-qwen25-extra-dinosiglip-224px-0_5b" in script
    assert 'pretrained_models/configs' in script
    assert '--vlm_path "${ROBOTWIN_VLM_PATH}"' in script
    assert '--config_file_path "${ROBOTWIN_CONFIG_PATH}"' in script
    assert '--run_id_override "${RUN_ID}"' in script
    assert '--run_id "${RUN_ID}"' not in script
    assert '"$@"' in script


def test_robotwin_eval_runner_imports_without_sapien():
    import experiments.robot.robotwin.eval_policy_runner as runner

    assert runner.RoboTwinEvalConfig.__name__ == "RoboTwinEvalConfig"


def test_robotwin_tasks_expose_easy_and_hard_configs():
    from experiments.robot.robotwin.tasks import eval_task_configs_for_protocol

    assert eval_task_configs_for_protocol("robotwin_lerobot_unified") == ("demo_clean", "demo_randomized")
    assert eval_task_configs_for_protocol("robotwin_generated_single_setting", trained_task_config="demo_clean") == ("demo_clean",)


def test_robotwin_materialization_uses_precomputed_release_statistics(tmp_path, monkeypatch):
    from prismatic.vla.datasets.rlds.oxe import materialize

    monkeypatch.setattr(materialize, "ACTION_DIM", 14)
    monkeypatch.setattr(materialize, "PROPRIO_DIM", 14)

    statistics_path = tmp_path / "robotwin_dataset_statistics.json"
    statistics_path.write_text("{}", encoding="utf-8")
    kwargs = materialize.make_oxe_dataset_kwargs(
        "robotwin_lerobot_unified",
        tmp_path,
        load_camera_views=("primary", "left_wrist", "right_wrist"),
        load_proprio=False,
    )

    assert kwargs["dataset_statistics"] == str(statistics_path)


def test_evaluate_robotwin_defaults_match_training_contract():
    script = read_script("vla-scripts/evaluate_robotwin.sh")
    assert 'ROBOTWIN_PROTOCOL="${ROBOTWIN_PROTOCOL:-auto}"' in script
    assert 'ROBOTWIN_NUM_IMAGES="${ROBOTWIN_NUM_IMAGES:-3}"' in script
    assert 'ROBOTWIN_USE_PROPRIO="${ROBOTWIN_USE_PROPRIO:-False}"' in script
    assert 'ROBOTWIN_CENTER_CROP="${ROBOTWIN_CENTER_CROP:-True}"' in script
    assert "--manifest-path" in script
    assert "--num-images-in-input" in script
    assert "--center-crop" in script
    assert "--no-center-crop" in script
    assert "EXTRA_ARGS" in script
    assert 'ROBOTWIN_DRY_RUN="${ROBOTWIN_DRY_RUN:-False}"' in script
    assert 'export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"' in script
    assert 'ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-${ROBOTWIN_CONDA_PREFIX}/bin/python}"' in script
    assert '"${ROBOTWIN_PYTHON}" vla-scripts/verify_robotwin_setup.py' in script
    assert '"${ROBOTWIN_PYTHON}" experiments/robot/robotwin/eval_policy_runner.py' in script
    assert '--robotwin-python "${ROBOTWIN_PYTHON}"' in script
    assert 'ROBOTWIN_UNNORM_KEY="${ROBOTWIN_UNNORM_KEY:-}"' in script
    assert '--unnorm-key "${ROBOTWIN_UNNORM_KEY}"' in script
    assert "convert_robotwin_lerobot_to_rlds.py" in script


def test_batch_eval_records_easy_and_hard_settings():
    script = read_script("vla-scripts/evaluate_robotwin_batch.sh")
    assert "demo_clean" in script
    assert "demo_randomized" in script
    assert "planned_eval_jobs.json" in script




def test_robotwin_eval_command_matches_robotwin_eval_policy_args(tmp_path):
    from experiments.robot.robotwin.eval_policy_runner import (
        RoboTwinEvalConfig,
        build_robotwin_eval_command,
    )

    cfg = RoboTwinEvalConfig(
        robotwin_path=tmp_path / "RoboTwin",
        manifest_path=tmp_path / "manifest.json",
        checkpoint_path=tmp_path / "checkpoint",
        output_dir=tmp_path / "out",
        deploy_policy_template=tmp_path / "deploy_policy.yml",
        ckpt_setting="demo_clean",
        seed=7,
        robotwin_python="python3",
    )
    command = build_robotwin_eval_command(
        cfg,
        {
            "task_name": "adjust_bottle",
            "task_config": "demo_clean",
            "unnorm_key": "robotwin_lerobot_unified",
        },
    )
    assert command[:2] == ["python3", str(tmp_path / "RoboTwin" / "script" / "eval_policy.py")]
    assert command.index("--policy_ckpt_path") > command.index("--overrides")
    assert "--task_name" in command
    assert "adjust_bottle" in command
    assert "--task_config" in command
    assert "demo_clean" in command
    assert "--ckpt_setting" in command
    assert "demo_clean" in command
    assert "--policy_name" in command
    assert command[command.index("--unnorm_key") + 1] == "robotwin_lerobot_unified"
    assert command[command.index("--dataset_name") + 1] == "robotwin_lerobot_unified"
    assert "--num_images_in_input" in command
    assert "--use_proprio" in command
    assert "--actions_are_normalized" in command
    assert command[command.index("--center_crop") + 1] == "True"

    template = read_script("experiments/robot/robotwin/deploy_policy.yml")
    assert "unnorm_key: null" in template
    assert "dataset_name: robotwin_lerobot_unified" in template
    assert "num_images_in_input: 3" in template
    assert "use_proprio: false" in template
    assert "use_pro_version: true" in template
    assert "center_crop: true" in template
    assert "num_open_loop_steps: 16" in template
    assert "n_action_steps: 16" in template


def test_robotwin_eval_env_exposes_adapter_and_repo_paths(tmp_path):
    from experiments.robot.robotwin.eval_policy_runner import RoboTwinEvalConfig, build_robotwin_eval_env

    cfg = RoboTwinEvalConfig(
        robotwin_path=tmp_path / "RoboTwin",
        manifest_path=tmp_path / "manifest.json",
        checkpoint_path=tmp_path / "checkpoint",
        output_dir=tmp_path / "out",
    )
    env = build_robotwin_eval_env(cfg)
    pythonpath = env["PYTHONPATH"]
    assert str(ROOT / "experiments" / "robot" / "robotwin") in pythonpath
    assert str(ROOT) in pythonpath
    assert str(tmp_path / "RoboTwin") in pythonpath


def test_robotwin_eval_command_resolves_paths_before_switching_to_robotwin_cwd(tmp_path, monkeypatch):
    from experiments.robot.robotwin.eval_policy_runner import RoboTwinEvalConfig, build_robotwin_eval_command

    monkeypatch.chdir(ROOT)
    cfg = RoboTwinEvalConfig(
        robotwin_path=tmp_path / "RoboTwin",
        manifest_path=tmp_path / "manifest.json",
        checkpoint_path=Path("relative-checkpoint"),
        output_dir=tmp_path / "out",
        deploy_policy_template=Path("experiments/robot/robotwin/deploy_policy.yml"),
    )

    command = build_robotwin_eval_command(
        cfg,
        {
            "task_name": "adjust_bottle",
            "task_config": "demo_clean",
            "episodes": 100,
            "unnorm_key": "robotwin_lerobot_unified",
        },
    )

    assert Path(command[1]).is_absolute()
    assert Path(command[command.index("--config") + 1]).is_absolute()
    assert Path(command[command.index("--policy_ckpt_path") + 1]).is_absolute()


def test_robotwin_eval_rejects_episode_count_unsupported_by_official_runner(tmp_path, monkeypatch):
    import experiments.robot.robotwin.eval_policy_runner as runner

    manifest = type("Manifest", (), {"dataset_name": "robotwin_lerobot_unified", "trained_task_config": None})()
    profile = type("Profile", (), {"name": "robotwin_lerobot_unified", "eval_episodes": 100})()
    monkeypatch.setattr(runner, "load_robotwin_manifest", lambda _: manifest)
    monkeypatch.setattr(runner, "resolve_robotwin_protocol", lambda *_: profile)
    cfg = runner.RoboTwinEvalConfig(
        robotwin_path=tmp_path / "RoboTwin",
        manifest_path=tmp_path / "manifest.json",
        checkpoint_path=tmp_path / "checkpoint",
        output_dir=tmp_path / "out",
        episodes=10,
    )

    with pytest.raises(ValueError, match="hardcodes 100 episodes"):
        runner.planned_eval_jobs(cfg)


def test_robotwin_eval_jobs_use_manifest_dataset_for_checkpoint_stats(tmp_path, monkeypatch):
    import experiments.robot.robotwin.eval_policy_runner as runner

    manifest = type("Manifest", (), {"dataset_name": "robotwin_lerobot_unified", "trained_task_config": None})()
    profile = type("Profile", (), {"name": "robotwin_lerobot_unified", "eval_episodes": 100})()
    monkeypatch.setattr(runner, "load_robotwin_manifest", lambda _: manifest)
    monkeypatch.setattr(runner, "resolve_robotwin_protocol", lambda *_: profile)
    cfg = runner.RoboTwinEvalConfig(
        robotwin_path=tmp_path / "RoboTwin",
        manifest_path=tmp_path / "manifest.json",
        checkpoint_path=tmp_path / "checkpoint",
        output_dir=tmp_path / "out",
        task_name="adjust_bottle",
        task_config="demo_clean",
    )

    assert runner.planned_eval_jobs(cfg) == [
        {
            "task_name": "adjust_bottle",
            "task_config": "demo_clean",
            "episodes": 100,
            "protocol": "robotwin_lerobot_unified",
            "unnorm_key": "robotwin_lerobot_unified",
        }
    ]


def test_robotwin_policy_exports_eval_and_reorders_action(monkeypatch):
    import numpy as np
    import experiments.robot.robotwin.model2robotwin_interface as interface

    cfg = interface.RoboTwinPolicyConfig(pretrained_checkpoint="checkpoint", unnorm_key="robotwin")
    assert cfg.use_pro_version is True
    policy = interface.VLAAdapterRoboTwinPolicy(
        cfg=cfg,
        model=object(),
        processor=None,
        action_min=np.full(14, -1.0, dtype=np.float32),
        action_max=np.full(14, 1.0, dtype=np.float32),
    )

    def fake_get_action(*args, **kwargs):
        return np.tile(np.arange(14, dtype=np.float32), (16, 1))

    monkeypatch.setattr(interface, "get_action", fake_get_action)

    class FakeEnv:
        take_action_cnt = 0

        def __init__(self):
            self.taken = None

        def get_instruction(self):
            return "adjust the bottle"

        def take_action(self, action):
            self.taken = action

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    observation = {
        "observation": {
            "head_camera": {"rgb": image},
            "left_camera": {"rgb": image},
            "right_camera": {"rgb": image},
        },
        "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
    }

    env = FakeEnv()
    interface.eval(env, policy, observation)
    np.testing.assert_array_equal(env.taken, np.array([0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13], dtype=np.float32))
    assert len(policy._chunk) == 15
    interface.reset_model(policy)
    assert policy._chunk == []


def test_robotwin_policy_rejects_checkpoint_with_wrong_action_chunk(monkeypatch):
    import numpy as np
    import experiments.robot.robotwin.model2robotwin_interface as interface

    cfg = interface.RoboTwinPolicyConfig(pretrained_checkpoint="checkpoint", unnorm_key="robotwin")
    policy = interface.VLAAdapterRoboTwinPolicy(
        cfg=cfg,
        model=object(),
        processor=None,
        action_min=np.full(14, -1.0, dtype=np.float32),
        action_max=np.full(14, 1.0, dtype=np.float32),
    )
    monkeypatch.setattr(interface, "get_action", lambda *args, **kwargs: np.zeros((8, 14), dtype=np.float32))

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    observation = {
        "observation": {
            "head_camera": {"rgb": image},
            "left_camera": {"rgb": image},
            "right_camera": {"rgb": image},
        },
        "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
    }

    with pytest.raises(ValueError, match=r"expected \(16, 14\)"):
        policy.step(observation, "adjust the bottle")


def test_robotwin_policy_rejects_eval_chunk_override_that_differs_from_training():
    import numpy as np
    import experiments.robot.robotwin.model2robotwin_interface as interface

    assert interface.ROBOTWIN_ACTION_CHUNK == 16
    assert interface.ROBOTWIN_ACTION_DIM == 14

    cfg = interface.RoboTwinPolicyConfig(
        pretrained_checkpoint="checkpoint",
        unnorm_key="robotwin",
        num_open_loop_steps=8,
        n_action_steps=8,
    )

    with pytest.raises(ValueError, match="must match training"):
        interface.VLAAdapterRoboTwinPolicy(
            cfg=cfg,
            model=object(),
            processor=None,
            action_min=np.full(14, -1.0, dtype=np.float32),
            action_max=np.full(14, 1.0, dtype=np.float32),
        )


def test_robotwin_action_bounds_accept_min_max_and_quantile_stats():
    import numpy as np
    import experiments.robot.robotwin.model2robotwin_interface as interface

    minimum = np.arange(14, dtype=np.float32)
    maximum = minimum + 1.0

    low, high = interface._action_bounds_from_stats(
        {"robotwin": {"action": {"min": minimum.tolist(), "max": maximum.tolist()}}},
        "robotwin",
    )
    np.testing.assert_array_equal(low, minimum)
    np.testing.assert_array_equal(high, maximum)

    low, high = interface._action_bounds_from_stats(
        {"robotwin": {"abs": {"action": {"q01": minimum.tolist(), "q99": maximum.tolist()}}}},
        "robotwin",
    )
    np.testing.assert_array_equal(low, minimum)
    np.testing.assert_array_equal(high, maximum)


@pytest.mark.parametrize(
    ("stats", "error"),
    [
        ({"robotwin": {"action": {}}}, "missing action bounds"),
        ({"robotwin": {"action": {"min": [0.0] * 13, "max": [1.0] * 13}}}, "expected 14 values"),
        (
            {"robotwin": {"action": {"min": [0.0] * 13 + [float("nan")], "max": [1.0] * 14}}},
            "non-finite",
        ),
        ({"robotwin": {"action": {"min": [0.0] * 14, "max": [0.0] * 14}}}, "strictly greater"),
    ],
)
def test_robotwin_action_bounds_reject_invalid_checkpoint_stats(stats, error):
    import experiments.robot.robotwin.model2robotwin_interface as interface

    with pytest.raises(ValueError, match=error):
        interface._action_bounds_from_stats(stats, "robotwin")






def test_finetune_robotwin_runs_readonly_setup_verifier_before_training():
    script = read_script("vla-scripts/finetune_robotwin.sh")
    verifier_idx = script.index("vla-scripts/verify_robotwin_setup.py")
    torchrun_idx = script.index('"${ROBOTWIN_TORCHRUN}" --standalone')
    assert verifier_idx < torchrun_idx
    assert "--data-root" in script
    assert "--manifest-path" in script
    assert "--require-checkpoint" not in script.split("torchrun --standalone", 1)[0]


def test_evaluate_robotwin_runs_readonly_setup_verifier_before_eval():
    script = read_script("vla-scripts/evaluate_robotwin.sh")
    verifier_idx = script.index("vla-scripts/verify_robotwin_setup.py")
    eval_idx = script.index("experiments/robot/robotwin/eval_policy_runner.py")
    assert verifier_idx < eval_idx
    assert "--checkpoint-path" in script
    assert "--require-checkpoint" in script
    assert "--require-simulator --require-assets" in script
    assert 'ROBOTWIN_DRY_RUN}" != "True"' in script
