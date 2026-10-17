import json
import subprocess
import tarfile
from argparse import Namespace
from pathlib import Path

import pytest

from vla_scripts_import import load_script_module


prepare = load_script_module("prepare_robotwin_data", "vla-scripts/prepare_robotwin_data.py")


def write_fastwam_lerobot_metadata(
    root: Path,
    *,
    fps=50,
    action_shape=None,
    camera_keys=None,
    episode_lengths=(20, 20),
):
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    camera_keys = camera_keys or (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )
    features = {
        key: {"dtype": "video", "shape": [480, 640, 3]}
        for key in camera_keys
    }
    features.update(
        {
            "observation.state": {"dtype": "float32", "shape": [14]},
            "action": {"dtype": "float32", "shape": action_shape or [14]},
        }
    )
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "fps": fps,
                "total_episodes": len(episode_lengths),
                "total_frames": sum(episode_lengths),
                "features": features,
            }
        ),
        encoding="utf-8",
    )
    (meta_dir / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": index, "tasks": ["adjust bottle"], "length": length}) + "\n"
            for index, length in enumerate(episode_lengths)
        ),
        encoding="utf-8",
    )
    (meta_dir / "tasks.jsonl").write_text('{"task_index": 0, "task": "adjust bottle"}\n', encoding="utf-8")


def write_robotwin_rlds_metadata(root: Path, *, num_episodes=2, camera_keys=None):
    dataset_dir = root / "robotwin_lerobot_unified" / "0.1.0"
    dataset_dir.mkdir(parents=True)
    camera_keys = camera_keys or ("cam_high", "cam_left_wrist", "cam_right_wrist")
    observation_features = {
        camera: {
            "image": {"dtype": "uint8", "encodingFormat": "jpeg", "shape": {"dimensions": ["32", "32", "3"]}}
        }
        for camera in camera_keys
    }
    observation_features["state"] = {
        "tensor": {"dtype": "float32", "shape": {"dimensions": ["14"]}}
    }
    step_features = {
        "action": {"tensor": {"dtype": "float32", "shape": {"dimensions": ["14"]}}},
        "observation": {"featuresDict": {"features": observation_features}},
        "language_instruction": {"text": {}},
        "is_first": {"tensor": {"dtype": "bool", "shape": {}}},
        "is_last": {"tensor": {"dtype": "bool", "shape": {}}},
        "is_terminal": {"tensor": {"dtype": "bool", "shape": {}}},
    }
    (dataset_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "fileFormat": "tfrecord",
                "name": "robotwin_lerobot_unified",
                "version": "0.1.0",
                "splits": [{"name": "train", "shardLengths": [str(num_episodes)]}],
            }
        ),
        encoding="utf-8",
    )
    (dataset_dir / "features.json").write_text(
        json.dumps(
            {
                "featuresDict": {
                    "features": {
                        "steps": {
                            "sequence": {
                                "feature": {"featuresDict": {"features": step_features}}
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (dataset_dir / "robotwin_lerobot_unified-train.tfrecord-00000-of-00001").write_bytes(b"tfrecord")


def test_infer_lerobot_unified_profile():
    profile = prepare.infer_profile(
        source="lerobot/robotwin_unified",
        task_config=None,
        clean_count=2500,
        randomized_count=2500,
    )
    assert profile == "robotwin_lerobot_unified"


def test_infer_single_setting_profile():
    profile = prepare.infer_profile(
        source="local_hdf5",
        task_config="demo_randomized",
        clean_count=0,
        randomized_count=500,
    )
    assert profile == "robotwin_generated_single_setting"


def test_write_manifest_records_three_cameras_and_no_normalized_actions(tmp_path):
    manifest_path = tmp_path / "robotwin_manifest.json"
    prepare.write_manifest(
        manifest_path=manifest_path,
        dataset_name="robotwin_lerobot_unified",
        source="lerobot/robotwin_unified",
        profile="robotwin_lerobot_unified",
        task_count=50,
        trajectory_counts={"demo_clean": 2500, "demo_randomized": 2500},
        trained_task_config=None,
        action_bounds={"min": [-1.0] * 14, "max": [1.0] * 14},
        actions_are_normalized=False,
        gripper_convention="width_0_1",
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["camera_keys"] == ["head_camera", "left_camera", "right_camera"]
    assert payload["action_dim"] == 14
    assert payload["state_dim"] == 14
    assert payload["actions_are_normalized"] is False
    assert payload["action_order"] == "model_left6_right6_left_gripper_right_gripper"
    assert payload["robotwin_action_order"] == "left6_left_gripper_right6_right_gripper"


def test_load_robotwin_release_stats_uses_total_episodes_and_model_action_order(tmp_path):
    stats_path = tmp_path / "dataset_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "num_episodes": 27500,
                "action": {
                    "default": {
                        "global_min": list(range(14)),
                        "global_max": list(range(100, 114)),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    trajectory_counts, action_bounds = prepare.load_robotwin_release_stats(stats_path)

    assert trajectory_counts == {"total": 27500}
    assert action_bounds["min"] == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
    assert action_bounds["max"] == [100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 106, 113]


def test_load_robotwin_release_stats_rejects_wrong_action_dimension(tmp_path):
    stats_path = tmp_path / "dataset_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "num_episodes": 1,
                "action": {"default": {"global_min": [0] * 13, "global_max": [1] * 13}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="14"):
        prepare.load_robotwin_release_stats(stats_path)


def test_robotwin_training_statistics_use_release_values_in_model_order(tmp_path):
    stats_path = tmp_path / "dataset_stats.json"
    source_vector = list(range(14))
    payload = {
        "num_episodes": 2,
        "num_transition": 20,
        "action": {"default": {}},
        "state": {"default": {}},
    }
    for field_name in ("action", "state"):
        for statistic_name in (
            "global_mean",
            "global_std",
            "global_max",
            "global_min",
            "global_q01",
            "global_q99",
        ):
            payload[field_name]["default"][statistic_name] = source_vector
    stats_path.write_text(json.dumps(payload), encoding="utf-8")

    statistics = prepare.load_robotwin_training_statistics(stats_path)

    expected = [*range(6), *range(7, 13), 6, 13]
    assert statistics["action"]["min"] == expected
    assert statistics["proprio"]["mean"] == expected
    assert statistics["num_trajectories"] == 2
    assert statistics["num_transitions"] == 20


def test_write_robotwin_training_statistics_uses_stable_output_name(tmp_path):
    stats_path = tmp_path / "dataset_stats.json"
    vector = [float(index) for index in range(14)]
    field = {
        name: vector
        for name in (
            "global_mean",
            "global_std",
            "global_max",
            "global_min",
            "global_q01",
            "global_q99",
        )
    }
    stats_path.write_text(
        json.dumps(
            {
                "num_episodes": 1,
                "num_transition": 16,
                "action": {"default": field},
                "state": {"default": field},
            }
        ),
        encoding="utf-8",
    )

    output_path = prepare.write_robotwin_training_statistics(stats_path, tmp_path / "rlds")

    assert output_path.name == "robotwin_dataset_statistics.json"
    assert json.loads(output_path.read_text())["num_trajectories"] == 1


def test_lerobot_converter_command_uses_direct_robotwin_wrapper_and_any4lerobot_base(tmp_path, monkeypatch):
    base_converter = tmp_path / "lerobot2rlds.py"
    base_converter.touch()
    monkeypatch.setattr(prepare, "ANY4LEROBOT_CONVERTER", base_converter)
    converter_python = tmp_path / "lerobot-v21" / "bin" / "python"
    converter_python.parent.mkdir(parents=True)
    converter_python.write_text("", encoding="utf-8")
    command = prepare.build_lerobot_converter_command(
        source_dir=Path("/path/to/datasets/robotwin_lerobot"),
        output_dir=tmp_path / "rlds",
        dataset_name="robotwin_lerobot_unified",
        converter_python=converter_python,
    )
    assert command[0] == str(converter_python)
    assert command[1].endswith("/vla-scripts/convert_robotwin_lerobot_to_rlds.py")
    assert "--src-dir" in command
    assert "--output-dir" in command
    assert "--task-name" in command
    assert command[command.index("--base-converter") + 1] == str(base_converter)
    assert "--input-dir" not in command


def test_lerobot_converter_command_rejects_missing_converter_python(tmp_path, monkeypatch):
    base_converter = tmp_path / "lerobot2rlds.py"
    base_converter.touch()
    monkeypatch.setattr(prepare, "ANY4LEROBOT_CONVERTER", base_converter)
    with pytest.raises(FileNotFoundError, match="converter Python"):
        prepare.build_lerobot_converter_command(
            source_dir=tmp_path / "robotwin_lerobot",
            output_dir=tmp_path / "rlds",
            dataset_name="robotwin_lerobot_unified",
            converter_python=tmp_path / "missing-python",
        )


def test_hf_download_kwargs_default_to_lerobot_compatible_training_dataset(tmp_path):
    kwargs = prepare.build_hf_dataset_download_kwargs(
        repo_id=prepare.DEFAULT_HF_REPO_ID,
        local_dir=tmp_path / "robotwin_lerobot",
        allow_patterns=["meta/*"],
    )

    assert prepare.DEFAULT_HF_REPO_ID == "yuanty/robotwin2.0-fastwam"
    assert prepare.OFFICIAL_ROBOTWIN_HF_REPO_ID == "TianxingChen/RoboTwin2.0"
    assert kwargs == {
        "repo_id": "yuanty/robotwin2.0-fastwam",
        "repo_type": "dataset",
        "local_dir": str(tmp_path / "robotwin_lerobot"),
        "allow_patterns": ["meta/*"],
    }


def test_ensure_source_dir_skips_download_when_source_has_files(tmp_path, monkeypatch):
    source_dir = tmp_path / "robotwin_lerobot"
    (source_dir / "meta").mkdir(parents=True)
    calls = []

    monkeypatch.setattr(prepare, "download_hf_dataset", lambda **kwargs: calls.append(kwargs))
    prepare.ensure_source_dir(
        Namespace(
            source_dir=source_dir,
            skip_download=False,
            hf_repo_id="yuanty/robotwin2.0-fastwam",
            hf_allow_pattern=None,
            hf_endpoint="https://hf-mirror.com",
        )
    )

    assert calls == []


def test_ensure_source_dir_resumes_partial_split_archive(tmp_path, monkeypatch):
    source_dir = tmp_path / "robotwin_lerobot"
    source_dir.mkdir()
    (source_dir / "robotwin2.0.tar.gz.part-00").write_text("partial", encoding="utf-8")
    calls = []

    monkeypatch.setattr(prepare, "download_hf_dataset", lambda **kwargs: calls.append(kwargs))
    prepare.ensure_source_dir(
        Namespace(
            source_dir=source_dir,
            skip_download=False,
            hf_repo_id="yuanty/robotwin2.0-fastwam",
            hf_allow_pattern=None,
            hf_endpoint="https://huggingface.co",
            http_proxy="http://localhost:7895",
            https_proxy="http://localhost:7895",
        )
    )

    assert len(calls) == 1


def test_ensure_source_dir_ignores_huggingface_cache_marker(tmp_path, monkeypatch):
    source_dir = tmp_path / "robotwin_lerobot"
    cache_dir = source_dir / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True)
    (cache_dir / ".gitignore").write_text("*", encoding="utf-8")
    calls = []

    monkeypatch.setattr(prepare, "download_hf_dataset", lambda **kwargs: calls.append(kwargs))
    prepare.ensure_source_dir(
        Namespace(
            source_dir=source_dir,
            skip_download=False,
            hf_repo_id="yuanty/robotwin2.0-fastwam",
            hf_allow_pattern=None,
            hf_endpoint="https://huggingface.co",
            http_proxy="http://localhost:7895",
            https_proxy="http://localhost:7895",
        )
    )

    assert len(calls) == 1


def test_configure_hf_download_env_uses_official_endpoint_and_proxy_defaults(monkeypatch):
    for key in ("HF_ENDPOINT", "HF_HUB_DISABLE_XET", "http_proxy", "https_proxy"):
        monkeypatch.delenv(key, raising=False)

    prepare.configure_hf_download_env(
        hf_endpoint=prepare.DEFAULT_HF_ENDPOINT,
        http_proxy="http://localhost:7895",
        https_proxy="http://localhost:7895",
    )

    import os

    assert prepare.DEFAULT_HF_ENDPOINT == "https://huggingface.co"
    assert os.environ["HF_ENDPOINT"] == "https://huggingface.co"
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert os.environ["http_proxy"] == "http://localhost:7895"
    assert os.environ["https_proxy"] == "http://localhost:7895"


def test_configure_hf_download_env_overrides_broken_mirror(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")

    prepare.configure_hf_download_env(
        hf_endpoint="https://huggingface.co",
        http_proxy=None,
        https_proxy=None,
    )

    import os

    assert os.environ["HF_ENDPOINT"] == "https://huggingface.co"


def test_validate_lerobot_source_dir_accepts_meta_data_videos_root(tmp_path):
    source_dir = tmp_path / "robotwin_lerobot"
    for child in ("meta", "data", "videos"):
        (source_dir / child).mkdir(parents=True)

    prepare.validate_lerobot_source_dir(source_dir)


def test_validate_lerobot_source_dir_rejects_missing_meta(tmp_path):
    source_dir = tmp_path / "robotwin_lerobot"
    source_dir.mkdir()

    try:
        prepare.validate_lerobot_source_dir(source_dir)
    except FileNotFoundError as exc:
        assert "LeRobot source" in str(exc)
        assert "meta" in str(exc)
    else:
        raise AssertionError("expected missing LeRobot meta directory to fail")


def test_validate_fastwam_lerobot_contract_accepts_three_cameras_and_14d_vectors(tmp_path):
    source_dir = tmp_path / "robotwin2.0"
    write_fastwam_lerobot_metadata(source_dir)

    summary = prepare.validate_robotwin_lerobot_contract(source_dir)

    assert summary == {"fps": 50.0, "num_episodes": 2, "num_frames": 40}


@pytest.mark.parametrize(
    ("metadata_kwargs", "error"),
    [
        ({"fps": 30}, "50 FPS"),
        ({"action_shape": [13]}, "action.*14"),
        (
            {
                "camera_keys": (
                    "observation.images.cam_high",
                    "observation.images.cam_left_wrist",
                )
            },
            "cam_right_wrist",
        ),
    ],
)
def test_validate_fastwam_lerobot_contract_rejects_training_schema_mismatch(tmp_path, metadata_kwargs, error):
    source_dir = tmp_path / "robotwin2.0"
    write_fastwam_lerobot_metadata(source_dir, **metadata_kwargs)

    with pytest.raises(ValueError, match=error):
        prepare.validate_robotwin_lerobot_contract(source_dir)


def test_validate_fastwam_lerobot_contract_requires_episode_and_task_metadata(tmp_path):
    source_dir = tmp_path / "robotwin2.0"
    write_fastwam_lerobot_metadata(source_dir)
    (source_dir / "meta" / "tasks.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="tasks.jsonl"):
        prepare.validate_robotwin_lerobot_contract(source_dir)


def test_validate_fastwam_lerobot_contract_rejects_episode_shorter_than_action_chunk(tmp_path):
    source_dir = tmp_path / "robotwin2.0"
    write_fastwam_lerobot_metadata(source_dir, episode_lengths=(15, 20))

    with pytest.raises(ValueError, match="at least 16 frames"):
        prepare.validate_robotwin_lerobot_contract(source_dir)


def test_validate_fastwam_lerobot_contract_rejects_truncated_episode_metadata(tmp_path):
    source_dir = tmp_path / "robotwin2.0"
    write_fastwam_lerobot_metadata(source_dir)
    episodes_path = source_dir / "meta" / "episodes.jsonl"
    episodes_path.write_text(episodes_path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="episode count"):
        prepare.validate_robotwin_lerobot_contract(source_dir)


def test_validate_robotwin_rlds_output_accepts_three_views_and_matching_episode_count(tmp_path):
    write_robotwin_rlds_metadata(tmp_path)

    summary = prepare.validate_robotwin_rlds_output(
        tmp_path,
        dataset_name="robotwin_lerobot_unified",
        expected_episodes=2,
    )

    assert summary == {"num_episodes": 2, "version": "0.1.0"}


def test_validate_robotwin_rlds_output_rejects_missing_camera(tmp_path):
    write_robotwin_rlds_metadata(tmp_path, camera_keys=("cam_high", "cam_left_wrist"))

    with pytest.raises(ValueError, match="cam_right_wrist"):
        prepare.validate_robotwin_rlds_output(
            tmp_path,
            dataset_name="robotwin_lerobot_unified",
            expected_episodes=2,
        )


def test_validate_robotwin_rlds_output_rejects_lost_episode(tmp_path):
    write_robotwin_rlds_metadata(tmp_path, num_episodes=1)

    with pytest.raises(ValueError, match="episode count"):
        prepare.validate_robotwin_rlds_output(
            tmp_path,
            dataset_name="robotwin_lerobot_unified",
            expected_episodes=2,
        )


def test_extract_split_archive_recovers_lerobot_root(tmp_path, monkeypatch):
    archive_source = tmp_path / "archive_source"
    for child in ("meta", "data", "videos"):
        (archive_source / "robotwin2.0" / child).mkdir(parents=True)
    (archive_source / "robotwin2.0" / "meta" / "info.json").write_text("{}", encoding="utf-8")
    archive_path = tmp_path / "robotwin2.0.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(archive_source / "robotwin2.0", arcname="robotwin2.0")

    payload = archive_path.read_bytes()
    midpoint = len(payload) // 2
    download_dir = tmp_path / "download"
    download_dir.mkdir()
    (download_dir / "robotwin2.0.tar.gz.part-00").write_bytes(payload[:midpoint])
    (download_dir / "robotwin2.0.tar.gz.part-01").write_bytes(payload[midpoint:])
    monkeypatch.setattr(prepare, "DEFAULT_SPLIT_ARCHIVE_PART_SIZES", (midpoint, len(payload) - midpoint))

    root = prepare.extract_robotwin_split_archive(download_dir)

    assert root == download_dir / "robotwin2.0"
    assert (root / "meta" / "info.json").read_text(encoding="utf-8") == "{}"


def test_resolve_lerobot_source_root_accepts_extracted_subdir(tmp_path):
    download_dir = tmp_path / "download"
    extracted_root = download_dir / "robotwin2.0"
    for child in ("meta", "data", "videos"):
        (extracted_root / child).mkdir(parents=True)
    (extracted_root / prepare.EXTRACTION_COMPLETE_MARKER).write_text("complete", encoding="utf-8")

    root = prepare.resolve_lerobot_source_root(download_dir)

    assert root == download_dir / "robotwin2.0"


def test_main_download_only_does_not_write_rlds_manifest(tmp_path, monkeypatch):
    source_dir = tmp_path / "download"
    extracted_root = source_dir / "robotwin2.0"
    for child in ("meta", "data", "videos"):
        (extracted_root / child).mkdir(parents=True)
    (extracted_root / prepare.EXTRACTION_COMPLETE_MARKER).write_text("complete", encoding="utf-8")
    output_dir = tmp_path / "rlds"
    args = Namespace(
        source="lerobot/robotwin_unified",
        source_dir=source_dir,
        output_dir=output_dir,
        dataset_name="robotwin_lerobot_unified",
        task_config=None,
        task_count=50,
        clean_count=2500,
        randomized_count=25000,
        skip_convert=False,
        download_only=True,
        actions_are_normalized=False,
        gripper_convention="width_0_1",
    )
    validated = []
    monkeypatch.setattr(prepare, "parse_args", lambda: args)
    monkeypatch.setattr(prepare, "ensure_source_dir", lambda parsed_args: None)
    monkeypatch.setattr(
        prepare,
        "validate_robotwin_lerobot_contract",
        lambda source: validated.append(source)
        or {"fps": 50.0, "num_episodes": 27500, "num_frames": 6075103},
    )

    prepare.main()

    assert validated == [extracted_root]
    assert not (output_dir / "robotwin_manifest.json").exists()


def test_snapshot_download_retries_transient_failures():
    attempts = []

    def flaky_download(**kwargs):
        attempts.append(kwargs)
        if len(attempts) < 3:
            raise RuntimeError("transient download failure")
        return "/tmp/robotwin"

    result = prepare.run_snapshot_download_with_retries(
        flaky_download,
        {"repo_id": "yuanty/robotwin2.0-fastwam"},
        max_attempts=3,
        retry_delay_s=0,
    )

    assert result == "/tmp/robotwin"
    assert len(attempts) == 3


def test_build_robotwin_split_archive_specs_has_eight_public_parts(tmp_path):
    specs = prepare.build_robotwin_split_archive_specs(
        repo_id="yuanty/robotwin2.0-fastwam",
        local_dir=tmp_path,
        hf_endpoint="https://huggingface.co",
    )

    assert len(specs) == 8
    assert specs[0][0] == tmp_path / "robotwin2.0.tar.gz.part-00"
    assert specs[-1][0] == tmp_path / "robotwin2.0.tar.gz.part-07"
    assert specs[0][1].endswith("/resolve/main/robotwin2.0.tar.gz.part-00?download=true")
    assert specs[0][2] == 10737418240
    assert specs[-1][2] == 3924503300
    assert sum(expected_size for _, _, expected_size in specs) == 79086430980


def test_split_archive_part_completion_requires_exact_release_size(tmp_path):
    part = tmp_path / "part-00"
    assert prepare.split_archive_part_is_complete(part, 3) is False

    part.write_bytes(b"ab")
    assert prepare.split_archive_part_is_complete(part, 3) is False

    part.write_bytes(b"abc")
    assert prepare.split_archive_part_is_complete(part, 3) is True

    part.write_bytes(b"abcd")
    with pytest.raises(ValueError, match="larger than expected"):
        prepare.split_archive_part_is_complete(part, 3)


def test_build_robotwin_release_metadata_specs_includes_stats_and_readme(tmp_path):
    specs = prepare.build_robotwin_release_metadata_specs(
        repo_id="yuanty/robotwin2.0-fastwam",
        local_dir=tmp_path,
        hf_endpoint="https://huggingface.co",
    )

    assert specs == [
        (
            tmp_path / "README.md",
            "https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam/resolve/main/README.md?download=true",
        ),
        (
            tmp_path / "dataset_stats.json",
            "https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam/resolve/main/dataset_stats.json?download=true",
        ),
    ]


def test_build_curl_resume_command_enables_resume_and_retry(tmp_path):
    command = prepare.build_curl_resume_command(
        "https://huggingface.co/datasets/example/repo/resolve/main/part-00?download=true",
        tmp_path / "part-00",
    )

    assert command[0] == "curl"
    assert "--http1.1" in command
    assert ["--continue-at", "-"] == command[command.index("--continue-at") : command.index("--continue-at") + 2]
    assert "--retry" not in command
    assert "--retry-all-errors" not in command
    assert "--output" in command


def test_build_curl_overwrite_command_does_not_resume(tmp_path):
    command = prepare.build_curl_overwrite_command(
        "https://huggingface.co/datasets/example/repo/resolve/main/dataset_stats.json?download=true",
        tmp_path / "dataset_stats.json",
    )

    assert "--continue-at" not in command
    assert "--output" in command


def test_split_downloader_fetches_release_metadata_before_archives(tmp_path, monkeypatch):
    calls = []
    completed_parts = set()

    def fake_download(url, output_path, **kwargs):
        calls.append((output_path.name, kwargs.get("resume", True)))
        if output_path.name.startswith("robotwin2.0.tar.gz.part-"):
            completed_parts.add(output_path)

    monkeypatch.setattr(prepare, "run_curl_download_with_retries", fake_download)
    monkeypatch.setattr(
        prepare,
        "split_archive_part_is_complete",
        lambda output_path, expected_size: output_path in completed_parts,
    )

    prepare.download_robotwin_split_archives_with_curl(
        repo_id=prepare.DEFAULT_HF_REPO_ID,
        local_dir=tmp_path,
        hf_endpoint="https://huggingface.co",
        max_workers=1,
    )

    assert calls[:2] == [("README.md", False), ("dataset_stats.json", False)]
    assert [name for name, _ in calls[2:]] == [
        f"robotwin2.0.tar.gz.part-{index:02d}" for index in range(8)
    ]
    assert all(resume is True for _, resume in calls[2:])


def test_split_downloader_skips_parts_that_are_already_complete(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        prepare,
        "run_curl_download_with_retries",
        lambda url, output_path, **kwargs: calls.append(output_path.name),
    )
    monkeypatch.setattr(
        prepare,
        "split_archive_part_is_complete",
        lambda output_path, expected_size: output_path.name.startswith("robotwin2.0.tar.gz.part-"),
    )

    prepare.download_robotwin_split_archives_with_curl(
        repo_id=prepare.DEFAULT_HF_REPO_ID,
        local_dir=tmp_path,
        hf_endpoint="https://huggingface.co",
        max_workers=8,
    )

    assert calls == ["README.md", "dataset_stats.json"]


def test_python_retry_restarts_curl_after_each_failure(tmp_path):
    return_codes = iter([18, 18, 0])
    commands = []

    def fake_run(command, check):
        commands.append((command, check))
        return Namespace(returncode=next(return_codes))

    prepare.run_curl_download_with_retries(
        "https://huggingface.co/example",
        tmp_path / "part-00",
        max_attempts=3,
        retry_delay_s=0,
        run_fn=fake_run,
        sleep_fn=lambda _: None,
    )

    assert len(commands) == 3
    assert all(check is False for _, check in commands)


def test_python_retry_stops_on_http_error(tmp_path):
    commands = []

    def fake_run(command, check):
        commands.append((command, check))
        return Namespace(returncode=22)

    with pytest.raises(subprocess.CalledProcessError) as error:
        prepare.run_curl_download_with_retries(
            "https://huggingface.co/missing",
            tmp_path / "part-00",
            max_attempts=1000,
            retry_delay_s=0,
            run_fn=fake_run,
            sleep_fn=lambda _: None,
        )

    assert error.value.returncode == 22
    assert len(commands) == 1


def test_download_hf_dataset_uses_curl_for_default_split_repo(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        prepare,
        "download_robotwin_split_archives_with_curl",
        lambda **kwargs: calls.append(kwargs),
    )

    prepare.download_hf_dataset(
        repo_id=prepare.DEFAULT_HF_REPO_ID,
        local_dir=tmp_path,
        allow_patterns=None,
        hf_endpoint="https://huggingface.co",
        http_proxy=None,
        https_proxy=None,
        max_attempts=8,
        retry_delay_s=0,
    )

    assert len(calls) == 1
    assert calls[0]["repo_id"] == prepare.DEFAULT_HF_REPO_ID
    assert calls[0]["max_workers"] == prepare.DEFAULT_CURL_DOWNLOAD_WORKERS
