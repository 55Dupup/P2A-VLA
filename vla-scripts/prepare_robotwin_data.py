#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROBOTWIN_LEROBOT_CONVERTER = Path(__file__).with_name("convert_robotwin_lerobot_to_rlds.py")
ANY4LEROBOT_CONVERTER = Path("/path/to/any4lerobot/lerobot2rlds/lerobot2rlds.py")
DEFAULT_LEROBOT_V21_PYTHON = Path("python")
DEFAULT_HF_REPO_ID = "yuanty/robotwin2.0-fastwam"
OFFICIAL_ROBOTWIN_HF_REPO_ID = "TianxingChen/RoboTwin2.0"
STARVLA_CLEAN_HF_REPO_ID = "StarVLA/RoboTwin-Clean"
STARVLA_RANDOMIZED_HF_REPO_ID = "StarVLA/RoboTwin-Randomized"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
DEFAULT_HTTP_PROXY = "http://localhost:7895"
DEFAULT_HTTPS_PROXY = "http://localhost:7895"
DEFAULT_EXTRACTED_SUBDIR = "robotwin2.0"
EXTRACTION_COMPLETE_MARKER = ".vla_adapter_extract_complete"
DEFAULT_SPLIT_ARCHIVE_GLOB = "robotwin2.0.tar.gz.part-*"
DEFAULT_SPLIT_ARCHIVE_PART_COUNT = 8
DEFAULT_SPLIT_ARCHIVE_PART_SIZES = (10737418240,) * 7 + (3924503300,)
DEFAULT_RELEASE_METADATA_FILENAMES = ("README.md", "dataset_stats.json")
DEFAULT_RLDS_VERSION = "0.1.0"
DEFAULT_CURL_DOWNLOAD_WORKERS = 8
DEFAULT_CURL_MAX_ATTEMPTS = 1000
DEFAULT_CURL_RETRY_DELAY_S = 2.0
DEFAULT_DOWNLOAD_MAX_ATTEMPTS = 8
DEFAULT_DOWNLOAD_RETRY_DELAY_S = 15.0
DEFAULT_CAMERA_KEYS = ["head_camera", "left_camera", "right_camera"]
ROBOTWIN_TRAINING_STATISTICS_FILENAME = "robotwin_dataset_statistics.json"
FASTWAM_LEROBOT_CAMERA_FEATURES = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
FASTWAM_LEROBOT_ACTION_FEATURES = ("action", "action.default")
FASTWAM_LEROBOT_STATE_FEATURES = ("observation.state", "observation.state.default")
FASTWAM_LEROBOT_FPS = 50.0
DEFAULT_ACTION_ORDER = "model_left6_right6_left_gripper_right_gripper"
DEFAULT_ROBOTWIN_ACTION_ORDER = "left6_left_gripper_right6_right_gripper"


def infer_profile(source: str, task_config: str | None, clean_count: int, randomized_count: int) -> str:
    if source == "lerobot/robotwin_unified":
        return "robotwin_lerobot_unified"
    if task_config in {"demo_clean", "demo_randomized"} and (clean_count == 0 or randomized_count == 0):
        return "robotwin_generated_single_setting"
    if clean_count == 50 and randomized_count == 0:
        return "robotwin_leaderboard_clean50"
    if clean_count > 0 and randomized_count > 0:
        return "robotwin_mixed_clean_randomized"
    raise ValueError(
        f"Cannot infer RoboTwin profile from source={source}, task_config={task_config}, "
        f"clean_count={clean_count}, randomized_count={randomized_count}"
    )


def write_manifest(
    *,
    manifest_path: Path,
    dataset_name: str,
    source: str,
    profile: str,
    task_count: int,
    trajectory_counts: dict[str, int],
    trained_task_config: str | None,
    action_bounds: dict[str, list[float]],
    actions_are_normalized: bool,
    gripper_convention: str,
) -> None:
    payload = {
        "dataset_name": dataset_name,
        "source": source,
        "profile": profile,
        "task_count": task_count,
        "trajectory_counts": trajectory_counts,
        "camera_keys": list(DEFAULT_CAMERA_KEYS),
        "action_dim": 14,
        "state_dim": 14,
        "action_order": DEFAULT_ACTION_ORDER,
        "robotwin_action_order": DEFAULT_ROBOTWIN_ACTION_ORDER,
        "action_normalization": "bounds",
        "action_bounds": action_bounds,
        "actions_are_normalized": actions_are_normalized,
        "gripper_convention": gripper_convention,
        "trained_task_config": trained_task_config,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _robotwin_vector_to_model_order(values: list[float], *, field_name: str) -> list[float]:
    if not isinstance(values, list) or len(values) != 14:
        raise ValueError(f"RoboTwin release statistic {field_name} must be a 14-element list")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError(f"RoboTwin release statistic {field_name} must contain only numbers")
    return [*values[:6], *values[7:13], values[6], values[13]]


def load_robotwin_release_stats(stats_path: Path) -> tuple[dict[str, int], dict[str, list[float]]]:
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing RoboTwin release statistics: {stats_path}")
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    num_episodes = payload.get("num_episodes")
    if isinstance(num_episodes, bool) or not isinstance(num_episodes, int) or num_episodes < 1:
        raise ValueError("RoboTwin release statistic num_episodes must be a positive integer")
    try:
        action_stats = payload["action"]["default"]
        action_min = action_stats["global_min"]
        action_max = action_stats["global_max"]
    except (KeyError, TypeError) as exc:
        raise ValueError("RoboTwin release statistics must contain action.default.global_min/global_max") from exc

    return (
        {"total": num_episodes},
        {
            "min": _robotwin_vector_to_model_order(action_min, field_name="action.default.global_min"),
            "max": _robotwin_vector_to_model_order(action_max, field_name="action.default.global_max"),
        },
    )


def load_robotwin_training_statistics(stats_path: Path) -> dict[str, Any]:
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing RoboTwin release statistics: {stats_path}")
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    num_episodes = payload.get("num_episodes")
    num_transitions = payload.get("num_transition")
    if isinstance(num_episodes, bool) or not isinstance(num_episodes, int) or num_episodes < 1:
        raise ValueError("RoboTwin release statistic num_episodes must be a positive integer")
    if (
        isinstance(num_transitions, bool)
        or not isinstance(num_transitions, int)
        or num_transitions < num_episodes
    ):
        raise ValueError("RoboTwin release statistic num_transition must be a valid positive integer")

    def vector_statistics(field_name: str) -> dict[str, list[float]]:
        try:
            source = payload[field_name]["default"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"RoboTwin release statistics are missing {field_name}.default") from exc
        output = {}
        for target_name, source_name in (
            ("mean", "global_mean"),
            ("std", "global_std"),
            ("max", "global_max"),
            ("min", "global_min"),
            ("q01", "global_q01"),
            ("q99", "global_q99"),
        ):
            try:
                values = source[source_name]
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"RoboTwin release statistics are missing {field_name}.default.{source_name}"
                ) from exc
            output[target_name] = _robotwin_vector_to_model_order(
                values,
                field_name=f"{field_name}.default.{source_name}",
            )
        return output

    return {
        "action": vector_statistics("action"),
        "proprio": vector_statistics("state"),
        "num_transitions": num_transitions,
        "num_trajectories": num_episodes,
    }


def write_robotwin_training_statistics(stats_path: Path, output_dir: Path) -> Path:
    output_path = output_dir / ROBOTWIN_TRAINING_STATISTICS_FILENAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(load_robotwin_training_statistics(stats_path), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def build_lerobot_converter_command(
    *,
    source_dir: Path,
    output_dir: Path,
    dataset_name: str,
    converter_python: Path = DEFAULT_LEROBOT_V21_PYTHON,
) -> list[str]:
    if not ROBOTWIN_LEROBOT_CONVERTER.exists():
        raise FileNotFoundError(f"Missing RoboTwin LeRobot converter: {ROBOTWIN_LEROBOT_CONVERTER}")
    if not ANY4LEROBOT_CONVERTER.exists():
        raise FileNotFoundError(f"Missing any4lerobot base converter: {ANY4LEROBOT_CONVERTER}")
    if not converter_python.is_file():
        raise FileNotFoundError(f"Missing LeRobot v2.1 converter Python: {converter_python}")
    return [
        str(converter_python),
        str(ROBOTWIN_LEROBOT_CONVERTER),
        "--src-dir",
        str(source_dir),
        "--output-dir",
        str(output_dir),
        "--task-name",
        dataset_name,
        "--base-converter",
        str(ANY4LEROBOT_CONVERTER),
    ]


def validate_lerobot_source_dir(source_dir: Path) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing LeRobot source directory: {source_dir}")
    if not (source_dir / "meta").exists():
        raise FileNotFoundError(
            f"LeRobot source directory must contain a `meta` directory before RLDS conversion: {source_dir}. "
            "Use a LeRobot-compatible RoboTwin dataset or adjust --source-dir to the dataset root."
        )


def _find_lerobot_feature(features: dict, candidates: tuple[str, ...], label: str) -> tuple[str, dict]:
    for candidate in candidates:
        feature = features.get(candidate)
        if isinstance(feature, dict):
            return candidate, feature
    raise ValueError(f"Fast-WAM LeRobot metadata is missing {label}; expected one of {list(candidates)}")


def _validate_vector_feature_shape(feature_name: str, feature: dict, expected_dim: int) -> None:
    shape = feature.get("shape")
    if shape != [expected_dim]:
        raise ValueError(
            f"Fast-WAM LeRobot feature `{feature_name}` must have shape [{expected_dim}], got {shape}"
        )


def _load_jsonl_objects(path: Path, label: str) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Fast-WAM LeRobot metadata `{path.name}` must exist and be non-empty")
    records = []
    try:
        with path.open("r", encoding="utf-8") as metadata_file:
            for line_number, line in enumerate(metadata_file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"record {line_number} is not a JSON object")
                records.append(record)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        raise ValueError(f"Unable to read Fast-WAM LeRobot {label}: {path}") from exc
    if not records:
        raise ValueError(f"Fast-WAM LeRobot metadata `{path.name}` must contain at least one record")
    return records


def validate_robotwin_lerobot_contract(source_dir: Path) -> dict[str, int | float]:
    info_path = source_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing Fast-WAM LeRobot metadata: {info_path}")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Unable to read Fast-WAM LeRobot metadata: {info_path}") from exc

    fps = info.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or float(fps) != FASTWAM_LEROBOT_FPS:
        raise ValueError(f"Fast-WAM RoboTwin data must use 50 FPS, got {fps}")

    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("Fast-WAM LeRobot metadata must contain a `features` object")
    for camera_feature in FASTWAM_LEROBOT_CAMERA_FEATURES:
        feature = features.get(camera_feature)
        if not isinstance(feature, dict):
            raise ValueError(f"Fast-WAM LeRobot metadata is missing camera feature `{camera_feature}`")
        shape = feature.get("shape")
        if not isinstance(shape, list) or len(shape) != 3 or 3 not in (shape[0], shape[-1]):
            raise ValueError(
                f"Fast-WAM LeRobot camera feature `{camera_feature}` must be a three-channel image, got {shape}"
            )

    action_name, action_feature = _find_lerobot_feature(features, FASTWAM_LEROBOT_ACTION_FEATURES, "action")
    state_name, state_feature = _find_lerobot_feature(features, FASTWAM_LEROBOT_STATE_FEATURES, "state")
    _validate_vector_feature_shape(action_name, action_feature, 14)
    _validate_vector_feature_shape(state_name, state_feature, 14)

    num_episodes = info.get("total_episodes")
    num_frames = info.get("total_frames")
    for field_name, value in (("total_episodes", num_episodes), ("total_frames", num_frames)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Fast-WAM LeRobot metadata `{field_name}` must be a positive integer, got {value}")

    episodes = _load_jsonl_objects(source_dir / "meta" / "episodes.jsonl", "episode metadata")
    _load_jsonl_objects(source_dir / "meta" / "tasks.jsonl", "task metadata")
    if len(episodes) != num_episodes:
        raise ValueError(
            f"Fast-WAM LeRobot episode count mismatch: episodes.jsonl has {len(episodes)}, "
            f"info.json reports {num_episodes}"
        )

    episode_lengths = []
    for episode in episodes:
        length = episode.get("length")
        episode_index = episode.get("episode_index")
        if isinstance(length, bool) or not isinstance(length, int) or length < 16:
            raise ValueError(
                f"Fast-WAM LeRobot episode {episode_index} must contain at least 16 frames for the action chunk, "
                f"got {length}"
            )
        episode_lengths.append(length)
    if sum(episode_lengths) != num_frames:
        raise ValueError(
            f"Fast-WAM LeRobot frame count mismatch: episode lengths sum to {sum(episode_lengths)}, "
            f"info.json reports {num_frames}"
        )

    return {"fps": float(fps), "num_episodes": num_episodes, "num_frames": num_frames}


def _load_json_object(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Unable to read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _rlds_dimensions(feature: dict) -> tuple[int, ...]:
    shape = feature.get("shape", {})
    dimensions = shape.get("dimensions") if isinstance(shape, dict) else None
    if not isinstance(dimensions, list):
        return ()
    try:
        return tuple(int(dimension) for dimension in dimensions)
    except (TypeError, ValueError):
        return ()


def validate_robotwin_rlds_output(
    output_dir: Path,
    *,
    dataset_name: str,
    expected_episodes: int,
) -> dict[str, int | str]:
    dataset_dir = output_dir / dataset_name / DEFAULT_RLDS_VERSION
    info = _load_json_object(dataset_dir / "dataset_info.json", "RLDS dataset info")
    features_payload = _load_json_object(dataset_dir / "features.json", "RLDS feature schema")

    if info.get("name") != dataset_name or info.get("version") != DEFAULT_RLDS_VERSION:
        raise ValueError(
            f"RLDS output identity mismatch: expected {dataset_name}/{DEFAULT_RLDS_VERSION}, "
            f"got {info.get('name')}/{info.get('version')}"
        )
    if info.get("fileFormat") != "tfrecord":
        raise ValueError(f"RoboTwin RLDS output must use TFRecord, got {info.get('fileFormat')}")

    train_splits = [split for split in info.get("splits", []) if split.get("name") == "train"]
    if len(train_splits) != 1:
        raise ValueError("RoboTwin RLDS output must contain exactly one train split")
    try:
        num_episodes = sum(int(length) for length in train_splits[0]["shardLengths"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("RoboTwin RLDS train split has invalid shard lengths") from exc
    if num_episodes != expected_episodes:
        raise ValueError(
            f"RoboTwin RLDS episode count mismatch: converted {num_episodes}, expected {expected_episodes}"
        )

    try:
        step_features = features_payload["featuresDict"]["features"]["steps"]["sequence"]["feature"][
            "featuresDict"
        ]["features"]
        observation_features = step_features["observation"]["featuresDict"]["features"]
    except (KeyError, TypeError) as exc:
        raise ValueError("RoboTwin RLDS feature schema is missing the RLDS steps/observation structure") from exc

    for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        image = observation_features.get(camera, {}).get("image", {})
        dimensions = _rlds_dimensions(image)
        if image.get("dtype") != "uint8" or len(dimensions) != 3 or dimensions[-1] != 3:
            raise ValueError(f"RoboTwin RLDS camera `{camera}` must be an HWC uint8 RGB image, got {image}")

    action = step_features.get("action", {}).get("tensor", {})
    state = observation_features.get("state", {}).get("tensor", {})
    for label, tensor in (("action", action), ("state", state)):
        if tensor.get("dtype") != "float32" or _rlds_dimensions(tensor) != (14,):
            raise ValueError(f"RoboTwin RLDS {label} must be a 14-dimensional float32 tensor, got {tensor}")

    if "text" not in step_features.get("language_instruction", {}):
        raise ValueError("RoboTwin RLDS output is missing language_instruction text")
    for flag in ("is_first", "is_last", "is_terminal"):
        tensor = step_features.get(flag, {}).get("tensor", {})
        if tensor.get("dtype") != "bool":
            raise ValueError(f"RoboTwin RLDS output is missing boolean `{flag}`")

    tfrecord_paths = list(dataset_dir.glob(f"{dataset_name}-train.tfrecord-*"))
    if not tfrecord_paths or any(path.stat().st_size == 0 for path in tfrecord_paths):
        raise ValueError(f"RoboTwin RLDS output has no non-empty train TFRecord shards in {dataset_dir}")

    return {"num_episodes": num_episodes, "version": DEFAULT_RLDS_VERSION}


def resolve_lerobot_source_root(download_dir: Path) -> Path:
    if (download_dir / "meta").exists():
        return download_dir

    extracted_root = download_dir / DEFAULT_EXTRACTED_SUBDIR
    required_paths = [extracted_root / child for child in ("meta", "data", "videos")]
    if all(path.exists() for path in required_paths) and (extracted_root / EXTRACTION_COMPLETE_MARKER).exists():
        return extracted_root
    raise FileNotFoundError(
        f"No complete LeRobot root found under {download_dir}. Expected `meta` at the download root or a completed "
        f"`{DEFAULT_EXTRACTED_SUBDIR}/{{data,meta,videos}}` extraction."
    )


def extract_robotwin_split_archive(download_dir: Path) -> Path:
    archive_parts = validate_robotwin_split_archive_parts(download_dir)

    process = subprocess.Popen(
        ["tar", "-xzf", "-", "--no-same-owner", "--no-same-permissions", "-C", str(download_dir)],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for archive_part in archive_parts:
            with archive_part.open("rb") as source:
                shutil.copyfileobj(source, process.stdin)
    finally:
        process.stdin.close()

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, process.args)
    extracted_root = download_dir / DEFAULT_EXTRACTED_SUBDIR
    required_paths = [extracted_root / child for child in ("meta", "data", "videos")]
    if not all(path.exists() for path in required_paths):
        raise FileNotFoundError(
            f"Extracted RoboTwin archive is missing one or more LeRobot directories under {extracted_root}."
        )
    (extracted_root / EXTRACTION_COMPLETE_MARKER).write_text("complete\n", encoding="utf-8")
    return resolve_lerobot_source_root(download_dir)


def build_hf_dataset_download_kwargs(*, repo_id: str, local_dir: Path, allow_patterns: list[str] | None = None) -> dict:
    kwargs = {"repo_id": repo_id, "repo_type": "dataset", "local_dir": str(local_dir)}
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    return kwargs


def configure_hf_download_env(*, hf_endpoint: str, http_proxy: str | None, https_proxy: str | None) -> None:
    os.environ["HF_ENDPOINT"] = hf_endpoint
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if http_proxy:
        os.environ.setdefault("http_proxy", http_proxy)
    if https_proxy:
        os.environ.setdefault("https_proxy", https_proxy)


def run_snapshot_download_with_retries(
    download_fn,
    download_kwargs: dict,
    *,
    max_attempts: int,
    retry_delay_s: float,
):
    if max_attempts < 1:
        raise ValueError("download max attempts must be at least 1")
    for attempt in range(1, max_attempts + 1):
        try:
            return download_fn(**download_kwargs)
        except Exception:
            if attempt == max_attempts:
                raise
            print(f"RoboTwin download attempt {attempt}/{max_attempts} failed; retrying in {retry_delay_s}s")
            time.sleep(retry_delay_s)


def build_robotwin_split_archive_specs(
    *,
    repo_id: str,
    local_dir: Path,
    hf_endpoint: str,
) -> list[tuple[Path, str, int]]:
    base_url = f"{hf_endpoint.rstrip('/')}/datasets/{repo_id}/resolve/main"
    specs = []
    for part_index, expected_size in enumerate(DEFAULT_SPLIT_ARCHIVE_PART_SIZES):
        filename = f"robotwin2.0.tar.gz.part-{part_index:02d}"
        specs.append((local_dir / filename, f"{base_url}/{filename}?download=true", expected_size))
    return specs


def split_archive_part_is_complete(output_path: Path, expected_size: int) -> bool:
    if not output_path.exists():
        return False
    actual_size = output_path.stat().st_size
    if actual_size > expected_size:
        raise ValueError(
            f"RoboTwin archive part {output_path} is larger than expected: {actual_size} > {expected_size} bytes"
        )
    return actual_size == expected_size


def validate_robotwin_split_archive_parts(download_dir: Path) -> list[Path]:
    archive_parts = sorted(download_dir.glob(DEFAULT_SPLIT_ARCHIVE_GLOB))
    if len(archive_parts) != len(DEFAULT_SPLIT_ARCHIVE_PART_SIZES):
        raise ValueError(
            f"Expected {len(DEFAULT_SPLIT_ARCHIVE_PART_SIZES)} RoboTwin archive parts in {download_dir}, "
            f"found {len(archive_parts)}"
        )
    for part_index, expected_size in enumerate(DEFAULT_SPLIT_ARCHIVE_PART_SIZES):
        expected_path = download_dir / f"robotwin2.0.tar.gz.part-{part_index:02d}"
        if not split_archive_part_is_complete(expected_path, expected_size):
            actual_size = expected_path.stat().st_size if expected_path.exists() else 0
            raise ValueError(
                f"RoboTwin archive part {expected_path.name} is incomplete: "
                f"{actual_size} != {expected_size} bytes"
            )
    return archive_parts


def build_robotwin_release_metadata_specs(
    *,
    repo_id: str,
    local_dir: Path,
    hf_endpoint: str,
) -> list[tuple[Path, str]]:
    base_url = f"{hf_endpoint.rstrip('/')}/datasets/{repo_id}/resolve/main"
    return [
        (local_dir / filename, f"{base_url}/{filename}?download=true")
        for filename in DEFAULT_RELEASE_METADATA_FILENAMES
    ]


def _build_curl_download_command(url: str, output_path: Path, *, resume: bool) -> list[str]:
    command = [
        "curl",
        "--fail",
        "--location",
        "--http1.1",
        "--silent",
        "--show-error",
    ]
    if resume:
        command.extend(["--continue-at", "-"])
    command.extend(
        [
            "--connect-timeout",
            "30",
            "--speed-limit",
            "1024",
            "--speed-time",
            "120",
            "--output",
            str(output_path),
            url,
        ]
    )
    return command


def build_curl_resume_command(url: str, output_path: Path) -> list[str]:
    return _build_curl_download_command(url, output_path, resume=True)


def build_curl_overwrite_command(url: str, output_path: Path) -> list[str]:
    return _build_curl_download_command(url, output_path, resume=False)


def run_curl_download_with_retries(
    url: str,
    output_path: Path,
    *,
    max_attempts: int,
    retry_delay_s: float,
    resume: bool = True,
    run_fn=None,
    sleep_fn=None,
) -> None:
    if max_attempts < 1:
        raise ValueError("curl max attempts must be at least 1")
    run_fn = subprocess.run if run_fn is None else run_fn
    sleep_fn = time.sleep if sleep_fn is None else sleep_fn
    command = (
        build_curl_resume_command(url, output_path)
        if resume
        else build_curl_overwrite_command(url, output_path)
    )
    for attempt in range(1, max_attempts + 1):
        completed = run_fn(command, check=False)
        if completed.returncode == 0:
            return
        if completed.returncode == 22:
            raise subprocess.CalledProcessError(completed.returncode, command)
        if attempt == max_attempts:
            raise subprocess.CalledProcessError(completed.returncode, command)
        print(
            f"RoboTwin curl download failed for {output_path.name} "
            f"({attempt}/{max_attempts}); resuming in {retry_delay_s}s"
        )
        sleep_fn(retry_delay_s)


def download_robotwin_split_archives_with_curl(
    *,
    repo_id: str,
    local_dir: Path,
    hf_endpoint: str,
    max_workers: int,
) -> None:
    if max_workers < 1:
        raise ValueError("curl download workers must be at least 1")
    local_dir.mkdir(parents=True, exist_ok=True)
    specs = build_robotwin_split_archive_specs(
        repo_id=repo_id,
        local_dir=local_dir,
        hf_endpoint=hf_endpoint,
    )
    metadata_specs = build_robotwin_release_metadata_specs(
        repo_id=repo_id,
        local_dir=local_dir,
        hf_endpoint=hf_endpoint,
    )

    for output_path, url in metadata_specs:
        run_curl_download_with_retries(
            url,
            output_path,
            max_attempts=DEFAULT_CURL_MAX_ATTEMPTS,
            retry_delay_s=DEFAULT_CURL_RETRY_DELAY_S,
            resume=False,
        )

    def download_part(spec: tuple[Path, str, int]) -> None:
        output_path, url, expected_size = spec
        if split_archive_part_is_complete(output_path, expected_size):
            print(f"RoboTwin archive part already complete: {output_path.name}")
            return
        run_curl_download_with_retries(
            url,
            output_path,
            max_attempts=DEFAULT_CURL_MAX_ATTEMPTS,
            retry_delay_s=DEFAULT_CURL_RETRY_DELAY_S,
        )
        if not split_archive_part_is_complete(output_path, expected_size):
            actual_size = output_path.stat().st_size if output_path.exists() else 0
            raise ValueError(
                f"RoboTwin archive part {output_path.name} has unexpected size after download: "
                f"{actual_size} != {expected_size} bytes"
            )

    with ThreadPoolExecutor(max_workers=min(max_workers, len(specs))) as executor:
        list(executor.map(download_part, specs))


def download_hf_dataset(
    *,
    repo_id: str,
    local_dir: Path,
    allow_patterns: list[str] | None,
    hf_endpoint: str,
    http_proxy: str | None,
    https_proxy: str | None,
    max_attempts: int,
    retry_delay_s: float,
    download_workers: int = DEFAULT_CURL_DOWNLOAD_WORKERS,
) -> None:
    configure_hf_download_env(hf_endpoint=hf_endpoint, http_proxy=http_proxy, https_proxy=https_proxy)
    local_dir.mkdir(parents=True, exist_ok=True)
    if repo_id == DEFAULT_HF_REPO_ID and allow_patterns is None:
        download_robotwin_split_archives_with_curl(
            repo_id=repo_id,
            local_dir=local_dir,
            hf_endpoint=hf_endpoint,
            max_workers=download_workers,
        )
        return

    from huggingface_hub import snapshot_download

    run_snapshot_download_with_retries(
        snapshot_download,
        build_hf_dataset_download_kwargs(repo_id=repo_id, local_dir=local_dir, allow_patterns=allow_patterns),
        max_attempts=max_attempts,
        retry_delay_s=retry_delay_s,
    )


def ensure_source_dir(args: argparse.Namespace) -> None:
    try:
        resolve_lerobot_source_root(args.source_dir)
        has_complete_source = True
    except FileNotFoundError:
        has_complete_source = False
    if has_complete_source or args.skip_download:
        return
    download_hf_dataset(
        repo_id=args.hf_repo_id,
        local_dir=args.source_dir,
        allow_patterns=args.hf_allow_pattern,
        hf_endpoint=args.hf_endpoint,
        http_proxy=args.http_proxy,
        https_proxy=args.https_proxy,
        max_attempts=getattr(args, "download_max_attempts", DEFAULT_DOWNLOAD_MAX_ATTEMPTS),
        retry_delay_s=getattr(args, "download_retry_delay", DEFAULT_DOWNLOAD_RETRY_DELAY_S),
        download_workers=getattr(args, "download_workers", DEFAULT_CURL_DOWNLOAD_WORKERS),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RoboTwin RLDS data and manifest for VLA-Adapter")
    parser.add_argument("--source", default="lerobot/robotwin_unified")
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT)
    parser.add_argument("--hf-allow-pattern", action="append", default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--http-proxy", default=DEFAULT_HTTP_PROXY)
    parser.add_argument("--https-proxy", default=DEFAULT_HTTPS_PROXY)
    parser.add_argument("--download-max-attempts", type=int, default=DEFAULT_DOWNLOAD_MAX_ATTEMPTS)
    parser.add_argument("--download-retry-delay", type=float, default=DEFAULT_DOWNLOAD_RETRY_DELAY_S)
    parser.add_argument("--download-workers", type=int, default=DEFAULT_CURL_DOWNLOAD_WORKERS)
    parser.add_argument("--source-dir", type=Path, default=Path("/path/to/datasets/robotwin_lerobot"))
    parser.add_argument("--output-dir", type=Path, default=Path("/path/to/datasets/robotwin_rlds"))
    parser.add_argument("--dataset-stats-path", type=Path, default=None)
    parser.add_argument("--converter-python", type=Path, default=DEFAULT_LEROBOT_V21_PYTHON)
    parser.add_argument("--dataset-name", default="robotwin_lerobot_unified")
    parser.add_argument("--task-config", choices=["demo_clean", "demo_randomized"], default=None)
    parser.add_argument("--task-count", type=int, default=50)
    parser.add_argument("--clean-count", type=int, default=2500)
    parser.add_argument("--randomized-count", type=int, default=2500)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--actions-are-normalized", action="store_true")
    parser.add_argument("--gripper-convention", default="width_0_1", choices=["width_0_1", "binary_open_close"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = infer_profile(args.source, args.task_config, args.clean_count, args.randomized_count)
    ensure_source_dir(args)
    conversion_source_dir = None
    source_contract = None
    if args.source == "lerobot/robotwin_unified" and (args.download_only or not args.skip_convert):
        try:
            conversion_source_dir = resolve_lerobot_source_root(args.source_dir)
        except FileNotFoundError:
            conversion_source_dir = extract_robotwin_split_archive(args.source_dir)
        validate_lerobot_source_dir(conversion_source_dir)
        source_contract = validate_robotwin_lerobot_contract(conversion_source_dir)
        print(
            "Validated Fast-WAM LeRobot contract: "
            f"{source_contract['num_episodes']} episodes, {source_contract['num_frames']} frames, "
            f"{source_contract['fps']:g} FPS"
        )

    if args.download_only:
        if conversion_source_dir is None:
            raise ValueError("--download-only currently requires --source=lerobot/robotwin_unified")
        print(f"Prepared LeRobot source: {conversion_source_dir}")
        return

    if conversion_source_dir is not None:
        subprocess.run(
            build_lerobot_converter_command(
                source_dir=conversion_source_dir,
                output_dir=args.output_dir,
                dataset_name=args.dataset_name,
                converter_python=args.converter_python,
            ),
            check=True,
        )
        assert source_contract is not None
        rlds_contract = validate_robotwin_rlds_output(
            args.output_dir,
            dataset_name=args.dataset_name,
            expected_episodes=int(source_contract["num_episodes"]),
        )
        print(
            f"Validated RoboTwin RLDS output: {rlds_contract['num_episodes']} episodes, "
            f"version {rlds_contract['version']}"
        )
    if args.source == "lerobot/robotwin_unified":
        stats_path = args.dataset_stats_path or args.source_dir / "dataset_stats.json"
        trajectory_counts, action_bounds = load_robotwin_release_stats(stats_path)
        training_statistics_path = write_robotwin_training_statistics(stats_path, args.output_dir)
        print(f"Wrote RoboTwin training statistics: {training_statistics_path}")
        if source_contract is not None and trajectory_counts["total"] != source_contract["num_episodes"]:
            raise ValueError(
                "RoboTwin release statistics episode count does not match LeRobot metadata: "
                f"{trajectory_counts['total']} != {source_contract['num_episodes']}"
            )
    else:
        trajectory_counts = {"demo_clean": args.clean_count, "demo_randomized": args.randomized_count}
        action_bounds = {"min": [-1.0] * 14, "max": [1.0] * 14}

    write_manifest(
        manifest_path=args.output_dir / "robotwin_manifest.json",
        dataset_name=args.dataset_name,
        source=args.source,
        profile=profile,
        task_count=args.task_count,
        trajectory_counts=trajectory_counts,
        trained_task_config=args.task_config,
        action_bounds=action_bounds,
        actions_are_normalized=args.actions_are_normalized,
        gripper_convention=args.gripper_convention,
    )


if __name__ == "__main__":
    main()
