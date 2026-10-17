#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np


DEFAULT_BASE_CONVERTER = Path("/path/to/any4lerobot/lerobot2rlds/lerobot2rlds.py")
VIDEO_FEATURE_PREFIX = "observation.images."


def load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def load_task_map(raw_dir: Path) -> dict[int, str]:
    return {
        int(record["task_index"]): str(record["task"])
        for record in load_jsonl(raw_dir / "meta" / "tasks.jsonl")
    }


def load_episode_lengths(raw_dir: Path, max_episodes: int | None) -> list[tuple[int, int]]:
    episodes = [
        (int(record["episode_index"]), int(record["length"]))
        for record in load_jsonl(raw_dir / "meta" / "episodes.jsonl")
    ]
    return episodes if max_episodes is None else episodes[:max_episodes]


def resolve_episode_path(
    raw_dir: Path,
    template: str,
    *,
    episode_index: int,
    chunks_size: int,
    video_key: str | None = None,
) -> Path:
    values = {
        "episode_chunk": episode_index // chunks_size,
        "episode_index": episode_index,
        "video_key": video_key,
    }
    return raw_dir / template.format(**values)


def validate_episode_dataframe(dataframe, *, episode_index: int, expected_length: int):
    if len(dataframe) != expected_length:
        raise ValueError(
            f"Episode {episode_index} parquet has {len(dataframe)} frames, expected {expected_length}"
        )
    dataframe = dataframe.sort_values("frame_index", kind="stable").reset_index(drop=True)
    frame_indices = dataframe["frame_index"].to_numpy()
    if not np.array_equal(frame_indices, np.arange(expected_length)):
        raise ValueError(f"Episode {episode_index} has non-contiguous frame_index values")
    episode_indices = dataframe["episode_index"].to_numpy()
    if not np.all(episode_indices == episode_index):
        raise ValueError(f"Episode {episode_index} parquet contains another episode index")
    return dataframe


def build_episode_steps(
    dataframe,
    frames_by_camera: dict[str, list[np.ndarray]],
    task_map: dict[int, str],
) -> list[dict[str, Any]]:
    episode_length = len(dataframe)
    states = dataframe["observation.state"].to_numpy()
    actions = dataframe["action"].to_numpy()
    task_indices = dataframe["task_index"].to_numpy()
    steps = []
    for step_index in range(episode_length):
        is_last = step_index == episode_length - 1
        observation = {
            camera_name: frames[step_index]
            for camera_name, frames in frames_by_camera.items()
        }
        observation["state"] = np.asarray(states[step_index], dtype=np.float32)
        task_index = int(task_indices[step_index])
        if task_index not in task_map:
            raise KeyError(f"Missing language task_index {task_index}")
        steps.append(
            {
                "observation": observation,
                "action": np.asarray(actions[step_index], dtype=np.float32),
                "language_instruction": task_map[task_index],
                "is_first": step_index == 0,
                "is_last": is_last,
                "is_terminal": is_last,
            }
        )
    return steps


def bounded_parallel_map(function, items, *, workers: int):
    if workers < 1:
        raise ValueError("decode workers must be at least 1")
    if workers == 1:
        yield from map(function, items)
        return

    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        for _ in range(workers):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass


def load_base_converter(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing any4lerobot base converter: {path}")
    spec = importlib.util.spec_from_file_location("robotwin_any4lerobot_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import any4lerobot base converter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install_direct_generator(base_converter, *, decode_workers: int) -> None:
    def _generate_examples(self):
        import pandas as pd

        raw_dir = Path(self.raw_dir)
        info = json.loads((raw_dir / "meta" / "info.json").read_text(encoding="utf-8"))
        chunks_size = int(info["chunks_size"])
        data_path_template = str(info["data_path"])
        video_path_template = str(info["video_path"])
        video_features = [
            name
            for name, feature in info["features"].items()
            if name.startswith(VIDEO_FEATURE_PREFIX) and feature.get("dtype") == "video"
        ]
        if len(video_features) != 3:
            raise ValueError(f"Expected three RoboTwin video features, got {video_features}")
        task_map = load_task_map(raw_dir)
        episode_lengths = load_episode_lengths(raw_dir, self.max_episodes)

        def decode_episode(episode_metadata):
            episode_index, expected_length = episode_metadata
            parquet_path = resolve_episode_path(
                raw_dir,
                data_path_template,
                episode_index=episode_index,
                chunks_size=chunks_size,
            )
            dataframe = validate_episode_dataframe(
                pd.read_parquet(parquet_path),
                episode_index=episode_index,
                expected_length=expected_length,
            )
            frames_by_camera = {}
            for video_feature in video_features:
                video_path = resolve_episode_path(
                    raw_dir,
                    video_path_template,
                    episode_index=episode_index,
                    chunks_size=chunks_size,
                    video_key=video_feature,
                )
                camera_name = video_feature.removeprefix(VIDEO_FEATURE_PREFIX)
                frames_by_camera[camera_name] = base_converter._decode_video_rgb(
                    video_path, expected_length
                )
            return episode_index, {
                "steps": build_episode_steps(dataframe, frames_by_camera, task_map)
            }

        def generate():
            decoded_episodes = bounded_parallel_map(
                decode_episode,
                episode_lengths,
                workers=decode_workers,
            )
            for position, (episode_index, episode) in enumerate(decoded_episodes, start=1):
                parquet_path = resolve_episode_path(
                    raw_dir, data_path_template, episode_index=episode_index, chunks_size=chunks_size
                )
                if position == 1 or position % 100 == 0:
                    logging.info(
                        "Direct RoboTwin conversion: episode %d/%d (%s)",
                        position,
                        len(episode_lengths),
                        parquet_path.name,
                    )
                yield str(episode_index), episode

        return generate()

    base_converter.DatasetBuilder._generate_examples = _generate_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct Fast-WAM LeRobot to RLDS converter")
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--base-converter", type=Path, default=DEFAULT_BASE_CONVERTER)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--encoding-format", choices=["jpeg", "png"], default="jpeg")
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--decode-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_converter = load_base_converter(args.base_converter)
    install_direct_generator(base_converter, decode_workers=args.decode_workers)
    base_converter.main(
        src_dir=args.src_dir,
        output_dir=args.output_dir,
        task_name=args.task_name,
        version=args.version,
        encoding_format=args.encoding_format,
        enable_beam=False,
        max_episodes=args.max_episodes,
        direct_robocasa_v20=False,
        direct_robocasa_merged_panda=False,
        beam_run_mode="multi_processing",
        beam_num_workers=1,
    )


if __name__ == "__main__":
    main()
