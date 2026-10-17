"""Dump RoboCasa train/eval image and language alignment artifacts.

This script does not load the policy. It compares the exact image/task fields
that the training pipeline and RoboCasa eval pipeline feed into the model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image, ImageDraw

import robocasa  # noqa: F401 - registers RoboCasa Gymnasium envs

from experiments.robot.robocasa.simulation_env import (
    ROBOCASA_IMAGE_ALIASES,
    ROBOCASA_INSTRUCTION_ALIASES,
    _lookup_alias,
    _parse_csv_items,
)


TRAIN_MODEL_INPUT_ORDER = (
    {
        "model_key": "full_image",
        "train_raw_key": "right_image",
        "train_canonical_key": "image_primary",
        "eval_canonical_key": "video.right_image",
        "eval_raw_key": "video.robot0_agentview_right",
    },
    {
        "model_key": "wrist_image",
        "train_raw_key": "left_image",
        "train_canonical_key": "image_left_wrist",
        "eval_canonical_key": "video.left_image",
        "eval_raw_key": "video.robot0_agentview_left",
    },
    {
        "model_key": "wrist_image_1",
        "train_raw_key": "wrist_image",
        "train_canonical_key": "image_right_wrist",
        "eval_canonical_key": "video.wrist_image",
        "eval_raw_key": "video.robot0_eye_in_hand",
    },
)


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _as_text(value.item())
        if value.size == 1:
            return _as_text(value.reshape(-1)[0])
    return str(value)


def _image_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > 3:
        array = array[-1]
    if array.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {array.shape}")
    return array.astype(np.uint8)


def _image_stats(image: np.ndarray) -> Dict[str, Any]:
    image = _image_array(image)
    return {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "min": int(image.min()),
        "max": int(image.max()),
        "mean": float(image.mean()),
        "std": float(image.std()),
        "channel_mean_rgb": [float(image[..., idx].mean()) for idx in range(image.shape[-1])],
    }


def _save_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(_image_array(image)).save(path)


def _draw_sheet(path: Path, rows: List[Tuple[str, Dict[str, np.ndarray]]]) -> None:
    tile_w, tile_h = 256, 256
    label_h = 44
    cols = len(TRAIN_MODEL_INPUT_ORDER)
    sheet = Image.new("RGB", (cols * tile_w, len(rows) * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)

    for row_idx, (row_name, images) in enumerate(rows):
        y0 = row_idx * (tile_h + label_h)
        for col_idx, spec in enumerate(TRAIN_MODEL_INPUT_ORDER):
            model_key = spec["model_key"]
            image = Image.fromarray(_image_array(images[model_key])).resize((tile_w, tile_h))
            x0 = col_idx * tile_w
            sheet.paste(image, (x0, y0 + label_h))
            draw.text((x0 + 6, y0 + 4), row_name, fill=(0, 0, 0))
            draw.text((x0 + 6, y0 + 22), model_key, fill=(0, 0, 0))

    sheet.save(path)


def _qwen_prompt(task: str) -> str:
    task = task.lower()
    return (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"What action should the robot take to {task}?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _first_matching_train_episode(
    data_root_dir: str,
    dataset_name: str,
    target_instruction: str,
    max_episodes: int,
    frame_index: int,
) -> Tuple[Dict[str, np.ndarray], str, Dict[str, Any]]:
    tf.config.set_visible_devices([], "GPU")
    builder = tfds.builder(dataset_name, data_dir=data_root_dir)
    dataset = builder.as_dataset(split="train", shuffle_files=False)
    target = target_instruction.strip().lower()
    fallback = None

    for episode_idx, episode in enumerate(dataset.take(max_episodes)):
        steps = list(episode["steps"].as_numpy_iterator())
        if not steps:
            continue
        first_instruction = _as_text(steps[0]["language_instruction"])
        selected_idx = min(max(frame_index, 0), len(steps) - 1)
        candidate = (
            {
                "full_image": _image_array(steps[selected_idx]["observation"]["right_image"]),
                "wrist_image": _image_array(steps[selected_idx]["observation"]["left_image"]),
                "wrist_image_1": _image_array(steps[selected_idx]["observation"]["wrist_image"]),
            },
            first_instruction,
            {
                "episode_index": int(episode_idx),
                "frame_index": int(selected_idx),
                "episode_length": int(len(steps)),
                "matched_target_instruction": first_instruction.strip().lower() == target,
            },
        )
        if fallback is None:
            fallback = candidate
        if first_instruction.strip().lower() == target:
            return candidate

    if fallback is None:
        raise RuntimeError(f"No episodes found in {dataset_name}.")
    return fallback


def _eval_reset_images(
    env_name: str,
    task_description: str,
    env_split: str,
    obj_registries: str,
    seed: int,
    flip_images: bool,
) -> Tuple[Dict[str, np.ndarray], str, Dict[str, Any]]:
    env = gym.make(
        env_name,
        enable_render=True,
        split=env_split,
        obj_registries=tuple(_parse_csv_items(obj_registries)) or None,
    )
    try:
        observation, info = env.reset(seed=seed)
        images = {}
        for spec in TRAIN_MODEL_INPUT_ORDER:
            canonical_key = spec["eval_canonical_key"]
            image = _lookup_alias(observation, ROBOCASA_IMAGE_ALIASES[canonical_key])
            if image is None:
                raise KeyError(f"Unable to find eval image for {canonical_key}; keys={sorted(observation.keys())}")
            image_array = _image_array(image)
            if flip_images:
                image_array = np.flipud(image_array).copy()
            images[spec["model_key"]] = image_array

        raw_instruction = _lookup_alias(observation, ROBOCASA_INSTRUCTION_ALIASES)
        model_instruction = None
        if task_description and task_description != "perform the task":
            model_instruction = task_description
        if model_instruction is None:
            model_instruction = raw_instruction
        if model_instruction is None:
            model_instruction = task_description

        return images, _as_text(model_instruction), {
            "raw_env_instruction": _as_text(raw_instruction) if raw_instruction is not None else None,
            "model_instruction_source": "task_description_arg"
            if task_description and task_description != "perform the task"
            else "raw_env_instruction",
            "reset_info": {key: _as_text(value) for key, value in info.items()},
        }
    finally:
        env.close()


def _write_artifacts(
    output_dir: Path,
    train_images: Dict[str, np.ndarray],
    eval_images: Dict[str, np.ndarray],
    train_instruction: str,
    eval_instruction: str,
    metadata: Dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in TRAIN_MODEL_INPUT_ORDER:
        model_key = spec["model_key"]
        _save_image(output_dir / f"train_{model_key}__{spec['train_raw_key']}.png", train_images[model_key])
        _save_image(output_dir / f"eval_{model_key}__{spec['eval_raw_key'].replace('.', '_')}.png", eval_images[model_key])

    _draw_sheet(
        output_dir / "train_eval_model_input_order.png",
        [("train", train_images), ("eval", eval_images)],
    )

    payload = {
        **metadata,
        "model_input_order": list(TRAIN_MODEL_INPUT_ORDER),
        "train_instruction": train_instruction,
        "eval_instruction": eval_instruction,
        "train_prompt": _qwen_prompt(train_instruction),
        "eval_prompt": _qwen_prompt(eval_instruction),
        "prompt_equal": _qwen_prompt(train_instruction) == _qwen_prompt(eval_instruction),
        "train_image_stats": {key: _image_stats(value) for key, value in train_images.items()},
        "eval_image_stats": {key: _image_stats(value) for key, value in eval_images.items()},
        "note": "Model inputs are image-only plus task text when USE_PROPRIO=False; state is intentionally not dumped here.",
    }
    metadata_path = output_dir / "alignment_metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root_dir", default="/path/to/datasets/robocasa_merged_24_tasks_100demos_v1_rlds")
    parser.add_argument("--dataset_name", default="robocasa_merged_24_tasks_100demos_v1")
    parser.add_argument("--env_name", default="robocasa/TurnOffSinkFaucet")
    parser.add_argument("--task_description", default="turn off the sink faucet")
    parser.add_argument("--env_split", default="target")
    parser.add_argument("--obj_registries", default="lightwheel")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--flip_images", action="store_true")
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=500)
    parser.add_argument("--output_dir", default="experiments/robocasa/tmp/alignment_debug")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_images, train_instruction, train_meta = _first_matching_train_episode(
        args.data_root_dir,
        args.dataset_name,
        args.task_description,
        args.max_episodes,
        args.frame_index,
    )
    eval_images, eval_instruction, eval_meta = _eval_reset_images(
        args.env_name,
        args.task_description,
        args.env_split,
        args.obj_registries,
        args.seed,
        args.flip_images,
    )
    metadata_path = _write_artifacts(
        Path(args.output_dir),
        train_images,
        eval_images,
        train_instruction,
        eval_instruction,
        {
            "data_root_dir": args.data_root_dir,
            "dataset_name": args.dataset_name,
            "env_name": args.env_name,
            "task_description_arg": args.task_description,
            "env_split": args.env_split,
            "obj_registries": args.obj_registries,
            "seed": args.seed,
            "flip_images": args.flip_images,
            "train_sample": train_meta,
            "eval_reset": eval_meta,
        },
    )
    print(metadata_path)


if __name__ == "__main__":
    main()
