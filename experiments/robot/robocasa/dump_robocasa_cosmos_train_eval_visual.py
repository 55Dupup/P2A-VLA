"""Dump RoboCasa-Cosmos training/eval visual alignment artifacts.

This compares the exact model-facing image order used by the current
RoboCasa-Cosmos training and eval setup:

    full_image     <- left camera
    wrist_image    <- right camera
    wrist_image_1  <- wrist camera

Eval images are vertically flipped when ``flip_images`` is true, matching the
current policy adapter path.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image, ImageDraw

from experiments.robot.robocasa.run_robocasa_cosmos_eval import (
    RoboCasaCosmosEvalConfig,
    _proprio_from_raw_obs,
    _task_description,
    create_robocasa_env,
)


MODEL_INPUT_SPECS = (
    {
        "model_key": "full_image",
        "train_key": "left_image",
        "eval_raw_key": "robot0_agentview_left_image",
        "camera": "left",
    },
    {
        "model_key": "wrist_image",
        "train_key": "right_image",
        "eval_raw_key": "robot0_agentview_right_image",
        "camera": "right",
    },
    {
        "model_key": "wrist_image_1",
        "train_key": "wrist_image",
        "eval_raw_key": "robot0_eye_in_hand_image",
        "camera": "wrist",
    },
)


DEFAULT_TASKS = (
    "PnPCounterToCab",
    "PnPCabToCounter",
    "PnPCounterToSink",
    "PnPSinkToCounter",
    "PnPCounterToMicrowave",
    "PnPMicrowaveToCounter",
    "PnPCounterToStove",
    "PnPStoveToCounter",
    "OpenSingleDoor",
    "CloseSingleDoor",
    "OpenDoubleDoor",
    "CloseDoubleDoor",
    "OpenDrawer",
    "CloseDrawer",
    "TurnOnStove",
    "TurnOffStove",
    "TurnOnSinkFaucet",
    "TurnOffSinkFaucet",
    "TurnSinkSpout",
    "CoffeeSetupMug",
    "CoffeeServeMug",
    "CoffeePressButton",
    "TurnOnMicrowave",
    "TurnOffMicrowave",
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


def _normalize_instruction(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _image_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > 3:
        array = array[-1]
    if array.ndim != 3:
        raise ValueError(f"Expected HWC image, got {array.shape}")
    return array.astype(np.uint8)


def _maybe_flip(image: np.ndarray, flip_images: bool) -> np.ndarray:
    image = _image_array(image)
    if not flip_images:
        return image
    return np.flipud(image).copy()


def _image_stats(image: np.ndarray) -> Dict[str, Any]:
    image = _image_array(image)
    return {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "min": int(image.min()),
        "max": int(image.max()),
        "mean": float(image.mean()),
        "std": float(image.std()),
        "channel_mean_rgb": [float(image[..., idx].mean()) for idx in range(3)],
    }


def _qwen_prompt(task: str) -> str:
    task = task.lower()
    return (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"What action should the robot take to {task}?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _eval_model_inputs(
    task_name: str,
    seed: int,
    episode_idx: int,
    warmup_steps: int,
    flip_images: bool,
) -> Dict[str, Any]:
    cfg = RoboCasaCosmosEvalConfig(task_name=task_name, flip_images=flip_images)
    env = create_robocasa_env(cfg, seed=seed, episode_idx=episode_idx)
    try:
        obs = env.reset()
        for _ in range(warmup_steps):
            obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape, dtype=np.float32))

        instruction = _task_description(env, cfg)
        images = {}
        raw_images = {}
        for spec in MODEL_INPUT_SPECS:
            raw = _image_array(obs[spec["eval_raw_key"]])
            raw_images[spec["model_key"]] = raw
            images[spec["model_key"]] = _maybe_flip(raw, flip_images)

        return {
            "task_name": task_name,
            "instruction": instruction,
            "images": images,
            "raw_images": raw_images,
            "state": _proprio_from_raw_obs(obs),
            "env_action_dim": int(env.action_spec[0].shape[-1]),
        }
    finally:
        env.close()


def _collect_eval_samples(
    task_names: Iterable[str],
    seed: int,
    warmup_steps: int,
    flip_images: bool,
) -> List[Dict[str, Any]]:
    samples = []
    for episode_idx, task_name in enumerate(task_names):
        samples.append(
            _eval_model_inputs(
                task_name=task_name,
                seed=seed,
                episode_idx=episode_idx,
                warmup_steps=warmup_steps,
                flip_images=flip_images,
            )
        )
    return samples


def _collect_train_samples(
    data_root_dir: str,
    dataset_name: str,
    target_instructions: Iterable[str],
    max_episodes: int,
    frame_index: int,
) -> Dict[str, Dict[str, Any]]:
    tf.config.set_visible_devices([], "GPU")
    targets = {_normalize_instruction(value) for value in target_instructions}
    found: Dict[str, Dict[str, Any]] = {}

    builder = tfds.builder(dataset_name, data_dir=data_root_dir)
    dataset = builder.as_dataset(split="train", shuffle_files=False)

    for episode_idx, episode in enumerate(dataset.take(max_episodes)):
        if len(found) == len(targets):
            break
        steps = list(episode["steps"].as_numpy_iterator())
        if not steps:
            continue
        instruction = _as_text(steps[0]["language_instruction"])
        instruction_key = _normalize_instruction(instruction)
        if instruction_key not in targets or instruction_key in found:
            continue

        selected_idx = min(max(frame_index, 0), len(steps) - 1)
        step = steps[selected_idx]
        observation = step["observation"]
        found[instruction_key] = {
            "instruction": instruction,
            "episode_index": int(episode_idx),
            "frame_index": int(selected_idx),
            "episode_length": int(len(steps)),
            "images": {
                spec["model_key"]: _image_array(observation[spec["train_key"]])
                for spec in MODEL_INPUT_SPECS
            },
            "state": np.asarray(observation["state"], dtype=np.float32),
            "action": np.asarray(step["action"], dtype=np.float32),
        }

    return found


def _load_hdf5_image_array(obs_group, raw_key: str) -> np.ndarray:
    if raw_key in obs_group:
        return obs_group[raw_key][:].astype(np.uint8)
    jpeg_key = f"{raw_key}_jpeg"
    if jpeg_key not in obs_group:
        raise KeyError(f"Missing source image key {raw_key} or {jpeg_key}")
    frames = []
    for item in obs_group[jpeg_key]:
        raw = item.tobytes() if isinstance(item, np.ndarray) else bytes(item)
        frames.append(np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8))
    return np.stack(frames, axis=0)


def _demo_language(demo_group) -> str:
    for key in ("task_description", "language_instruction", "lang"):
        if key in demo_group.attrs:
            return _as_text(demo_group.attrs[key])
    return ""


def _sorted_demo_keys(h5_file) -> List[str]:
    def key_fn(name: str):
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return name
    return sorted(h5_file["data"].keys(), key=key_fn)


def _collect_source_task_fallbacks(source_dir: str, task_names: Iterable[str], frame_index: int) -> Dict[str, Dict[str, Any]]:
    source_root = Path(source_dir)
    result: Dict[str, Dict[str, Any]] = {}
    for task_name in task_names:
        task_dirs = sorted(path for path in source_root.rglob(task_name) if path.is_dir() and path.name == task_name)
        if not task_dirs:
            continue
        hdf5_files = sorted(task_dirs[0].rglob("*.hdf5"))
        if not hdf5_files:
            continue
        hdf5_path = hdf5_files[0]
        with h5py.File(hdf5_path, "r") as h5_file:
            demo_key = _sorted_demo_keys(h5_file)[0]
            demo_group = h5_file[f"data/{demo_key}"]
            obs_group = demo_group["obs"]
            left = _load_hdf5_image_array(obs_group, "robot0_agentview_left_rgb")
            right = _load_hdf5_image_array(obs_group, "robot0_agentview_right_rgb")
            wrist = _load_hdf5_image_array(obs_group, "robot0_eye_in_hand_rgb")
            idx = min(max(frame_index, 0), len(left) - 1)
            actions = np.asarray(demo_group["actions"], dtype=np.float32)
            states = np.asarray(demo_group["robot_states"], dtype=np.float32)
            result[task_name] = {
                "instruction": _demo_language(demo_group),
                "episode_index": None,
                "frame_index": int(idx),
                "episode_length": int(len(left)),
                "source_hdf5": str(hdf5_path),
                "source_demo_key": demo_key,
                "images": {
                    "full_image": _image_array(left[idx]),
                    "wrist_image": _image_array(right[idx]),
                    "wrist_image_1": _image_array(wrist[idx]),
                },
                "state": states[idx].astype(np.float32),
                "action": actions[idx, :7].astype(np.float32),
            }
    return result


def _save_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(_image_array(image)).save(path)


def _draw_task_sheet(
    path: Path,
    task_name: str,
    train: Optional[Dict[str, Any]],
    eval_sample: Dict[str, Any],
) -> None:
    tile = 224
    label_h = 86
    rows = ["train", "eval_model_input"]
    cols = len(MODEL_INPUT_SPECS)
    sheet = Image.new("RGB", (cols * tile, len(rows) * (tile + label_h)), "white")
    draw = ImageDraw.Draw(sheet)

    for row_idx, row_name in enumerate(rows):
        source = train if row_name == "train" else eval_sample
        y0 = row_idx * (tile + label_h)
        for col_idx, spec in enumerate(MODEL_INPUT_SPECS):
            x0 = col_idx * tile
            if source is None:
                image = Image.new("RGB", (tile, tile), (230, 230, 230))
            else:
                image = Image.fromarray(_image_array(source["images"][spec["model_key"]])).resize((tile, tile))
            sheet.paste(image, (x0, y0 + label_h))
            draw.text((x0 + 5, y0 + 4), task_name[:34], fill=(0, 0, 0))
            draw.text((x0 + 5, y0 + 22), f"{row_name}: {spec['model_key']}", fill=(0, 0, 0))
            draw.text((x0 + 5, y0 + 40), f"camera={spec['camera']}", fill=(0, 0, 0))
            if source is not None:
                draw.text((x0 + 5, y0 + 58), source["instruction"][:34], fill=(0, 0, 0))

    sheet.save(path)


def _draw_overview(path: Path, rows: List[Tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]]) -> None:
    tile = 112
    label_w = 260
    label_h = 24
    row_h = tile * 2 + label_h
    cols = len(MODEL_INPUT_SPECS)
    sheet = Image.new("RGB", (label_w + cols * tile, max(1, len(rows)) * row_h), "white")
    draw = ImageDraw.Draw(sheet)

    for row_idx, (task_name, train, eval_sample) in enumerate(rows):
        y0 = row_idx * row_h
        train_ok = train is not None
        draw.text((6, y0 + 6), task_name, fill=(0, 0, 0))
        draw.text((6, y0 + 28), f"train_match={train_ok}", fill=(0, 0, 0))
        draw.text((6, y0 + 50), eval_sample["instruction"][:38], fill=(0, 0, 0))
        for col_idx, spec in enumerate(MODEL_INPUT_SPECS):
            x0 = label_w + col_idx * tile
            draw.text((x0 + 4, y0 + 4), spec["model_key"], fill=(0, 0, 0))
            if train is not None:
                train_img = Image.fromarray(_image_array(train["images"][spec["model_key"]])).resize((tile, tile))
                sheet.paste(train_img, (x0, y0 + label_h))
            eval_img = Image.fromarray(_image_array(eval_sample["images"][spec["model_key"]])).resize((tile, tile))
            sheet.paste(eval_img, (x0, y0 + label_h + tile))

    sheet.save(path)


def _write_artifacts(
    output_dir: Path,
    eval_samples: List[Dict[str, Any]],
    train_by_instruction: Dict[str, Dict[str, Any]],
    train_by_task_fallback: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    task_rows = []
    metadata: Dict[str, Any] = {
        "data_root_dir": args.data_root_dir,
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "warmup_steps": args.warmup_steps,
        "frame_index": args.frame_index,
        "flip_eval_images": args.flip_images,
        "model_input_order": list(MODEL_INPUT_SPECS),
        "training_setup": {
            "num_images_in_input": 3,
            "use_proprio": True,
            "proprio_dim": 9,
            "action_dim": 7,
            "action_chunk": 32,
        },
        "eval_setup": {
            "num_images_in_input": 3,
            "use_proprio": True,
            "proprio_dim": 9,
            "model_action_dim": 7,
            "env_action_dim": 12,
            "fixed_mobile_base_action": [0.0, 0.0, 0.0, 0.0, -1.0],
        },
        "tasks": {},
    }

    for eval_sample in eval_samples:
        instruction_key = _normalize_instruction(eval_sample["instruction"])
        task_name = eval_sample["task_name"]
        train = train_by_instruction.get(instruction_key)
        train_match_source = "exact_instruction"
        if train is None:
            train = train_by_task_fallback.get(task_name)
            train_match_source = "source_task_fallback" if train is not None else "missing"
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        for spec in MODEL_INPUT_SPECS:
            model_key = spec["model_key"]
            if train is not None:
                _save_image(task_dir / f"train_{model_key}__{spec['camera']}.png", train["images"][model_key])
            _save_image(task_dir / f"eval_model_{model_key}__{spec['camera']}.png", eval_sample["images"][model_key])
            _save_image(task_dir / f"eval_raw_{model_key}__{spec['camera']}.png", eval_sample["raw_images"][model_key])

        _draw_task_sheet(task_dir / "train_vs_eval_model_input.png", task_name, train, eval_sample)
        task_rows.append((task_name, train, eval_sample))

        metadata["tasks"][task_name] = {
            "eval_instruction": eval_sample["instruction"],
            "train_instruction": train["instruction"] if train else None,
            "instruction_equal": bool(train and _qwen_prompt(train["instruction"]) == _qwen_prompt(eval_sample["instruction"])),
            "train_match_source": train_match_source,
            "train_episode": {
                "episode_index": train["episode_index"],
                "frame_index": train["frame_index"],
                "episode_length": train["episode_length"],
                "source_hdf5": train.get("source_hdf5"),
                "source_demo_key": train.get("source_demo_key"),
            }
            if train
            else None,
            "train_action_7d": train["action"].astype(float).tolist() if train else None,
            "train_state_shape": list(train["state"].shape) if train else None,
            "eval_state_shape": list(eval_sample["state"].shape),
            "eval_env_action_dim": eval_sample["env_action_dim"],
            "train_image_stats": {k: _image_stats(v) for k, v in train["images"].items()} if train else None,
            "eval_image_stats": {k: _image_stats(v) for k, v in eval_sample["images"].items()},
        }

    _draw_overview(output_dir / "all_tasks_train_eval_overview.png", task_rows)
    metadata_path = output_dir / "alignment_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root_dir", default="/path/to/datasets/robocasa_cosmos_success_only_rlds")
    parser.add_argument("--source_dir", default="/path/to/datasets/RoboCasa-Cosmos-Policy/success_only")
    parser.add_argument("--dataset_name", default="robocasa_cosmos_success_only")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=2000)
    parser.add_argument("--flip_images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output_dir",
        default="/path/to/P2A-VLA/experiments/robocasa/tmp/robocasa_cosmos_train_eval_visual",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_names = [item.strip() for item in args.tasks.split(",") if item.strip()]
    eval_samples = _collect_eval_samples(
        task_names=task_names,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        flip_images=args.flip_images,
    )
    train_by_instruction = _collect_train_samples(
        data_root_dir=args.data_root_dir,
        dataset_name=args.dataset_name,
        target_instructions=[sample["instruction"] for sample in eval_samples],
        max_episodes=args.max_episodes,
        frame_index=args.frame_index,
    )
    train_by_task_fallback = _collect_source_task_fallbacks(
        source_dir=args.source_dir,
        task_names=task_names,
        frame_index=args.frame_index,
    )
    metadata_path = _write_artifacts(
        output_dir=Path(args.output_dir),
        eval_samples=eval_samples,
        train_by_instruction=train_by_instruction,
        train_by_task_fallback=train_by_task_fallback,
        args=args,
    )
    print(metadata_path)


if __name__ == "__main__":
    main()
