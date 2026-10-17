#!/usr/bin/env python
import argparse
import io
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from PIL import Image


DATASET_NAME = "robocasa_cosmos_success_only"
DEFAULT_SOURCE_DIR = Path("/path/to/datasets/RoboCasa-Cosmos-Policy/success_only")
DEFAULT_OUTPUT_DIR = Path("/path/to/datasets/robocasa_cosmos_success_only_rlds")
COSMOS_MOBILE_BASE_ACTION = [0.0, 0.0, 0.0, 0.0, -1.0]
LEFT_IMAGE_KEY = "robot0_agentview_left_rgb"
RIGHT_IMAGE_KEY = "robot0_agentview_right_rgb"
WRIST_IMAGE_KEY = "robot0_eye_in_hand_rgb"
STATE_KEY = "robot_states"
ACTION_KEY = "actions"


@dataclass(frozen=True)
class CosmosEpisodeArrays:
    left_images: np.ndarray
    right_images: np.ndarray
    wrist_images: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    language: str


def extract_cosmos_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] not in (7, 12):
        raise ValueError(f"RoboCasa Cosmos action must be 7D or 12D, got shape {action.shape}")
    return action[..., :7].astype(np.float32)


def validate_episode_arrays(episode: CosmosEpisodeArrays) -> None:
    lengths = {
        "left_images": len(episode.left_images),
        "right_images": len(episode.right_images),
        "wrist_images": len(episode.wrist_images),
        "states": len(episode.states),
        "actions": len(episode.actions),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"RoboCasa Cosmos episode arrays must have aligned lengths, got {lengths}")
    for name, images in (
        ("left_images", episode.left_images),
        ("right_images", episode.right_images),
        ("wrist_images", episode.wrist_images),
    ):
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"{name} must be shaped (T, H, W, 3), got {images.shape}")
        if images.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8, got {images.dtype}")
    if episode.states.ndim != 2 or episode.states.shape[-1] != 9:
        raise ValueError(f"states must be shaped (T, 9), got {episode.states.shape}")
    if episode.actions.ndim != 2 or episode.actions.shape[-1] not in (7, 12):
        raise ValueError(f"actions must be shaped (T, 7) or (T, 12), got {episode.actions.shape}")
    if not episode.language:
        raise ValueError("language instruction must be non-empty")


def _decode_jpeg_dataset(dataset) -> np.ndarray:
    frames = []
    for item in dataset:
        raw = item.tobytes() if isinstance(item, np.ndarray) else bytes(item)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        frames.append(np.asarray(image, dtype=np.uint8))
    return np.stack(frames, axis=0)


def _load_image_array(obs_group, raw_key: str) -> np.ndarray:
    if raw_key in obs_group:
        return obs_group[raw_key][:].astype(np.uint8)
    jpeg_key = f"{raw_key}_jpeg"
    if jpeg_key in obs_group:
        return _decode_jpeg_dataset(obs_group[jpeg_key])
    raise KeyError(f"Missing image dataset `{raw_key}` or `{jpeg_key}`")


def _decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _language_from_demo_group(demo_group) -> str:
    for key in ("task_description", "language_instruction", "lang"):
        if key in demo_group.attrs:
            return _decode_attr(demo_group.attrs[key])
    raise KeyError(f"Missing task description attrs in demo group `{demo_group.name}`")


def _sorted_demo_keys(h5_file) -> list[str]:
    def sort_key(name: str):
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return name

    return sorted(h5_file["data"].keys(), key=sort_key)


def iter_hdf5_files(source_dir: Path) -> list[Path]:
    return sorted(path for path in source_dir.rglob("*.hdf5") if path.is_file())


def load_episode(h5_file, demo_key: str) -> CosmosEpisodeArrays:
    demo_group = h5_file[f"data/{demo_key}"]
    obs_group = demo_group["obs"]
    episode = CosmosEpisodeArrays(
        left_images=_load_image_array(obs_group, LEFT_IMAGE_KEY),
        right_images=_load_image_array(obs_group, RIGHT_IMAGE_KEY),
        wrist_images=_load_image_array(obs_group, WRIST_IMAGE_KEY),
        states=demo_group[STATE_KEY][:].astype(np.float32),
        actions=demo_group[ACTION_KEY][:].astype(np.float32),
        language=_language_from_demo_group(demo_group),
    )
    validate_episode_arrays(episode)
    return episode


def build_steps(episode: CosmosEpisodeArrays) -> Iterable[dict]:
    actions = extract_cosmos_action(episode.actions)
    num_steps = actions.shape[0]
    for step_idx in range(num_steps):
        yield {
            "observation": {
                "left_image": episode.left_images[step_idx],
                "right_image": episode.right_images[step_idx],
                "wrist_image": episode.wrist_images[step_idx],
                "state": episode.states[step_idx].astype(np.float32),
            },
            "action": actions[step_idx].astype(np.float32),
            "language_instruction": episode.language,
            "is_first": step_idx == 0,
            "is_last": step_idx == num_steps - 1,
            "is_terminal": step_idx == num_steps - 1,
        }


def make_builder(*, source_dir: Path, output_dir: Path, max_episodes: int | None, image_size: int):
    import tensorflow_datasets as tfds

    class RobocasaCosmosSuccessOnly(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {"1.0.0": "RoboCasa Cosmos success_only converted to 7D-action RLDS."}

        def __init__(self, **kwargs):
            self.source_dir = Path(source_dir)
            self.max_episodes = max_episodes
            self.image_size = image_size
            super().__init__(**kwargs)

        def _info(self):
            return tfds.core.DatasetInfo(
                builder=self,
                description="RoboCasa Cosmos success_only converted for VLA-Adapter with 7D actions.",
                features=tfds.features.FeaturesDict(
                    {
                        "steps": tfds.features.Dataset(
                            {
                                "observation": tfds.features.FeaturesDict(
                                    {
                                        "left_image": tfds.features.Image(
                                            shape=(self.image_size, self.image_size, 3),
                                            dtype=np.uint8,
                                            encoding_format="jpeg",
                                        ),
                                        "right_image": tfds.features.Image(
                                            shape=(self.image_size, self.image_size, 3),
                                            dtype=np.uint8,
                                            encoding_format="jpeg",
                                        ),
                                        "wrist_image": tfds.features.Image(
                                            shape=(self.image_size, self.image_size, 3),
                                            dtype=np.uint8,
                                            encoding_format="jpeg",
                                        ),
                                        "state": tfds.features.Tensor(shape=(9,), dtype=np.float32),
                                    }
                                ),
                                "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                                "language_instruction": tfds.features.Text(),
                                "is_first": tfds.features.Tensor(shape=(), dtype=np.bool_),
                                "is_last": tfds.features.Tensor(shape=(), dtype=np.bool_),
                                "is_terminal": tfds.features.Tensor(shape=(), dtype=np.bool_),
                            }
                        )
                    }
                ),
            )

        def _split_generators(self, dl_manager):
            del dl_manager
            return {"train": self._generate_examples()}

        def _generate_examples(self):
            episode_count = 0
            hdf5_files = iter_hdf5_files(self.source_dir)
            if not hdf5_files:
                raise FileNotFoundError(f"No HDF5 files found under {self.source_dir}")

            for hdf5_path in hdf5_files:
                with h5py.File(hdf5_path, "r") as h5_file:
                    for demo_key in _sorted_demo_keys(h5_file):
                        if self.max_episodes is not None and episode_count >= self.max_episodes:
                            return
                        episode = load_episode(h5_file, demo_key)
                        rel_path = hdf5_path.relative_to(self.source_dir)
                        example_key = f"{rel_path.as_posix()}::{demo_key}"
                        yield example_key, {"steps": list(build_steps(episode))}
                        episode_count += 1

    return RobocasaCosmosSuccessOnly(data_dir=str(output_dir))


def write_manifest(*, output_dir: Path, source_dir: Path, max_episodes: int | None, image_size: int) -> None:
    manifest = {
        "dataset_name": DATASET_NAME,
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "max_episodes": max_episodes,
        "image_size": image_size,
        "source_action_dim": 12,
        "action_dim": 7,
        "proprio_dim": 9,
        "image_order": ["left_image", "right_image", "wrist_image"],
        "source_image_keys": [LEFT_IMAGE_KEY, RIGHT_IMAGE_KEY, WRIST_IMAGE_KEY],
        "fixed_mobile_base_action_for_eval": COSMOS_MOBILE_BASE_ACTION,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "robocasa_cosmos_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert NVIDIA RoboCasa-Cosmos-Policy HDF5 data to VLA-Adapter RLDS.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default=DATASET_NAME, choices=[DATASET_NAME])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = make_builder(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
        image_size=args.image_size,
    )
    if args.overwrite and Path(builder.data_dir).exists():
        shutil.rmtree(builder.data_dir)
    builder.download_and_prepare()
    write_manifest(
        output_dir=args.output_dir,
        source_dir=args.source_dir,
        max_episodes=args.max_episodes,
        image_size=args.image_size,
    )
    print(f"Converted {args.source_dir} to RLDS dataset {DATASET_NAME} at {builder.data_dir}")


if __name__ == "__main__":
    main()
