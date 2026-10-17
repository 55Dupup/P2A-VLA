"""Render the post-geometry VLA inputs for the exported RoboCasa Cosmos gallery.

The source gallery preserves the RLDS training images and the flipped evaluation
renders. This utility applies the same 0.9 crop geometry used by training and
evaluation so their model-facing images can be inspected at the same stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image, ImageDraw

from experiments.robot.openvla_utils import center_crop_image


VIEW_SPECS = (
    ("full_image", "left"),
    ("wrist_image", "right"),
    ("wrist_image_1", "wrist"),
)


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _crop(image: np.ndarray) -> np.ndarray:
    """Use the exact eval center-crop implementation for a visual reference."""
    return np.asarray(center_crop_image(image), dtype=np.uint8)


def _save(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image).save(path)


def _sheet(path: Path, task_name: str, train_images: Dict[str, np.ndarray], eval_images: Dict[str, np.ndarray]) -> None:
    tile_size, label_height, title_height = 224, 40, 28
    width = tile_size * len(VIEW_SPECS)
    height = title_height + 2 * (label_height + tile_size)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), task_name, fill="black")

    rows = (
        ("train reference: 0.9 center crop", train_images),
        ("eval: flip vertical, then 0.9 center crop", eval_images),
    )
    for row_idx, (row_label, images) in enumerate(rows):
        y = title_height + row_idx * (label_height + tile_size)
        for col_idx, (model_key, camera_name) in enumerate(VIEW_SPECS):
            x = col_idx * tile_size
            image = Image.fromarray(images[model_key]).resize((tile_size, tile_size), Image.Resampling.BICUBIC)
            canvas.paste(image, (x, y + label_height))
            draw.text((x + 4, y + 3), row_label if col_idx == 0 else camera_name, fill="black")

    canvas.save(path)


def _overview(path: Path, task_images: Dict[str, tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]) -> None:
    tile_size, label_height, task_height = 160, 22, 24
    width = tile_size * len(VIEW_SPECS)
    task_row_height = task_height + 2 * (label_height + tile_size)
    canvas = Image.new("RGB", (width, task_row_height * len(task_images)), "white")
    draw = ImageDraw.Draw(canvas)

    for task_idx, (task_name, (train_images, eval_images)) in enumerate(sorted(task_images.items())):
        y0 = task_idx * task_row_height
        draw.text((4, y0 + 4), task_name, fill="black")
        for row_idx, (row_label, images) in enumerate(
            (("train 0.9 center reference", train_images), ("eval flip + 0.9 center", eval_images))
        ):
            y = y0 + task_height + row_idx * (label_height + tile_size)
            for col_idx, (model_key, camera_name) in enumerate(VIEW_SPECS):
                x = col_idx * tile_size
                image = Image.fromarray(images[model_key]).resize((tile_size, tile_size), Image.Resampling.BICUBIC)
                canvas.paste(image, (x, y + label_height))
                draw.text((x + 3, y + 3), row_label if col_idx == 0 else camera_name, fill="black")

    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=Path("experiments/robocasa/tmp/robocasa_cosmos_all24_train_eval_visuals"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("experiments/robocasa/tmp/robocasa_cosmos_all24_final_vla_inputs"),
    )
    args = parser.parse_args()

    source_metadata_path = args.source_dir / "alignment_metadata.json"
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    task_images: Dict[str, tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = {}
    output_tasks: Dict[str, dict] = {}

    for task_dir in sorted(path for path in args.source_dir.iterdir() if path.is_dir()):
        train_images: Dict[str, np.ndarray] = {}
        eval_images: Dict[str, np.ndarray] = {}
        task_output_dir = args.output_dir / task_dir.name
        task_output_dir.mkdir(parents=True, exist_ok=True)

        for model_key, camera_name in VIEW_SPECS:
            train_raw = _load(task_dir / f"train_{model_key}__{camera_name}.png")
            # Training uses a random 0.9 crop. A center crop is a deterministic reference with identical scale.
            train_reference = _crop(train_raw)
            eval_flipped = _load(task_dir / f"eval_model_{model_key}__{camera_name}.png")
            eval_final = _crop(eval_flipped)

            train_images[model_key] = train_reference
            eval_images[model_key] = eval_final
            _save(task_output_dir / f"train_model_input_reference_{model_key}__{camera_name}.png", train_reference)
            _save(task_output_dir / f"eval_final_model_input_{model_key}__{camera_name}.png", eval_final)

        _sheet(task_output_dir / "train_eval_final_input_geometry.png", task_dir.name, train_images, eval_images)
        task_images[task_dir.name] = (train_images, eval_images)
        source_task = source_metadata["tasks"][task_dir.name]
        output_tasks[task_dir.name] = {
            "eval_instruction": source_task["eval_instruction"],
            "train_instruction": source_task["train_instruction"],
            "instruction_equal": source_task["instruction_equal"],
            "train_match_source": source_task["train_match_source"],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _overview(args.output_dir / "all_tasks_final_model_input_overview.png", task_images)
    output_metadata = {
        "source_gallery": str(args.source_dir),
        "num_tasks": len(task_images),
        "view_order": ["left", "right", "wrist"],
        "training_geometry": {
            "storage": "Converted RLDS images are 224x224; source HDF5 files are demo_gentex_im128_randcams.",
            "actual_augmentation": "random_resized_crop(scale=[0.9, 0.9], ratio=[1.0, 1.0]) plus color jitter.",
            "visualized_reference": "Deterministic center crop at scale 0.9; crop scale matches training, crop offset and color jitter do not.",
        },
        "eval_geometry": {
            "raw_render": "224x224 RoboCasa render.",
            "actual_preprocessing": "vertical flip followed by deterministic center crop at scale 0.9.",
        },
        "shared_processor": "Both training and evaluation invoke processor.image_processor.apply_transform after the geometry above.",
        "tasks": output_tasks,
    }
    (args.output_dir / "alignment_metadata.json").write_text(
        json.dumps(output_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(task_images)} task visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
