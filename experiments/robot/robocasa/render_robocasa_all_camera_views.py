import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from experiments.robot.robocasa.dump_robocasa_cosmos_train_eval_visual import (
    _demo_language,
    _load_hdf5_image_array,
    _sorted_demo_keys,
)
from experiments.robot.robocasa.run_robocasa_cosmos_eval import (
    RoboCasaCosmosEvalConfig,
    create_robocasa_env,
)


USED_CAMERA_NAMES = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)

TRAINING_CAMERA_KEYS = (
    ("train_left", "robot0_agentview_left_rgb"),
    ("train_right", "robot0_agentview_right_rgb"),
    ("train_wrist", "robot0_eye_in_hand_rgb"),
)

VIEW_MATCHES = {
    "robot0_agentview_left": "train_left (exact camera family; training pose is randomized)",
    "robot0_agentview_right": "train_right (exact camera family; training pose is randomized)",
    "robot0_eye_in_hand": "train_wrist (exact wrist camera family)",
    "robot0_agentview_center": "no direct training view; central third-person alternative",
    "robot0_frontview": "no direct training view; centered frontal alternative",
    "robot0_robotview": "no direct training view; rear/robot-centric alternative",
}


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def _save_sheet(path: Path, rows, tile_size: int = 224, label_height: int = 54) -> None:
    columns = max(len(row) for row in rows)
    sheet = Image.new("RGB", (columns * tile_size, len(rows) * (tile_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(rows):
        y = row_index * (tile_size + label_height)
        for column_index, (label, image) in enumerate(row):
            x = column_index * tile_size
            sheet.paste(Image.fromarray(_resize(image, tile_size)), (x, y + label_height))
            draw.multiline_text((x + 5, y + 4), label, fill="black", spacing=2)
    sheet.save(path)


def _load_training_views(hdf5_path: Path, frame_index: int):
    with h5py.File(hdf5_path, "r") as h5_file:
        demo_key = _sorted_demo_keys(h5_file)[0]
        demo_group = h5_file[f"data/{demo_key}"]
        obs_group = demo_group["obs"]
        arrays = {
            label: _load_hdf5_image_array(obs_group, key)
            for label, key in TRAINING_CAMERA_KEYS
        }
        episode_length = len(next(iter(arrays.values())))
        index = min(max(frame_index, 0), episode_length - 1)
        return (
            {label: images[index] for label, images in arrays.items()},
            {
                "source_hdf5": str(hdf5_path),
                "demo_key": demo_key,
                "frame_index": index,
                "episode_length": episode_length,
                "language": _demo_language(demo_group),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", default="PnPCounterToCab")
    parser.add_argument("--training-hdf5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--training-frame-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "eval_raw_observation_orientation"
    model_dir = args.output_dir / "eval_model_input_orientation"
    train_dir = args.output_dir / "training_views"
    for directory in (raw_dir, model_dir, train_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cfg = RoboCasaCosmosEvalConfig(
        task_name=args.task_name,
        use_proprio=False,
        env_img_res=args.resolution,
    )
    env = create_robocasa_env(cfg, seed=args.seed, episode_idx=0)
    try:
        obs = env.reset()
        for _ in range(10):
            obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape, dtype=np.float32))

        camera_rows = []
        metadata_cameras = []
        for camera_id in range(env.sim.model.ncam):
            camera_name = env.sim.model.camera_id2name(camera_id)
            raw_image = np.asarray(
                env.sim.render(
                    camera_name=camera_name,
                    width=args.resolution,
                    height=args.resolution,
                ),
                dtype=np.uint8,
            )
            model_image = np.flipud(raw_image).copy()
            Image.fromarray(raw_image).save(raw_dir / f"{camera_id:02d}_{camera_name}.png")
            Image.fromarray(model_image).save(model_dir / f"{camera_id:02d}_{camera_name}.png")
            camera_rows.append(
                (
                    f"{camera_id}: {camera_name}\n"
                    f"{'USED' if camera_name in USED_CAMERA_NAMES else 'not used'}",
                    model_image,
                )
            )
            metadata_cameras.append(
                {
                    "camera_id": camera_id,
                    "camera_name": camera_name,
                    "used_by_current_eval": camera_name in USED_CAMERA_NAMES,
                    "closest_training_view": VIEW_MATCHES.get(camera_name, "unknown"),
                    "world_position": env.sim.data.cam_xpos[camera_id].astype(float).tolist(),
                    "world_rotation_matrix": env.sim.data.cam_xmat[camera_id].reshape(3, 3).astype(float).tolist(),
                    "vertical_fov_degrees": float(env.sim.model.cam_fovy[camera_id]),
                }
            )

        task_description = env.get_ep_meta().get("lang", "")
    finally:
        env.close()

    training_views, training_metadata = _load_training_views(
        args.training_hdf5, args.training_frame_index
    )
    for label, image in training_views.items():
        Image.fromarray(image).save(train_dir / f"{label}_raw128.png")
        Image.fromarray(_resize(image, args.resolution)).save(
            train_dir / f"{label}_resized{args.resolution}.png"
        )

    used_eval = [item for item in camera_rows if "USED" in item[0]]
    unused_eval = [item for item in camera_rows if "not used" in item[0]]
    training_row = [
        (f"{label}\nsource 128x128", training_views[label])
        for label, _ in TRAINING_CAMERA_KEYS
    ]
    _save_sheet(
        args.output_dir / "all_eval_views_model_input_orientation.png",
        [camera_rows[:3], camera_rows[3:]],
        tile_size=args.resolution,
    )
    _save_sheet(
        args.output_dir / "train_vs_current_and_extra_eval_views.png",
        [training_row, used_eval, unused_eval],
        tile_size=args.resolution,
    )

    metadata = {
        "task_name": args.task_name,
        "eval_task_description": task_description,
        "seed": args.seed,
        "resolution": args.resolution,
        "flip_applied_for_model_input": True,
        "current_eval_camera_order": list(USED_CAMERA_NAMES),
        "training_camera_order": [label for label, _ in TRAINING_CAMERA_KEYS],
        "training": training_metadata,
        "cameras": metadata_cameras,
    }
    (args.output_dir / "camera_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
