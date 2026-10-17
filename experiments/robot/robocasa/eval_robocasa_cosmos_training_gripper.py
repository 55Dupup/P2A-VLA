import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from experiments.robot.robocasa.run_robocasa_cosmos_eval import (
    RoboCasaCosmosEvalConfig,
    _build_policy,
)


IMAGE_KEYS = (
    ("left", "robot0_agentview_left_rgb"),
    ("right", "robot0_agentview_right_rgb"),
    ("wrist", "robot0_eye_in_hand_rgb"),
)


def _demo_key_sort(key: str):
    try:
        return int(key.rsplit("_", 1)[-1])
    except ValueError:
        return key


def _decode_text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _load_image(group: h5py.Group, key: str) -> np.ndarray:
    if key not in group:
        raise KeyError(f"Missing training image key: {key}")
    return np.asarray(group[key], dtype=np.uint8)


def _transition_indices(gripper: np.ndarray) -> tuple[int, int]:
    close = np.flatnonzero((gripper[:-1] < 0) & (gripper[1:] > 0)) + 1
    release = np.flatnonzero((gripper[:-1] > 0) & (gripper[1:] < 0)) + 1
    if close.size == 0 or release.size == 0:
        raise ValueError("Selected demo must contain both open-to-close and close-to-open transitions")
    return int(close[0]), int(release[0])


def _sample_frames(length: int, close_idx: int, release_idx: int) -> list[tuple[str, int]]:
    candidates = [
        ("open_early", 0),
        ("pre_grasp_8", close_idx - 8),
        ("pre_grasp_1", close_idx - 1),
        ("grasp_switch", close_idx),
        ("post_grasp_1", close_idx + 1),
        ("holding_8", close_idx + 8),
        ("pre_release_8", release_idx - 8),
        ("pre_release_1", release_idx - 1),
        ("release_switch", release_idx),
        ("post_release_1", release_idx + 1),
        ("open_final", length - 1),
    ]
    result = []
    used = set()
    for label, index in candidates:
        index = min(max(index, 0), length - 1)
        if index not in used:
            result.append((label, index))
            used.add(index)
    return result


def _action_chunk(actions: np.ndarray, frame_idx: int, chunk_size: int) -> np.ndarray:
    chunk = actions[frame_idx : frame_idx + chunk_size]
    if len(chunk) < chunk_size:
        chunk = np.concatenate([chunk, np.repeat(chunk[-1:], chunk_size - len(chunk), axis=0)], axis=0)
    return chunk


def _observation(images: dict[str, np.ndarray], instruction: str, frame_idx: int) -> dict:
    # Regenerated Cosmos HDF5 images are already upright, so this diagnostic uses flip_images=False.
    return {
        "video.left_image": images["left"][frame_idx][None],
        "video.right_image": images["right"][frame_idx][None],
        "video.wrist_image": images["wrist"][frame_idx][None],
        "annotation.human.coarse_action": np.asarray([instruction]),
    }


def _save_contact_sheet(rows: list[dict], images: dict[str, np.ndarray], path: Path) -> None:
    tile_size = 224
    label_height = 86
    sheet = Image.new("RGB", (tile_size * 3, (tile_size + label_height) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_idx, row in enumerate(rows):
        y = row_idx * (tile_size + label_height)
        gt = " ".join(f"{value:+.0f}" for value in row["ground_truth_gripper_chunk"])
        pred = " ".join(f"{value:+.2f}" for value in row["predicted_gripper_chunk"])
        draw.text((4, y + 4), f"{row['label']} frame={row['frame_idx']} current_gt={row['ground_truth_current_gripper']:+.0f}", fill="black")
        draw.text((4, y + 24), f"gt8:   {gt}", fill="black")
        draw.text((4, y + 44), f"pred8: {pred}", fill="black")
        draw.text((4, y + 64), f"pred_sign_match={row['chunk_sign_accuracy']:.3f}", fill="black")
        for col, (view_name, _) in enumerate(IMAGE_KEYS):
            image = Image.fromarray(images[view_name][row["frame_idx"]]).resize(
                (tile_size, tile_size), Image.Resampling.BILINEAR
            )
            sheet.paste(image, (col * tile_size, y + label_height))
            draw.text((col * tile_size + 4, y + label_height + 4), view_name, fill="white", stroke_width=2, stroke_fill="black")
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate gripper predictions on exact RoboCasa Cosmos training frames")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-hdf5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--demo-key", default="demo_1")
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.training_hdf5, "r") as handle:
        demo_keys = sorted(handle["data"].keys(), key=_demo_key_sort)
        if args.demo_key not in demo_keys:
            raise KeyError(f"Unknown demo key {args.demo_key}; available examples: {demo_keys[:10]}")
        demo = handle[f"data/{args.demo_key}"]
        observation_group = demo["obs"]
        images = {name: _load_image(observation_group, key) for name, key in IMAGE_KEYS}
        actions = np.asarray(demo["actions"], dtype=np.float32)[..., :7]
        instruction = _decode_text(demo.attrs["task_description"])

    close_idx, release_idx = _transition_indices(actions[:, 6])
    samples = _sample_frames(len(actions), close_idx, release_idx)

    cfg = RoboCasaCosmosEvalConfig(
        pretrained_checkpoint=str(args.checkpoint),
        dataset_name="robocasa_cosmos_success_only",
        unnorm_key="robocasa_cosmos_success_only",
        use_film=False,
        use_minivlm=True,
        use_proprio=False,
        proprio_dim=9,
        num_images_in_input=3,
        center_crop=True,
        flip_images=False,
        use_pro_version=True,
        use_gate=True,
        use_latent_outputs=True,
        n_action_steps=args.chunk_size,
        image_obs_key="video.left_image",
        extra_image_obs_keys="video.right_image,video.wrist_image",
        state_obs_keys="state",
        instruction_obs_key="annotation.human.coarse_action",
    )
    policy = _build_policy(cfg)

    rows = []
    predicted_chunks = []
    ground_truth_chunks = []
    for label, frame_idx in samples:
        predicted = np.asarray(policy.step(_observation(images, instruction, frame_idx))["actions"][0], dtype=np.float32)
        ground_truth = _action_chunk(actions, frame_idx, args.chunk_size)
        predicted = predicted[: args.chunk_size]
        predicted_gripper = predicted[:, 6]
        ground_truth_gripper = ground_truth[:, 6]
        sign_match = np.sign(predicted_gripper) == np.sign(ground_truth_gripper)
        rows.append(
            {
                "label": label,
                "frame_idx": frame_idx,
                "ground_truth_current_gripper": float(ground_truth_gripper[0]),
                "predicted_current_gripper": float(predicted_gripper[0]),
                "ground_truth_gripper_chunk": ground_truth_gripper.tolist(),
                "predicted_gripper_chunk": predicted_gripper.tolist(),
                "chunk_sign_accuracy": float(np.mean(sign_match)),
            }
        )
        predicted_chunks.append(predicted)
        ground_truth_chunks.append(ground_truth)

    all_predicted_gripper = np.concatenate([chunk[:, 6] for chunk in predicted_chunks])
    all_ground_truth_gripper = np.concatenate([chunk[:, 6] for chunk in ground_truth_chunks])
    prediction_sign = all_predicted_gripper > 0
    target_sign = all_ground_truth_gripper > 0
    summary = {
        "checkpoint": str(args.checkpoint),
        "training_hdf5": str(args.training_hdf5),
        "demo_key": args.demo_key,
        "instruction": instruction,
        "source_image_shape": list(images["left"].shape[1:]),
        "model_camera_order": [name for name, _ in IMAGE_KEYS],
        "flip_images": False,
        "close_transition_frame": close_idx,
        "release_transition_frame": release_idx,
        "num_sampled_frames": len(rows),
        "chunk_size": args.chunk_size,
        "overall_chunk_sign_accuracy": float(np.mean(prediction_sign == target_sign)),
        "ground_truth_close_count": int(target_sign.sum()),
        "predicted_close_count": int(prediction_sign.sum()),
        "close_recall": float(np.mean(prediction_sign[target_sign])) if np.any(target_sign) else None,
        "open_recall": float(np.mean(~prediction_sign[~target_sign])) if np.any(~target_sign) else None,
        "samples": rows,
    }

    (args.output_dir / "gripper_training_frame_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "gripper_training_frame_actions.npz",
        sampled_frame_indices=np.asarray([frame_idx for _, frame_idx in samples]),
        predicted_action_chunks=np.stack(predicted_chunks),
        ground_truth_action_chunks=np.stack(ground_truth_chunks),
    )
    _save_contact_sheet(rows, images, args.output_dir / "gripper_training_frame_visuals.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
