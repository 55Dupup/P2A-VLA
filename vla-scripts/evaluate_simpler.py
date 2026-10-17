#!/usr/bin/env python
"""Run SimplerEnv evaluation with a VLA-Adapter checkpoint."""

import argparse
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sapien.core import Pose
from transforms3d.euler import euler2axangle, euler2quat

from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from simpler_env.utils.io import DictAction

from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector
from experiments.robot.robot_utils import get_action, get_model


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_range_tuple(values):
    return np.linspace(values[0], values[1], int(values[2]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Evaluate VLA-Adapter on SimplerEnv")

    # SimplerEnv arguments
    parser.add_argument("--policy-model", type=str, default="openvla")
    parser.add_argument("--policy-setup", type=str, default="google_robot", choices=["google_robot", "widowx_bridge"])
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to VLA-Adapter checkpoint directory")
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument("--additional-env-save-tags", type=str, default=None)
    parser.add_argument("--scene-name", type=str, default="google_pick_coke_can_1_v4")
    parser.add_argument("--enable-raytracing", action="store_true")
    parser.add_argument("--robot", type=str, default="google_robot_static")
    parser.add_argument("--obs-camera-name", type=str, default=None)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--control-freq", type=int, default=3)
    parser.add_argument("--sim-freq", type=int, default=513)
    parser.add_argument("--max-episode-steps", type=int, default=80)
    parser.add_argument("--rgb-overlay-path", type=str, default=None)
    parser.add_argument("--robot-init-x-range", type=float, nargs=3, default=[0.35, 0.35, 1], help="[xmin, xmax, num]")
    parser.add_argument("--robot-init-y-range", type=float, nargs=3, default=[0.20, 0.20, 1], help="[ymin, ymax, num]")
    parser.add_argument(
        "--robot-init-rot-quat-center",
        type=float,
        nargs=4,
        default=[1, 0, 0, 0],
        help="[x, y, z, w]",
    )
    parser.add_argument(
        "--robot-init-rot-rpy-range",
        type=float,
        nargs=9,
        default=[0, 0, 1, 0, 0, 1, 0, 0, 1],
        help="[rmin, rmax, rnum, pmin, pmax, pnum, ymin, ymax, ynum]",
    )
    parser.add_argument("--obj-variation-mode", type=str, default="xy", choices=["xy", "episode"])
    parser.add_argument("--obj-episode-range", type=int, nargs=2, default=[0, 60], help="[start, end]")
    parser.add_argument("--obj-init-x-range", type=float, nargs=3, default=[-0.35, -0.12, 5], help="[xmin, xmax, num]")
    parser.add_argument("--obj-init-y-range", type=float, nargs=3, default=[-0.02, 0.42, 5], help="[ymin, ymax, num]")
    parser.add_argument(
        "--additional-env-build-kwargs",
        nargs="+",
        action=DictAction,
        help=(
            "Additional env build kwargs in xxx=yyy format. "
            "If value is a list, use key=\"[a,b]\" or key=a,b."
        ),
    )
    parser.add_argument("--logging-dir", type=str, default=None)
    parser.add_argument("--tf-memory-limit", type=int, default=3072, help="Tensorflow memory limit (MB)")

    # VLA-Adapter inference arguments
    parser.add_argument("--model-family", type=str, default="openvla")
    parser.add_argument("--unnorm-key", type=str, default="")
    parser.add_argument("--num-open-loop-steps", type=int, default=8)
    parser.add_argument("--num-images-in-input", type=int, default=1)
    parser.add_argument("--use-l1-regression", type=str2bool, default=True)
    parser.add_argument("--use-proprio", type=str2bool, default=False)
    parser.add_argument("--proprio-dim", type=int, default=8)
    parser.add_argument("--use-film", type=str2bool, default=False)
    parser.add_argument("--use-minivlm", type=str2bool, default=False)
    parser.add_argument("--center-crop", type=str2bool, default=False)
    parser.add_argument("--load-in-8bit", type=str2bool, default=False)
    parser.add_argument("--load-in-4bit", type=str2bool, default=False)
    parser.add_argument("--save-version", type=str, default="Pro")
    parser.add_argument("--use-gate", type=str2bool, default=False)
    parser.add_argument("--use-latent-outputs", type=str2bool, default=False)

    # Action ensembling arguments
    parser.add_argument("--action-ensemble", type=str2bool, default=True)
    parser.add_argument("--action-ensemble-horizon", type=int, default=None)
    parser.add_argument("--adaptive-ensemble-alpha", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224], help="Policy image size used in plots")
    return parser


@dataclass
class VLAAdapterConfig:
    model_family: str
    pretrained_checkpoint: str
    unnorm_key: str
    num_open_loop_steps: int
    num_images_in_input: int
    use_l1_regression: bool
    use_proprio: bool
    proprio_dim: int
    use_film: bool
    use_minivlm: bool
    center_crop: bool
    load_in_8bit: bool
    load_in_4bit: bool
    save_version: str
    use_gate: bool
    use_latent_outputs: bool


class AdaptiveEnsembler:
    def __init__(self, pred_action_horizon: int, adaptive_ensemble_alpha: float = 0.0):
        self.pred_action_horizon = pred_action_horizon
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_history = deque(maxlen=self.pred_action_horizon)

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action: np.ndarray) -> np.ndarray:
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )

        ref = curr_act_preds[num_actions - 1, :]
        dot_product = np.sum(curr_act_preds * ref, axis=1)
        norm_previous_pred = np.linalg.norm(curr_act_preds, axis=1)
        norm_ref = np.linalg.norm(ref)
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)

        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()
        return np.sum(weights[:, None] * curr_act_preds, axis=0)


class VLAAdapterSimplerPolicy:
    def __init__(
        self,
        cfg: VLAAdapterConfig,
        policy_setup: str = "google_robot",
        action_scale: float = 1.0,
        image_size: Sequence[int] = (224, 224),
        action_ensemble: bool = True,
        action_ensemble_horizon: Optional[int] = None,
        adaptive_ensemble_alpha: float = 0.1,
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.cfg = cfg
        self.policy_setup = policy_setup
        self.image_size = tuple(image_size)
        self.action_scale = action_scale

        if policy_setup == "widowx_bridge":
            self._setup_default_unnorm_key = "bridge_orig"
            self.sticky_gripper_num_repeat = 1
            if action_ensemble_horizon is None:
                action_ensemble_horizon = 7
        elif policy_setup == "google_robot":
            self._setup_default_unnorm_key = "fractal20220817_data"
            self.sticky_gripper_num_repeat = 10
            if action_ensemble_horizon is None:
                action_ensemble_horizon = 2
        else:
            raise NotImplementedError(f"Unsupported policy setup: {policy_setup}")

        self.model = get_model(cfg)
        if hasattr(self.model, "set_version"):
            self.model.set_version(cfg.save_version)
        self.action_head = get_action_head(cfg, self.model.llm_dim) if cfg.use_l1_regression else None
        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(cfg, self.model.llm_dim, cfg.proprio_dim)
        self.processor = get_processor(cfg) if cfg.model_family == "openvla" else None

        self._resolve_unnorm_key()
        self.action_ensemble = action_ensemble
        self.action_ensembler = None
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(action_ensemble_horizon, adaptive_ensemble_alpha)

        self.task_description = None
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def _resolve_unnorm_key(self):
        norm_stats = getattr(self.model, "norm_stats", {})
        if not norm_stats:
            raise ValueError("Model has no `norm_stats`; cannot run action de-normalization for SimplerEnv.")

        if self.cfg.unnorm_key:
            if self.cfg.unnorm_key not in norm_stats:
                raise ValueError(
                    f"`unnorm_key={self.cfg.unnorm_key}` not found in checkpoint statistics. "
                    f"Available keys: {list(norm_stats.keys())}"
                )
            return

        if self._setup_default_unnorm_key in norm_stats:
            self.cfg.unnorm_key = self._setup_default_unnorm_key
            return

        if len(norm_stats) == 1:
            self.cfg.unnorm_key = next(iter(norm_stats.keys()))
            return

        raise ValueError(
            f"Cannot infer `unnorm_key` automatically for policy_setup={self.policy_setup}. "
            f"Available keys: {list(norm_stats.keys())}. Please pass `--unnorm-key`."
        )

    def reset(self, task_description: str):
        self.task_description = task_description
        if self.action_ensembler is not None:
            self.action_ensembler.reset()
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs):
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image, got dtype={image.dtype}")

        obs = {"full_image": image}
        if self.cfg.use_proprio:
            obs["state"] = np.zeros((self.cfg.proprio_dim,), dtype=np.float32)

        chunk_actions = get_action(
            self.cfg,
            self.model,
            obs,
            self.task_description,
            processor=self.processor,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            noisy_action_projector=None,
            use_film=self.cfg.use_film,
            use_minivlm=self.cfg.use_minivlm,
        )

        raw_actions = np.stack(chunk_actions, axis=0).astype(np.float64)
        if self.action_ensembler is not None:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]
        else:
            raw_actions = raw_actions[:1]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),
        }

        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        roll, pitch, yaw = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        axis, angle = euler2axangle(roll, pitch, yaw)
        action["rot_axangle"] = axis * angle * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0.0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action
        else:
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

        action["terminate_episode"] = np.array([0.0])
        return raw_action, action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, self.image_size, interpolation=cv.INTER_AREA)

    def visualize_epoch(self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str):
        resized_images = [self._resize_image(image) for image in images]
        action_labels = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
        img_strip = np.concatenate(np.array(resized_images[::3]), axis=1)

        figure_layout = [["image"] * len(action_labels), action_labels]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(action_labels):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)


def parse_args():
    parser = build_parser()
    args = parser.parse_args()
    if not Path(args.ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {args.ckpt_path}")

    # SimplerEnv robot initialization grid
    args.robot_init_xs = parse_range_tuple(args.robot_init_x_range)
    args.robot_init_ys = parse_range_tuple(args.robot_init_y_range)
    args.robot_init_quats = []
    for r in parse_range_tuple(args.robot_init_rot_rpy_range[:3]):
        for p in parse_range_tuple(args.robot_init_rot_rpy_range[3:6]):
            for y in parse_range_tuple(args.robot_init_rot_rpy_range[6:]):
                pose = Pose(q=euler2quat(r, p, y)) * Pose(q=args.robot_init_rot_quat_center)
                args.robot_init_quats.append(pose.q)

    if args.obj_variation_mode == "xy":
        args.obj_init_xs = parse_range_tuple(args.obj_init_x_range)
        args.obj_init_ys = parse_range_tuple(args.obj_init_y_range)

    if args.obs_camera_name is not None:
        tag = f"obs_camera_{args.obs_camera_name}"
        args.additional_env_save_tags = tag if args.additional_env_save_tags is None else f"{args.additional_env_save_tags}_{tag}"

    if args.logging_dir is None:
        ckpt_path = Path(args.ckpt_path)
        if ckpt_path.is_dir():
            args.logging_dir = str(ckpt_path / "eval_simpler")
        else:
            args.logging_dir = str(ckpt_path.parent / "eval_simpler")
    os.makedirs(args.logging_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    os.environ["DISPLAY"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if os.path.exists("/usr/share/vulkan/icd.d/nvidia_icd.json"):
        os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus and args.tf_memory_limit > 0:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
            )
        except RuntimeError as err:
            print(f"[Warn] TensorFlow memory limit not applied: {err}")

    cfg = VLAAdapterConfig(
        model_family=args.model_family,
        pretrained_checkpoint=args.ckpt_path,
        unnorm_key=args.unnorm_key,
        num_open_loop_steps=args.num_open_loop_steps,
        num_images_in_input=args.num_images_in_input,
        use_l1_regression=args.use_l1_regression,
        use_proprio=args.use_proprio,
        proprio_dim=args.proprio_dim,
        use_film=args.use_film,
        use_minivlm=args.use_minivlm,
        center_crop=args.center_crop,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        save_version=args.save_version,
        use_gate=args.use_gate,
        use_latent_outputs=args.use_latent_outputs,
    )

    model = VLAAdapterSimplerPolicy(
        cfg=cfg,
        policy_setup=args.policy_setup,
        action_scale=args.action_scale,
        image_size=args.image_size,
        action_ensemble=args.action_ensemble,
        action_ensemble_horizon=args.action_ensemble_horizon,
        adaptive_ensemble_alpha=args.adaptive_ensemble_alpha,
    )

    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", float(np.mean(success_arr)))


if __name__ == "__main__":
    main()
