from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
LOGGER = logging.getLogger(__name__)


DATASET_TASK_INSTRUCTIONS = {
    "assembly-v3": "pick up a nut and place it onto a peg",
    "basketball-v3": "dunk the basketball into the basket",
    "bin-picking-v3": "grasp the puck from one bin and place it into another bin",
    "box-close-v3": "grasp the cover and close the box with it",
    "button-press-topdown-v3": "press a button from the top",
    "button-press-topdown-wall-v3": "bypass a wall and press a button from the top",
    "button-press-v3": "press a button",
    "button-press-wall-v3": "bypass a wall and press a button",
    "coffee-button-v3": "push a button on the coffee machine",
    "coffee-pull-v3": "pull a mug from a coffee machine",
    "coffee-push-v3": "push a mug under a coffee machine",
    "dial-turn-v3": "rotate a dial 180 degrees",
    "disassemble-v3": "pick a nut out of a peg",
    "door-close-v3": "close a door with a revolving joint",
    "door-lock-v3": "lock the door by rotating the lock clockwise",
    "door-open-v3": "open a door with a revolving joint",
    "door-unlock-v3": "unlock the door by rotating the lock counter-clockwise",
    "drawer-close-v3": "push and close a drawer",
    "drawer-open-v3": "open a drawer",
    "faucet-close-v3": "rotate the faucet clockwise",
    "faucet-open-v3": "rotate the faucet counter-clockwise",
    "hammer-v3": "hammer a screw on the wall",
    "hand-insert-v3": "insert the gripper into a hole",
    "handle-press-side-v3": "press a handle down sideways",
    "handle-press-v3": "press a handle down",
    "handle-pull-side-v3": "pull a handle up sideways",
    "handle-pull-v3": "pull a handle up",
    "lever-pull-v3": "pull a lever down 90 degrees",
    "peg-insert-side-v3": "insert a peg sideways",
    "peg-unplug-side-v3": "unplug a peg sideways",
    "pick-out-of-hole-v3": "pick up a puck from a hole",
    "pick-place-v3": "pick and place a puck to a goal",
    "pick-place-wall-v3": "pick a puck, bypass a wall and place the puck",
    "plate-slide-back-v3": "get a plate from the cabinet",
    "plate-slide-back-side-v3": "get a plate from the cabinet sideways",
    "plate-slide-side-v3": "slide a plate into a cabinet sideways",
    "plate-slide-v3": "slide a plate into a cabinet",
    "push-v3": "push the puck to a goal",
    "push-wall-v3": "bypass a wall and push a puck to a goal",
    "reach-v3": "reach a goal position",
    "reach-wall-v3": "bypass a wall and reach a goal",
    "shelf-place-v3": "pick and place a puck onto a shelf",
    "soccer-v3": "kick a soccer into the goal",
    "stick-pull-v3": "grasp a stick and pull a box with the stick",
    "stick-push-v3": "grasp a stick and push a box using the stick",
    "sweep-into-v3": "sweep a puck into a hole",
    "sweep-v3": "sweep a puck off the table",
    "window-close-v3": "push and close a window",
    "window-open-v3": "push and open a window",
}

FALLBACK_TASK_INSTRUCTIONS = {
    "push-back-v3": "push the puck backward",
}

SUPPORTED_TASK_INSTRUCTION_ENVS = set(DATASET_TASK_INSTRUCTIONS) | set(FALLBACK_TASK_INSTRUCTIONS)
FALLBACK_ONLY_ENVS = set(FALLBACK_TASK_INSTRUCTIONS) - set(DATASET_TASK_INSTRUCTIONS)

METAWORLD_TASK_DIFFICULTY = {
    "button-press-v3": "easy",
    "button-press-topdown-v3": "easy",
    "button-press-topdown-wall-v3": "easy",
    "button-press-wall-v3": "easy",
    "coffee-button-v3": "easy",
    "dial-turn-v3": "easy",
    "door-close-v3": "easy",
    "door-lock-v3": "easy",
    "door-open-v3": "easy",
    "door-unlock-v3": "easy",
    "drawer-close-v3": "easy",
    "drawer-open-v3": "easy",
    "faucet-close-v3": "easy",
    "faucet-open-v3": "easy",
    "handle-press-v3": "easy",
    "handle-press-side-v3": "easy",
    "handle-pull-v3": "easy",
    "handle-pull-side-v3": "easy",
    "lever-pull-v3": "easy",
    "peg-unplug-side-v3": "easy",
    "plate-slide-v3": "easy",
    "plate-slide-back-v3": "easy",
    "plate-slide-back-side-v3": "easy",
    "plate-slide-side-v3": "easy",
    "reach-v3": "easy",
    "reach-wall-v3": "easy",
    "window-close-v3": "easy",
    "window-open-v3": "easy",
    "basketball-v3": "medium",
    "bin-picking-v3": "medium",
    "box-close-v3": "medium",
    "coffee-pull-v3": "medium",
    "coffee-push-v3": "medium",
    "hammer-v3": "medium",
    "peg-insert-side-v3": "medium",
    "push-wall-v3": "medium",
    "soccer-v3": "medium",
    "sweep-v3": "medium",
    "sweep-into-v3": "medium",
    "assembly-v3": "hard",
    "hand-insert-v3": "hard",
    "pick-out-of-hole-v3": "hard",
    "pick-place-v3": "hard",
    "push-v3": "hard",
    "push-back-v3": "hard",
    "disassemble-v3": "very_hard",
    "pick-place-wall-v3": "very_hard",
    "shelf-place-v3": "very_hard",
    "stick-pull-v3": "very_hard",
    "stick-push-v3": "very_hard",
}

DIFFICULTY_ORDER = ("easy", "medium", "hard", "very_hard", "unknown")
LEROBOT_METAWORLD_CAMERA_POS = (0.75, 0.075, 0.7)
METAWORLD_ACTION_DIM = 4
METAWORLD_PROPRIO_DIM = 4
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


class BlackRenderFrameError(RuntimeError):
    """Raised when MuJoCo returns persistent black rendered frames."""


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def yn_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered not in {"y", "n"}:
        raise ValueError(f"Expected 'y' or 'n', got {value!r}")
    return lowered == "y"


def strict_bool_arg(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "t", "1", "yes", "y"}:
        return True
    if lowered in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value!r}. Expected one of: true/false, yes/no, y/n, 1/0."
    )


def strict_nonnegative_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value!r}.") from exc

    if parsed < 0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative integer, got {value!r}.")
    return parsed


def strict_positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value!r}.") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}.")
    return parsed


def normalize_reset_output(reset_output: Any) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(reset_output, tuple):
        if len(reset_output) == 2:
            obs, info = reset_output
            return obs, info if isinstance(info, dict) else {}
        if len(reset_output) == 1:
            return reset_output[0], {}
    return reset_output, {}


def normalize_step_output(step_output: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    if not isinstance(step_output, tuple):
        raise TypeError(f"Expected env.step(...) to return a tuple, got {type(step_output)}")

    if len(step_output) == 5:
        obs, reward, terminated, truncated, info = step_output
        return obs, float(reward), bool(terminated), bool(truncated), info if isinstance(info, dict) else {}

    if len(step_output) == 4:
        obs, reward, done, info = step_output
        return obs, float(reward), bool(done), False, info if isinstance(info, dict) else {}

    raise ValueError(f"Unsupported env.step(...) return length: {len(step_output)}")


def sanitize_filename(text: str, max_len: int = 80) -> str:
    text = text.replace("/", "_").replace(" ", "_")
    text = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return text[:max_len]


def resize_image(img: np.ndarray, resize_size: int | tuple[int, int]) -> np.ndarray:
    import tensorflow as tf

    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    with tf.device("/CPU:0"):
        tensor = tf.convert_to_tensor(img, dtype=tf.uint8)
        tensor = tf.image.resize(tensor, resize_size, method="lanczos3", antialias=True)
        tensor = tf.cast(tf.clip_by_value(tf.round(tensor), 0, 255), tf.uint8)
        return tensor.numpy()


def preprocess_metaworld_image(
    frame: np.ndarray,
    resize_size: int | tuple[int, int] | None,
    flip_mode: str,
) -> np.ndarray:
    img = np.asarray(frame, dtype=np.uint8)
    if flip_mode == "horizontal":
        img = img[:, ::-1, :]
    elif flip_mode == "vertical":
        img = img[::-1, :, :]
    elif flip_mode == "both":
        img = img[::-1, ::-1, :]
    if resize_size is None or resize_size == 0:
        return img
    return resize_image(img, resize_size)


def is_black_frame(frame: np.ndarray | None, threshold: float = 2.0) -> bool:
    if frame is None:
        return True
    arr = np.asarray(frame)
    if arr.size == 0:
        return True
    return float(arr.mean()) < threshold and float(arr.std()) < 1e-6


def to_jsonable_info(info: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()
        elif isinstance(value, (np.floating, float)):
            payload[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            payload[key] = int(value)
        elif isinstance(value, (np.bool_, bool)):
            payload[key] = bool(value)
        else:
            payload[key] = value
    return payload


def get_task_instruction(env_name: str, override: str = "") -> str:
    if override:
        return override.strip().lower()
    if env_name in DATASET_TASK_INSTRUCTIONS:
        return DATASET_TASK_INSTRUCTIONS[env_name]
    if env_name in FALLBACK_TASK_INSTRUCTIONS:
        return FALLBACK_TASK_INSTRUCTIONS[env_name]
    return env_name.replace("-v3", "").replace("-", " ")


def get_task_difficulty(env_name: str) -> str:
    return METAWORLD_TASK_DIFFICULTY.get(env_name, "unknown")


def save_rollout_video(
    output_dir: Path,
    file_name: str,
    benchmark_name: str,
    env_name: str,
    episode_idx: int,
    success: bool,
    frames: list[np.ndarray],
) -> Path | None:
    if not frames:
        return None

    import imageio.v2 as imageio

    rollout_dir = output_dir / "videos" / benchmark_name / env_name
    rollout_dir.mkdir(parents=True, exist_ok=True)
    video_name = (
        f"{sanitize_filename(file_name)}-{DATE_TIME}"
        f"--episode={episode_idx}"
        f"--success={success}"
        f"--env={sanitize_filename(env_name)}.mp4"
    )
    video_path = rollout_dir / video_name

    writer = imageio.get_writer(str(video_path), fps=30)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return video_path


def ensure_metaworld_constants_context() -> None:
    if any("metaworld" in arg.lower() for arg in sys.argv):
        return
    sys.argv.append("metaworld")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MetaWorld evaluation for VLA-Adapter / OpenVLA checkpoints")

    parser.add_argument("--seed", type=int, default=7, help="Random seed")

    parser.add_argument(
        "--benchmark",
        type=str,
        default="MT50",
        choices=["MT1", "MT10", "MT25", "MT50", "ML1", "ML10", "ML25", "ML45"],
        help="MetaWorld benchmark to evaluate",
    )
    parser.add_argument("--ml_split", type=str, default="train", choices=["train", "test"], help="Split for ML benchmarks")
    parser.add_argument("--env_name", type=str, default="", help="Environment name for MT1 / ML1")
    parser.add_argument("--task_instruction", type=str, default="", help="Override the task instruction text")
    parser.add_argument("--num_trials_per_task", type=int, default=50, help="Episodes to evaluate per environment")
    parser.add_argument("--max_envs", type=int, default=0, help="Optional debug cap on number of environments")
    parser.add_argument("--max_episode_steps", type=int, default=0, help="0 uses the environment horizon")
    parser.add_argument("--terminate_on_success", type=str, default="y", choices=["y", "n"], help="Stop an episode once success is reached")

    parser.add_argument("--model_family", type=str, default="openvla", choices=["openvla"], help="Policy family")
    parser.add_argument("--pretrained_checkpoint", type=str, required=True, help="Checkpoint directory or Hub id")
    parser.add_argument("--file_name", type=str, default="metaworld-eval", help="Run tag used in saved filenames")
    parser.add_argument("--unnorm_key", type=str, default="", help="Optional normalization key override")
    parser.add_argument("--num_open_loop_steps", type=int, default=8, help="How many predicted actions to execute per policy call")
    parser.add_argument("--num_images_in_input", type=int, default=1, help="Number of camera images expected by the checkpoint")
    parser.add_argument("--center_crop", type=strict_bool_arg, default=True, help="Whether to center crop images inside the OpenVLA helper")
    parser.add_argument("--use_proprio", type=strict_bool_arg, default=False, help="Whether to pass 4D MetaWorld state into the checkpoint")
    parser.add_argument("--use_l1_regression", type=strict_bool_arg, default=True, help="Load the continuous action head")
    parser.add_argument("--use_film", type=strict_bool_arg, default=False, help="Enable FiLM if the checkpoint uses it")
    parser.add_argument("--use_minivlm", type=strict_bool_arg, default=True, help="Use the MiniVLM prompt format")
    parser.add_argument("--use_pro_version", type=strict_bool_arg, default=True, help="Use the Pro action head variant")
    parser.add_argument("--use_gate", type=strict_bool_arg, default=True, help="Enable gating in the action head")
    parser.add_argument("--use_latent_outputs", type=strict_bool_arg, default=True, help="Enable latent-output action head mode")
    parser.add_argument("--load_in_8bit", type=strict_bool_arg, default=False, help="Load the checkpoint in 8-bit mode")
    parser.add_argument("--load_in_4bit", type=strict_bool_arg, default=False, help="Load the checkpoint in 4-bit mode")

    parser.add_argument("--camera_name", type=str, default="corner2", help="MetaWorld camera name")
    parser.add_argument("--render_width", type=int, default=480, help="Rendered frame width")
    parser.add_argument("--render_height", type=int, default=480, help="Rendered frame height")
    parser.add_argument("--image_resize", type=int, default=224, help="Optional pre-policy resize; 0 lets the VLA inference pipeline resize automatically")
    parser.add_argument(
        "--image_flip",
        type=str,
        default="both",
        choices=["none", "horizontal", "vertical", "both"],
        help="Image flip applied before policy inference",
    )
    parser.add_argument(
        "--apply_lerobot_camera_patch",
        type=str,
        default="y",
        choices=["y", "n"],
        help="Apply the LeRobot MetaWorld camera patch before rendering",
    )
    parser.add_argument("--lerobot_camera_id", type=int, default=2, help="Camera id used for the LeRobot camera patch")
    parser.add_argument(
        "--lerobot_camera_pos",
        type=str,
        default=",".join(str(v) for v in LEROBOT_METAWORLD_CAMERA_POS),
        help="Comma-separated xyz position used by the LeRobot camera patch",
    )
    parser.add_argument("--reward_function_version", type=str, default="v2", choices=["v1", "v2"], help="MetaWorld reward function version")
    parser.add_argument("--mujoco_gl", type=str, default="egl", choices=["egl", "osmesa", "glfw"], help="MuJoCo OpenGL backend")
    parser.add_argument(
        "--render_black_retry_attempts",
        type=strict_positive_int_arg,
        default=5,
        help="Positive number of render attempts before treating the frame as failed",
    )
    parser.add_argument("--black_frame_mean_threshold", type=float, default=2.0, help="Mean-pixel threshold for black frame detection")
    parser.add_argument(
        "--render_failure_max_restarts",
        type=strict_nonnegative_int_arg,
        default=3,
        help="How many times to recreate the env after a persistent render failure",
    )
    parser.add_argument(
        "--on_render_failure",
        type=str,
        default="skip_episode",
        choices=["skip_episode", "abort_task", "raise"],
        help="Action to take after restart attempts are exhausted",
    )

    parser.add_argument("--position_action_scale", type=float, default=1.0, help="Scale factor applied to xyz deltas before env.step")
    parser.add_argument("--gripper_action_index", type=int, default=-1, help="Index to use for the gripper dimension when model output is wider than 4D")
    parser.add_argument(
        "--gripper_mapping",
        type=str,
        default="clip_to_signed",
        choices=["raw", "zero_one_to_signed", "clip_to_signed"],
        help="Mapping applied to the checkpoint gripper output before env.step",
    )
    parser.add_argument("--invert_gripper", type=str, default="n", choices=["y", "n"], help="Invert the gripper sign before env.step")
    parser.add_argument("--clip_env_action", type=str, default="y", choices=["y", "n"], help="Clip mapped MetaWorld actions to [-1, 1]")

    parser.add_argument("--save_video", type=str, default="n", choices=["y", "n"], help="Save rollout videos")
    parser.add_argument("--save_action", type=str, default="n", choices=["y", "n"], help="Save executed actions")
    parser.add_argument("--save_debug_frame", type=str, default="n", choices=["y", "n"], help="Save the first raw and processed frame for each episode")
    parser.add_argument("--output_dir", type=str, default="", help="Optional output directory override")
    parser.add_argument(
        "--skip_unsupported_envs",
        type=str,
        default="y",
        choices=["y", "n"],
        help="Skip benchmark envs that are not covered by the MetaWorld instruction table",
    )
    parser.add_argument(
        "--include_fallback_only_envs",
        type=str,
        default="n",
        choices=["y", "n"],
        help="Include fallback-only envs that are not dataset-backed, such as push-back-v3",
    )
    return parser


class MetaWorldEvaluator:
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg

        ensure_metaworld_constants_context()
        from experiments.robot.openvla_utils import (
            get_action_head,
            get_processor,
            get_proprio_projector,
            get_vla,
            get_vla_action,
        )

        self._get_vla_action = get_vla_action
        self.model = get_vla(cfg)
        self.action_head = get_action_head(cfg, self.model.llm_dim) if cfg.use_l1_regression else None
        self.proprio_projector = (
            get_proprio_projector(cfg, self.model.llm_dim, METAWORLD_PROPRIO_DIM) if cfg.use_proprio else None
        )
        self.processor = get_processor(cfg)
        self._resolve_unnorm_key()

        self.run_id = self._build_run_id()
        base_output_dir = Path(cfg.output_dir) if cfg.output_dir else PROJECT_ROOT / "experiments" / "metaworld"
        self.output_dir = base_output_dir if base_output_dir.name == self.run_id else base_output_dir / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / f"{self.run_id}.log"
        self.log_file = self.log_path.open("w", encoding="utf-8")

        self._log(f"Run ID: {self.run_id}")
        self._log(f"Output dir: {self.output_dir}")
        self._log(f"Checkpoint: {cfg.pretrained_checkpoint}")
        self._log(f"Normalization key: {self.cfg.unnorm_key}")

    def _build_run_id(self) -> str:
        name_parts = [
            "EVAL",
            "metaworld",
            self.cfg.benchmark.lower(),
            sanitize_filename(self.cfg.file_name or self.cfg.env_name or "all"),
            DATE_TIME,
        ]
        return "-".join(part for part in name_parts if part)

    def _resolve_unnorm_key(self) -> None:
        norm_stats = getattr(self.model, "norm_stats", {})
        if not norm_stats:
            raise ValueError("Checkpoint has no `norm_stats`; cannot resolve an action un-normalization key.")

        if self.cfg.unnorm_key:
            if self.cfg.unnorm_key not in norm_stats:
                raise ValueError(
                    f"`unnorm_key={self.cfg.unnorm_key}` not found in checkpoint statistics. "
                    f"Available keys: {list(norm_stats.keys())}"
                )
            return

        preferred_keys = ("metaworld_mt50", "metaworld_mt50_no_noops")
        for key in preferred_keys:
            if key in norm_stats:
                self.cfg.unnorm_key = key
                return

        if len(norm_stats) == 1:
            self.cfg.unnorm_key = next(iter(norm_stats.keys()))
            return

        raise ValueError(
            "Cannot infer `unnorm_key` automatically for MetaWorld evaluation. "
            f"Available keys: {list(norm_stats.keys())}. Please pass --unnorm_key explicitly."
        )

    def _log(self, message: str) -> None:
        LOGGER.info(message)
        self.log_file.write(message + "\n")
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()

    def _apply_lerobot_camera_patch(self, env) -> None:
        if not yn_to_bool(self.cfg.apply_lerobot_camera_patch):
            return

        camera_pos = np.asarray([float(v.strip()) for v in self.cfg.lerobot_camera_pos.split(",")], dtype=np.float64)
        if camera_pos.shape != (3,):
            raise ValueError(
                f"Expected three comma-separated floats for --lerobot_camera_pos, got {self.cfg.lerobot_camera_pos!r}"
            )
        env.model.cam_pos[int(self.cfg.lerobot_camera_id)][:] = camera_pos

    def _safe_render(self, env) -> np.ndarray:
        max_attempts = int(self.cfg.render_black_retry_attempts)
        last_frame = None
        last_render_exception: Exception | None = None
        for _ in range(max_attempts):
            self._apply_lerobot_camera_patch(env)
            try:
                frame = env.render()
            except Exception as exc:
                last_render_exception = exc
                frame = None
            last_frame = frame
            if frame is not None and not is_black_frame(frame, threshold=self.cfg.black_frame_mean_threshold):
                return np.asarray(frame, dtype=np.uint8)

            if hasattr(env, "sim") and hasattr(env.sim, "forward"):
                try:
                    env.sim.forward()
                except Exception as exc:
                    last_render_exception = exc
            elif hasattr(env, "mj_forward"):
                try:
                    env.mj_forward()
                except Exception as exc:
                    last_render_exception = exc

        if last_render_exception is not None and last_frame is None:
            raise BlackRenderFrameError(
                f"MetaWorld render raised after {max_attempts} retries: {last_render_exception!r}"
            )

        if last_frame is None:
            raise BlackRenderFrameError(
                f"MetaWorld render returned None after {max_attempts} retries."
            )

        arr = np.asarray(last_frame, dtype=np.uint8)
        mean = float(arr.mean()) if arr.size else 0.0
        std = float(arr.std()) if arr.size else 0.0
        raise BlackRenderFrameError(
            f"Black render frame detected after {max_attempts} retries; "
            f"mean={mean:.3f}, std={std:.3f}"
        )

    def _make_observation(self, obs: np.ndarray, frame: np.ndarray) -> dict[str, Any]:
        return {
            "full_image": preprocess_metaworld_image(frame, self.cfg.image_resize, self.cfg.image_flip),
            "state": np.asarray(obs[:METAWORLD_PROPRIO_DIM], dtype=np.float32),
        }

    def _get_action_chunks(self, observation: dict[str, Any], task_instruction: str) -> list[np.ndarray]:
        actions = self._get_vla_action(
            cfg=self.cfg,
            vla=self.model,
            processor=self.processor,
            obs=observation,
            task_label=task_instruction,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            noisy_action_projector=None,
            use_film=self.cfg.use_film,
            use_minivlm=self.cfg.use_minivlm,
        )
        return [np.asarray(action, dtype=np.float32) for action in actions]

    def _map_model_action_to_metaworld(self, raw_action: np.ndarray) -> np.ndarray:
        action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if action.shape[0] < METAWORLD_ACTION_DIM:
            padded = np.zeros(METAWORLD_ACTION_DIM, dtype=np.float32)
            padded[: action.shape[0]] = action
            action = padded

        xyz = action[:3] * self.cfg.position_action_scale
        if action.shape[0] == METAWORLD_ACTION_DIM:
            gripper = float(action[3])
        else:
            gripper_index = self.cfg.gripper_action_index
            if gripper_index < 0:
                gripper_index = action.shape[0] + gripper_index
            gripper_index = min(max(gripper_index, 0), action.shape[0] - 1)
            gripper = float(action[gripper_index])

        if self.cfg.gripper_mapping == "zero_one_to_signed":
            gripper = 1.0 - 2.0 * float(np.clip(gripper, 0.0, 1.0))
        elif self.cfg.gripper_mapping == "clip_to_signed":
            gripper = float(np.clip(gripper, -1.0, 1.0))

        if yn_to_bool(self.cfg.invert_gripper):
            gripper = -gripper

        env_action = np.concatenate([xyz, np.array([gripper], dtype=np.float32)], axis=0)
        if yn_to_bool(self.cfg.clip_env_action):
            env_action = np.clip(env_action, -1.0, 1.0)
        return env_action.astype(np.float32)

    def _seed_env(self, env) -> None:
        if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
            env.action_space.seed(self.cfg.seed)

    def _reset_env(self, env, episode_seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        try:
            return normalize_reset_output(env.reset(seed=episode_seed))
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(episode_seed)
            return normalize_reset_output(env.reset())

    def _init_env(self, env_cls):
        candidate_kwargs = [
            {
                "render_mode": "rgb_array",
                "camera_name": self.cfg.camera_name,
                "width": self.cfg.render_width,
                "height": self.cfg.render_height,
                "reward_function_version": self.cfg.reward_function_version,
            },
            {
                "render_mode": "rgb_array",
                "camera_name": self.cfg.camera_name,
                "width": self.cfg.render_width,
                "height": self.cfg.render_height,
            },
            {
                "render_mode": "rgb_array",
                "camera_name": self.cfg.camera_name,
            },
            {},
        ]

        last_error: Exception | None = None
        for kwargs in candidate_kwargs:
            try:
                env = env_cls(**kwargs)
                self._apply_lerobot_camera_patch(env)
                self._seed_env(env)
                return env
            except TypeError as exc:
                last_error = exc

        raise RuntimeError(f"Failed to initialize MetaWorld env with supported kwargs: {last_error}")

    def _get_benchmark_spec(self):
        import metaworld

        if self.cfg.benchmark in {"MT1", "ML1"} and not self.cfg.env_name:
            raise ValueError(f"--env_name is required for {self.cfg.benchmark}")

        if self.cfg.benchmark in {"MT1", "ML1"}:
            benchmark = getattr(metaworld, self.cfg.benchmark)(self.cfg.env_name, seed=self.cfg.seed)
        else:
            benchmark = getattr(metaworld, self.cfg.benchmark)(seed=self.cfg.seed)

        if self.cfg.benchmark.startswith("MT"):
            env_classes = benchmark.train_classes
            tasks = benchmark.train_tasks
        else:
            if self.cfg.ml_split == "train":
                env_classes = benchmark.train_classes
                tasks = benchmark.train_tasks
            else:
                env_classes = benchmark.test_classes
                tasks = benchmark.test_tasks

        tasks_by_env: dict[str, list[Any]] = defaultdict(list)
        for task in tasks:
            tasks_by_env[task.env_name].append(task)

        original_env_names = list(env_classes.keys())
        env_names = list(original_env_names)
        if yn_to_bool(self.cfg.skip_unsupported_envs):
            env_names = [env_name for env_name in env_names if env_name in SUPPORTED_TASK_INSTRUCTION_ENVS]
        if not yn_to_bool(self.cfg.include_fallback_only_envs):
            env_names = [env_name for env_name in env_names if env_name not in FALLBACK_ONLY_ENVS]
        if self.cfg.max_envs > 0:
            env_names = env_names[: self.cfg.max_envs]

        skipped_env_names = [env_name for env_name in original_env_names if env_name not in env_names]
        return self.cfg.benchmark, env_names, skipped_env_names, env_classes, tasks_by_env

    def _save_episode_artifacts(self, benchmark_name: str, env_name: str, result: dict[str, Any]) -> None:
        if yn_to_bool(self.cfg.save_video):
            video_path = save_rollout_video(
                self.output_dir,
                self.cfg.file_name or "metaworld-eval",
                benchmark_name,
                env_name,
                result["episode_idx"],
                result["success"],
                result["replay_frames"],
            )
            if video_path is not None:
                self._log(f"Saved video: {video_path}")

        if yn_to_bool(self.cfg.save_action):
            action_dir = self.output_dir / "actions" / benchmark_name / env_name
            action_dir.mkdir(parents=True, exist_ok=True)
            action_path = action_dir / f"episode_{result['episode_idx']:04d}.json"
            payload = {
                "env_name": env_name,
                "task_instruction": result["task_instruction"],
                "episode_idx": result["episode_idx"],
                "success": result["success"],
                "num_steps": result["num_steps"],
                "latency_sec": result["latency_sec"],
                "actions": result["actions"],
            }
            with action_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self._log(f"Saved actions: {action_path}")

        if yn_to_bool(self.cfg.save_debug_frame) and result.get("debug_frame_payload") is not None:
            from PIL import Image

            payload = result["debug_frame_payload"]
            debug_dir = self.output_dir / "debug_frames" / benchmark_name / env_name / f"episode_{result['episode_idx']:04d}"
            debug_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(payload["raw_frame"]).save(debug_dir / "raw_frame.png")
            Image.fromarray(payload["processed_frame"]).save(debug_dir / "processed_frame.png")
            with (debug_dir / "metadata.json").open("w", encoding="utf-8") as f:
                json.dump(payload["metadata"], f, indent=2)
            self._log(f"Saved debug frames: {debug_dir}")

    def _run_episode(self, env, task, env_name: str, task_instruction: str, episode_idx: int) -> dict[str, Any]:
        env.set_task(task)
        obs, _ = self._reset_env(env, self.cfg.seed + episode_idx)
        self._apply_lerobot_camera_patch(env)
        raw_frame = self._safe_render(env)

        env_default_horizon = getattr(env, "max_path_length", None)
        if env_default_horizon is None:
            env_default_horizon = getattr(env, "_max_episode_steps", 500)
        max_episode_steps = self.cfg.max_episode_steps if self.cfg.max_episode_steps > 0 else env_default_horizon

        replay_frames: list[np.ndarray] = []
        action_records: list[dict[str, Any]] = []
        chunk_actions: list[np.ndarray] = []
        chunk_step_idx = 0
        total_inference_time = 0.0
        num_policy_calls = 0
        success = False
        terminated = False
        truncated = False
        debug_frame_payload = None
        pending_frame = True

        for step_idx in range(max_episode_steps):
            if pending_frame:
                replay_frames.append(np.asarray(raw_frame, dtype=np.uint8))
                pending_frame = False
            observation = self._make_observation(obs, raw_frame)

            if debug_frame_payload is None and yn_to_bool(self.cfg.save_debug_frame):
                debug_frame_payload = {
                    "raw_frame": np.asarray(raw_frame, dtype=np.uint8).copy(),
                    "processed_frame": np.asarray(observation["full_image"], dtype=np.uint8).copy(),
                    "metadata": {
                        "env_name": env_name,
                        "episode_idx": episode_idx,
                        "task_instruction": task_instruction,
                        "image_flip": self.cfg.image_flip,
                        "image_resize": self.cfg.image_resize,
                        "camera_name": self.cfg.camera_name,
                        "render_width": self.cfg.render_width,
                        "render_height": self.cfg.render_height,
                        "apply_lerobot_camera_patch": self.cfg.apply_lerobot_camera_patch,
                        "lerobot_camera_id": self.cfg.lerobot_camera_id,
                        "lerobot_camera_pos": self.cfg.lerobot_camera_pos,
                    },
                }

            if chunk_step_idx >= len(chunk_actions):
                start_time = time.time()
                chunk_actions = self._get_action_chunks(observation, task_instruction)
                total_inference_time += time.time() - start_time
                num_policy_calls += 1
                chunk_step_idx = 0

                if not chunk_actions:
                    raise RuntimeError("Policy returned zero actions for a rollout step.")

            raw_action = chunk_actions[chunk_step_idx]
            env_action = self._map_model_action_to_metaworld(raw_action)
            next_obs, reward, terminated, truncated, info = normalize_step_output(env.step(env_action))
            step_success = bool(info.get("success", info.get("is_success", False)))
            success = success or step_success

            action_records.append(
                {
                    "step": step_idx,
                    "raw_model_action": np.asarray(raw_action, dtype=np.float32).tolist(),
                    "env_action": env_action.tolist(),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "success": success,
                    "info": to_jsonable_info(info),
                }
            )

            obs = next_obs
            chunk_step_idx += 1

            # Preserve successful terminal transitions even if a post-terminal render would fail.
            if success and yn_to_bool(self.cfg.terminate_on_success):
                break
            if terminated or truncated:
                break

            raw_frame = self._safe_render(env)
            pending_frame = True

        if pending_frame:
            replay_frames.append(np.asarray(raw_frame, dtype=np.uint8))
        return {
            "env_name": env_name,
            "task_instruction": task_instruction,
            "episode_idx": episode_idx,
            "success": success,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "num_steps": len(action_records),
            "latency_sec": total_inference_time / max(num_policy_calls, 1),
            "actions": action_records,
            "replay_frames": replay_frames,
            "debug_frame_payload": debug_frame_payload,
        }

    def evaluate(self) -> dict[str, Any]:
        benchmark_name, env_names, skipped_env_names, env_classes, tasks_by_env = self._get_benchmark_spec()
        self._log(f"Benchmark: {benchmark_name}")
        self._log(f"Evaluating envs: {env_names}")
        if skipped_env_names:
            self._log(f"Skipped envs: {skipped_env_names}")

        summary: dict[str, Any] = {
            "benchmark": benchmark_name,
            "env_names": env_names,
            "config": vars(self.cfg),
            "per_env": {},
            "per_difficulty": {
                difficulty: {
                    "num_envs": 0,
                    "num_episodes": 0,
                    "num_successes": 0,
                    "num_render_failures": 0,
                    "success_rate": 0.0,
                    "render_failure_rate": 0.0,
                    "env_names": [],
                }
                for difficulty in DIFFICULTY_ORDER
            },
            "total_episodes": 0,
            "total_successes": 0,
            "total_render_failures": 0,
            "render_failures": [],
        }

        for env_idx, env_name in enumerate(env_names, start=1):
            env_tasks = tasks_by_env.get(env_name, [])
            if not env_tasks:
                self._log(f"Skip {env_name}: no tasks found in benchmark split")
                continue

            difficulty = get_task_difficulty(env_name)
            task_instruction = get_task_instruction(env_name, self.cfg.task_instruction)
            self._log(
                f"[{env_idx}/{len(env_names)}] Start env {env_name} "
                f"difficulty={difficulty} with {len(env_tasks)} benchmark tasks"
            )

            env_successes = 0
            env_render_failures = 0
            env_latencies: list[float] = []
            env_results: list[dict[str, Any]] = []
            env = self._init_env(env_classes[env_name])
            num_trials = self.cfg.num_trials_per_task
            max_restarts = max(0, int(self.cfg.render_failure_max_restarts))

            for episode_idx in range(num_trials):
                task = env_tasks[episode_idx % len(env_tasks)]
                result = None
                last_exc: Exception | None = None

                for restart_idx in range(max_restarts + 1):
                    try:
                        result = self._run_episode(env, task, env_name, task_instruction, episode_idx)
                        break
                    except BlackRenderFrameError as exc:
                        last_exc = exc
                        self._log(
                            f"Render failure in {env_name} episode {episode_idx + 1}/{num_trials}; "
                            f"restart {restart_idx + 1}/{max_restarts}: {exc}"
                        )
                        try:
                            env.close()
                        except Exception:
                            pass
                        if restart_idx < max_restarts:
                            env = self._init_env(env_classes[env_name])
                    except Exception:
                        try:
                            env.close()
                        except Exception:
                            pass
                        raise

                if result is None:
                    env_render_failures += 1
                    summary["total_render_failures"] += 1
                    summary["per_difficulty"][difficulty]["num_render_failures"] += 1
                    summary["render_failures"].append(
                        {
                            "env_name": env_name,
                            "difficulty": difficulty,
                            "episode_idx": episode_idx,
                            "task_instruction": task_instruction,
                            "error": repr(last_exc),
                            "action": self.cfg.on_render_failure,
                        }
                    )

                    self._log(
                        f"Render failure persisted in {env_name} episode {episode_idx + 1}/{num_trials} "
                        f"after {max_restarts} restarts. "
                        f"Action: {self.cfg.on_render_failure}"
                    )

                    if self.cfg.on_render_failure == "abort_task":
                        self._log(f"Abort remaining episodes for env {env_name} due to persistent render failure.")
                        break
                    if self.cfg.on_render_failure == "raise":
                        raise RuntimeError(
                            f"Persistent render failure in {env_name} episode {episode_idx}"
                        ) from last_exc

                    env = self._init_env(env_classes[env_name])
                    continue

                env_results.append(
                    {
                        "episode_idx": result["episode_idx"],
                        "success": result["success"],
                        "num_steps": result["num_steps"],
                        "latency_sec": result["latency_sec"],
                        "difficulty": difficulty,
                        "render_failed": False,
                    }
                )
                env_successes += int(result["success"])
                env_latencies.append(result["latency_sec"])
                summary["total_episodes"] += 1
                summary["total_successes"] += int(result["success"])
                summary["per_difficulty"][difficulty]["num_episodes"] += 1
                summary["per_difficulty"][difficulty]["num_successes"] += int(result["success"])

                self._save_episode_artifacts(benchmark_name, env_name, result)
                self._log(
                    f"{env_name} episode {episode_idx + 1}/{num_trials}: "
                    f"success={result['success']} steps={result['num_steps']} "
                    f"latency={result['latency_sec']:.4f}s"
                )

            try:
                env.close()
            except Exception:
                pass

            completed_trials = len(env_results)
            success_rate = env_successes / max(completed_trials, 1)
            avg_latency = float(np.mean(env_latencies)) if env_latencies else 0.0
            summary["per_env"][env_name] = {
                "difficulty": difficulty,
                "task_instruction": task_instruction,
                "num_trials_requested": num_trials,
                "num_trials_completed": completed_trials,
                "num_render_failures": env_render_failures,
                "num_successes": env_successes,
                "success_rate": success_rate,
                "avg_latency_sec": avg_latency,
                "episodes": env_results,
            }
            summary["per_difficulty"][difficulty]["num_envs"] += 1
            summary["per_difficulty"][difficulty]["env_names"].append(env_name)
            self._log(
                f"{env_name} finished: success_rate={success_rate:.3f} "
                f"({env_successes}/{completed_trials}), render_failures={env_render_failures}, "
                f"avg_latency={avg_latency:.4f}s"
            )

        for difficulty, stats in summary["per_difficulty"].items():
            stats["success_rate"] = stats["num_successes"] / max(stats["num_episodes"], 1)
            stats["render_failure_rate"] = stats["num_render_failures"] / max(
                stats["num_episodes"] + stats["num_render_failures"], 1
            )

        summary["overall_success_rate"] = summary["total_successes"] / max(summary["total_episodes"], 1)
        summary["overall_render_failure_rate"] = summary["total_render_failures"] / max(
            summary["total_episodes"] + summary["total_render_failures"], 1
        )

        summary_path = self.output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self._log(f"Summary saved to: {summary_path}")
        self._log(
            f"Overall success rate: {summary['overall_success_rate']:.3f} "
            f"({summary['total_successes']}/{summary['total_episodes']}), "
            f"render_failure_rate={summary['overall_render_failure_rate']:.3f} "
            f"({summary['total_render_failures']} failures)"
        )
        return summary


def main() -> None:
    cfg = build_parser().parse_args()

    os.environ["MUJOCO_GL"] = cfg.mujoco_gl
    os.environ.setdefault("PYOPENGL_PLATFORM", cfg.mujoco_gl)

    set_seed_everywhere(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    evaluator = MetaWorldEvaluator(cfg)
    try:
        evaluator.evaluate()
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
