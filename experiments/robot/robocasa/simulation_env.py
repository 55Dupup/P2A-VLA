"""
Run RoboCasa simulations with a local VLA-Adapter checkpoint.

This follows the same high-level structure as starVLA's RoboCasa example:
1. build a Gymnasium RoboCasa environment
2. wrap it with multistep/video utilities
3. adapt environment observations into VLA-Adapter inference inputs
4. execute chunked actions inside the simulator
"""

import importlib
import json
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import draccus
import numpy as np

from experiments.robot.robot_utils import DATE_TIME, set_seed_everywhere
from experiments.robot.robocasa.model2robocasa_interface import (
    DEFAULT_STATE_KEYS,
    ObservationAdapterConfig,
    VLAAdapterRoboCasaPolicy,
)
from prismatic.vla.datasets.rlds.oxe.robocasa_registry import infer_robocasa_dataset_name_from_env

try:
    import gymnasium as gym
except ImportError:
    gym = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


DEFAULT_DICT_ACTION_KEYS = (
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
)
DEFAULT_DICT_ACTION_DIMS = (3, 3, 1, 4, 1)


@dataclass
class VideoConfig:
    video_dir: Optional[str] = None
    steps_per_render: int = 2
    fps: int = 10
    codec: str = "libx264"


@dataclass
class MultiStepConfig:
    video_delta_indices: Tuple[int, ...] = (0,)
    state_delta_indices: Tuple[int, ...] = (0,)
    n_action_steps: int = 16
    max_episode_steps: int = 720


@dataclass
class RoboCasaEvalConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: str = ""
    dataset_name: str = ""
    unnorm_key: str = ""
    use_l1_regression: bool = True
    use_minivlm: bool = True
    use_film: bool = False
    use_proprio: bool = True
    proprio_dim: int = 29
    num_images_in_input: int = 1
    center_crop: bool = True
    flip_images: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_pro_version: bool = False
    use_gate: bool = False
    use_latent_outputs: bool = False
    save_version: str = "vla-adapter-robocasa"
    phase: str = "Inference"

    env_name: str = "robocasa/TurnOffSinkFaucet"
    env_split: str = "target"
    obj_registries: str = "lightwheel"
    n_episodes: int = 10
    n_envs: int = 1
    seed: int = 7
    save_video: bool = False
    enable_render: bool = True
    local_log_dir: str = "./experiments/logs"
    run_id_note: Optional[str] = None

    action_space_mode: str = "auto"
    image_obs_key: str = "video.ego_view"
    extra_image_obs_keys: str = ""
    instruction_obs_key: str = "annotation.human.coarse_action"
    task_description: str = "perform the task"
    state_obs_keys: str = ",".join(DEFAULT_STATE_KEYS)
    dataset_name_suffix: str = "_1000"
    dict_action_keys: str = ",".join(DEFAULT_DICT_ACTION_KEYS)
    dict_action_dims: str = ",".join(str(dim) for dim in DEFAULT_DICT_ACTION_DIMS)

    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)

    @property
    def n_action_steps(self) -> int:
        return self.multistep.n_action_steps


def _require_robocasa_runtime():
    missing = []
    if gym is None:
        missing.append("gymnasium")

    robocasa_available = True
    for module_name in ("robocasa", "robosuite"):
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)
            if module_name == "robocasa":
                robocasa_available = False

    if robocasa_available:
        try:
            importlib.import_module("robocasa.wrappers.gym_wrapper")
        except ImportError:
            missing.append("robocasa.wrappers.gym_wrapper")

    if missing:
        raise ImportError(
            "RoboCasa evaluation dependencies are missing: "
            + ", ".join(missing)
            + ". Install RoboCasa in the current environment first, then rerun this script."
        )


def _parse_csv_items(value: str) -> List[str]:
    if value is None or value == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_ints(value: str) -> List[int]:
    return [int(item) for item in _parse_csv_items(value)]


ROBOCASA_IMAGE_ALIASES = {
    "video.right_image": (
        "video.right_image",
        "right_image",
        "observation.images.robot0_agentview_right",
        "video.robot0_agentview_right",
        "video.robot0_agentview_right_image",
        "images.robot0_agentview_right",
        "robot0_agentview_right",
        "robot0_agentview_right_image",
    ),
    "video.left_image": (
        "video.left_image",
        "left_image",
        "observation.images.robot0_agentview_left",
        "video.robot0_agentview_left",
        "video.robot0_agentview_left_image",
        "images.robot0_agentview_left",
        "robot0_agentview_left",
        "robot0_agentview_left_image",
    ),
    "video.wrist_image": (
        "video.wrist_image",
        "wrist_image",
        "observation.images.robot0_eye_in_hand",
        "video.robot0_eye_in_hand",
        "video.robot0_eye_in_hand_image",
        "images.robot0_eye_in_hand",
        "robot0_eye_in_hand",
        "robot0_eye_in_hand_image",
    ),
}
ROBOCASA_STATE_ALIASES = ("state", "observation.state")
ROBOCASA_INSTRUCTION_ALIASES = (
    "annotation.human.coarse_action",
    "annotation.human.task_description",
    "language",
    "language_instruction",
    "task.language_instruction",
    "observation.task",
)


def _dict_items(container: Any):
    if isinstance(container, dict):
        return container.items()
    if hasattr(container, "spaces"):
        return container.spaces.items()
    return ()


def _lookup_path(container: Any, key: str):
    if isinstance(container, dict) and key in container:
        return container[key]
    if hasattr(container, "spaces") and key in container.spaces:
        return container.spaces[key]

    value = container
    for part in key.split("."):
        if isinstance(value, dict):
            if part not in value:
                return None
            value = value[part]
        elif hasattr(value, "spaces"):
            if part not in value.spaces:
                return None
            value = value.spaces[part]
        else:
            return None
    return value


def _lookup_alias(container: Any, aliases: Tuple[str, ...]):
    for alias in aliases:
        value = _lookup_path(container, alias)
        if value is not None:
            return value
    return None


def _available_keys(container: Any, prefix: str = "") -> List[str]:
    keys: List[str] = []
    for key, value in _dict_items(container):
        full_key = f"{prefix}.{key}" if prefix else str(key)
        keys.append(full_key)
        if isinstance(value, dict) or hasattr(value, "spaces"):
            keys.extend(_available_keys(value, full_key))
    return keys


_BaseObservationWrapper = gym.Wrapper if gym is not None else object


class RoboCasaObservationAdapterWrapper(_BaseObservationWrapper):
    """Normalize official/native RoboCasa observations to the merged Panda training schema."""

    def __init__(self, env, task_description: str, use_proprio: bool):
        if gym is None:
            raise ImportError("gymnasium is required to create RoboCasaObservationAdapterWrapper.")
        super().__init__(env)
        self.task_description = task_description
        self.use_proprio = use_proprio
        self.observation_space = self._build_observation_space(env.observation_space)

    def _build_observation_space(self, source_space):
        canonical_spaces = {}
        for canonical_key, aliases in ROBOCASA_IMAGE_ALIASES.items():
            image_space = _lookup_alias(source_space, aliases)
            if image_space is None:
                raise KeyError(
                    f"Unable to find RoboCasa image observation for `{canonical_key}`. "
                    f"Tried aliases {aliases}. Available keys: {_available_keys(source_space)}"
                )
            canonical_spaces[canonical_key] = image_space

        if self.use_proprio:
            state_space = _lookup_alias(source_space, ROBOCASA_STATE_ALIASES)
            if state_space is None:
                raise KeyError(
                    "Unable to find RoboCasa state observation. "
                    f"Tried aliases {ROBOCASA_STATE_ALIASES}. Available keys: {_available_keys(source_space)}"
                )
            canonical_spaces["state"] = state_space

        instruction_space = _lookup_alias(source_space, ROBOCASA_INSTRUCTION_ALIASES)
        if instruction_space is None:
            instruction_space = gym.spaces.Text(max_length=1024)
        canonical_spaces["annotation.human.coarse_action"] = instruction_space
        return gym.spaces.Dict(canonical_spaces)

    def _normalize_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {}
        for canonical_key, aliases in ROBOCASA_IMAGE_ALIASES.items():
            image = _lookup_alias(observation, aliases)
            if image is None:
                raise KeyError(
                    f"Unable to find RoboCasa image observation for `{canonical_key}`. "
                    f"Tried aliases {aliases}. Available keys: {_available_keys(observation)}"
                )
            normalized[canonical_key] = image

        if self.use_proprio:
            state = _lookup_alias(observation, ROBOCASA_STATE_ALIASES)
            if state is None:
                raise KeyError(
                    "Unable to find RoboCasa state observation. "
                    f"Tried aliases {ROBOCASA_STATE_ALIASES}. Available keys: {_available_keys(observation)}"
                )
            normalized["state"] = state

        instruction = None
        if self.task_description and self.task_description != "perform the task":
            instruction = self.task_description
        if instruction is None:
            instruction = _lookup_alias(observation, ROBOCASA_INSTRUCTION_ALIASES)
        if instruction is None:
            instruction = self.task_description
        normalized["annotation.human.coarse_action"] = instruction
        return normalized

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._normalize_observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize_observation(observation), reward, terminated, truncated, info


def _setup_logging(cfg: RoboCasaEvalConfig):
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    run_id = f"EVAL-robocasa-{Path(cfg.pretrained_checkpoint).name}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    log_path = Path(cfg.local_log_dir) / f"{run_id}.txt"
    if cfg.save_video and cfg.video.video_dir is None:
        cfg.video.video_dir = str(Path(cfg.local_log_dir) / run_id)
    logger.info("Logging RoboCasa evaluation to %s", log_path)
    return run_id, log_path


class SimulationInferenceEnv:
    def __init__(self, cfg: RoboCasaEvalConfig, model: VLAAdapterRoboCasaPolicy):
        self.cfg = cfg
        self.model = model
        self.env = None

    def setup_environment(self):
        from experiments.robot.robocasa.wrappers.video_recording_wrapper import (
            VideoRecorder,
            VideoRecordingWrapper,
        )

        env_fns = [partial(self._create_single_env, idx=i, recorder_cls=VideoRecorder, wrapper_cls=VideoRecordingWrapper)
                   for i in range(self.cfg.n_envs)]
        if self.cfg.n_envs == 1:
            return gym.vector.SyncVectorEnv(env_fns)
        return gym.vector.AsyncVectorEnv(env_fns, shared_memory=False, context="spawn")

    def _create_single_env(self, idx, recorder_cls, wrapper_cls):
        from experiments.robot.robocasa.wrappers.multistep_wrapper import MultiStepWrapper

        env = gym.make(
            self.cfg.env_name,
            enable_render=self.cfg.enable_render,
            split=self.cfg.env_split,
            obj_registries=tuple(_parse_csv_items(self.cfg.obj_registries)) or None,
        )
        env = RoboCasaObservationAdapterWrapper(
            env,
            task_description=self.cfg.task_description,
            use_proprio=self.cfg.use_proprio,
        )
        if self.cfg.save_video and self.cfg.video.video_dir is not None and idx == 0:
            recorder = recorder_cls.create_h264(fps=self.cfg.video.fps, codec=self.cfg.video.codec)
            env = wrapper_cls(
                env,
                recorder,
                video_dir=Path(self.cfg.video.video_dir),
                steps_per_render=self.cfg.video.steps_per_render,
            )

        env = MultiStepWrapper(
            env,
            video_delta_indices=np.asarray(self.cfg.multistep.video_delta_indices),
            state_delta_indices=(np.asarray(self.cfg.multistep.state_delta_indices) if self.cfg.use_proprio else None),
            n_action_steps=self.cfg.multistep.n_action_steps,
            max_episode_steps=self.cfg.multistep.max_episode_steps,
        )
        return env

    def run(self) -> Tuple[str, List[bool]]:
        self.env = self.setup_environment()
        obs, _ = self.env.reset(seed=self.cfg.seed)
        completed_episodes = 0
        current_lengths = [0] * self.cfg.n_envs
        current_successes = [False] * self.cfg.n_envs
        episode_successes: List[bool] = []

        while completed_episodes < self.cfg.n_episodes:
            model_outputs = self.model.step(obs)
            actions = self._format_actions_for_env(model_outputs["actions"])
            next_obs, _, terminations, truncations, infos = self.env.step(actions)

            for env_idx in range(self.cfg.n_envs):
                current_lengths[env_idx] += 1
                current_successes[env_idx] |= self._extract_success_flag(infos, env_idx)
                if terminations[env_idx] or truncations[env_idx]:
                    episode_successes.append(current_successes[env_idx])
                    logger.info(
                        "Env %d finished episode %d | success=%s | steps=%d",
                        env_idx,
                        completed_episodes + 1,
                        current_successes[env_idx],
                        current_lengths[env_idx],
                    )
                    current_lengths[env_idx] = 0
                    current_successes[env_idx] = False
                    completed_episodes += 1
                    if completed_episodes >= self.cfg.n_episodes:
                        break

            obs = next_obs

        self.env.close()
        self.env = None
        return self.cfg.env_name, episode_successes[: self.cfg.n_episodes]

    def _format_actions_for_env(self, actions: np.ndarray):
        single_action_space = self.env.single_action_space if hasattr(self.env, "single_action_space") else self.env.action_space

        if self.cfg.action_space_mode == "box":
            return actions

        if self.cfg.action_space_mode == "dict" or single_action_space.__class__.__name__ == "Dict":
            action_keys = _parse_csv_items(self.cfg.dict_action_keys)
            action_dims = _parse_csv_ints(self.cfg.dict_action_dims)
            if not action_keys or not action_dims or len(action_keys) != len(action_dims):
                raise ValueError("Dict action mode requires matching `dict_action_keys` and `dict_action_dims`.")
            if sum(action_dims) != actions.shape[-1]:
                raise ValueError(
                    f"Dict action dims sum to {sum(action_dims)}, but the policy returned {actions.shape[-1]} dims. "
                    "This usually means the checkpoint is still a 7D LIBERO/CALVIN-style policy rather than a "
                    "RoboCasa multi-head policy."
                )
            action_dict = {}
            start = 0
            for key, dim in zip(action_keys, action_dims):
                action_dict[key] = actions[..., start : start + dim]
                start += dim
            return action_dict

        return actions

    @staticmethod
    def _extract_success_flag(infos: Dict[str, np.ndarray], env_idx: int) -> bool:
        success_key = "success" if "success" in infos else "is_success" if "is_success" in infos else None
        if success_key is None:
            final_infos = infos.get("final_info")
            if final_infos is None:
                return False
            if isinstance(final_infos, np.ndarray):
                final_infos = final_infos.tolist()
            if isinstance(final_infos, (list, tuple)) and env_idx < len(final_infos):
                final_info = final_infos[env_idx]
                if isinstance(final_info, dict):
                    if "success" in final_info:
                        value = final_info["success"]
                    elif "is_success" in final_info:
                        value = final_info["is_success"]
                    else:
                        return False
                else:
                    return False
            else:
                return False
        else:
            value = infos[success_key][env_idx]
        value = np.asarray(value).reshape(-1)
        return bool(value[-1]) if value.size > 0 else False


def _build_policy(cfg: RoboCasaEvalConfig) -> VLAAdapterRoboCasaPolicy:
    if not cfg.dataset_name:
        inferred_dataset_name = infer_robocasa_dataset_name_from_env(cfg.env_name, cfg.dataset_name_suffix)
        if inferred_dataset_name is not None:
            cfg.dataset_name = inferred_dataset_name
    if not cfg.unnorm_key and cfg.dataset_name:
        cfg.unnorm_key = cfg.dataset_name
    logger.info("Resolved RoboCasa dataset_name=%s, unnorm_key=%s", cfg.dataset_name or "<unset>", cfg.unnorm_key or "<unset>")

    state_keys = tuple(_parse_csv_items(cfg.state_obs_keys))
    cfg.num_open_loop_steps = cfg.multistep.n_action_steps
    adapter_cfg = ObservationAdapterConfig(
        image_obs_key=cfg.image_obs_key,
        instruction_obs_key=cfg.instruction_obs_key,
        state_obs_keys=state_keys or DEFAULT_STATE_KEYS,
        extra_image_obs_keys=tuple(_parse_csv_items(cfg.extra_image_obs_keys)),
        flip_images=cfg.flip_images,
    )
    return VLAAdapterRoboCasaPolicy(cfg, adapter_cfg)


def _validate_config(cfg: RoboCasaEvalConfig):
    if not cfg.pretrained_checkpoint:
        raise ValueError("`pretrained_checkpoint` must be set for RoboCasa evaluation.")
    if cfg.load_in_8bit and cfg.load_in_4bit:
        raise ValueError("Only one of `load_in_8bit` and `load_in_4bit` can be enabled.")
    if cfg.action_space_mode not in {"auto", "box", "dict"}:
        raise ValueError("`action_space_mode` must be one of: auto, box, dict.")
    if cfg.use_proprio and cfg.proprio_dim <= 0:
        raise ValueError("`use_proprio=True` requires a positive `proprio_dim` for the RoboCasa projector.")


@draccus.wrap()
def run_robocasa_eval(cfg: RoboCasaEvalConfig):
    _validate_config(cfg)
    _require_robocasa_runtime()
    set_seed_everywhere(cfg.seed)

    run_id, log_path = _setup_logging(cfg)
    policy = _build_policy(cfg)
    runner = SimulationInferenceEnv(cfg, policy)
    env_name, episode_successes = runner.run()

    success_rate = float(np.mean(episode_successes)) if episode_successes else 0.0
    payload = {
        "run_id": run_id,
        "env_name": env_name,
        "n_episodes": len(episode_successes),
        "success_rate": success_rate,
        "episode_successes": episode_successes,
        "config": {
            "pretrained_checkpoint": cfg.pretrained_checkpoint,
            "dataset_name": cfg.dataset_name,
            "unnorm_key": cfg.unnorm_key,
            "action_space_mode": cfg.action_space_mode,
            "image_obs_key": cfg.image_obs_key,
            "extra_image_obs_keys": cfg.extra_image_obs_keys,
            "state_obs_keys": cfg.state_obs_keys,
            "task_description": cfg.task_description,
            "n_action_steps": cfg.n_action_steps,
            "use_proprio": cfg.use_proprio,
            "flip_images": cfg.flip_images,
            "enable_render": cfg.enable_render,
            "env_split": cfg.env_split,
            "obj_registries": cfg.obj_registries,
        },
    }
    log_path.write_text(json.dumps(payload, indent=2))
    logger.info("RoboCasa success rate: %.4f", success_rate)
    logger.info("Saved RoboCasa results to %s", log_path)
    return success_rate


if __name__ == "__main__":
    run_robocasa_eval()
