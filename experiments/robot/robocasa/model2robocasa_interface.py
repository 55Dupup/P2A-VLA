from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np

from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import get_action, get_image_resize_size, get_model
from experiments.robot.robocasa.base_config import BasePolicy, ModalityConfig


DEFAULT_STATE_KEYS = (
    "state.left_arm",
    "state.right_arm",
    "state.left_hand",
    "state.right_hand",
    "state.waist",
)


@dataclass
class ObservationAdapterConfig:
    image_obs_key: str = "video.ego_view"
    instruction_obs_key: str = "annotation.human.coarse_action"
    state_obs_keys: Sequence[str] = DEFAULT_STATE_KEYS
    extra_image_obs_keys: Sequence[str] = field(default_factory=tuple)
    flip_images: bool = False


class VLAAdapterRoboCasaPolicy(BasePolicy):
    def __init__(self, cfg, adapter_cfg: ObservationAdapterConfig):
        self.cfg = cfg
        self.adapter_cfg = adapter_cfg
        self.model = get_model(cfg)
        self.model.set_version(cfg.save_version)
        self.action_head = get_action_head(cfg, self.model.llm_dim) if cfg.use_l1_regression else None
        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(cfg, self.model.llm_dim, cfg.proprio_dim)
        self.noisy_action_projector = None
        self.processor = get_processor(cfg) if cfg.model_family == "openvla" else None
        self.resize_size = get_image_resize_size(cfg)
        self._ensure_unnorm_key()

    def _ensure_unnorm_key(self):
        norm_stats = getattr(self.model, "norm_stats", {})
        if getattr(self.cfg, "unnorm_key", ""):
            if self.cfg.unnorm_key not in norm_stats:
                raise ValueError(
                    f"`unnorm_key={self.cfg.unnorm_key}` was not found in checkpoint statistics. "
                    f"Available keys: {list(norm_stats.keys())}"
                )
            return

        if len(norm_stats) == 1:
            self.cfg.unnorm_key = next(iter(norm_stats.keys()))
            return

        if getattr(self.cfg, "dataset_name", "") in norm_stats:
            self.cfg.unnorm_key = self.cfg.dataset_name
            return

        raise ValueError(
            "Unable to infer `unnorm_key` for RoboCasa evaluation. "
            "Please pass `--unnorm_key` explicitly for checkpoints trained on multiple datasets."
        )

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        state_keys = list(self.adapter_cfg.state_obs_keys) if self.cfg.use_proprio else []
        video_keys = [self.adapter_cfg.image_obs_key, *self.adapter_cfg.extra_image_obs_keys]
        return {
            "video": ModalityConfig(delta_indices=[0], modality_keys=video_keys),
            "state": ModalityConfig(delta_indices=[0], modality_keys=state_keys),
            "language": ModalityConfig(delta_indices=[0], modality_keys=[self.adapter_cfg.instruction_obs_key]),
        }

    def step(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        batch_size = self._infer_batch_size(observations)
        batch_actions = []

        for env_idx in range(batch_size):
            task_label = self._extract_instruction(observations, env_idx)
            policy_obs = self._build_policy_observation(observations, env_idx)
            actions = get_action(
                self.cfg,
                self.model,
                policy_obs,
                task_label,
                processor=self.processor,
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                noisy_action_projector=self.noisy_action_projector,
                use_film=self.cfg.use_film,
                use_minivlm=self.cfg.use_minivlm,
            )
            batch_actions.append(self._stack_or_pad_action_chunk(actions))

        return {"actions": np.stack(batch_actions, axis=0)}

    def _infer_batch_size(self, observations: Dict[str, Any]) -> int:
        for value in observations.values():
            if isinstance(value, np.ndarray) and value.ndim > 0:
                return int(value.shape[0])
            if isinstance(value, (list, tuple)) and len(value) > 0:
                return len(value)
        return 1

    def _build_policy_observation(self, observations: Dict[str, Any], env_idx: int) -> Dict[str, Any]:
        image_obs_key = self._resolve_image_obs_key(observations)
        image = self._extract_last_rgb(self._select_env_entry(observations[image_obs_key], env_idx))
        image = self._maybe_flip_image(image)
        policy_obs: Dict[str, Any] = {"full_image": image}

        if self.cfg.num_images_in_input > 1:
            extra_video_keys = self._resolve_extra_image_obs_keys(observations, image_obs_key)
            extra_video_keys = extra_video_keys[: max(0, self.cfg.num_images_in_input - 1)]
            for wrist_idx, key in enumerate(extra_video_keys):
                wrist_image = self._extract_last_rgb(self._select_env_entry(observations[key], env_idx))
                wrist_image = self._maybe_flip_image(wrist_image)
                obs_key = "wrist_image" if wrist_idx == 0 else f"wrist_image_{wrist_idx}"
                policy_obs[obs_key] = wrist_image

        if self.cfg.use_proprio:
            state = self._extract_state(observations, env_idx)
            if self.cfg.proprio_dim and state.shape[-1] != self.cfg.proprio_dim:
                raise ValueError(
                    f"RoboCasa proprio dim mismatch: expected {self.cfg.proprio_dim}, got {state.shape[-1]}. "
                    "Adjust `--proprio_dim` or `--state_obs_keys` to match the training setup."
                )
            policy_obs["state"] = state

        return policy_obs

    def _maybe_flip_image(self, image: np.ndarray) -> np.ndarray:
        if not self.adapter_cfg.flip_images:
            return image
        return np.flipud(image).copy()

    def _resolve_image_obs_key(self, observations: Dict[str, Any]) -> str:
        if self.adapter_cfg.image_obs_key in observations:
            return self.adapter_cfg.image_obs_key

        fallback_key = "video.ego_view_bg_crop_pad_res256_freq20"
        if fallback_key in observations:
            return fallback_key

        available_video_keys = [key for key in observations.keys() if key.startswith("video")]
        raise KeyError(
            f"Image observation key `{self.adapter_cfg.image_obs_key}` was not found. "
            f"Available video keys: {available_video_keys}"
        )

    def _resolve_extra_image_obs_keys(self, observations: Dict[str, Any], image_obs_key: str) -> List[str]:
        if self.adapter_cfg.extra_image_obs_keys:
            missing = [key for key in self.adapter_cfg.extra_image_obs_keys if key not in observations]
            if missing:
                image_keys = [key for key in observations if "image" in key or key.startswith("video")]
                raise KeyError(
                    f"Configured extra image keys {missing} were not found. "
                    f"Available image/video keys: {image_keys}"
                )
            return list(self.adapter_cfg.extra_image_obs_keys)

        return [key for key in observations.keys() if key.startswith("video") and key != image_obs_key]

    def _extract_state(self, observations: Dict[str, Any], env_idx: int) -> np.ndarray:
        state_parts: List[np.ndarray] = []
        for key in self.adapter_cfg.state_obs_keys:
            if key not in observations:
                continue
            value = self._select_env_entry(observations[key], env_idx)
            state_parts.append(self._extract_last_vector(value))

        if len(state_parts) == 0:
            raise ValueError(
                "No RoboCasa state keys were found in the observation dict. "
                "Either disable `--use_proprio` or update `--state_obs_keys`."
            )

        return np.concatenate(state_parts, axis=-1)

    def _extract_instruction(self, observations: Dict[str, Any], env_idx: int) -> str:
        raw_instruction = observations.get(self.adapter_cfg.instruction_obs_key)
        if raw_instruction is None:
            return getattr(self.cfg, "task_description", "perform the task")

        item = self._select_env_entry(raw_instruction, env_idx)
        if isinstance(item, np.ndarray):
            if item.ndim == 0:
                item = item.item()
            elif item.size == 1:
                item = item.reshape(-1)[0]
        if isinstance(item, (list, tuple)) and len(item) > 0:
            item = item[-1]
        return str(item)

    def _stack_or_pad_action_chunk(self, actions: Sequence[np.ndarray]) -> np.ndarray:
        stacked = np.stack(actions, axis=0)
        if stacked.shape[0] >= self.cfg.n_action_steps:
            return stacked[: self.cfg.n_action_steps]

        pad_count = self.cfg.n_action_steps - stacked.shape[0]
        pad = np.repeat(stacked[-1:], pad_count, axis=0)
        return np.concatenate([stacked, pad], axis=0)

    @staticmethod
    def _select_env_entry(value: Any, env_idx: int) -> Any:
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            return value[env_idx]
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return value
            return value[env_idx]
        return value

    @staticmethod
    def _extract_last_rgb(value: Any) -> np.ndarray:
        array = np.asarray(value)
        while array.ndim > 3:
            array = array[-1]
        if array.ndim != 3:
            raise ValueError(f"Expected an RGB image with 3 dims, got shape {array.shape}")
        return array.astype(np.uint8)

    @staticmethod
    def _extract_last_vector(value: Any) -> np.ndarray:
        array = np.asarray(value)
        while array.ndim > 1:
            array = array[-1]
        return array.astype(np.float32)
