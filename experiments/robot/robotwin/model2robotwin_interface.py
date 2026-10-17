from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector
from experiments.robot.robot_utils import get_action, get_model as load_robot_model
from experiments.robot.robotwin.action_mapping import (
    GripperConvention,
    RobotwinActionMetadata,
    convert_model_action_to_robotwin,
)


ROBOTWIN_ACTION_DIM = 14
ROBOTWIN_ACTION_CHUNK = 16


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


@dataclass
class RoboTwinPolicyConfig:
    pretrained_checkpoint: str
    unnorm_key: str | None = None
    num_images_in_input: int = 3
    use_proprio: bool = False
    actions_are_normalized: bool = False
    gripper_convention: str = "width_0_1"
    model_family: str = "openvla"
    dataset_name: str = "robotwin"
    save_version: str = "robotwin"
    num_open_loop_steps: int = 16
    n_action_steps: int = 16
    proprio_dim: int = 14
    use_l1_regression: bool = True
    use_pro_version: bool = True
    use_film: bool = False
    use_minivlm: bool = True
    use_gate: bool = False
    use_latent_outputs: bool = False
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False


def _coerce_policy_config(usr_args: dict[str, Any]) -> RoboTwinPolicyConfig:
    checkpoint = usr_args.get("policy_ckpt_path") or usr_args.get("pretrained_checkpoint")
    if checkpoint is None:
        raise ValueError("RoboTwin evaluation requires `policy_ckpt_path` in deploy_policy overrides.")

    return RoboTwinPolicyConfig(
        pretrained_checkpoint=str(checkpoint),
        unnorm_key=usr_args.get("unnorm_key"),
        num_images_in_input=int(usr_args.get("num_images_in_input", 3)),
        use_proprio=_to_bool(usr_args.get("use_proprio", False)),
        actions_are_normalized=_to_bool(usr_args.get("actions_are_normalized", False)),
        gripper_convention=str(usr_args.get("gripper_convention", "width_0_1")),
        dataset_name=str(usr_args.get("dataset_name", "robotwin")),
        save_version=str(usr_args.get("save_version", "robotwin")),
        num_open_loop_steps=int(usr_args.get("num_open_loop_steps", usr_args.get("n_action_steps", 16))),
        n_action_steps=int(usr_args.get("n_action_steps", usr_args.get("num_open_loop_steps", 16))),
        proprio_dim=int(usr_args.get("proprio_dim", 14)),
        use_l1_regression=_to_bool(usr_args.get("use_l1_regression", True)),
        use_pro_version=_to_bool(usr_args.get("use_pro_version", True)),
        use_film=_to_bool(usr_args.get("use_film", False)),
        use_minivlm=_to_bool(usr_args.get("use_minivlm", True)),
        use_gate=_to_bool(usr_args.get("use_gate", False)),
        use_latent_outputs=_to_bool(usr_args.get("use_latent_outputs", False)),
        center_crop=_to_bool(usr_args.get("center_crop", True)),
        load_in_8bit=_to_bool(usr_args.get("load_in_8bit", False)),
        load_in_4bit=_to_bool(usr_args.get("load_in_4bit", False)),
    )


def _as_runtime_cfg(cfg: RoboTwinPolicyConfig) -> SimpleNamespace:
    return SimpleNamespace(**cfg.__dict__)


def _resolve_unnorm_key(cfg: RoboTwinPolicyConfig, norm_stats: dict[str, Any]) -> str:
    if cfg.unnorm_key:
        if cfg.unnorm_key not in norm_stats:
            raise ValueError(
                f"`unnorm_key={cfg.unnorm_key}` was not found in checkpoint statistics. "
                f"Available keys: {list(norm_stats.keys())}"
            )
        return cfg.unnorm_key

    if cfg.dataset_name in norm_stats:
        return cfg.dataset_name
    if len(norm_stats) == 1:
        return next(iter(norm_stats.keys()))
    raise ValueError(
        "Unable to infer `unnorm_key` for RoboTwin evaluation. "
        "Pass `--unnorm_key` when evaluating checkpoints trained on multiple datasets."
    )


def _action_bounds_from_stats(norm_stats: dict[str, Any], unnorm_key: str) -> tuple[np.ndarray, np.ndarray]:
    stats = norm_stats.get(unnorm_key, {})
    if "abs" in stats:
        stats = stats["abs"]
    action_stats = stats.get("action", stats)
    low = action_stats.get("min", action_stats.get("q01"))
    high = action_stats.get("max", action_stats.get("q99"))
    if low is None or high is None:
        raise ValueError(
            f"Checkpoint statistics for `unnorm_key={unnorm_key}` are missing action bounds; "
            "expected `min`/`max` or `q01`/`q99`."
        )

    try:
        low_array = np.asarray(low, dtype=np.float32)
        high_array = np.asarray(high, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Checkpoint action bounds for `unnorm_key={unnorm_key}` must be numeric."
        ) from exc

    if low_array.shape != (14,) or high_array.shape != (14,):
        raise ValueError(
            f"Checkpoint action bounds for `unnorm_key={unnorm_key}` expected 14 values; "
            f"got min shape {low_array.shape} and max shape {high_array.shape}."
        )
    if not np.all(np.isfinite(low_array)) or not np.all(np.isfinite(high_array)):
        raise ValueError(f"Checkpoint action bounds for `unnorm_key={unnorm_key}` contain non-finite values.")
    if np.any(high_array <= low_array):
        invalid_dims = np.flatnonzero(high_array <= low_array).tolist()
        raise ValueError(
            f"Checkpoint action bounds for `unnorm_key={unnorm_key}` require max to be strictly greater "
            f"than min in every dimension; invalid dimensions: {invalid_dims}."
        )
    return low_array, high_array


class VLAAdapterRoboTwinPolicy:
    def __init__(
        self,
        cfg: RoboTwinPolicyConfig,
        model: Any,
        processor: Any,
        action_min: np.ndarray,
        action_max: np.ndarray,
        action_head: Any = None,
        proprio_projector: Any = None,
        noisy_action_projector: Any = None,
    ):
        if cfg.num_open_loop_steps != ROBOTWIN_ACTION_CHUNK or cfg.n_action_steps != ROBOTWIN_ACTION_CHUNK:
            raise ValueError(
                "RoboTwin evaluation chunk settings must match training: "
                f"num_open_loop_steps=n_action_steps={ROBOTWIN_ACTION_CHUNK}"
            )
        self.cfg = cfg
        self.runtime_cfg = _as_runtime_cfg(cfg)
        self.model = model
        self.processor = processor
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        self.noisy_action_projector = noisy_action_projector
        self.metadata = RobotwinActionMetadata(
            action_min=np.asarray(action_min, dtype=np.float32),
            action_max=np.asarray(action_max, dtype=np.float32),
            gripper_convention=GripperConvention(cfg.gripper_convention),
        )
        self._chunk: list[np.ndarray] = []

    @classmethod
    def from_usr_args(cls, usr_args: dict[str, Any]) -> "VLAAdapterRoboTwinPolicy":
        cfg = _coerce_policy_config(usr_args)
        runtime_cfg = _as_runtime_cfg(cfg)
        model = load_robot_model(runtime_cfg)
        if hasattr(model, "set_version"):
            model.set_version(cfg.save_version)

        norm_stats = getattr(model, "norm_stats", {})
        runtime_cfg.unnorm_key = _resolve_unnorm_key(cfg, norm_stats)
        cfg.unnorm_key = runtime_cfg.unnorm_key
        action_min, action_max = _action_bounds_from_stats(norm_stats, cfg.unnorm_key)

        action_head = get_action_head(runtime_cfg, model.llm_dim) if cfg.use_l1_regression else None
        proprio_projector = None
        if cfg.use_proprio:
            proprio_projector = get_proprio_projector(runtime_cfg, model.llm_dim, cfg.proprio_dim)
        processor = get_processor(runtime_cfg) if cfg.model_family == "openvla" else None

        return cls(
            cfg=cfg,
            model=model,
            processor=processor,
            action_min=action_min,
            action_max=action_max,
            action_head=action_head,
            proprio_projector=proprio_projector,
        )

    def reset(self, task_description: str = "") -> None:
        self._chunk.clear()

    def _format_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        obs = observation["observation"] if "observation" in observation else observation
        formatted = {"full_image": obs["head_camera"]["rgb"]}
        if self.cfg.num_images_in_input > 1:
            formatted["wrist_image"] = obs["left_camera"]["rgb"]
            formatted["wrist_image_1"] = obs["right_camera"]["rgb"]
        if self.cfg.use_proprio:
            formatted["state"] = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        return formatted

    def _next_model_action(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        if not self._chunk:
            raw_actions = np.asarray(
                get_action(
                    self.runtime_cfg,
                    self.model,
                    self._format_observation(observation),
                    instruction,
                    processor=self.processor,
                    action_head=self.action_head,
                    proprio_projector=self.proprio_projector,
                    noisy_action_projector=self.noisy_action_projector,
                    use_film=self.cfg.use_film,
                    use_minivlm=self.cfg.use_minivlm,
                ),
                dtype=np.float32,
            )
            expected_shape = (ROBOTWIN_ACTION_CHUNK, ROBOTWIN_ACTION_DIM)
            if raw_actions.shape != expected_shape:
                raise ValueError(
                    f"RoboTwin policy expected {expected_shape} action chunk, got {raw_actions.shape}"
                )
            self._chunk = list(raw_actions)
        return self._chunk.pop(0)

    def step(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        model_action = self._next_model_action(observation, instruction)
        return convert_model_action_to_robotwin(
            model_action,
            self.metadata,
            actions_are_normalized=self.cfg.actions_are_normalized,
        )


def get_model(usr_args: dict[str, Any]) -> VLAAdapterRoboTwinPolicy:
    return VLAAdapterRoboTwinPolicy.from_usr_args(usr_args)


def reset_model(model: VLAAdapterRoboTwinPolicy) -> None:
    model.reset(task_description="")


def _extract_instruction(task_env: Any, observation: dict[str, Any]) -> str:
    if hasattr(task_env, "get_instruction"):
        return str(task_env.get_instruction())
    if "instruction" in observation:
        return str(observation["instruction"])
    return str(getattr(task_env, "task_description", "perform the task"))


def eval(TASK_ENV: Any, model: VLAAdapterRoboTwinPolicy, observation: dict[str, Any]) -> None:
    instruction = _extract_instruction(TASK_ENV, observation)
    action = model.step(observation, instruction)
    TASK_ENV.take_action(action)
