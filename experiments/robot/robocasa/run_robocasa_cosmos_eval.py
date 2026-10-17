import ast
import json
import logging
import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import draccus
import numpy as np

from experiments.robot.robot_utils import DATE_TIME, set_seed_everywhere
from experiments.robot.robocasa.model2robocasa_interface import ObservationAdapterConfig, VLAAdapterRoboCasaPolicy


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


DEFAULT_CONTROLLER_CONFIGS_PATH = (
    "/path/to/cosmos-policy/cosmos_policy/experiments/robot/robocasa/robocasa_controller_configs.pkl"
)
COSMOS_MOBILE_BASE_ACTION = np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

TASK_MAX_STEPS = {
    "PnPCounterToCab": 500,
    "PnPCabToCounter": 500,
    "PnPCounterToSink": 700,
    "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600,
    "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500,
    "PnPStoveToCounter": 500,
    "OpenSingleDoor": 500,
    "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000,
    "CloseDoubleDoor": 700,
    "OpenDrawer": 500,
    "CloseDrawer": 500,
    "TurnOnStove": 500,
    "TurnOffStove": 500,
    "TurnOnSinkFaucet": 500,
    "TurnOffSinkFaucet": 500,
    "TurnSinkSpout": 500,
    "CoffeeSetupMug": 600,
    "CoffeeServeMug": 600,
    "CoffeePressButton": 300,
    "TurnOnMicrowave": 500,
    "TurnOffMicrowave": 500,
}


@dataclass
class RoboCasaCosmosEvalConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: str = ""
    dataset_name: str = "robocasa_cosmos_success_only"
    unnorm_key: str = "robocasa_cosmos_success_only"
    use_l1_regression: bool = True
    use_minivlm: bool = True
    use_film: bool = False
    use_proprio: bool = True
    proprio_dim: int = 9
    num_images_in_input: int = 3
    center_crop: bool = True
    flip_images: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_pro_version: bool = True
    use_gate: bool = True
    use_latent_outputs: bool = True
    save_version: str = "vla-adapter-robocasa-cosmos"
    phase: str = "Inference"

    task_name: str = "TurnOffMicrowave"
    task_description: str = ""
    num_trials_per_task: int = 50
    env_img_res: int = 224
    robots: str = "PandaMobile"
    controllers: str = "OSC_POSE"
    obj_instance_split: str = "B"
    layout_and_style_ids: str = "((1,1),(2,2),(4,4),(6,9),(7,10))"
    randomize_cameras: bool = False
    seed: int = 195
    deterministic: bool = True
    deterministic_reset: bool = False
    deterministic_reset_seed: Optional[int] = None
    enable_render: bool = True
    save_video: bool = True
    save_action_trace: bool = False
    local_log_dir: str = "experiments/robocasa/tmp/robocasa_cosmos_single"
    run_id_note: Optional[str] = None
    controller_configs_path: str = DEFAULT_CONTROLLER_CONFIGS_PATH

    # Official Cosmos eval executed 16 open-loop steps; temporarily default to 8.
    n_action_steps: int = 8
    append_fixed_mobile_base_action: bool = True
    image_obs_key: str = "video.left_image"
    extra_image_obs_keys: str = "video.right_image,video.wrist_image"
    instruction_obs_key: str = "annotation.human.coarse_action"
    state_obs_keys: str = "state"


def _parse_csv_items(value: str) -> List[str]:
    if value is None or value == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def append_fixed_mobile_base_action(action: np.ndarray, env_action_dim: int) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] == 7 and env_action_dim == 12:
        return np.concatenate([action, COSMOS_MOBILE_BASE_ACTION], axis=-1).astype(np.float32)
    return action


def _maybe_flip_image(image: np.ndarray, flip_images: bool) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    if not flip_images:
        return image
    return np.flipud(image).copy()


def _proprio_from_raw_obs(obs: Dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
            np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1),
        ],
        axis=-1,
    ).astype(np.float32)


def _policy_observation_from_raw_obs(obs: Dict[str, Any], task_description: str) -> Dict[str, Any]:
    return {
        "video.left_image": np.asarray(obs["robot0_agentview_left_image"], dtype=np.uint8)[None],
        "video.right_image": np.asarray(obs["robot0_agentview_right_image"], dtype=np.uint8)[None],
        "video.wrist_image": np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)[None],
        "state": _proprio_from_raw_obs(obs)[None],
        "annotation.human.coarse_action": [task_description],
    }


def _video_images_from_raw_obs(obs: Dict[str, Any], flip_images: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        _maybe_flip_image(obs["robot0_agentview_left_image"], flip_images),
        _maybe_flip_image(obs["robot0_agentview_right_image"], flip_images),
        _maybe_flip_image(obs["robot0_eye_in_hand_image"], flip_images),
    )


def _layout_and_style_ids(cfg: RoboCasaCosmosEvalConfig, episode_idx: int | None):
    if not cfg.layout_and_style_ids:
        return None
    all_layout_style_ids = ast.literal_eval(cfg.layout_and_style_ids)
    if episode_idx is None:
        return all_layout_style_ids
    scene_index = (episode_idx // 10) % len(all_layout_style_ids)
    return (all_layout_style_ids[scene_index],)


def create_robocasa_env(cfg: RoboCasaCosmosEvalConfig, seed: int | None, episode_idx: int | None):
    import robocasa  # noqa: F401 - registers RoboCasa tasks with robosuite
    import robosuite

    controller_path = Path(cfg.controller_configs_path)
    if not controller_path.exists():
        raise FileNotFoundError(f"Missing RoboCasa controller config: {controller_path}")
    with controller_path.open("rb") as handle:
        controller_configs = pickle.load(handle)

    env_kwargs = dict(
        env_name=cfg.task_name,
        robots=cfg.robots,
        controller_configs=controller_configs,
        camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        camera_widths=cfg.env_img_res,
        camera_heights=cfg.env_img_res,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split=cfg.obj_instance_split,
        generative_textures=None,
        randomize_cameras=cfg.randomize_cameras,
        layout_and_style_ids=_layout_and_style_ids(cfg, episode_idx),
        translucent_robot=False,
    )
    return robosuite.make(**env_kwargs)


def _save_rollout_video(
    primary_images: List[np.ndarray],
    secondary_images: List[np.ndarray],
    wrist_images: List[np.ndarray],
    episode_idx: int,
    success: bool,
    task_description: str,
    rollout_dir: Path,
) -> Path:
    import imageio

    rollout_dir.mkdir(parents=True, exist_ok=True)
    processed_task = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:40]
    path = rollout_dir / f"{DATE_TIME}--episode={episode_idx}--success={success}--task={processed_task}.mp4"
    writer = imageio.get_writer(str(path), fps=30)
    for primary, secondary, wrist in zip(primary_images, secondary_images, wrist_images):
        writer.append_data(np.concatenate([primary, secondary, wrist], axis=1))
    writer.close()
    return path


def _task_description(env, cfg: RoboCasaCosmosEvalConfig) -> str:
    try:
        ep_meta = env.get_ep_meta()
        if isinstance(ep_meta, dict) and ep_meta.get("lang"):
            return str(ep_meta["lang"])
    except Exception:
        pass
    return cfg.task_description or "perform the task"


def _env_action_dim(env) -> int:
    if hasattr(env, "action_dim"):
        return int(env.action_dim)
    action_spec = getattr(env, "action_spec", None)
    if action_spec is not None:
        return int(action_spec[0].shape[-1])
    return 0


def run_episode(cfg: RoboCasaCosmosEvalConfig, env, policy: VLAAdapterRoboCasaPolicy, episode_idx: int):
    for _ in range(10):
        dummy_action = np.zeros(env.action_spec[0].shape, dtype=np.float32)
        obs, _, _, _ = env.step(dummy_action)

    task_description = _task_description(env, cfg)
    max_steps = TASK_MAX_STEPS.get(cfg.task_name, 500)
    env_action_dim = _env_action_dim(env)
    action_queue = deque()
    success = False
    primary_images: List[np.ndarray] = []
    secondary_images: List[np.ndarray] = []
    wrist_images: List[np.ndarray] = []
    query_timesteps: List[int] = []
    predicted_action_chunks: List[np.ndarray] = []
    executed_model_actions: List[np.ndarray] = []
    executed_env_actions: List[np.ndarray] = []

    for timestep in range(max_steps):
        primary, secondary, wrist = _video_images_from_raw_obs(obs, cfg.flip_images)
        primary_images.append(primary)
        secondary_images.append(secondary)
        wrist_images.append(wrist)

        if not action_queue:
            policy_obs = _policy_observation_from_raw_obs(obs, task_description)
            model_outputs = policy.step(policy_obs)
            action_chunk = np.asarray(model_outputs["actions"][0], dtype=np.float32)[: cfg.n_action_steps]
            query_timesteps.append(timestep)
            predicted_action_chunks.append(action_chunk.copy())
            action_queue.extend(action_chunk)

        model_action = np.asarray(action_queue.popleft(), dtype=np.float32)
        action = model_action
        if cfg.append_fixed_mobile_base_action:
            action = append_fixed_mobile_base_action(action, env_action_dim)
        executed_model_actions.append(model_action.copy())
        executed_env_actions.append(np.asarray(action, dtype=np.float32).copy())
        logger.debug("episode=%d timestep=%d action=%s", episode_idx, timestep, np.array2string(action, precision=4))
        obs, _, _, info = env.step(action)

        if env._check_success():
            success = True
            break

    episode_length = len(primary_images)
    if cfg.save_action_trace:
        trace_dir = Path(cfg.local_log_dir) / "action_traces" / f"{cfg.task_name}--{DATE_TIME}"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{DATE_TIME}--episode={episode_idx}--success={success}.npz"
        np.savez_compressed(
            trace_path,
            task_name=np.asarray(cfg.task_name),
            task_description=np.asarray(task_description),
            episode_idx=np.asarray(episode_idx, dtype=np.int32),
            success=np.asarray(success),
            query_timesteps=np.asarray(query_timesteps, dtype=np.int32),
            predicted_action_chunks=np.asarray(predicted_action_chunks, dtype=np.float32),
            executed_model_actions=np.asarray(executed_model_actions, dtype=np.float32),
            executed_env_actions=np.asarray(executed_env_actions, dtype=np.float32),
        )
    return success, episode_length, task_description, primary_images, secondary_images, wrist_images


def _build_policy(cfg: RoboCasaCosmosEvalConfig) -> VLAAdapterRoboCasaPolicy:
    if not cfg.unnorm_key and cfg.dataset_name:
        cfg.unnorm_key = cfg.dataset_name
    cfg.num_open_loop_steps = cfg.n_action_steps
    adapter_cfg = ObservationAdapterConfig(
        image_obs_key=cfg.image_obs_key,
        instruction_obs_key=cfg.instruction_obs_key,
        state_obs_keys=tuple(_parse_csv_items(cfg.state_obs_keys)) or ("state",),
        extra_image_obs_keys=tuple(_parse_csv_items(cfg.extra_image_obs_keys)),
        flip_images=cfg.flip_images,
    )
    return VLAAdapterRoboCasaPolicy(cfg, adapter_cfg)


def _setup_logging(cfg: RoboCasaCosmosEvalConfig) -> Tuple[str, Path]:
    log_dir = Path(cfg.local_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"EVAL-robocasa-cosmos-{Path(cfg.pretrained_checkpoint).name}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    return run_id, log_dir / f"{run_id}.txt"


def _validate_config(cfg: RoboCasaCosmosEvalConfig) -> None:
    if not cfg.pretrained_checkpoint:
        raise ValueError("`pretrained_checkpoint` must be set for RoboCasa Cosmos evaluation.")
    if cfg.load_in_8bit and cfg.load_in_4bit:
        raise ValueError("Only one of `load_in_8bit` and `load_in_4bit` can be enabled.")
    if cfg.use_proprio and cfg.proprio_dim != 9:
        raise ValueError("RoboCasa Cosmos proprio is 9D; use `--proprio_dim 9`.")
    if cfg.num_images_in_input != 3:
        raise ValueError("RoboCasa Cosmos eval is aligned for exactly three images: left, right, wrist.")


def run_task(cfg: RoboCasaCosmosEvalConfig, policy: VLAAdapterRoboCasaPolicy):
    successes: List[bool] = []
    lengths: List[int] = []
    video_paths: List[str] = []

    for episode_idx in range(cfg.num_trials_per_task):
        seed = cfg.seed * episode_idx * 256 if (cfg.deterministic or cfg.deterministic_reset) else None
        env = create_robocasa_env(cfg, seed=seed, episode_idx=episode_idx)
        try:
            if cfg.deterministic_reset:
                reset_seed = cfg.deterministic_reset_seed if cfg.deterministic_reset_seed is not None else cfg.seed
                set_seed_everywhere(reset_seed)
            env.reset()
            success, length, task_description, primary_images, secondary_images, wrist_images = run_episode(
                cfg, env, policy, episode_idx
            )
            successes.append(success)
            lengths.append(length)
            logger.info(
                "Task %s episode %d/%d success=%s length=%d",
                cfg.task_name,
                episode_idx + 1,
                cfg.num_trials_per_task,
                success,
                length,
            )
            if cfg.save_video:
                rollout_dir = Path(cfg.local_log_dir) / "rollout_data" / f"{cfg.task_name}--{DATE_TIME}"
                video_path = _save_rollout_video(
                    primary_images,
                    secondary_images,
                    wrist_images,
                    episode_idx,
                    success,
                    task_description,
                    rollout_dir,
                )
                video_paths.append(str(video_path))
        finally:
            env.close()

    return successes, lengths, video_paths


@draccus.wrap()
def run_robocasa_cosmos_eval(cfg: RoboCasaCosmosEvalConfig):
    _validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    run_id, log_path = _setup_logging(cfg)
    policy = _build_policy(cfg)
    successes, lengths, video_paths = run_task(cfg, policy)
    success_rate = float(np.mean(successes)) if successes else 0.0
    payload = {
        "run_id": run_id,
        "task_name": cfg.task_name,
        "n_episodes": len(successes),
        "success_rate": success_rate,
        "episode_successes": successes,
        "episode_lengths": lengths,
        "video_paths": video_paths,
        "config": {
            "pretrained_checkpoint": cfg.pretrained_checkpoint,
            "dataset_name": cfg.dataset_name,
            "unnorm_key": cfg.unnorm_key,
            "image_obs_key": cfg.image_obs_key,
            "extra_image_obs_keys": cfg.extra_image_obs_keys,
            "state_obs_keys": cfg.state_obs_keys,
            "num_images_in_input": cfg.num_images_in_input,
            "use_proprio": cfg.use_proprio,
            "proprio_dim": cfg.proprio_dim,
            "flip_images": cfg.flip_images,
            "n_action_steps": cfg.n_action_steps,
            "append_fixed_mobile_base_action": cfg.append_fixed_mobile_base_action,
            "save_action_trace": cfg.save_action_trace,
            "robots": cfg.robots,
            "controllers": cfg.controllers,
            "obj_instance_split": cfg.obj_instance_split,
            "layout_and_style_ids": cfg.layout_and_style_ids,
            "seed": cfg.seed,
            "deterministic": cfg.deterministic,
        },
    }
    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("RoboCasa Cosmos success rate: %.4f", success_rate)
    logger.info("Saved RoboCasa Cosmos results to %s", log_path)
    return success_rate


if __name__ == "__main__":
    run_robocasa_cosmos_eval()
