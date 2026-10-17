#!/usr/bin/env python
"""Save per-layer VLA attention heatmaps on a fixed LIBERO observation."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_CL_CHECKPOINT = (
    "/path/to/checkpoint"
)
DEFAULT_NO_CL_CHECKPOINT = (
    "/path/to/checkpoint"
)
IGNORE_INDEX = -100
ACTION_DIM_NAME_TO_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
    "roll": 3,
    "pitch": 4,
    "yaw": 5,
    "gripper": 6,
}



@dataclass(frozen=True)
class ObservationBundle:
    task_description: str
    prompt: str
    raw_primary: np.ndarray
    raw_wrist: np.ndarray
    policy_primary: np.ndarray
    policy_wrist: np.ndarray | None
    state: np.ndarray


def infer_patch_grid(num_patches: int) -> tuple[int, int]:
    side = int(math.sqrt(num_patches))
    if side * side != num_patches:
        raise ValueError(f"Patch count must be a square for image heatmaps, got {num_patches}.")
    return side, side


def normalize_unit(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    min_value = float(np.nanmin(values))
    max_value = float(np.nanmax(values))
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        raise ValueError("Heatmap contains non-finite values.")
    if max_value - min_value < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / (max_value - min_value)).astype(np.float32)


def parse_action_dim(target: str) -> int:
    if target in ACTION_DIM_NAME_TO_INDEX:
        return ACTION_DIM_NAME_TO_INDEX[target]
    try:
        dim = int(target)
    except ValueError as exc:
        valid = ", ".join(["mean_abs_action", *ACTION_DIM_NAME_TO_INDEX.keys(), "0..6"])
        raise ValueError(f"Unsupported CAM target {target!r}; expected one of: {valid}") from exc
    if dim < 0 or dim > 6:
        raise ValueError(f"Action dimension index must be in [0, 6], got {dim}.")
    return dim


def select_action_target_value(actions: np.ndarray, target: str, action_step: int) -> float:
    actions = np.asarray(actions, dtype=np.float32)
    if target == "mean_abs_action":
        return float(np.abs(actions).mean())

    dim = parse_action_dim(target)
    if action_step >= 0:
        if action_step >= actions.shape[0]:
            raise ValueError(f"action_step={action_step} exceeds action chunk length {actions.shape[0]}.")
        return float(abs(actions[action_step, dim]))
    return float(np.abs(actions[:, dim]).mean())


def select_action_target_tensor(actions, target: str, action_step: int):
    if target == "mean_abs_action":
        return actions.abs().mean()

    dim = parse_action_dim(target)
    if action_step >= 0:
        if action_step >= actions.shape[0]:
            raise ValueError(f"action_step={action_step} exceeds action chunk length {actions.shape[0]}.")
        return actions[action_step, dim].abs()
    return actions[:, dim].abs().mean()


def gradcam_vector_from_features(features: np.ndarray, gradients: np.ndarray) -> np.ndarray:
    if features.shape != gradients.shape:
        raise ValueError(f"features shape {features.shape} does not match gradients shape {gradients.shape}.")
    weights = gradients.astype(np.float32, copy=False).mean(axis=0)
    cam = (features.astype(np.float32, copy=False) * weights[None, :]).sum(axis=1)
    return np.maximum(cam, 0.0).astype(np.float32)




def grad_x_activation_vector_from_features(features: np.ndarray, gradients: np.ndarray) -> np.ndarray:
    if features.shape != gradients.shape:
        raise ValueError(f"features shape {features.shape} does not match gradients shape {gradients.shape}.")
    relevance = np.abs(features.astype(np.float32, copy=False) * gradients.astype(np.float32, copy=False))
    return relevance.sum(axis=1).astype(np.float32)




def cam_vector_from_features(features: np.ndarray, gradients: np.ndarray, method: str) -> np.ndarray:
    if method == "gradcam":
        return gradcam_vector_from_features(features, gradients)
    if method == "grad_x_activation":
        return grad_x_activation_vector_from_features(features, gradients)
    raise ValueError(f"Unsupported CAM method for gradient extraction: {method}")


def attention_vector_to_heatmap(attention: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    grid_h, grid_w = infer_patch_grid(int(attention.size))
    heatmap = normalize_unit(attention.reshape(grid_h, grid_w))
    image = Image.fromarray(np.uint8(heatmap * 255), mode="L")
    image = image.resize((output_size[1], output_size[0]), resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def attention_vector_to_heatmap_with_range(
    attention: np.ndarray, output_size: tuple[int, int], value_min: float, value_max: float
) -> np.ndarray:
    grid_h, grid_w = infer_patch_grid(int(attention.size))
    values = attention.astype(np.float32, copy=False).reshape(grid_h, grid_w)
    if value_max - value_min < 1e-12:
        heatmap = np.zeros_like(values, dtype=np.float32)
    else:
        heatmap = np.clip((values - value_min) / (value_max - value_min), 0.0, 1.0).astype(np.float32)
    image = Image.fromarray(np.uint8(heatmap * 255), mode="L")
    image = image.resize((output_size[1], output_size[0]), resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def signed_delta_to_rgb(delta: np.ndarray) -> np.ndarray:
    from matplotlib import cm

    delta = delta.astype(np.float32, copy=False)
    scale = float(np.max(np.abs(delta)))
    if scale < 1e-12:
        normed = np.zeros_like(delta, dtype=np.float32) + 0.5
    else:
        normed = np.clip(delta / (2.0 * scale) + 0.5, 0.0, 1.0).astype(np.float32)
    rgb = cm.get_cmap("coolwarm")(normed)[..., :3] * 255.0
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def signed_delta_vector_to_rgb(delta: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    grid_h, grid_w = infer_patch_grid(int(delta.size))
    delta_grid = delta.astype(np.float32, copy=False).reshape(grid_h, grid_w)
    image = Image.fromarray(signed_delta_to_rgb(delta_grid))
    image = image.resize((output_size[1], output_size[0]), resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def overlay_heatmap_on_image(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}.")
    if heatmap.shape != image.shape[:2]:
        raise ValueError(f"Heatmap shape {heatmap.shape} does not match image shape {image.shape[:2]}.")

    from matplotlib import cm

    heatmap = normalize_unit(heatmap)
    color = cm.get_cmap("magma")(heatmap)[..., :3]
    overlay = (1.0 - alpha) * image.astype(np.float32) + alpha * (color * 255.0)
    return np.clip(np.round(overlay), 0, 255).astype(np.uint8)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def build_prompt(task_description: str, use_minivlm: bool) -> str:
    task = task_description.lower()
    if not use_minivlm:
        return f"In: What action should the robot take to {task}?\nOut:"
    return (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"What action should the robot take to {task}?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def source_layer_name(cam_source: str, layer_idx: int) -> str:
    if cam_source in {"vision_backbone", "projected_patch"}:
        return cam_source
    return f"layer_{layer_idx:02d}"


def resolve_unnorm_key(unnorm_key: str, norm_stats: dict[str, Any]) -> str:
    if unnorm_key in norm_stats:
        return unnorm_key
    no_noops_key = f"{unnorm_key}_no_noops"
    if no_noops_key in norm_stats:
        return no_noops_key
    available = ", ".join(sorted(norm_stats.keys()))
    raise KeyError(f"Action/proprio stats key {unnorm_key!r} not found. Available keys: {available}")


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    label, path = value.split("=", 1)
    return label, Path(path)


def make_cfg(args: argparse.Namespace, checkpoint: Path) -> SimpleNamespace:
    return SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=str(checkpoint),
        use_l1_regression=True,
        use_minivlm=args.use_minivlm,
        use_film=args.use_film,
        num_images_in_input=args.num_images_in_input,
        use_proprio=args.use_proprio,
        center_crop=args.center_crop,
        num_open_loop_steps=8,
        unnorm_key=args.unnorm_key or args.task_suite_name,
        load_in_8bit=False,
        load_in_4bit=False,
        task_suite_name=args.task_suite_name,
        env_img_res=args.env_img_res,
        seed=args.seed,
        save_version="attention-heatmaps",
        use_pro_version=args.use_pro_version,
        use_gate=args.use_gate,
        use_latent_outputs=args.use_latent_outputs,
        phase="Inference",
    )


def make_observation_bundle_from_libero_obs(env_obs, task_description: str, args: argparse.Namespace) -> ObservationBundle:
    from experiments.robot.libero.libero_utils import get_libero_image, get_libero_wrist_image, quat2axisangle
    from experiments.robot.openvla_utils import prepare_images_for_vla, resize_image_for_policy

    raw_primary = get_libero_image(env_obs)
    raw_wrist = get_libero_wrist_image(env_obs)
    resized_primary = resize_image_for_policy(raw_primary, 224)
    resized_wrist = resize_image_for_policy(raw_wrist, 224)
    state = np.concatenate(
        (env_obs["robot0_eef_pos"], quat2axisangle(env_obs["robot0_eef_quat"]), env_obs["robot0_gripper_qpos"])
    )
    prep_cfg = SimpleNamespace(center_crop=args.center_crop)
    policy_images = prepare_images_for_vla([resized_primary, resized_wrist], prep_cfg)
    prompt = build_prompt(task_description, args.use_minivlm)
    return ObservationBundle(
        task_description=task_description,
        prompt=prompt,
        raw_primary=raw_primary,
        raw_wrist=raw_wrist,
        policy_primary=np.asarray(policy_images[0], dtype=np.uint8),
        policy_wrist=np.asarray(policy_images[1], dtype=np.uint8),
        state=state.astype(np.float32),
    )


def process_openvla_env_action(action: np.ndarray) -> np.ndarray:
    from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action

    action = normalize_gripper_action(action, binarize=True)
    return invert_gripper_action(action)


def advance_with_policy(env, obs, task_description: str, args: argparse.Namespace):
    if args.capture_policy_steps <= 0:
        return obs

    from experiments.robot.openvla_utils import get_vla_action

    capture_ckpt = Path(args.capture_policy_checkpoint or DEFAULT_CL_CHECKPOINT)
    cfg = make_cfg(args, capture_ckpt)
    model, processor, device = load_vla_readonly(cfg)
    action_head, proprio_projector = load_local_action_components(cfg, model, device)
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    try:
        for _ in range(args.capture_policy_steps):
            if not action_queue:
                bundle = make_observation_bundle_from_libero_obs(obs, task_description, args)
                policy_obs = {
                    "full_image": bundle.policy_primary.copy(),
                    "wrist_image": bundle.policy_wrist.copy() if bundle.policy_wrist is not None else None,
                    "state": bundle.state.copy(),
                }
                actions = get_vla_action(
                    cfg=cfg,
                    vla=model,
                    processor=processor,
                    obs=policy_obs,
                    task_label=task_description,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=None,
                    use_film=args.use_film,
                    use_minivlm=args.use_minivlm,
                )
                action_queue.extend(actions)
            action = process_openvla_env_action(action_queue.popleft())
            obs, _, done, _ = env.step(action.tolist())
            if done:
                break
        return obs
    finally:
        del model
        del processor
        del action_head
        del proprio_projector
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def capture_observation(args: argparse.Namespace) -> ObservationBundle:
    from libero.libero import benchmark

    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
    )
    from experiments.robot.robot_utils import set_seed_everywhere

    set_seed_everywhere(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    if args.episode_idx >= len(initial_states):
        raise ValueError(f"episode_idx={args.episode_idx} exceeds {len(initial_states)} initial states.")

    env, task_description = get_libero_env(task, "openvla", resolution=args.env_img_res)
    try:
        env.reset()
        obs = env.set_init_state(initial_states[args.episode_idx])
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))

        obs = advance_with_policy(env, obs, task_description, args)
        return make_observation_bundle_from_libero_obs(obs, task_description, args)
    finally:
        env.close()


def load_vla_readonly(cfg: SimpleNamespace):
    import torch
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    kwargs: dict[str, Any] = dict(
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        trust_remote_code=False,
    )
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            cfg.pretrained_checkpoint,
            attn_implementation="eager",
            **kwargs,
        )
    except TypeError:
        model = AutoModelForVision2Seq.from_pretrained(cfg.pretrained_checkpoint, **kwargs)

    model.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    model.eval()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    stats_path = Path(cfg.pretrained_checkpoint) / "dataset_statistics.json"
    if stats_path.is_file():
        with stats_path.open("r", encoding="utf-8") as f:
            model.norm_stats = json.load(f)
        cfg.unnorm_key = resolve_unnorm_key(cfg.unnorm_key, model.norm_stats)
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=False)
    return model, processor, device


def find_local_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    checkpoint_dir = Path(pretrained_checkpoint)
    matches = sorted(
        str(path)
        for path in checkpoint_dir.iterdir()
        if path.is_file() and file_pattern in path.name and "checkpoint" in path.name
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {file_pattern} checkpoint in {checkpoint_dir}, found {len(matches)}."
        )
    return matches[0]


def load_component_state_dict_local(checkpoint_path: str):
    import torch

    state_dict = torch.load(checkpoint_path, weights_only=True)
    return {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def load_local_action_components(cfg: SimpleNamespace, model, device):
    import torch

    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from prismatic.vla.constants import ACTION_DIM

    action_head = L1RegressionActionHead(
        input_dim=model.llm_dim,
        hidden_dim=model.llm_dim,
        action_dim=ACTION_DIM,
        use_pro_version=cfg.use_pro_version,
        use_gate=cfg.use_gate,
        use_latent_outputs=cfg.use_latent_outputs,
    ).to(device=device, dtype=torch.bfloat16)
    action_head.load_state_dict(
        load_component_state_dict_local(find_local_checkpoint_file(cfg.pretrained_checkpoint, "action_head")),
        strict=False,
    )
    action_head.eval()

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = ProprioProjector(llm_dim=model.llm_dim, proprio_dim=8).to(
            device=device, dtype=torch.bfloat16
        )
        proprio_projector.load_state_dict(
            load_component_state_dict_local(find_local_checkpoint_file(cfg.pretrained_checkpoint, "proprio_projector"))
        )
        proprio_projector.eval()

    return action_head, proprio_projector


def build_model_inputs(processor, model, device, obs: ObservationBundle, args: argparse.Namespace):
    import torch

    inputs = processor(obs.prompt, Image.fromarray(obs.policy_primary).convert("RGB")).to(device, dtype=torch.bfloat16)
    if args.num_images_in_input > 1 and obs.policy_wrist is not None:
        wrist_inputs = processor(obs.prompt, Image.fromarray(obs.policy_wrist).convert("RGB")).to(
            device, dtype=torch.bfloat16
        )
        inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1)

    labels = inputs["input_ids"].clone()
    labels[:] = IGNORE_INDEX
    input_ids, attention_mask = model._prepare_input_for_action_prediction(inputs["input_ids"], inputs["attention_mask"])
    labels = model._prepare_labels_for_action_prediction(labels, input_ids)
    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = attention_mask
    return inputs, labels


def extract_attention_vectors(model, processor, device, obs: ObservationBundle, args: argparse.Namespace) -> dict[str, Any]:
    import torch

    inputs, labels = build_model_inputs(processor, model, device, obs, args)
    original_prompt_len = int(processor(obs.prompt, Image.fromarray(obs.policy_primary).convert("RGB"))["input_ids"].shape[-1])
    patches_per_image = int(model.vision_backbone.get_num_patches())
    num_images = int(model.vision_backbone.get_num_images_in_input())
    total_visual_patches = patches_per_image * num_images
    visual_start = 1
    visual_end = visual_start + total_visual_patches
    action_start = original_prompt_len + total_visual_patches
    action_end = action_start + int(model.action_queries.weight.shape[0])

    with torch.inference_mode():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            labels=labels,
            output_attentions=True,
            return_dict=True,
            use_film=args.use_film,
        )

    if outputs.attentions is None:
        raise RuntimeError("Model returned no attentions. Try a Transformers build that supports eager attentions.")

    layers = []
    for layer_idx, attention in enumerate(outputs.attentions):
        layer_attention = attention[0, :, action_start:action_end, visual_start:visual_end]
        vector = layer_attention.float().mean(dim=(0, 1)).detach().cpu().numpy()
        image_vectors = [
            vector[i * patches_per_image : (i + 1) * patches_per_image].astype(np.float32)
            for i in range(num_images)
        ]
        layers.append({"layer": layer_idx, "vectors": image_vectors})

    return {
        "layers": layers,
        "patches_per_image": patches_per_image,
        "num_images": num_images,
        "total_visual_patches": total_visual_patches,
        "original_prompt_len": original_prompt_len,
        "action_query_count": int(model.action_queries.weight.shape[0]),
    }


def predict_normalized_actions_and_visual_states(
    model,
    inputs: dict[str, Any],
    labels,
    action_head,
    proprio,
    proprio_projector,
    args: argparse.Namespace,
):
    import torch

    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, NUM_TOKENS

    input_embeddings = model.get_input_embeddings()(inputs["input_ids"])
    all_actions_mask = model._process_action_masks(labels)
    language_embeddings = input_embeddings[~all_actions_mask].reshape(
        input_embeddings.shape[0], -1, input_embeddings.shape[2]
    )

    raw_visual_features = None
    projected_patch_embeddings = None
    if args.cam_source == "vision_backbone":
        if args.use_film:
            raw_visual_features = model.vision_backbone(inputs["pixel_values"], language_embeddings)
        else:
            raw_visual_features = model.vision_backbone(inputs["pixel_values"])
        raw_visual_features = raw_visual_features.detach().requires_grad_(True)
        projected_patch_embeddings = model.projector(raw_visual_features)
    else:
        projected_patch_embeddings = model._process_vision_features(
            inputs["pixel_values"], language_embeddings, args.use_film
        )
        if args.cam_source == "projected_patch":
            projected_patch_embeddings = projected_patch_embeddings.detach().requires_grad_(True)

    action_queries = model.action_queries.weight
    action_queries = action_queries.view(1, action_queries.shape[0], action_queries.shape[1]).repeat(
        input_embeddings.shape[0], 1, 1
    )
    input_embeddings = model._replace_input_embeddings(input_embeddings.clone(), all_actions_mask, action_queries)
    multimodal_embeddings, multimodal_attention_mask = model._build_multimodal_attention(
        input_embeddings, projected_patch_embeddings, inputs["attention_mask"]
    )

    language_model_output = model.language_model(
        input_ids=None,
        attention_mask=multimodal_attention_mask,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=multimodal_embeddings,
        labels=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    )

    num_patches = model.vision_backbone.get_num_patches() * model.vision_backbone.get_num_images_in_input()
    num_prompt_tokens = inputs["input_ids"].shape[-1] - NUM_TOKENS - 2

    mixed_layers = []
    prompt_layers = []
    visual_layers = []
    action_layers = []
    visual_tensors_for_cam = []
    for item in language_model_output.hidden_states:
        batch_size = item.shape[0]
        vis_start = 1
        vis_end = vis_start + num_patches
        prompt_start = vis_end
        prompt_end = prompt_start + num_prompt_tokens
        act_start = prompt_end
        act_end = act_start + NUM_TOKENS

        visual_hidden_states = item[:, vis_start:vis_end, :].reshape(batch_size, 1, num_patches, -1).to(torch.bfloat16)
        visual_hidden_states.retain_grad()
        action_hidden_states = item[:, act_start:act_end, :].reshape(batch_size, 1, NUM_TOKENS, -1).to(torch.bfloat16)
        prompt_hidden_states = item[:, prompt_start:prompt_end, :].reshape(batch_size, 1, num_prompt_tokens, -1).to(
            torch.bfloat16
        )
        mixed_layers.append(torch.cat((visual_hidden_states, action_hidden_states), dim=2))
        visual_layers.append(visual_hidden_states)
        action_layers.append(action_hidden_states)
        prompt_layers.append(prompt_hidden_states)
        visual_tensors_for_cam.append(visual_hidden_states)

    multi_layer_hidden_states = torch.cat(mixed_layers, dim=1)
    multi_layer_prompt_states = torch.cat(prompt_layers, dim=1)
    multi_layer_visual_states = torch.cat(visual_layers, dim=1)
    multi_layer_action_states = torch.cat(action_layers, dim=1)

    normalized_actions = action_head.predict_action(
        multi_layer_hidden_states,
        proprio=proprio,
        proprio_projector=proprio_projector,
        latent_outputs=language_model_output.hidden_states[-1],
        multi_layer_prompt_states=multi_layer_prompt_states,
        multi_layer_visual_states=multi_layer_visual_states,
        multi_layer_action_states=multi_layer_action_states,
    )
    normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM).float()
    if args.cam_source == "vision_backbone":
        return normalized_actions, [raw_visual_features]
    if args.cam_source == "projected_patch":
        return normalized_actions, [projected_patch_embeddings]
    return normalized_actions, visual_tensors_for_cam


def extract_gradcam_vectors(
    model,
    processor,
    device,
    obs: ObservationBundle,
    args: argparse.Namespace,
    cfg: SimpleNamespace,
    action_head,
    proprio_projector,
) -> dict[str, Any]:
    import torch

    from experiments.robot.openvla_utils import normalize_proprio

    model.requires_grad_(False)
    action_head.requires_grad_(False)
    if proprio_projector is not None:
        proprio_projector.requires_grad_(False)

    inputs, labels = build_model_inputs(processor, model, device, obs, args)
    patches_per_image = int(model.vision_backbone.get_num_patches())
    num_images = int(model.vision_backbone.get_num_images_in_input())
    total_visual_patches = patches_per_image * num_images

    proprio = None
    if cfg.use_proprio:
        proprio_stats = model.norm_stats[cfg.unnorm_key]["proprio"]
        proprio_np = normalize_proprio(obs.state.copy(), proprio_stats)
        proprio = torch.tensor(proprio_np, device=device, dtype=torch.bfloat16)

    model.zero_grad(set_to_none=True)
    action_head.zero_grad(set_to_none=True)
    if proprio_projector is not None:
        proprio_projector.zero_grad(set_to_none=True)

    normalized_actions, visual_tensors = predict_normalized_actions_and_visual_states(
        model, inputs, labels, action_head, proprio, proprio_projector, args
    )
    target = select_action_target_tensor(normalized_actions, args.cam_target, args.action_step)
    target.backward()

    layers = []
    if args.cam_source in {"vision_backbone", "projected_patch"}:
        tensors_for_output = [(0, visual_tensors[0])]
    else:
        # hidden_states[0] is the embedding output; skip it so layer_00 corresponds to transformer layer 0.
        tensors_for_output = list(enumerate(visual_tensors[1:]))

    for layer_idx, visual_tensor in tensors_for_output:
        if visual_tensor.grad is None:
            raise RuntimeError(f"Missing gradient for visual source {args.cam_source} layer {layer_idx}.")
        if visual_tensor.ndim == 4:
            features = visual_tensor.detach()[0, 0].float().cpu().numpy()
            gradients = visual_tensor.grad.detach()[0, 0].float().cpu().numpy()
        else:
            features = visual_tensor.detach()[0].float().cpu().numpy()
            gradients = visual_tensor.grad.detach()[0].float().cpu().numpy()
        vector = cam_vector_from_features(features, gradients, args.cam_method)
        image_vectors = [
            vector[i * patches_per_image : (i + 1) * patches_per_image].astype(np.float32)
            for i in range(num_images)
        ]
        layers.append({"layer": layer_idx, "layer_name": source_layer_name(args.cam_source, layer_idx), "vectors": image_vectors})

    actions_np = normalized_actions.detach().cpu().numpy().astype(np.float32)
    return {
        "method": args.cam_method,
        "layers": layers,
        "patches_per_image": patches_per_image,
        "num_images": num_images,
        "total_visual_patches": total_visual_patches,
        "original_prompt_len": int(processor(obs.prompt, Image.fromarray(obs.policy_primary).convert("RGB"))["input_ids"].shape[-1]),
        "action_query_count": int(model.action_queries.weight.shape[0]),
        "cam_source": args.cam_source,
        "cam_target": args.cam_target,
        "action_step": args.action_step,
        "target_value": float(target.detach().cpu().item()),
        "normalized_actions": actions_np.tolist(),
        "selected_target_value_check": select_action_target_value(actions_np, args.cam_target, args.action_step),
    }


def save_checkpoint_heatmaps(
    label: str,
    checkpoint_dir: Path,
    attention_data: dict[str, Any],
    obs: ObservationBundle,
    output_dir: Path,
    alpha: float,
) -> dict[str, Any]:
    method = attention_data.get("method", "attention")
    ckpt_out = output_dir / "checkpoints" / label
    ckpt_out.mkdir(parents=True, exist_ok=True)
    base_images = [obs.policy_primary]
    view_names = ["primary"]
    if attention_data["num_images"] > 1 and obs.policy_wrist is not None:
        base_images.append(obs.policy_wrist)
        view_names.append("wrist")

    layer_records = []
    for layer in attention_data["layers"]:
        layer_idx = int(layer["layer"])
        record = {"layer": layer_idx, "views": {}}
        for view_idx, (view_name, base_image) in enumerate(zip(view_names, base_images)):
            vector = layer["vectors"][view_idx]
            heatmap = attention_vector_to_heatmap(vector, output_size=base_image.shape[:2])
            overlay = overlay_heatmap_on_image(base_image, heatmap, alpha=alpha)
            heatmap_png = ckpt_out / f"layer_{layer_idx:02d}_{view_name}_heatmap.png"
            overlay_png = ckpt_out / f"layer_{layer_idx:02d}_{view_name}_overlay.png"
            npy_path = ckpt_out / f"layer_{layer_idx:02d}_{view_name}_{method}.npy"
            save_image(heatmap_png, np.uint8(np.round(heatmap * 255)))
            save_image(overlay_png, overlay)
            np.save(npy_path, vector)
            record["views"][view_name] = {
                "heatmap_png": str(heatmap_png),
                "overlay_png": str(overlay_png),
                f"{method}_npy": str(npy_path),
                "vector_sum": float(vector.sum()),
                "vector_max": float(vector.max()),
            }
        layer_records.append(record)

    summary = {
        "label": label,
        "checkpoint": str(checkpoint_dir),
        "method": method,
        "cam_source": attention_data.get("cam_source"),
        "cam_target": attention_data.get("cam_target"),
        "action_step": attention_data.get("action_step"),
        "target_value": attention_data.get("target_value"),
        "normalized_actions": attention_data.get("normalized_actions"),
        "patches_per_image": attention_data["patches_per_image"],
        "num_images": attention_data["num_images"],
        "total_visual_patches": attention_data["total_visual_patches"],
        "original_prompt_len": attention_data["original_prompt_len"],
        "action_query_count": attention_data["action_query_count"],
        "num_layers": len(layer_records),
        "layers": layer_records,
    }
    with (ckpt_out / "attention_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def add_title(image: Image.Image, title: str) -> Image.Image:
    title_h = 28
    canvas = Image.new("RGB", (image.width, image.height + title_h), "white")
    canvas.paste(image, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill=(0, 0, 0))
    return canvas


def save_pairwise_comparisons(summaries: list[dict[str, Any]], output_dir: Path) -> None:
    if len(summaries) < 2:
        return
    comp_dir = output_dir / "comparisons"
    shared_dir = output_dir / "comparisons_shared_scale"
    diff_dir = output_dir / "differences"
    comp_dir.mkdir(parents=True, exist_ok=True)
    shared_dir.mkdir(parents=True, exist_ok=True)
    diff_dir.mkdir(parents=True, exist_ok=True)
    first, second = summaries[0], summaries[1]
    method = first.get("method") or "attention"
    layer_count = min(first["num_layers"], second["num_layers"])
    metric_rows = []

    for layer_idx in range(layer_count):
        for view_name in first["layers"][layer_idx]["views"].keys():
            if view_name not in second["layers"][layer_idx]["views"]:
                continue
            left_path = Path(first["layers"][layer_idx]["views"][view_name]["overlay_png"])
            right_path = Path(second["layers"][layer_idx]["views"][view_name]["overlay_png"])
            left = add_title(Image.open(left_path).convert("RGB"), first["label"])
            right = add_title(Image.open(right_path).convert("RGB"), second["label"])
            canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), "white")
            canvas.paste(left, (0, 0))
            canvas.paste(right, (left.width, 0))
            canvas.save(comp_dir / f"layer_{layer_idx:02d}_{view_name}_compare.png")

            left_npy = Path(first["layers"][layer_idx]["views"][view_name][f"{method}_npy"])
            right_npy = Path(second["layers"][layer_idx]["views"][view_name][f"{method}_npy"])
            left_vec = np.load(left_npy).astype(np.float32)
            right_vec = np.load(right_npy).astype(np.float32)
            value_min = float(min(left_vec.min(), right_vec.min()))
            value_max = float(max(left_vec.max(), right_vec.max()))

            base_path = output_dir / "observation" / f"policy_{view_name}.png"
            base_image = np.asarray(Image.open(base_path).convert("RGB"))
            left_shared = overlay_heatmap_on_image(
                base_image, attention_vector_to_heatmap_with_range(left_vec, base_image.shape[:2], value_min, value_max)
            )
            right_shared = overlay_heatmap_on_image(
                base_image, attention_vector_to_heatmap_with_range(right_vec, base_image.shape[:2], value_min, value_max)
            )
            left_shared_img = add_title(Image.fromarray(left_shared), f"{first['label']} shared scale")
            right_shared_img = add_title(Image.fromarray(right_shared), f"{second['label']} shared scale")
            shared_canvas = Image.new(
                "RGB",
                (left_shared_img.width + right_shared_img.width, max(left_shared_img.height, right_shared_img.height)),
                "white",
            )
            shared_canvas.paste(left_shared_img, (0, 0))
            shared_canvas.paste(right_shared_img, (left_shared_img.width, 0))
            shared_canvas.save(shared_dir / f"layer_{layer_idx:02d}_{view_name}_shared_compare.png")

            left_norm = normalize_unit(left_vec)
            right_norm = normalize_unit(right_vec)
            delta = left_norm - right_norm
            delta_rgb = signed_delta_vector_to_rgb(delta, base_image.shape[:2])
            delta_img = add_title(Image.fromarray(delta_rgb), f"delta: {first['label']} - {second['label']}")
            diff_canvas = Image.new("RGB", (shared_canvas.width + delta_img.width, max(shared_canvas.height, delta_img.height)), "white")
            diff_canvas.paste(shared_canvas, (0, 0))
            diff_canvas.paste(delta_img, (shared_canvas.width, 0))
            diff_canvas.save(diff_dir / f"layer_{layer_idx:02d}_{view_name}_delta_compare.png")

            if left_norm.std() > 0 and right_norm.std() > 0:
                corr = float(np.corrcoef(left_norm, right_norm)[0, 1])
            else:
                corr = float("nan")
            metric_rows.append(
                {
                    "layer": layer_idx,
                    "view": view_name,
                    "corr": corr,
                    "mean_abs_norm_diff": float(np.mean(np.abs(delta))),
                    "max_abs_norm_diff": float(np.max(np.abs(delta))),
                    "left_max": float(left_vec.max()),
                    "right_max": float(right_vec.max()),
                }
            )

    with (diff_dir / "difference_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metric_rows, f, indent=2)


def write_observation_artifacts(obs: ObservationBundle, args: argparse.Namespace, output_dir: Path) -> None:
    image_dir = output_dir / "observation"
    save_image(image_dir / "raw_primary.png", obs.raw_primary)
    save_image(image_dir / "raw_wrist.png", obs.raw_wrist)
    save_image(image_dir / "policy_primary.png", obs.policy_primary)
    if obs.policy_wrist is not None:
        save_image(image_dir / "policy_wrist.png", obs.policy_wrist)
    np.savez_compressed(
        image_dir / "observation_arrays.npz",
        raw_primary=obs.raw_primary,
        raw_wrist=obs.raw_wrist,
        policy_primary=obs.policy_primary,
        policy_wrist=obs.policy_wrist,
        state=obs.state,
    )
    metadata = {
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "episode_idx": args.episode_idx,
        "seed": args.seed,
        "num_steps_wait": args.num_steps_wait,
        "capture_policy_steps": args.capture_policy_steps,
        "capture_policy_checkpoint": args.capture_policy_checkpoint,
        "env_img_res": args.env_img_res,
        "center_crop": args.center_crop,
        "num_images_in_input": args.num_images_in_input,
        "task_description": obs.task_description,
        "prompt": obs.prompt,
        "state": obs.state.tolist(),
        "cam_method": args.cam_method,
        "cam_source": args.cam_source,
        "cam_target": args.cam_target,
        "action_step": args.action_step,
        "overlay_note": "Overlays are drawn on policy_* images after resize/center-crop, matching model input.",
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=None, help="LABEL=/path/to/checkpoint")
    parser.add_argument("--output_dir", default="experiments/libero_attention_heatmaps/latest")
    parser.add_argument("--task_suite_name", default="libero_object")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--capture_policy_steps", type=int, default=0, help="Advance this many env steps with capture_policy_checkpoint before saving the observation.")
    parser.add_argument("--capture_policy_checkpoint", default=DEFAULT_CL_CHECKPOINT, help="Checkpoint used only to roll out to the capture observation.")
    parser.add_argument("--env_img_res", type=int, default=256)
    parser.add_argument("--num_images_in_input", type=int, default=2)
    parser.add_argument("--unnorm_key", default="")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--cam_method", choices=["gradcam", "grad_x_activation", "attention"], default="gradcam")
    parser.add_argument("--cam_source", choices=["vlm_layers", "projected_patch", "vision_backbone"], default="vlm_layers")
    parser.add_argument("--cam_target", default="mean_abs_action", help="mean_abs_action or action dim: x/y/z/roll/pitch/yaw/gripper/0..6")
    parser.add_argument("--action_step", type=int, default=-1, help="Action chunk index for dim targets; -1 means average over chunk.")
    parser.add_argument("--center_crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_minivlm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_proprio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_film", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_pro_version", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_latent_outputs", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    checkpoint_specs = args.checkpoint or [
        f"contrastive={DEFAULT_CL_CHECKPOINT}",
        f"no_contrastive={DEFAULT_NO_CL_CHECKPOINT}",
    ]
    checkpoints = [parse_checkpoint(value) for value in checkpoint_specs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    obs = capture_observation(args)
    write_observation_artifacts(obs, args, output_dir)

    summaries = []
    for label, checkpoint_dir in checkpoints:
        cfg = make_cfg(args, checkpoint_dir)
        model, processor, device = load_vla_readonly(cfg)
        try:
            if args.cam_method == "attention":
                heatmap_data = extract_attention_vectors(model, processor, device, obs, args)
            else:
                action_head, proprio_projector = load_local_action_components(cfg, model, device)
                try:
                    heatmap_data = extract_gradcam_vectors(
                        model, processor, device, obs, args, cfg, action_head, proprio_projector
                    )
                finally:
                    del action_head
                    del proprio_projector
            summaries.append(save_checkpoint_heatmaps(label, checkpoint_dir, heatmap_data, obs, output_dir, args.alpha))
        finally:
            del model
            del processor
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    save_pairwise_comparisons(summaries, output_dir)
    with (output_dir / "checkpoint_summaries.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote LIBERO {args.cam_method} heatmaps to: {output_dir}")


if __name__ == "__main__":
    main()
