"""
finetune.py

Fine-tunes Qwen2.5-0.5B via LoRA.





"""

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type
import torch.nn.functional as F
import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
import wandb
import swanlab

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask
)
from prismatic.training.resume_finetune import (
    infer_resume_finetune_step,
    load_component_state_dict,
    resolve_resume_finetune_path,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    NUM_TOKENS
)
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.models import load, load_vla


# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class FinetuneConfig:
    # fmt: off
    config_file_path: str = "openvla/openvla-7b"     # Path to necessary config files of LA-Adapter
    vlm_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)
    use_minivlm: bool = False                        # 
    resum_vla_path: str = "openvla/openvla-7b"       # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 50_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur) default: 100_000

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for training 
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input
    phase1_path: str = "None"
    use_contrastive_loss: bool = False               # TODO If True, adds contrastive loss 
    contrastive_weight: float = 0.1                  # Weight for contrastive loss (relative to action loss)
    action_alignment_strategy: str = "original"      # original | improved
    action_use_manifold_regularization: bool = False # If True, adds local manifold regularization in the improved path
    action_manifold_topk: int = 4                    # Number of local neighbors used by manifold regularization
    action_manifold_temperature: float = 0.2         # Temperature for local neighborhood graph weights
    action_learnable_alignment_weights: bool = False # If True, learns the relative weights of alignment terms
    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0.1                       # Number of steps to warm up learning rate (from 10 percent to full LR)
    num_steps_before_decay: int = 100000             # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200000                          # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    resume_finetune: bool = False                    # If True, warm-starts a new finetune run from a completed checkpoint
    resume_finetune_path: Optional[Path] = None      # Completed VLA-Adapter checkpoint directory for warm-start finetune
    resume_finetune_step: Optional[int] = None       # Optional component step; inferred from checkpoint path if omitted
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

    # LoRA
    use_lora: bool = False                           # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = False         # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!

    # Full Finetune
    use_fz: bool = False                             # If True, uses LoRA fine-tuning

    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: str = ""                            # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # revision version
    use_pro_version: bool = True                             # the version number
    use_gate: bool = False              # whether to use gated fusion
    phase: str = "Training"
    use_latent_outputs: bool = False                    # whether to use the last layer latent outputs from VLM
    # fmt: on
    
    #训练数据的格式
    dataset_type: str = "rlds" # hdf5
    num_workers: int = 1
    epoch_num: int = 1000
    num_samples_per_traj: int = 4 # choose number of traj in single hdf5



def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict



def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_note:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_note
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.config_file_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.config_file_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_fz:
            run_id += f"+frozen+dropout-{cfg.lora_dropout}"
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
    return run_id



def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    Loads a checkpoint for a given module.

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)



def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wrap a module with DistributedDataParallel.

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)



def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        None.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    print(f"# trainable params in {name}: {num_params}")



def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device_id (str): Device ID.
        module_args (dict): Args for initializing the module.
        to_bf16 (bool): Whether to convert to torch.bfloat16 data type.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume_finetune:
        state_dict = load_component_state_dict(
            cfg.resume_finetune_path,
            module_name,
            cfg.resume_finetune_step,
        )
        if state_dict is not None:
            module.load_state_dict(remove_ddp_in_checkpoint(state_dict))
            print(f"Loaded resume-finetune `{module_name}` weights.")
        else:
            print(f"No resume-finetune `{module_name}` checkpoint found; using fresh initialization.")
    elif cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.resum_vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)
        print('loaded!!!!!!!!!')

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)



def run_forward_pass(
    vla,
    action_head,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_proprio,
    use_film,
    num_patches,
    compute_diffusion_l1=False,
    use_pro_version=True,
    cfg=None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        use_l1_regression (bool): Whether to use L1 regression.
        use_diffusion (bool): Whether to use diffusion.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.
        num_patches (int): Number of vision patches.
        compute_diffusion_l1 (bool): Whether to sample actions and compute L1 loss for diffusion (do this once every
                                    diffusion_sample_freq steps during training; do it every batch for validation)
        num_diffusion_steps (int): Number of diffusion steps (only used for diffusion).

    Returns:
        tuple: (loss, metrics_dict)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
    """
    metrics = {}

    # Get ground-truth action labels
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)
    noise, noisy_actions, diffusion_timestep_embeddings = None, None, None #暂时不用扩散模型

    # VLA forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        #CausalLM 输出 就是因果语言模型一次前向传播返回的所有结果结构体
        #e.g.loss 
        #logits:代码中用 output.logits[:, num_patches:-1].argmax(dim=2) 取出动作部分的预测 token
        #hidden_states:所有 transformer 层的隐藏状态列表 (B, seq_len, hidden_dim)
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id), #task text tokens + dummy action tokens
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=use_film,
            )

    # Get action masks needed for logging
    ground_truth_token_ids = batch["labels"][:,1:].to(device_id) #已经排除了BOS
    current_action_mask = get_current_action_mask(ground_truth_token_ids) # 当前步动作的位置
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids) # 未来步动作的位置

    # Compute metrics for discrete action representation (next-token prediction)
    if not (use_l1_regression): #离散动作预测模式 CE loss
        loss = output.loss #此时loss 为 LLM输出action token的交叉熵损失
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2) #取出动作部分的预测的概率最大的 token IDs
        #FIXME 计算loss
        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=current_action_mask
            )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, 
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=current_action_mask
            )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=next_actions_mask
            )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, 
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=next_actions_mask
            )
        
        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )
        
    # Compute metrics for continuous action representations (L1 regression)
    else:
        #初始化空列表，遍历所有层的 hidden states（output.hidden_states 是 tuple，每项 [B, seq_len, D]）
# 初始化列表（与推理阶段保持一致）
        multi_layer_hidden_states = [] # 存放 Visual + Action 混合
        multi_layer_visual_states = []
        multi_layer_prompt_states = []
        multi_layer_action_states = []

        # 获取 batch_size
        batch_size = batch["input_ids"].shape[0]
        
        # 遍历每一层
        for item in output.hidden_states:
            # item shape: [B, Seq_Len, D]
            
            # === 1. 提取 Visual 部分 (Visual Patches) ===
            # 假设布局: [BOS(1)] + [Visual(num_patches)] + [Text...]
            # 切片: 从 1 到 num_patches + 1
            visual_hidden_states = item[:, 1 : num_patches + 1].reshape(batch_size, 1, num_patches, -1)
            
            # === 2. 提取 Text + Action 混合部分 ===
            # 从 Visual 之后开始截取
            text_and_action_hidden = item[:, num_patches + 1 :] 
            
            # === 3. 提取 Action 部分 ===
            # 训练时建议使用 Mask 提取，比切片更安全（防止 Padding 干扰）
            # 逻辑：利用 mask 选出所有动作 token，然后 reshape 回 (B, 1, 64, D)
            # 注意：这里假设 batch 中每个样本的动作长度都是固定的 NUM_TOKENS (64)
            actions_hidden_states = text_and_action_hidden[current_action_mask | next_actions_mask].reshape(batch_size, 1, NUM_TOKENS, -1).to(torch.bfloat16)

            # === 4. 提取 Prompt (纯文本) 部分 ===
            # 逻辑：Text+Action 的总长度 减去 Action 的长度，剩下的就是 Prompt
            # 假设 Layout 是紧凑的: [Prompt] + [Action] (没有中间间隔)
            prompt_hidden_states = text_and_action_hidden[:, : -NUM_TOKENS].reshape(batch_size, 1, -1, item.shape[-1]).to(torch.bfloat16)
            
            # === 5. 构建混合特征 (Visual + Action) ===
            # 这是 VLA-Adapter 的核心输入：视觉 + 动作
            mixed_hidden_states = torch.cat((visual_hidden_states, actions_hidden_states), dim=2)
            
            # === 6. 存入列表 ===
            multi_layer_hidden_states.append(mixed_hidden_states)
            multi_layer_visual_states.append(visual_hidden_states)
            multi_layer_prompt_states.append(prompt_hidden_states)
            multi_layer_action_states.append(actions_hidden_states)

        # === 拼接层维度 (Dim 1) ===
        # 结果形状: [B, Layers, Seq_Len, D]
        multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim=1) # [B, K, 512+64, D]
        multi_layer_visual_states = torch.cat(multi_layer_visual_states, dim=1) # [B, K, 512, D]
        multi_layer_prompt_states = torch.cat(multi_layer_prompt_states, dim=1) # [B, K, Prompt_Len, D]
        multi_layer_action_states = torch.cat(multi_layer_action_states, dim=1) # [B, K, 64, D]

        
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)

        # TODO 增加对于action-head部分中的表征对齐损失（如果 use_contrastive_loss=True 的话）
        # 先前的工作都是在VLM部分做对齐，然而VLM一般提供的通识信息 对于动作做决策的时候才应该做对齐。
        # 所以可以考虑在action head部分增加一个对齐损失，来约束执行的动作与视觉与语义更一致
        # TODO 两个方案：1.在VLM->action head中的每个中间层做对齐（需要修改模型结构，增加 projection layers 来对齐不同层的特征）；
        # TODO 2.直接在最终的动作输出上做对齐（不需要修改模型结构，只需要增加一个 contrastive loss 来对齐 predicted_actions 与 visual features 或者 prompt features）
        # import ipdb;ipdb.set_trace()
        # TODO FIXME 调整 action head 输入
        # 这里的参数传递必须与推理代码完全一致
        outputs = action_head.module.predict_action(
            actions_hidden_states=multi_layer_hidden_states, 
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            phase=cfg.phase,
            latent_outputs=last_hidden_states, 
            multi_layer_prompt_states=multi_layer_prompt_states,
            multi_layer_visual_states=multi_layer_visual_states,
            multi_layer_action_states=multi_layer_action_states 
        )

        # ✅ 通过 use_contrastive_loss 和 phase 来判断返回值结构
        if cfg.use_contrastive_loss and cfg.phase == "Training":
            predicted_actions, contrastive_loss = outputs
        else:
            predicted_actions = outputs
            contrastive_loss = 0.0 # 推理阶段或不使用对比学习时，设置为浮点数 0.0

        # 1. 计算纯动作 L1 损失
        l1_loss = torch.nn.L1Loss()(predicted_actions, ground_truth_actions)
        total_loss = l1_loss
        
        # 2. 计算对比学习损失
        if cfg.use_contrastive_loss and cfg.phase == "Training":
            # 确保你的 config 中定义了 contrastive_weight，比如 0.01
            total_loss = l1_loss + cfg.contrastive_weight * contrastive_loss 
            
        alignment_stats = {}
        if action_head is not None:
            action_head_model = action_head.module if hasattr(action_head, "module") else action_head
            alignment_stats = getattr(action_head_model, "last_alignment_stats", {})

        # 3. 记录日志
        metrics.update(
            {
                "l1_loss_value": l1_loss.item(),  
                # 【修改这里】：判断 contrastive_loss 是否是 Tensor，如果是才调 .item()
                "contrastive_loss": contrastive_loss.item() if torch.is_tensor(contrastive_loss) else contrastive_loss,
                "total_loss": total_loss.item(),
            }
        )
        for name, value in alignment_stats.items():
            metrics[name] = value.item() if torch.is_tensor(value) else value

        # Get detailed L1 losses for logging
        should_log_l1_loss = use_l1_regression
        if should_log_l1_loss:
            ground_truth_curr_action = ground_truth_actions[:, 0] #[B, 8, 7]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            if compute_diffusion_l1:
                print('curr: ',curr_action_l1_loss.item())
                # print('next: ',next_actions_l1_loss.item())
            metrics.update(
                {
                    "curr_action_l1_loss": curr_action_l1_loss.item(),
                    "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )

    # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
    return total_loss, metrics



def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics



def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity (str): W&B entity instance

    Returns:
        None.
    """
    log_dict = {}
    for name, value in metrics.items():
        # Map loss_value to Loss for better readability in W&B
        if name == "total_loss":
            log_dict[f"{prefix}/Loss"] = value
        # Keep other metrics as is
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)
    
def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    action_head,
    train_dataset,
    distributed_state,
    new_state_dict,
    
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.

    Args:
        cfg (FinetuneConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        action_head (nn.Module): Action head module.
        train_dataset (RLDSDataset): Training dataset.
        distributed_state (PartialState): Distributed training state.

    Returns:
        None.
    """
    # Determine checkpoint paths and naming
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    # Create directories and save dataset statistics (main process only)
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    # Wait for directories to be created
    dist.barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)

        if cfg.use_fz:
            vla.module.save_pretrained(checkpoint_dir) # directly save checkpoint without lora
        else:
            vla.module.save_pretrained(adapter_dir)

        # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}"
            )

        if cfg.use_l1_regression and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

        if cfg.use_film:
            # To be safe, just save the entire vision backbone (not just FiLM components)
            torch.save(
                vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
            )

    # Wait for model components to be saved
    dist.barrier()

    # Merge LoRA weights into base model and save resulting model checkpoint
    # Note: Can be very slow on some devices; if so, we recommend merging offline
    # 只有主进程进入这个块执行耗时的合并操作
    if distributed_state.is_main_process:
        if cfg.use_lora and cfg.merge_lora_during_training :
            print(f"Rank 0: Merging LoRA weights at Step {log_step}...")
            resume_has_merged_model = (
                cfg.resume_finetune_path is not None
                and any((cfg.resume_finetune_path / name).exists() for name in ("model.safetensors", "pytorch_model.bin"))
            )
            if resume_has_merged_model:
                base_vla = AutoModelForVision2Seq.from_pretrained(
                    str(cfg.resume_finetune_path),
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=False,
                )
            elif cfg.use_minivlm:
                config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
                # 建议加上 to("cpu") 或 device_map，防止主进程合并时显存炸裂
                base_vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)
                new_state_dict['action_queries.weight'] = vla.state_dict()['module.base_model.model.action_queries.weight'].cpu()
                base_vla.load_state_dict(new_state_dict, strict=False)
            else:
                base_vla = AutoModelForVision2Seq.from_pretrained(
                    cfg.config_file_path, 
                    torch_dtype=torch.bfloat16, 
                    low_cpu_mem_usage=True, 
                    trust_remote_code=False
                )

            merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir,device_map="cpu")
            merged_vla = merged_vla.merge_and_unload()

            # 执行保存
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")
            
            # # 手动清理内存，防止后续训练 OOM
            # del base_vla, merged_vla
            # import gc; gc.collect(); torch.cuda.empty_cache()

    # --- 关键点：barrier 必须放在 if 块外面 ---
    # 所有进程（无论主从）都会运行到这一行并停下
    # 直到主进程完成上面的保存工作，大家才一起继续
    # if dist.is_initialized():
    dist.barrier()

def run_validation(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    val_dataloader,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,
    log_step,
    distributed_state,
    val_time_limit,
) -> None:
    """
    Compute validation set metrics for logging.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        val_dataloader (DataLoader): Validation data loader.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        cfg (FinetuneConfig): Training configuration.
        num_patches (int): Number of vision patches.
        log_step (int): Current logging step.
        distributed_state (PartialState): Distributed training state.
        val_time_limit (int): Time limit for computing validation metrics.

    Returns:
        None.
    """
    val_start_time = time.time()
    vla.eval()
    val_batches_count = 0

    # List to store validation metrics
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            # Always compute L1 loss for validation, even for diffusion
            _, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
                compute_diffusion_l1=True,
                use_pro_version=cfg.use_pro_version
            )

            # Add the loss value to the metrics
            metrics["loss"] = metrics["total_loss"] if "total_loss" in metrics else metrics.get("loss_value", 0.0)
            all_val_metrics.append(metrics)
            val_batches_count += 1

            # Cut testing on validation set short if it exceeds time limit
            if time.time() - val_start_time > val_time_limit:
                break

    # Compute average validation metrics
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    # Add batch count to metrics
    avg_val_metrics["val_batches_count"] = val_batches_count

    # Log validation metrics to W&B
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, swanlab)


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA.

    Allows toggling different action representations (discrete vs. continuous), different learning objectives
    (next-token prediction vs. L1 regression vs. diffusion), FiLM. Also allows for additional model inputs,
    such as additional camera images and robot proprioceptive state. Assumes parallel action generation with
    action chunking.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        None.
    """ 
    #TODO DEBUG
    # import ipdb; ipdb.set_trace()
    global RAW_STATE_DICT

    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.config_file_path = cfg.config_file_path.rstrip("/")
    if cfg.resume and cfg.resume_finetune:
        raise ValueError("`resume` and `resume_finetune` are different modes and cannot both be True.")

    cfg.resume_finetune_path = resolve_resume_finetune_path(cfg.resume_finetune, cfg.resume_finetune_path)
    if cfg.resume_finetune_path is not None:
        cfg.resume_finetune_step = infer_resume_finetune_step(
            cfg.resume_finetune_path,
            cfg.resume_finetune_step,
        )
        print(
            "Resume-finetune warm start enabled:\n"
            f"\tcheckpoint: {cfg.resume_finetune_path}\n"
            f"\tcomponent step: {cfg.resume_finetune_step}"
        )

    print(f"Fine-tuning OpenVLA Model `{cfg.config_file_path}` on `{cfg.dataset_name}`")

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # --- 增加以下排查代码 ---
    # is_main_process 默认为全局 Rank 0
    if distributed_state.is_main_process:
        print(f"确认：当前进程是主进程 (Global Rank: {distributed_state.process_index}, Local ID: {device_id})")
    else:
        print(f"确认：当前进程是子进程 (Global Rank: {distributed_state.process_index}, Local ID: {device_id})")
    
    # 强制同步一次，确保所有进程都打印完了再往下走
    distributed_state.wait_for_everyone()
 
    # Initialize wandb logging
    if distributed_state.is_main_process:
        print('wandb init')
        wandb.init(project=cfg.wandb_project, name=f"ft+{run_id}", mode="offline")
        swanlab.init(project=cfg.wandb_project, experiment_name=f"ft+{run_id}")
    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes
    # In both cases, we want to check whether any changes have been made to
    # the `modeling_prismatic.py` file in this codebase; if so, we will copy
    # the file to the downloaded or locally stored checkpoint directory so
    # that the user's changes to the VLA class logic go into effect

    model_source_path = str(cfg.resume_finetune_path) if cfg.resume_finetune_path is not None else cfg.config_file_path

    if cfg.resume_finetune_path is None and model_is_on_hf_hub(cfg.config_file_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.config_file_path)
        # Overwrite VLA path
        cfg.config_file_path = vla_download_path
        model_source_path = cfg.config_file_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


    # Update config.json and sync model files
    if distributed_state.is_main_process:
        update_auto_map(model_source_path)
        check_model_logic_mismatch(model_source_path)

    # Wait for model files to be synced
    dist.barrier()

    # Load processor and VLA
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    processor = AutoProcessor.from_pretrained(model_source_path, trust_remote_code=True)

    resume_finetune_loaded_peft_adapter = False
    if cfg.resume_finetune_path is not None:
        RAW_STATE_DICT = {}
        adapter_dir = cfg.resume_finetune_path / "lora_adapter"
        has_merged_model = any((cfg.resume_finetune_path / name).exists() for name in ("model.safetensors", "pytorch_model.bin"))

        if has_merged_model:
            print(f"Loading merged resume-finetune VLA weights from `{cfg.resume_finetune_path}`")
            vla = AutoModelForVision2Seq.from_pretrained(
                str(cfg.resume_finetune_path),
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
                trust_remote_code=False,
            ).to(device_id)
            if cfg.use_minivlm:
                RAW_STATE_DICT = {name: param.detach().cpu() for name, param in vla.state_dict().items()}
        elif cfg.use_lora and adapter_dir.is_dir():
            print(f"Loading resume-finetune LoRA adapter from `{adapter_dir}`")
            base_vla = AutoModelForVision2Seq.from_pretrained(
                cfg.config_file_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
                trust_remote_code=False,
            ).to(device_id)
            vla = PeftModel.from_pretrained(base_vla, adapter_dir, is_trainable=True)
            resume_finetune_loaded_peft_adapter = True
        else:
            raise FileNotFoundError(
                "resume_finetune_path must contain either `model.safetensors`, `pytorch_model.bin`, "
                f"or a `lora_adapter/` directory: {cfg.resume_finetune_path}"
            )

    elif cfg.use_minivlm:
        hf_token = ''
        if 'prism-qwen25-extra-dinosiglip-224px-0_5b' in cfg.vlm_path:
            
            vlm = load(cfg.vlm_path, hf_token=hf_token, load_for_training=True)
        else:
            vlm = load_vla(
                cfg.vlm_path,
                hf_token=hf_token,
                load_for_training=True,
                )
        config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16).to(device_id)  # Create a new model with configuration, the parameters are randomly initialized
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.shape}")
        replace_map = [
            ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
            ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
            ("llm_backbone.llm", "language_model"),
            ("projector.projector.0", "projector.fc1"),
            ("projector.projector.2", "projector.fc2"),
            ("projector.projector.4", "projector.fc3"),
            ("gamma", "scale_factor"),
            ]

        def rename_state_dict_keys(state_dict, replace_map):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                for old, new in replace_map:
                    if old in new_k:
                        new_k = new_k.replace(old, new)
                new_state_dict[new_k] = v
            return new_state_dict
        
        old_state_dict = vlm.state_dict()
        RAW_STATE_DICT = rename_state_dict_keys(old_state_dict, replace_map)
    
        missing_keys, unexpected_keys = vla.load_state_dict(RAW_STATE_DICT, strict=False)
        del old_state_dict

    else:
        RAW_STATE_DICT ={}
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.config_file_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=False,
            ).to(device_id)

    # Set number of images in VLA input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # vla.set_version(cfg.version)

    if cfg.use_lora and not resume_finetune_loaded_peft_adapter:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha= 2 * cfg.lora_rank,
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config) #为模型添加lora矩阵 #output = input @ W_original + (input @ (B @ A)) * scaling
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True
        vla.print_trainable_parameters()

    elif cfg.use_lora and resume_finetune_loaded_peft_adapter:
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True
        vla.print_trainable_parameters()

    else:
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        # Wrap vision backbone with FiLM wrapper
        # Important: For this, must specify `vla.model.vision_backbone` instead of just `vla.vision_backbone`, since the
        # latter would cause the new wrapped backbone to be saved as a new attribute of `vla` instead of overwriting the
        # original one (due to the LoRA wrapper)
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.config_file_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # If applicable, instantiate proprio projector
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
            to_bf16=True,
        )

    # If applicable, instantiate continuous action head for L1 regression
    if cfg.use_l1_regression:
        action_head = init_module(
        L1RegressionActionHead,
        "action_head",
        cfg,
        device_id,
        {
            "input_dim": vla.module.llm_dim, 
            "hidden_dim": vla.module.llm_dim, 
            "action_dim": ACTION_DIM,
            "use_pro_version": cfg.use_pro_version,
            "use_gate" : cfg.use_gate, # whether to use gated fusion
            "use_latent_outputs": cfg.use_latent_outputs, # whether to use the last layer latent outputs from VLM
            "use_contrastive_loss": cfg.use_contrastive_loss, # whether to use contrastive loss for better alignment between vision/language features and action representations
            "alignment_strategy": cfg.action_alignment_strategy,
            "use_manifold_regularization": cfg.action_use_manifold_regularization,
            "manifold_topk": cfg.action_manifold_topk,
            "manifold_temperature": cfg.action_manifold_temperature,
            "learnable_alignment_weights": cfg.action_learnable_alignment_weights,
            },
        to_bf16=True,
        )

    # Get number of vision patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings

    # Instantiate optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]

    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Record original learning rate
    original_lr = optimizer.param_groups[0]["lr"]

    # Create learning rate scheduler
    # 1. MultiStepLR
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
        gamma=0.1,  # Multiplicative factor of learning rate decay
    )
    # 2. CosineAnnealingLR
    # scheduler = CosineAnnealingLR(
    #         optimizer,
    #         T_max=cfg.num_steps_before_decay, 
    #         eta_min=0.0001,          
    #         )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # train_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder,
    # )
    # ---

    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    use_wrist_image = cfg.num_images_in_input > 1

    #TODO DEBUG 判断训练使用的数据格式 默认还是rlds
    assert cfg.dataset_type in ("rlds", "hdf5"), \
    "only support dataset_type rlds or hdf5, but get dataset_type: {}".format(cfg.dataset_type)

    # Create training and optional validation datasets
    # batch_transform = RLDSBatchTransform(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder,
    #     use_wrist_image=use_wrist_image,
    #     use_proprio=cfg.use_proprio,
    #     use_minivlm=cfg.use_minivlm
    #     )
    # train_dataset = RLDSDataset(
    #     cfg.data_root_dir,
    #     cfg.dataset_name,
    #     batch_transform,
    #     resize_resolution=tuple(vla.module.config.image_sizes),
    #     shuffle_buffer_size=cfg.shuffle_buffer_size,
    #     image_aug=cfg.image_aug,
    # )
    if cfg.dataset_type == "rlds":
        batch_transform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
            use_wrist_image=use_wrist_image,
            use_proprio=cfg.use_proprio,
            use_minivlm=cfg.use_minivlm
        )

        train_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            image_aug=cfg.image_aug,
        )
    elif cfg.dataset_type == "hdf5":
        batch_transform = HDF5BatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
            use_wrist_image=use_wrist_image,
            use_proprio=cfg.use_proprio,
            use_minivlm=cfg.use_minivlm
        )

        train_dataset = HDF5Dataset(
            data_dir=cfg.data_root_dir,
            dataset_name=cfg.dataset_name,
            resize_resolution=tuple(vla.module.config.image_sizes),
            train=True,
            image_aug=cfg.image_aug,
            image_augment_kwargs=dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            ),
            batch_transform=batch_transform,
            num_samples_per_traj=cfg.num_samples_per_traj

        )

    # if cfg.use_val_set:
    #     val_dataset = RLDSDataset(
    #         cfg.data_root_dir,
    #         cfg.dataset_name,
    #         batch_transform,
    #         resize_resolution=tuple(vla.module.config.image_sizes),
    #         shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
    #         image_aug=cfg.image_aug,
    #         train=False,
    #     )

    if cfg.use_val_set:
        if cfg.dataset_type == "rlds":
            batch_transform = RLDSBatchTransform(
                action_tokenizer,
                processor.tokenizer,
                image_transform=processor.image_processor.apply_transform,
                prompt_builder_fn=PurePromptBuilder,
                use_wrist_image=use_wrist_image,
                use_proprio=cfg.use_proprio,
                use_minivlm=cfg.use_minivlm
            )
            val_dataset = RLDSDataset(
                cfg.data_root_dir,
                cfg.dataset_name,
                batch_transform,
                resize_resolution=tuple(vla.module.config.image_sizes),
                shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
                image_aug=cfg.image_aug,
                train=False,
            )
        elif cfg.dataset_type == "hdf5":
            batch_transform = HDF5BatchTransform(
                action_tokenizer,
                processor.tokenizer,
                image_transform=processor.image_processor.apply_transform,
                prompt_builder_fn=PurePromptBuilder,
                use_wrist_image=use_wrist_image,
                use_proprio=cfg.use_proprio,
                use_minivlm=cfg.use_minivlm
            )

            val_dataset = HDF5Dataset(
                data_dir=cfg.data_root_dir,
                dataset_name=cfg.dataset_name,
                resize_resolution=tuple(vla.module.config.image_sizes),
                train=False,
                image_aug=False,
                batch_transform=batch_transform
            )

    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create collator and dataloader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    # TODO DEBUG 
    # dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=cfg.batch_size,
    #     sampler=None,
    #     collate_fn=collator,
    #     num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    # )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
        num_workers=cfg.num_workers,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    )
    print('Len of dataloader: ', len(dataloader))
    # if cfg.use_val_set:
    #     val_batch_size = cfg.batch_size
    #     val_dataloader = DataLoader(
    #         val_dataset,
    #         batch_size=val_batch_size,
    #         sampler=None,
    #         collate_fn=collator,
    #         num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    #     )
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            persistent_workers=True,
            pin_memory=True,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps), # 保留原始的，防止旧代码报错
        "l1_loss_value": deque(maxlen=cfg.grad_accumulation_steps),    # ✅ 新增：纯动作 L1 损失
        "contrastive_loss": deque(maxlen=cfg.grad_accumulation_steps), # ✅ 新增：对比损失
        "total_loss": deque(maxlen=cfg.grad_accumulation_steps),       # ✅ 新增：加权总损失
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # Start training
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader): 
            #FIXME 
            #batch.keys() = dict_keys(['pixel_values', 'proprio', 'input_ids', 'attention_mask', 'labels', 'actions', 'dataset_names'])
            # batch['proprio'].shape = torch.Size([8]) batch['attention_mask'].shape=batch[labels]=torch.Size([B, 113])
            # batch['actions'].shape=torch.Size([1, 8, 7]) #horizon =8
            # Compute training metrics and loss
            #只在某些 step（每 diffusion_sample_freq 步一次）计算额外的指标（这里可能是为未来 diffusion 准备的占位，目前没用到）。
            compute_diffusion_l1 = (cfg.use_l1_regression and batch_idx % cfg.diffusion_sample_freq == 0) or (cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0)
            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
                compute_diffusion_l1=compute_diffusion_l1,
                use_pro_version=cfg.use_pro_version,
                cfg=cfg,
            )

            # Normalize loss to account for gradient accumulation
            #梯度累积：将 loss 除以累积步数后反向传播（标准做法，避免显存爆炸）。
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Store recent train metrics
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)

            # Push Metrics to W&B (every wandb_log_freq gradient steps)
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, swanlab)
            # [If applicable] Linearly warm up learning rate from 10 percent to the original LR.
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)  # Cap at 1.0
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                # Log the learning rate
                # Make sure to do this AFTER any learning rate modifications (e.g., warmup/decay)
                wandb.log(
                    {
                        "VLA Train/Learning Rate": scheduler.get_last_lr()[0],
                    },
                    step=log_step,
                )
                swanlab.log(
                    {
                        "learning_rate": scheduler.get_last_lr()[0],
                    },
                    step=log_step,
                )
            # Optimizer and LR scheduler step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # Save model checkpoint: either keep latest checkpoint only or all checkpoints
            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    noisy_action_projector=None,
                    action_head=action_head,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                    new_state_dict=RAW_STATE_DICT,
                )

            # Test model on validation set
            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    val_dataloader=val_dataloader,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=log_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                )
                # Set model back to training mode after validation
                vla.train()

            # Stop training when max_steps is reached
            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
