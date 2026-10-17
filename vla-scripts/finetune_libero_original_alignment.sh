#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")/.."
data_name=libero_10_no_noops
current_time=$(date +"%Y%m%d_%H%M%S")
data_root_dir=/path/to/datasets/modified_libero_rlds

# Original contrastive alignment
use_contrastive_loss=True
contrastive_weight=0.1
action_alignment_strategy=original
action_use_manifold_regularization=False
action_manifold_topk=4
action_manifold_temperature=0.2
action_learnable_alignment_weights=True # Whether to use learnable weights for different components in the contrastive loss. If False, use fixed weights (e.g., 0.25 for each component if there are 4 components).

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nnodes 1 --nproc_per_node 4 vla-scripts/finetune.py \
--vlm_path ./pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path ./pretrained_models/configs \
--data_root_dir $data_root_dir \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 50000 \
--max_steps 80005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 8 \
--grad_accumulation_steps 8 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--use_gate True \
--use_latent_outputs True \
--use_contrastive_loss $use_contrastive_loss \
--contrastive_weight $contrastive_weight \
--action_alignment_strategy $action_alignment_strategy \
--action_use_manifold_regularization $action_use_manifold_regularization \
--action_manifold_topk $action_manifold_topk \
--action_manifold_temperature $action_manifold_temperature \
--action_learnable_alignment_weights $action_learnable_alignment_weights \
--wandb_entity "${WANDB_ENTITY:-your-wandb-entity}" \
--wandb_project "VLA-Adapter" \
--run_id_note "pro-libero_goal-original-alignment" \
