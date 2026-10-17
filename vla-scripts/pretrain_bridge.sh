#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

data_name=bridge_dataset
data_root_dir=${DATA_ROOT_DIR:-/path/to/datasets/OXE-data/bridge-rlds}

cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
nproc_per_node=${NPROC_PER_NODE:-4}
max_steps=${MAX_STEPS:-200005}
save_freq=${SAVE_FREQ:-5000}
batch_size=${BATCH_SIZE:-8}
grad_accumulation_steps=${GRAD_ACCUMULATION_STEPS:-8}
learning_rate=${LEARNING_RATE:-2e-4}
lora_rank=${LORA_RANK:-64}
num_workers=${NUM_WORKERS:-1}
run_id_note=${RUN_ID_NOTE:-bridge-pretrain/pro-bridge-rlds-bs${batch_size}-acc${grad_accumulation_steps}}

use_contrastive_loss=${USE_CONTRASTIVE_LOSS:-True}
contrastive_weight=${CONTRASTIVE_WEIGHT:-0.1}
action_alignment_strategy=${ACTION_ALIGNMENT_STRATEGY:-original}
action_use_manifold_regularization=${ACTION_USE_MANIFOLD_REGULARIZATION:-False}
action_manifold_topk=${ACTION_MANIFOLD_TOPK:-4}
action_manifold_temperature=${ACTION_MANIFOLD_TEMPERATURE:-0.2}
action_learnable_alignment_weights=${ACTION_LEARNABLE_ALIGNMENT_WEIGHTS:-True}

CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" torchrun --standalone --nnodes 1 --nproc_per_node "${nproc_per_node}" vla-scripts/finetune.py \
--vlm_path ./pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path ./pretrained_models/configs \
--data_root_dir "${data_root_dir}" \
--dataset_name "${data_name}" \
--run_root_dir outputs \
--dataset_type rlds \
--num_workers "${num_workers}" \
--use_l1_regression True \
--use_diffusion False \
--use_film False \
--num_images_in_input 1 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 100000 \
--max_steps "${max_steps}" \
--save_freq "${save_freq}" \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size "${batch_size}" \
--grad_accumulation_steps "${grad_accumulation_steps}" \
--learning_rate "${learning_rate}" \
--lora_rank "${lora_rank}" \
--use_pro_version True \
--use_gate True \
--use_latent_outputs True \
--use_contrastive_loss "${use_contrastive_loss}" \
--contrastive_weight "${contrastive_weight}" \
--action_alignment_strategy "${action_alignment_strategy}" \
--action_use_manifold_regularization "${action_use_manifold_regularization}" \
--action_manifold_topk "${action_manifold_topk}" \
--action_manifold_temperature "${action_manifold_temperature}" \
--action_learnable_alignment_weights "${action_learnable_alignment_weights}" \
--wandb_entity "${WANDB_ENTITY:-your-wandb-entity}" \
--wandb_project "VLA-Adapter" \
--run_id_note "${run_id_note}"
