#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

data_name=libero_mix
data_root_dir=${DATA_ROOT_DIR:-/path/to/datasets/libero_plus_rlds}

use_contrastive_loss=True
contrastive_weight=0.1
action_alignment_strategy=original
action_use_manifold_regularization=False
action_manifold_topk=4
action_manifold_temperature=0.2
action_learnable_alignment_weights=True

resume_finetune=${RESUME_FINETUNE:-False}
resume_finetune_path=${RESUME_FINETUNE_PATH:-}
resume_finetune_step=${RESUME_FINETUNE_STEP:-}
resume_finetune_args=()
if [[ "${resume_finetune}" == "True" || "${resume_finetune}" == "true" || "${resume_finetune}" == "1" ]]; then
  if [[ -z "${resume_finetune_path}" ]]; then
    echo "RESUME_FINETUNE_PATH must be set when RESUME_FINETUNE=${resume_finetune}" >&2
    exit 1
  fi
  resume_finetune_args+=(--resume_finetune True --resume_finetune_path "${resume_finetune_path}")
  if [[ -n "${resume_finetune_step}" ]]; then
    resume_finetune_args+=(--resume_finetune_step "${resume_finetune_step}")
  fi
fi

CUDA_VISIBLE_DEVICES="7" torchrun --standalone --nnodes 1 --nproc_per_node 1 vla-scripts/finetune.py \
--vlm_path ./pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path ./pretrained_models/configs \
--data_root_dir "${data_root_dir}" \
--dataset_name "${data_name}" \
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
--use_contrastive_loss "${use_contrastive_loss}" \
--contrastive_weight "${contrastive_weight}" \
--action_alignment_strategy "${action_alignment_strategy}" \
--action_use_manifold_regularization "${action_use_manifold_regularization}" \
--action_manifold_topk "${action_manifold_topk}" \
--action_manifold_temperature "${action_manifold_temperature}" \
--action_learnable_alignment_weights "${action_learnable_alignment_weights}" \
--wandb_entity "${WANDB_ENTITY:-your-wandb-entity}" \
--wandb_project "VLA-Adapter" \
--run_id_note "libero-plus/pro-libero-plus-libero_mix-original-8*8" \
"${resume_finetune_args[@]}"
