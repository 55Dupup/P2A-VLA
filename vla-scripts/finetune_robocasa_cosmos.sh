#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

data_name="${DATASET_NAME:-robocasa_cosmos}"
data_root_dir="${DATA_ROOT_DIR:-/path/to/datasets/robocasa_cosmos_success_only_rlds}"
python_bin="${PYTHON_BIN:-python}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
nproc_per_node="${NPROC_PER_NODE:-}"
vlm_path="${VLM_PATH:-./pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b}"
config_file_path="${CONFIG_FILE_PATH:-./pretrained_models/configs}"
run_root_dir="${RUN_ROOT_DIR:-${ROOT_DIR}/outputs/robocasa-cosmos}"
run_timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"

use_contrastive_loss="${USE_CONTRASTIVE_LOSS:-True}"
contrastive_weight="${CONTRASTIVE_WEIGHT:-0.1}"
action_alignment_strategy="${ACTION_ALIGNMENT_STRATEGY:-improved}"
action_use_manifold_regularization="${ACTION_USE_MANIFOLD_REGULARIZATION:-True}"
action_manifold_topk="${ACTION_MANIFOLD_TOPK:-8}"
action_manifold_temperature="${ACTION_MANIFOLD_TEMPERATURE:-0.2}"
action_learnable_alignment_weights="${ACTION_LEARNABLE_ALIGNMENT_WEIGHTS:-True}"

num_steps_before_decay="${NUM_STEPS_BEFORE_DECAY:-200000}"
max_steps="${MAX_STEPS:-150005}"
save_freq="${SAVE_FREQ:-5000}"
save_latest_checkpoint_only="${SAVE_LATEST_CHECKPOINT_ONLY:-False}"
merge_lora_during_training="${MERGE_LORA_DURING_TRAINING:-True}"
batch_size="${BATCH_SIZE:-4}"
grad_accumulation_steps="${GRAD_ACCUMULATION_STEPS:-4}"
learning_rate="${LEARNING_RATE:-2e-4}"
lora_rank="${LORA_RANK:-64}"
wandb_entity="${WANDB_ENTITY:-your-wandb-entity}"
wandb_project="${WANDB_PROJECT:-VLA-Adapter}"
run_id_note="${RUN_ID_NOTE:-robocasa_cosmos-7d-chunk8-3view-no-proprio-bs${batch_size}-acc${grad_accumulation_steps}-${run_timestamp}}"

if [[ -z "${nproc_per_node}" ]]; then
  IFS=',' read -r -a gpu_ids <<< "${cuda_visible_devices}"
  nproc_per_node="${#gpu_ids[@]}"
fi

CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" "${python_bin}" -m torch.distributed.run --standalone --nnodes 1 --nproc_per_node "${nproc_per_node}" vla-scripts/finetune.py \
  --vlm_path "${vlm_path}" \
  --config_file_path "${config_file_path}" \
  --data_root_dir "${data_root_dir}" \
  --dataset_name "${data_name}" \
  --run_root_dir "${run_root_dir}" \
  --run_id_note "${run_id_note}" \
  --use_film False \
  --num_images_in_input 3 \
  --use_proprio False \
  --use_lora True \
  --use_fz False \
  --use_minivlm True \
  --image_aug True \
  --num_steps_before_decay "${num_steps_before_decay}" \
  --max_steps "${max_steps}" \
  --save_freq "${save_freq}" \
  --save_latest_checkpoint_only "${save_latest_checkpoint_only}" \
  --merge_lora_during_training "${merge_lora_during_training}" \
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
  --wandb_entity "${wandb_entity}" \
  --wandb_project "${wandb_project}" \
  "$@"
