cd "$(dirname "${BASH_SOURCE[0]}")/.."
data_name=calvin_abc_rlds
current_time=$(date +"%Y%m%d_%H%M%S")
data_root_dir="${DATA_ROOT_DIR:-/path/to/datasets}"
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO

# Alignment strategy controls:
# 1) Original contrastive:
#    action_alignment_strategy=original
#    action_use_manifold_regularization=False
#    action_learnable_alignment_weights=False
# 2) Improved alignment:
#    action_alignment_strategy=improved
#    action_use_manifold_regularization=False
#    action_learnable_alignment_weights=True
# 3) Improved + manifold:
#    action_alignment_strategy=improved
#    action_use_manifold_regularization=True
#    action_learnable_alignment_weights=True
use_contrastive_loss=True
contrastive_weight=0.1
action_alignment_strategy=original
action_use_manifold_regularization=False
action_manifold_topk=8
action_manifold_temperature=0.2
action_learnable_alignment_weights=False
run_id_note="${RUN_ID_NOTE:-calvin-original-bs8-acc8-${run_timestamp}}"

CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nnodes 1 --nproc_per_node 2 vla-scripts/finetune.py \
--vlm_path ./pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path ./pretrained_models/configs \
--data_root_dir $data_root_dir \
--dataset_name $data_name \
--run_root_dir outputs/calvin \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 100000 \
--max_steps 150005 \
--save_freq 10000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 8 \
--grad_accumulation_steps 4 \
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
--run_id_note "$run_id_note" 
