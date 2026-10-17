data_name=pick_up_the_yellow_block_and_place_it_on_the_black_plate
current_time=$(date +"%Y%m%d_%H%M%S")
# data_root_dir=/path/to/vla-adapter-experiments/data/ for calvin
data_root_dir=/path/to/rlds

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
use_contrastive_loss=False
contrastive_weight=0.1
action_alignment_strategy=original
action_use_manifold_regularization=False
action_manifold_topk=4
action_manifold_temperature=0.2
action_learnable_alignment_weights=False

CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nnodes 1 --nproc_per_node 3 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
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
--num_steps_before_decay 200000 \
--max_steps 200005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 8 \
--grad_accumulation_steps 3 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--use_gate True \
--use_latent_outputs False \
--use_contrastive_loss $use_contrastive_loss \
--contrastive_weight $contrastive_weight \
--action_alignment_strategy $action_alignment_strategy \
--action_use_manifold_regularization $action_use_manifold_regularization \
--action_manifold_topk $action_manifold_topk \
--action_manifold_temperature $action_manifold_temperature \
--action_learnable_alignment_weights $action_learnable_alignment_weights \
--wandb_entity "real-world-pap" \
--wandb_project "vla-adapter-5090" \
--run_id_note "real-world-pap-rel-eef-chunk-25-gate"
