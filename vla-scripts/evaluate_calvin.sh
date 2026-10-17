cd "$(dirname "${BASH_SOURCE[0]}")/.."
CUDA_VISIBLE_DEVICES=0 python vla-scripts/evaluate_calvin.py \
  --pretrained_checkpoint /path/to/checkpoint \
  --local_log_dir /path/to/P2A-VLA/experiments/calvin_eval_logs \
  --save_video False \
  --use_pro_version True \
  --use_gate True \
  --use_latent_outputs True \
  --run_id_note "Eval-CALVIN-ABC-change_index-Gated-lattent-11w" 
#   > eval_logs/CALVIN--ABC.log 2>&1 &
