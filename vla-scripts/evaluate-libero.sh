# bash /path/to/P2A-VLA/vla-scripts/evaluate-libero.sh
# conda activate new-adapter
cd "$(dirname "${BASH_SOURCE[0]}")/.."


CUDA_VISIBLE_DEVICES=6 python ./experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --local_log_dir ./experiments/libero_eval_logs \
  --log_name change_index-gate-lattent-SigLip-all-new-6-5w \
  --save_video False \
  --save_version vla-adapter \
  --local_video_dir ./experiments/libero_eval_videos \
  --pretrained_checkpoint /path/to/checkpoint \
  --task_suite_name libero_goal \
  --use_pro_version True \
  --use_gate True \
  --use_latent_outputs True
