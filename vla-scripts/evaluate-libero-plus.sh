#!/usr/bin/env bash

set -euo pipefail

# Example:
#   conda activate vla-adapter-libero-plus
#   bash /path/to/P2A-VLA/vla-scripts/evaluate-libero-plus.sh

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Keep LIBERO-plus isolated from the original LIBERO/CALVIN setup.
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/path/to/LIBERO-plus/.libero}"

LOG_LABEL="libero-plus-eval-7W-$(date +"%Y%m%d_%H%M%S")"

LOG_ROOT="${LOG_ROOT:-/path/to/P2A-VLA/experiments/libero-plus_eval_logs}"
RUN_NAME="${RUN_NAME:-${LOG_LABEL}}"
RUN_DIR="${LOG_ROOT}/${RUN_NAME}"
TASK_SUITES="${TASK_SUITES:-libero_spatial libero_object libero_goal libero_10}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/path/to/checkpoint}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
TASK_CLASSIFICATION_PATH="${TASK_CLASSIFICATION_PATH:-/path/to/LIBERO-plus/libero/libero/benchmark/task_classification.json}"
UNNORM_KEY="${UNNORM_KEY:-libero_mix}"

mkdir -p "${RUN_DIR}"

for task_suite in ${TASK_SUITES}; do
  echo "Evaluating LIBERO-plus suite: ${task_suite}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python ./experiments/robot/libero/run_libero_eval.py \
    --use_proprio True \
    --num_images_in_input 2 \
    --use_film False \
    --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
    --local_log_dir "${RUN_DIR}" \
    --category_results_dir "${RUN_DIR}" \
    --task_classification_path "${TASK_CLASSIFICATION_PATH}" \
    --log_name "libero-plus" \
    --save_video False \
    --save_version vla-adapter \
    --local_video_dir ./experiments/libero-plus_eval_videos \
    --pretrained_checkpoint "${CHECKPOINT_PATH}" \
    --task_suite_name "${task_suite}" \
    --unnorm_key "${UNNORM_KEY}" \
    --use_pro_version True \
    --use_gate True \
    --use_latent_outputs True
done

python ./vla-scripts/aggregate_libero_plus_results.py --log-dir "${RUN_DIR}"

echo "LIBERO-plus logs and summary: ${RUN_DIR}"
