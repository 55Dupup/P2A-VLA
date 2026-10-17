#!/usr/bin/env bash

set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTHONPATH="/path/to/calvin/calvin_models:/path/to/calvin/calvin_env:/path/to/calvin:${PYTHONPATH:-}"

EVAL_LOG_DIR="/path/to/P2A-VLA/experiments/calvin_eval_logs"
LAUNCHER_LOG_DIR="/path/to/P2A-VLA/experiments/calvin_eval_auto_logs"
PYTHON_BIN="python"

mkdir -p "${EVAL_LOG_DIR}" "${LAUNCHER_LOG_DIR}"

run_eval() {
  local gpu="$1"
  local step="$2"
  local ckpt="/path/to/checkpoint${step}_chkpt"
  local run_id="Eval-CALVIN-ABC-auto-pro-calvin_abc-bs16-acc4-50902-change_index-Gated-lattent-improved-${step}"
  local log_file="${LAUNCHER_LOG_DIR}/${run_id}.launcher.log"

  {
    echo "START $(date '+%Y-%m-%d %H:%M:%S') gpu=${gpu} ckpt=${ckpt}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" vla-scripts/evaluate_calvin.py \
      --pretrained_checkpoint "${ckpt}" \
      --local_log_dir "${EVAL_LOG_DIR}" \
      --save_video False \
      --use_pro_version True \
      --use_gate True \
      --use_latent_outputs True \
      --run_id_note "${run_id}"
    status=$?
    echo "END $(date '+%Y-%m-%d %H:%M:%S') status=${status} gpu=${gpu} ckpt=${ckpt}"
    exit "${status}"
  } > "${log_file}" 2>&1
}

run_eval 5 60000 &
pid60000=$!

run_eval 6 70000 &
pid70000=$!

echo "started 60000 pid=${pid60000}"
echo "started 70000 pid=${pid70000}"

wait "${pid60000}" "${pid70000}"
