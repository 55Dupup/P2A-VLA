#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/path/to/P2A-VLA"
SERIES_PREFIX="${1:-/path/to/checkpoint}"
GPU_ID="${GPU_ID:-7}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"

AUTO_EVAL_DIR="${ROOT_DIR}/experiments/libero_eval_logs/auto_eval"
INTERNAL_DIR="${AUTO_EVAL_DIR}/internal"
STAGING_ROOT="${AUTO_EVAL_DIR}/staging"
RUN_ROOT="${AUTO_EVAL_DIR}/bg_runs"

series_name="$(basename "${SERIES_PREFIX}")"
series_parent="$(dirname "${SERIES_PREFIX}")"
run_dir="${RUN_ROOT}/${series_name}"
launcher_log="${run_dir}/launcher.log"

mkdir -p "${INTERNAL_DIR}" "${STAGING_ROOT}" "${run_dir}"

cd "${ROOT_DIR}"

prepare_staging_checkpoint() {
  local checkpoint_dir="$1"
  local checkpoint_name="$2"
  local staging_dir

  staging_dir="$(mktemp -d "${STAGING_ROOT}/${checkpoint_name}.XXXXXX")"
  while IFS= read -r -d '' item; do
    local base_name
    base_name="$(basename "$item")"
    case "${base_name}" in
      config.json|modeling_prismatic.py|configuration_prismatic.py)
        cp -a "${item}" "${staging_dir}/${base_name}"
        ;;
      *)
        ln -s "${item}" "${staging_dir}/${base_name}"
        ;;
    esac
  done < <(find "${checkpoint_dir}" -mindepth 1 -maxdepth 1 -print0)

  printf '%s\n' "${staging_dir}"
}

latest_internal_log_for_ckpt() {
  local checkpoint_name="$1"
  find "${INTERNAL_DIR}" -maxdepth 1 -type f -name "EVAL-libero_goal-auto-${checkpoint_name}-*.txt" | sort | tail -n 1
}

run_one_eval() {
  local checkpoint_dir="$1"
  local checkpoint_name
  local staging_dir
  local shell_log

  checkpoint_name="$(basename "${checkpoint_dir}")"
  shell_log="${run_dir}/${checkpoint_name}.gpu${GPU_ID}.log"

  if [[ -n "$(latest_internal_log_for_ckpt "${checkpoint_name}")" ]]; then
    printf '[SKIP] already evaluated: %s\n' "${checkpoint_name}" | tee -a "${launcher_log}"
    return 0
  fi

  staging_dir="$(prepare_staging_checkpoint "${checkpoint_dir}" "${checkpoint_name}")"
  printf '[START] %s | gpu=%s | %s\n' "$(date '+%F %T')" "${GPU_ID}" "${checkpoint_name}" | tee -a "${launcher_log}"

  if CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" ./experiments/robot/libero/run_libero_eval.py \
    --use_proprio True \
    --num_images_in_input 2 \
    --use_film False \
    --local_log_dir "${INTERNAL_DIR}" \
    --log_name "auto-${checkpoint_name}" \
    --save_video False \
    --save_version vla-adapter \
    --local_video_dir ./experiments/libero_eval_videos \
    --pretrained_checkpoint "${staging_dir}" \
    --task_suite_name libero_goal \
    --use_pro_version True \
    --use_gate True \
    --use_latent_outputs True \
    > "${shell_log}" 2>&1; then
    printf '[DONE] %s | %s\n' "$(date '+%F %T')" "${checkpoint_name}" | tee -a "${launcher_log}"
  else
    printf '[FAIL] %s | %s | log=%s\n' "$(date '+%F %T')" "${checkpoint_name}" "${shell_log}" | tee -a "${launcher_log}"
  fi

  rm -rf "${staging_dir}"
}

active_job_count() {
  local count
  count="$(jobs -pr | wc -l || true)"
  printf '%s\n' "${count:-0}"
}

mapfile -t checkpoints < <(find "${series_parent}" -maxdepth 1 -type d -name "${series_name}--*_chkpt" | sort -V)

if [[ "${#checkpoints[@]}" -eq 0 ]]; then
  echo "[ERROR] no checkpoints found for ${SERIES_PREFIX}" | tee -a "${launcher_log}"
  exit 1
fi

printf '[INFO] %s | series=%s | gpu=%s | parallel=%s | count=%s\n' \
  "$(date '+%F %T')" "${series_name}" "${GPU_ID}" "${MAX_PARALLEL}" "${#checkpoints[@]}" | tee -a "${launcher_log}"

for checkpoint_dir in "${checkpoints[@]}"; do
  while (( $(active_job_count) >= MAX_PARALLEL )); do
    wait -n || true
  done
  run_one_eval "${checkpoint_dir}" &
done

wait
printf '[ALL_DONE] %s | series=%s\n' "$(date '+%F %T')" "${series_name}" | tee -a "${launcher_log}"
