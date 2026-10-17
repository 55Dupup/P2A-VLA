#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

checkpoint_root="${CHECKPOINT_ROOT:-${ROOT_DIR}/outputs/metaworld}"
output_root="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/metaworld}"
batch_id="${BATCH_ID:-batch_eval_$(date +%Y%m%d-%H%M%S)}"
batch_dir="${output_root}/${batch_id}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-5}"
num_trials_per_task="${NUM_TRIALS_PER_TASK:-10}"

mkdir -p "${batch_dir}/logs"

mapfile -t checkpoints < <(
  find "${checkpoint_root}" -maxdepth 1 -mindepth 1 -type d -name '*_chkpt' |
    python -c 'import re,sys
paths=[p.strip() for p in sys.stdin if p.strip()]
def key(p):
    m=re.search(r"--(\d+)_chkpt$", p)
    return (int(m.group(1)) if m else 10**18, p)
for p in sorted(paths, key=key):
    print(p)'
)

if [[ "${#checkpoints[@]}" -eq 0 ]]; then
  echo "No checkpoints found under ${checkpoint_root}" >&2
  exit 1
fi

echo "Batch ID: ${batch_id}"
echo "Batch dir: ${batch_dir}"
echo "GPU: ${cuda_visible_devices}"
echo "Trials per task: ${num_trials_per_task}"
echo "Checkpoints: ${#checkpoints[@]}"

for checkpoint in "${checkpoints[@]}"; do
  checkpoint_name="$(basename "${checkpoint}")"
  eval_dir="${batch_dir}/${checkpoint_name}"
  log_path="${batch_dir}/logs/${checkpoint_name}.log"

  echo "[$(date '+%F %T')] Start ${checkpoint_name}"
  if PRETRAINED_CHECKPOINT="${checkpoint}" \
    CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" \
    OUTPUT_DIR="${eval_dir}" \
    NUM_TRIALS_PER_TASK="${num_trials_per_task}" \
    SAVE_VIDEO="${SAVE_VIDEO:-n}" \
    SAVE_ACTION="${SAVE_ACTION:-n}" \
    SAVE_DEBUG_FRAME="${SAVE_DEBUG_FRAME:-n}" \
    bash "${SCRIPT_DIR}/evaluate_metaworld.sh" >"${log_path}" 2>&1; then
    echo "[$(date '+%F %T')] Done ${checkpoint_name}; log=${log_path}"
  else
    echo "[$(date '+%F %T')] Failed ${checkpoint_name}; log=${log_path}" >&2
  fi
done

python "${SCRIPT_DIR}/summarize_metaworld_batch.py" "${batch_dir}" | tee "${batch_dir}/leaderboard.txt"
