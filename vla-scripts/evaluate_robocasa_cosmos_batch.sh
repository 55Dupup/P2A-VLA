#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_CHECKPOINT_PATH="${DEFAULT_CHECKPOINT_PATH:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DEFAULT_CHECKPOINT_PATH}}"
TASK_LIST_FILE="${TASK_LIST_FILE:-experiments/robot/robocasa/robocasa_cosmos_tasks.txt}"
N_TRIALS="${N_TRIALS:-50}"

if [[ $# -gt 0 ]]; then
  first_arg="$1"
  if [[ "${first_arg}" == *.txt || "${first_arg}" == *.list || ( -f "${first_arg}" && "${first_arg}" != *.pt ) ]]; then
    TASK_LIST_FILE="${first_arg}"
    N_TRIALS="${2:-${N_TRIALS}}"
  else
    CHECKPOINT_PATH="${first_arg}"
    TASK_LIST_FILE="${2:-${TASK_LIST_FILE}}"
    N_TRIALS="${3:-${N_TRIALS}}"
  fi
fi

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "CHECKPOINT_PATH is required. Usage: bash vla-scripts/evaluate_robocasa_cosmos_batch.sh CHECKPOINT [TASK_LIST_FILE] [N_TRIALS]" >&2
  exit 2
fi

if [[ ! -f "${TASK_LIST_FILE}" ]]; then
  echo "Missing TASK_LIST_FILE: ${TASK_LIST_FILE}" >&2
  echo "Each non-comment line should be a Cosmos RoboCasa task name, e.g. TurnOffMicrowave" >&2
  exit 2
fi

TIMESTAMP=$(date +"%Y_%m_%d-%H_%M_%S")
LOG_DIR=${LOG_DIR:-experiments/robocasa/tmp/robocasa_cosmos_batch_${TIMESTAMP}}
python_bin="${PYTHON_BIN:-python}"
mkdir -p "${LOG_DIR}"

RAN_ANY=false

while IFS= read -r LINE; do
  [[ -z "${LINE}" || "${LINE}" =~ ^# ]] && continue
  RAN_ANY=true
  TASK_NAME="${LINE%%|*}"
  TASK_TAG=${TASK_NAME//\//_}
  TASK_TAG=${TASK_TAG// /_}
  echo "Evaluating ${TASK_NAME}"
  TASK_NAME="${TASK_NAME}" N_TRIALS="${N_TRIALS}" LOG_DIR="${LOG_DIR}" \
    bash vla-scripts/evaluate_robocasa_cosmos.sh "${CHECKPOINT_PATH}" "${TASK_NAME}" "${N_TRIALS}" \
    > "${LOG_DIR}/${TASK_TAG}.log" 2>&1
done < "${TASK_LIST_FILE}"

if [[ "${RAN_ANY}" == "false" ]]; then
  echo "No evaluation tasks were found in TASK_LIST_FILE: ${TASK_LIST_FILE}" >&2
  exit 2
fi

"${python_bin}" -c '
import glob
import json
import os
import sys

log_dir = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(log_dir, "EVAL-robocasa-cosmos-*.txt"))):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows.append({
        "task_name": payload["task_name"],
        "success_rate": payload["success_rate"],
        "n_episodes": payload["n_episodes"],
        "result_file": os.path.basename(path),
        "video_paths": payload.get("video_paths", []),
    })
summary = {
    "num_tasks": len(rows),
    "mean_success_rate": sum(item["success_rate"] for item in rows) / len(rows) if rows else 0.0,
    "tasks": rows,
}
summary_path = os.path.join(log_dir, "summary.json")
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
print(f"Saved RoboCasa Cosmos batch summary to {summary_path}")
if rows:
    for item in rows:
        print(f"{item['task_name']}: {item['success_rate']:.4f} ({item['n_episodes']} episodes)")
    print(f"Mean success rate: {summary['mean_success_rate']:.4f}")
' "${LOG_DIR}"

echo "Finished RoboCasa Cosmos batch evaluation. Logs are in ${LOG_DIR}"
