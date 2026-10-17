#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_CHECKPOINT_PATH="${DEFAULT_CHECKPOINT_PATH:-/path/to/checkpoint}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DEFAULT_CHECKPOINT_PATH}}"
ENV_LIST_FILE="${ENV_LIST_FILE:-experiments/robot/robocasa/robocasa_merged_panda_envs.txt}"
N_EPISODES="${N_EPISODES:-20}"

if [[ $# -gt 0 ]]; then
  first_arg="$1"
  if [[ "${first_arg}" == *.txt || "${first_arg}" == *.list || ( -f "${first_arg}" && "${first_arg}" != *.pt ) ]]; then
    ENV_LIST_FILE="${first_arg}"
    N_EPISODES="${2:-${N_EPISODES}}"
  else
    CHECKPOINT_PATH="${first_arg}"
    ENV_LIST_FILE="${2:-${ENV_LIST_FILE}}"
    N_EPISODES="${3:-${N_EPISODES}}"
  fi
fi

TIMESTAMP=$(date +"%Y_%m_%d-%H_%M_%S")
LOG_DIR=${LOG_DIR:-experiments/robocasa/tmp/robocasa_merged_panda_batch_${TIMESTAMP}}
python_bin="${PYTHON_BIN:-python}"

if [[ ! -f "${ENV_LIST_FILE}" ]]; then
  echo "Missing ENV_LIST_FILE: ${ENV_LIST_FILE}" >&2
  echo "Each non-comment line should be: ENV_NAME|task language used during training" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"

RAN_ANY=false

while IFS= read -r LINE; do
  [[ -z "${LINE}" || "${LINE}" =~ ^# ]] && continue
  RAN_ANY=true

  ENV_NAME=${LINE%%|*}
  if [[ "${LINE}" == *"|"* ]]; then
    TASK_DESCRIPTION=${LINE#*|}
  else
    TASK_DESCRIPTION=${TASK_DESCRIPTION:-perform the task}
  fi

  TASK_TAG=${ENV_NAME//\//_}
  TASK_TAG=${TASK_TAG// /_}
  echo "Evaluating ${ENV_NAME} | ${TASK_DESCRIPTION}"
  ENV_NAME="${ENV_NAME}" TASK_DESCRIPTION="${TASK_DESCRIPTION}" N_EPISODES="${N_EPISODES}" LOG_DIR="${LOG_DIR}" \
    bash vla-scripts/evaluate_robocasa_merged_panda.sh "${CHECKPOINT_PATH}" "${ENV_NAME}" "${TASK_DESCRIPTION}" "${N_EPISODES}" \
    > "${LOG_DIR}/${TASK_TAG}.log" 2>&1
done < "${ENV_LIST_FILE}"

if [[ "${RAN_ANY}" == "false" ]]; then
  echo "No evaluation tasks were found in ENV_LIST_FILE: ${ENV_LIST_FILE}" >&2
  echo "Add lines in the format: ENV_NAME|task language used during training" >&2
  exit 2
fi

"${python_bin}" -c '
import glob
import json
import os
import sys

log_dir = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(log_dir, "EVAL-robocasa-*.txt"))):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows.append({
        "env_name": payload["env_name"],
        "success_rate": payload["success_rate"],
        "n_episodes": payload["n_episodes"],
        "result_file": os.path.basename(path),
        "task_description": payload["config"].get("task_description", ""),
    })

summary = {
    "num_tasks": len(rows),
    "mean_success_rate": sum(item["success_rate"] for item in rows) / len(rows) if rows else 0.0,
    "tasks": rows,
}
summary_path = os.path.join(log_dir, "summary.json")
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print(f"Saved RoboCasa merged Panda batch summary to {summary_path}")
if rows:
    for item in rows:
        print(f"{item['env_name']}: {item['success_rate']:.4f} ({item['n_episodes']} episodes)")
    print(f"Mean success rate: {summary['mean_success_rate']:.4f}")
' "${LOG_DIR}"

echo "Finished RoboCasa merged Panda batch evaluation. Logs are in ${LOG_DIR}"
