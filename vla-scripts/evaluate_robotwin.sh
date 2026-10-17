#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

DATA_ROOT_DIR="${DATA_ROOT_DIR:-/path/to/datasets/robotwin_rlds}"
MANIFEST_PATH="${MANIFEST_PATH:-${DATA_ROOT_DIR}/robotwin_manifest.json}"
ROBOTWIN_PATH="${ROBOTWIN_PATH:-/path/to/RoboTwin}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a VLA-Adapter checkpoint directory}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/robot/robotwin/eval_outputs}"
ROBOTWIN_PROTOCOL="${ROBOTWIN_PROTOCOL:-auto}"
ROBOTWIN_NUM_IMAGES="${ROBOTWIN_NUM_IMAGES:-3}"
ROBOTWIN_USE_PROPRIO="${ROBOTWIN_USE_PROPRIO:-False}"
ROBOTWIN_CENTER_CROP="${ROBOTWIN_CENTER_CROP:-True}"
ROBOTWIN_DRY_RUN="${ROBOTWIN_DRY_RUN:-False}"
ROBOTWIN_UNNORM_KEY="${ROBOTWIN_UNNORM_KEY:-}"
ROBOTWIN_CONDA_PREFIX="${ROBOTWIN_CONDA_PREFIX:-/path/to/conda_envs/vla-adapter-robotwin}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-${ROBOTWIN_CONDA_PREFIX}/bin/python}"
ROBOTWIN_CONVERTER_PATH="${ROBOTWIN_CONVERTER_PATH:-${ROOT_DIR}/vla-scripts/convert_robotwin_lerobot_to_rlds.py}"

if [[ ! -x "${ROBOTWIN_PYTHON}" ]]; then
  echo "Missing executable RoboTwin Python: ${ROBOTWIN_PYTHON}" >&2
  exit 1
fi

VERIFY_ARGS=(
  --robotwin-path "${ROBOTWIN_PATH}"
  --data-root "${DATA_ROOT_DIR}"
  --manifest-path "${MANIFEST_PATH}"
  --checkpoint-path "${CHECKPOINT_PATH}"
  --conda-prefix "${ROBOTWIN_CONDA_PREFIX}"
  --converter-path "${ROBOTWIN_CONVERTER_PATH}"
  --require-checkpoint
)
if [[ "${ROBOTWIN_DRY_RUN}" != "True" && "${ROBOTWIN_DRY_RUN}" != "true" ]]; then
  VERIFY_ARGS+=(--require-simulator --require-assets)
fi

"${ROBOTWIN_PYTHON}" vla-scripts/verify_robotwin_setup.py "${VERIFY_ARGS[@]}"

EXTRA_ARGS=()
if [[ "${ROBOTWIN_USE_PROPRIO}" == "True" || "${ROBOTWIN_USE_PROPRIO}" == "true" ]]; then
  EXTRA_ARGS+=(--use-proprio)
fi
if [[ "${ROBOTWIN_CENTER_CROP}" == "True" || "${ROBOTWIN_CENTER_CROP}" == "true" ]]; then
  EXTRA_ARGS+=(--center-crop)
else
  EXTRA_ARGS+=(--no-center-crop)
fi
if [[ -n "${TASK_NAME:-}" ]]; then
  EXTRA_ARGS+=(--task-name "${TASK_NAME}")
fi
if [[ -n "${TASK_CONFIG:-}" ]]; then
  EXTRA_ARGS+=(--task-config "${TASK_CONFIG}")
fi
if [[ -n "${EPISODES:-}" ]]; then
  EXTRA_ARGS+=(--episodes "${EPISODES}")
fi
if [[ -n "${ROBOTWIN_UNNORM_KEY}" ]]; then
  EXTRA_ARGS+=(--unnorm-key "${ROBOTWIN_UNNORM_KEY}")
fi
if [[ "${ROBOTWIN_DRY_RUN}" == "True" || "${ROBOTWIN_DRY_RUN}" == "true" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

"${ROBOTWIN_PYTHON}" experiments/robot/robotwin/eval_policy_runner.py \
  --robotwin-path "${ROBOTWIN_PATH}" \
  --manifest-path "${MANIFEST_PATH}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --protocol "${ROBOTWIN_PROTOCOL}" \
  --num-images-in-input "${ROBOTWIN_NUM_IMAGES}" \
  --robotwin-python "${ROBOTWIN_PYTHON}" \
  "${EXTRA_ARGS[@]}"
