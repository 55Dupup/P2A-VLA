#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_CHECKPOINT_PATH="${DEFAULT_CHECKPOINT_PATH:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DEFAULT_CHECKPOINT_PATH}}"
TASK_NAME="${TASK_NAME:-TurnOffMicrowave}"
N_TRIALS="${N_TRIALS:-50}"

if [[ $# -gt 0 ]]; then
  first_arg="$1"
  if [[ -e "${first_arg}" || "${first_arg}" == /* || "${first_arg}" == outputs/* || "${first_arg}" == ./* || "${first_arg}" == ../* || "${first_arg}" == *chkpt* || "${first_arg}" == *checkpoint* || "${first_arg}" == *.pt ]]; then
    CHECKPOINT_PATH="${first_arg}"
    TASK_NAME="${2:-${TASK_NAME}}"
    N_TRIALS="${3:-${N_TRIALS}}"
  else
    TASK_NAME="${first_arg}"
    N_TRIALS="${2:-${N_TRIALS}}"
  fi
fi

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "CHECKPOINT_PATH is required. Usage: bash vla-scripts/evaluate_robocasa_cosmos.sh CHECKPOINT [TASK_NAME] [N_TRIALS]" >&2
  echo "Alternatively set DEFAULT_CHECKPOINT_PATH or CHECKPOINT_PATH and pass TASK_NAME first." >&2
  exit 2
fi

python_bin="${PYTHON_BIN:-python}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"
LOG_DIR=${LOG_DIR:-experiments/robocasa/tmp/robocasa_cosmos_single}
DATASET_NAME=${DATASET_NAME:-robocasa_cosmos_success_only}
UNNORM_KEY=${UNNORM_KEY:-${DATASET_NAME}}
SAVE_VIDEO=${SAVE_VIDEO:-True}
FLIP_IMAGES=${FLIP_IMAGES:-True}
USE_PROPRIO=${USE_PROPRIO:-True}
PROPRIO_DIM=${PROPRIO_DIM:-9}
# Official Cosmos eval used 16 open-loop steps; temporarily default to 8.
N_ACTION_STEPS=${N_ACTION_STEPS:-8}
APPEND_FIXED_MOBILE_BASE_ACTION=${APPEND_FIXED_MOBILE_BASE_ACTION:-True}
IMAGE_OBS_KEY=${IMAGE_OBS_KEY:-video.left_image}
EXTRA_IMAGE_OBS_KEYS=${EXTRA_IMAGE_OBS_KEYS:-video.right_image,video.wrist_image}
STATE_OBS_KEYS=${STATE_OBS_KEYS:-state}
INSTRUCTION_OBS_KEY=${INSTRUCTION_OBS_KEY:-annotation.human.coarse_action}
ROBOTS=${ROBOTS:-PandaMobile}
CONTROLLERS=${CONTROLLERS:-OSC_POSE}
OBJ_INSTANCE_SPLIT=${OBJ_INSTANCE_SPLIT:-B}
LAYOUT_AND_STYLE_IDS=${LAYOUT_AND_STYLE_IDS:-((1,1),(2,2),(4,4),(6,9),(7,10))}
RANDOMIZE_CAMERAS=${RANDOMIZE_CAMERAS:-False}
SEED=${SEED:-195}
DETERMINISTIC=${DETERMINISTIC:-True}
CONTROLLER_CONFIGS_PATH=${CONTROLLER_CONFIGS_PATH:-/path/to/cosmos-policy/cosmos_policy/experiments/robot/robocasa/robocasa_controller_configs.pkl}

mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" "${python_bin}" experiments/robot/robocasa/run_robocasa_cosmos_eval.py \
  --pretrained_checkpoint "${CHECKPOINT_PATH}" \
  --task_name "${TASK_NAME}" \
  --num_trials_per_task "${N_TRIALS}" \
  --local_log_dir "${LOG_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --unnorm_key "${UNNORM_KEY}" \
  --use_film False \
  --use_minivlm True \
  --use_pro_version True \
  --use_gate True \
  --use_latent_outputs True \
  --flip_images "${FLIP_IMAGES}" \
  --use_proprio "${USE_PROPRIO}" \
  --proprio_dim "${PROPRIO_DIM}" \
  --num_images_in_input 3 \
  --image_obs_key "${IMAGE_OBS_KEY}" \
  --extra_image_obs_keys "${EXTRA_IMAGE_OBS_KEYS}" \
  --state_obs_keys "${STATE_OBS_KEYS}" \
  --instruction_obs_key "${INSTRUCTION_OBS_KEY}" \
  --n_action_steps "${N_ACTION_STEPS}" \
  --append_fixed_mobile_base_action "${APPEND_FIXED_MOBILE_BASE_ACTION}" \
  --robots "${ROBOTS}" \
  --controllers "${CONTROLLERS}" \
  --obj_instance_split "${OBJ_INSTANCE_SPLIT}" \
  --layout_and_style_ids "${LAYOUT_AND_STYLE_IDS}" \
  --randomize_cameras "${RANDOMIZE_CAMERAS}" \
  --seed "${SEED}" \
  --deterministic "${DETERMINISTIC}" \
  --controller_configs_path "${CONTROLLER_CONFIGS_PATH}" \
  --save_video "${SAVE_VIDEO}"
