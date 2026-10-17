#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_CHECKPOINT_PATH="${DEFAULT_CHECKPOINT_PATH:-/path/to/checkpoint}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DEFAULT_CHECKPOINT_PATH}}"
ENV_NAME="${ENV_NAME:-}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-perform the task}"
N_EPISODES="${N_EPISODES:-20}"

if [[ $# -gt 0 ]]; then
  first_arg="$1"
  if [[ -e "${first_arg}" || "${first_arg}" == /* || "${first_arg}" == outputs/* || "${first_arg}" == ./* || "${first_arg}" == ../* || "${first_arg}" == *chkpt* || "${first_arg}" == *checkpoint* || "${first_arg}" == *.pt ]]; then
    CHECKPOINT_PATH="${first_arg}"
    ENV_NAME="${2:-${ENV_NAME}}"
    TASK_DESCRIPTION="${3:-${TASK_DESCRIPTION}}"
    N_EPISODES="${4:-${N_EPISODES}}"
  else
    ENV_NAME="${first_arg}"
    TASK_DESCRIPTION="${2:-${TASK_DESCRIPTION}}"
    N_EPISODES="${3:-${N_EPISODES}}"
  fi
fi

if [[ -z "${ENV_NAME}" ]]; then
  echo "ENV_NAME is required. Usage: bash vla-scripts/evaluate_robocasa_merged_panda.sh [CHECKPOINT] ENV_NAME TASK_DESCRIPTION [N_EPISODES]" >&2
  echo "You can omit CHECKPOINT only when the first positional argument is ENV_NAME." >&2
  exit 2
fi

python_bin="${PYTHON_BIN:-python}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"
n_envs="${N_ENVS:-1}"
LOG_DIR=${LOG_DIR:-experiments/robocasa/tmp/robocasa_merged_panda_single}
DATASET_NAME=${DATASET_NAME:-robocasa_merged_24_tasks_100demos_v1}
UNNORM_KEY=${UNNORM_KEY:-${DATASET_NAME}}
SAVE_VIDEO=${SAVE_VIDEO:-False}
FLIP_IMAGES=${FLIP_IMAGES:-True}
ENV_SPLIT=${ENV_SPLIT:-target}
OBJ_REGISTRIES=${OBJ_REGISTRIES:-lightwheel}

IMAGE_OBS_KEY=${IMAGE_OBS_KEY:-video.right_image}
EXTRA_IMAGE_OBS_KEYS=${EXTRA_IMAGE_OBS_KEYS:-video.left_image,video.wrist_image}
STATE_OBS_KEYS=${STATE_OBS_KEYS:-state}
INSTRUCTION_OBS_KEY=${INSTRUCTION_OBS_KEY:-annotation.human.coarse_action}
action_space_mode="${ACTION_SPACE_MODE:-auto}"
use_proprio="${USE_PROPRIO:-False}"
proprio_dim="${PROPRIO_DIM:-9}"
DICT_ACTION_KEYS=${DICT_ACTION_KEYS:-action.end_effector_position,action.end_effector_rotation,action.gripper_close,action.base_motion,action.control_mode}
DICT_ACTION_DIMS=${DICT_ACTION_DIMS:-3,3,1,4,1}

mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" "${python_bin}" experiments/robot/robocasa/simulation_env.py \
  --pretrained_checkpoint "${CHECKPOINT_PATH}" \
  --env_name "${ENV_NAME}" \
  --env_split "${ENV_SPLIT}" \
  --obj_registries "${OBJ_REGISTRIES}" \
  --task_description "${TASK_DESCRIPTION}" \
  --n_episodes "${N_EPISODES}" \
  --n_envs "${n_envs}" \
  --local_log_dir "${LOG_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --unnorm_key "${UNNORM_KEY}" \
  --action_space_mode "${action_space_mode}" \
  --use_film False \
  --use_minivlm True \
  --use_pro_version True \
  --use_gate True \
  --use_latent_outputs True \
  --flip_images "${FLIP_IMAGES}" \
  --use_proprio "${use_proprio}" \
  --proprio_dim "${proprio_dim}" \
  --num_images_in_input 3 \
  --image_obs_key "${IMAGE_OBS_KEY}" \
  --extra_image_obs_keys "${EXTRA_IMAGE_OBS_KEYS}" \
  --state_obs_keys "${STATE_OBS_KEYS}" \
  --instruction_obs_key "${INSTRUCTION_OBS_KEY}" \
  --dict_action_keys "${DICT_ACTION_KEYS}" \
  --dict_action_dims "${DICT_ACTION_DIMS}" \
  --save_video "${SAVE_VIDEO}"
