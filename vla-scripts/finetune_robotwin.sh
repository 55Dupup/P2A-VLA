#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT_DIR="${DATA_ROOT_DIR:-/path/to/datasets/robotwin_rlds}"
MANIFEST_PATH="${MANIFEST_PATH:-${DATA_ROOT_DIR}/robotwin_manifest.json}"
DATASET_NAME="${DATASET_NAME:-robotwin_lerobot_unified}"
ROBOTWIN_PROTOCOL="${ROBOTWIN_PROTOCOL:-auto}"
ROBOTWIN_NUM_IMAGES="${ROBOTWIN_NUM_IMAGES:-3}"
ROBOTWIN_USE_PROPRIO="${ROBOTWIN_USE_PROPRIO:-False}"
ROBOTWIN_IMAGE_AUG="${ROBOTWIN_IMAGE_AUG:-True}"
RUN_ID="${RUN_ID:-robotwin-${DATASET_NAME}}"
ROBOTWIN_PATH="${ROBOTWIN_PATH:-/path/to/RoboTwin}"
ROBOTWIN_CONDA_PREFIX="${ROBOTWIN_CONDA_PREFIX:-/path/to/conda_envs/vla-adapter-robotwin}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-${ROBOTWIN_CONDA_PREFIX}/bin/python}"
ROBOTWIN_TORCHRUN="${ROBOTWIN_TORCHRUN:-${ROBOTWIN_CONDA_PREFIX}/bin/torchrun}"
ROBOTWIN_CONVERTER_PATH="${ROBOTWIN_CONVERTER_PATH:-${ROOT_DIR}/vla-scripts/convert_robotwin_lerobot_to_rlds.py}"
ROBOTWIN_VLM_PATH="${ROBOTWIN_VLM_PATH:-${ROOT_DIR}/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b}"
ROBOTWIN_CONFIG_PATH="${ROBOTWIN_CONFIG_PATH:-${ROOT_DIR}/pretrained_models/configs}"

if [[ ! -x "${ROBOTWIN_PYTHON}" || ! -x "${ROBOTWIN_TORCHRUN}" ]]; then
  echo "Missing RoboTwin training executables under ${ROBOTWIN_CONDA_PREFIX}" >&2
  exit 1
fi

"${ROBOTWIN_PYTHON}" vla-scripts/verify_robotwin_setup.py \
  --robotwin-path "${ROBOTWIN_PATH}" \
  --data-root "${DATA_ROOT_DIR}" \
  --manifest-path "${MANIFEST_PATH}" \
  --conda-prefix "${ROBOTWIN_CONDA_PREFIX}" \
  --converter-path "${ROBOTWIN_CONVERTER_PATH}"

"${ROBOTWIN_PYTHON}" - <<PY
from experiments.robot.robotwin.protocols import load_robotwin_manifest, resolve_robotwin_protocol
manifest = load_robotwin_manifest("${MANIFEST_PATH}")
profile = resolve_robotwin_protocol("${ROBOTWIN_PROTOCOL}", manifest)
print(f"RoboTwin protocol: {profile.name}")
print(f"Dataset source: {manifest.source}")
print(f"Trajectory counts: {dict(manifest.trajectory_counts)}")
print(f"Cameras: {profile.camera_names}")
print(f"num_images: ${ROBOTWIN_NUM_IMAGES}")
print(f"use_proprio: ${ROBOTWIN_USE_PROPRIO}")
print(f"image_aug: ${ROBOTWIN_IMAGE_AUG}")
print(f"action_normalization: {manifest.action_normalization}")
print(f"actions_are_normalized: {manifest.actions_are_normalized}")
PY

"${ROBOTWIN_PYTHON}" -c "import sys; sys.argv=['vla-scripts/finetune_robotwin.sh']; import prismatic.vla.constants as c; assert c.ROBOT_PLATFORM == 'ROBOTWIN', c.ROBOT_PLATFORM; print(c.ROBOT_PLATFORM, c.ACTION_DIM, c.PROPRIO_DIM, c.NUM_ACTIONS_CHUNK)"

"${ROBOTWIN_TORCHRUN}" --standalone --nnodes 1 --nproc-per-node "${NPROC_PER_NODE:-1}" \
  vla-scripts/finetune.py \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --run_id_override "${RUN_ID}" \
  --vlm_path "${ROBOTWIN_VLM_PATH}" \
  --config_file_path "${ROBOTWIN_CONFIG_PATH}" \
  --num_images_in_input "${ROBOTWIN_NUM_IMAGES}" \
  --use_proprio "${ROBOTWIN_USE_PROPRIO}" \
  --image_aug "${ROBOTWIN_IMAGE_AUG}" \
  --use_lora True \
  --use_minivlm True \
  --merge_lora_during_training True \
  "$@"
