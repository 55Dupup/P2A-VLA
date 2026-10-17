#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-conda}"
ROBOTWIN_ENV_PREFIX="${ROBOTWIN_ENV_PREFIX:-/path/to/conda_envs/vla-adapter-robotwin}"
ROBOTWIN_PYTHON="${ROBOTWIN_ENV_PREFIX}/bin/python"
CUROBO_VERSION="${CUROBO_VERSION:-v0.7.8}"
CUROBO_COMMIT="${CUROBO_COMMIT:-d64c4b005459db10c5dd867d8b30a87d5bda9bdb}"
CUROBO_SOURCE_DIR="${CUROBO_SOURCE_DIR:-${ROBOTWIN_ENV_PREFIX}/src/curobo-${CUROBO_VERSION}}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "vla-adapter" ]]; then
  echo "Refusing to install RoboTwin dependencies into vla-adapter"
  exit 1
fi

if [[ ! -d "${ROBOTWIN_ENV_PREFIX}" ]]; then
  "${CONDA_BIN}" create --prefix "${ROBOTWIN_ENV_PREFIX}" --clone vla-adapter -y
fi

if [[ ! -x "${ROBOTWIN_PYTHON}" ]]; then
  echo "Missing cloned RoboTwin Python: ${ROBOTWIN_PYTHON}" >&2
  exit 1
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export http_proxy="${http_proxy:-http://localhost:7895}"
export https_proxy="${https_proxy:-http://localhost:7895}"

"${ROBOTWIN_PYTHON}" -m pip install \
  "sapien==3.0.0b1" \
  "mplib==0.2.1" \
  "transforms3d==0.4.2" \
  "toppra==0.6.8" \
  "open3d==0.18.0"

mkdir -p "$(dirname "${CUROBO_SOURCE_DIR}")"
if [[ ! -d "${CUROBO_SOURCE_DIR}/.git" ]]; then
  git clone --branch "${CUROBO_VERSION}" --depth 1 \
    https://github.com/NVlabs/curobo.git "${CUROBO_SOURCE_DIR}"
fi

actual_curobo_commit="$(git -C "${CUROBO_SOURCE_DIR}" rev-parse HEAD)"
if [[ "${actual_curobo_commit}" != "${CUROBO_COMMIT}" ]]; then
  echo "Unexpected CuRobo commit: ${actual_curobo_commit} (expected ${CUROBO_COMMIT})" >&2
  exit 1
fi

"${ROBOTWIN_PYTHON}" -m pip install \
  "setuptools==69.5.1" \
  "warp-lang==1.12.0"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-8}"
"${ROBOTWIN_PYTHON}" -m pip install --no-build-isolation -e "${CUROBO_SOURCE_DIR}"
echo "RoboTwin dependency install finished in ${ROBOTWIN_ENV_PREFIX}"
