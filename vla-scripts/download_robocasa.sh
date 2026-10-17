#!/usr/bin/env bash

set -euo pipefail

LOCAL_DIR=${LOCAL_DIR:-/path/to/datasets/robocasa}
HF_ENDPOINT_VALUE=${HF_ENDPOINT_VALUE:-https://hf-mirror.com}
MAX_WORKERS=${MAX_WORKERS:-16}
MAX_RETRIES=${MAX_RETRIES:-20}

mkdir -p "${LOCAL_DIR}"

HF_ENDPOINT="${HF_ENDPOINT_VALUE}" python vla-scripts/download_robocasa_dataset.py \
  --local_dir "${LOCAL_DIR}" \
  --endpoint "${HF_ENDPOINT_VALUE}" \
  --max_workers "${MAX_WORKERS}" \
  --max_retries "${MAX_RETRIES}"
