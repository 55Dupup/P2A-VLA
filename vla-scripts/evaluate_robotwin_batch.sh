#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-experiments/robot/robotwin/eval_outputs/batch}"
mkdir -p "${OUTPUT_DIR}"

for TASK_CONFIG_VALUE in demo_clean demo_randomized; do
  TASK_CONFIG="${TASK_CONFIG_VALUE}" \
  OUTPUT_DIR="${OUTPUT_DIR}/${TASK_CONFIG_VALUE}" \
  bash vla-scripts/evaluate_robotwin.sh
done

python - <<PY
from pathlib import Path
for path in Path("${OUTPUT_DIR}").glob("*/planned_eval_jobs.json"):
    print(path)
PY
