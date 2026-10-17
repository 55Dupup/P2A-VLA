#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/path/to/P2A-VLA"
SERIES="pro-libero_goal-change_index-gate-lattent-SigLip-all-new-acc8-learnable-weights"
OUTPUT_DIR="${ROOT_DIR}/outputs"
AUTO_EVAL_DIR="${ROOT_DIR}/experiments/libero_eval_logs/auto_eval"
INTERNAL_DIR="${AUTO_EVAL_DIR}/internal"
STATUS_DIR="${AUTO_EVAL_DIR}/status"
STAGING_ROOT="${AUTO_EVAL_DIR}/staging"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"

mkdir -p "${INTERNAL_DIR}" "${STATUS_DIR}" "${STAGING_ROOT}"

remaining_ckpts=(
  "${SERIES}--50000_chkpt"
  "${SERIES}--55000_chkpt"
  "${SERIES}--60000_chkpt"
  "${SERIES}--65000_chkpt"
)

series_ckpts=(
  "${SERIES}--5000_chkpt"
  "${SERIES}--50000_chkpt"
  "${SERIES}--55000_chkpt"
  "${SERIES}--60000_chkpt"
  "${SERIES}--65000_chkpt"
)

prepare_staging_checkpoint() {
  local checkpoint_dir="$1"
  local checkpoint_name="$2"
  local staging_dir

  staging_dir="$(mktemp -d "${STAGING_ROOT}/${checkpoint_name}.XXXXXX")"
  while IFS= read -r -d '' item; do
    local base_name
    base_name="$(basename "$item")"
    case "$base_name" in
      config.json|modeling_prismatic.py|configuration_prismatic.py)
        cp -a "$item" "${staging_dir}/${base_name}"
        ;;
      *)
        ln -s "$item" "${staging_dir}/${base_name}"
        ;;
    esac
  done < <(find "$checkpoint_dir" -mindepth 1 -maxdepth 1 -print0)

  printf '%s\n' "$staging_dir"
}

latest_internal_log_for_ckpt() {
  local checkpoint_name="$1"
  find "${INTERNAL_DIR}" -maxdepth 1 -type f -name "EVAL-libero_goal-auto-${checkpoint_name}-*.txt" | sort | tail -n 1
}

extract_final_percent() {
  local log_file="$1"
  python - "$log_file" <<'PY'
import re, sys
path = sys.argv[1]
final_pat = re.compile(r'Overall success rate:\s*[0-9.]+\s*\(([-\d.]+)%\)')
episode_pat = re.compile(r'# successes:\s*\d+\s*\(([-\d.]+)%\)')
score = None
last = None
with open(path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        m = final_pat.search(line)
        if m:
            score = float(m.group(1))
        m2 = episode_pat.search(line)
        if m2:
            last = float(m2.group(1))
if score is None:
    score = last
if score is None:
    raise SystemExit(1)
print(f"{score:.4f}")
PY
}

run_one_eval() {
  local checkpoint_name="$1"
  local checkpoint_dir="${OUTPUT_DIR}/${checkpoint_name}"
  local staging_dir
  local shell_log="${AUTO_EVAL_DIR}/${checkpoint_name}.manual-running.log"

  if [[ ! -d "${checkpoint_dir}" ]]; then
    echo "[SKIP] missing checkpoint dir: ${checkpoint_dir}"
    return 0
  fi

  staging_dir="$(prepare_staging_checkpoint "${checkpoint_dir}" "${checkpoint_name}")"
  trap 'rm -rf "${staging_dir}"' RETURN

  echo "[START] ${checkpoint_name}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${ROOT_DIR}/experiments/robot/libero/run_libero_eval.py" \
    --use_proprio True \
    --num_images_in_input 2 \
    --use_film False \
    --local_log_dir "${INTERNAL_DIR}" \
    --log_name "auto-${checkpoint_name}" \
    --save_video False \
    --save_version vla-adapter \
    --local_video_dir "${ROOT_DIR}/experiments/libero_eval_videos" \
    --pretrained_checkpoint "${staging_dir}" \
    --task_suite_name libero_goal \
    --use_pro_version True \
    --use_gate True \
    --use_latent_outputs True \
    > "${shell_log}" 2>&1

  rm -rf "${staging_dir}"
  trap - RETURN
  echo "[DONE] ${checkpoint_name}"
}

collect_scores() {
  python - "$INTERNAL_DIR" "$OUTPUT_DIR" "$SERIES" <<'PY'
import glob, json, os, re, sys
internal_dir, output_dir, series = sys.argv[1:4]
pat = re.compile(r'^EVAL-libero_goal-auto-(.+?)-\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}(?:--sr([0-9]+(?:\.[0-9]+)?))?\.txt$')
final_pat = re.compile(r'Overall success rate:\s*[0-9.]+\s*\(([-\d.]+)%\)')
episode_pat = re.compile(r'# successes:\s*\d+\s*\(([-\d.]+)%\)')
rows = []
for path in sorted(glob.glob(os.path.join(internal_dir, "EVAL-libero_goal-auto-*.txt"))):
    name = os.path.basename(path)
    m = pat.match(name)
    if not m:
        continue
    ckpt = m.group(1)
    if not ckpt.startswith(series + "--"):
        continue
    score = None
    last = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            fm = final_pat.search(line)
            if fm:
                score = float(fm.group(1))
            em = episode_pat.search(line)
            if em:
                last = float(em.group(1))
    if score is None:
        score = last
    rows.append({
        "ckpt": ckpt,
        "score": score,
        "exists": os.path.isdir(os.path.join(output_dir, ckpt)),
        "log": path,
        "status": os.path.join(os.path.dirname(internal_dir), "status", ckpt + ".state"),
    })
rows.sort(key=lambda x: (-x["score"], x["ckpt"]))
print(json.dumps(rows, ensure_ascii=False))
PY
}

for checkpoint_name in "${remaining_ckpts[@]}"; do
  existing_log="$(latest_internal_log_for_ckpt "${checkpoint_name}")"
  if [[ -n "${existing_log}" ]]; then
    echo "[SKIP] already has internal log: ${checkpoint_name}"
    continue
  fi
  run_one_eval "${checkpoint_name}"
done

scores_json="$(collect_scores)"
echo "${scores_json}" > "${AUTO_EVAL_DIR}/${SERIES}__final_scores.json"

best_ckpt="$(python - "${AUTO_EVAL_DIR}/${SERIES}__final_scores.json" <<'PY'
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    rows = json.load(f)
existing = [r for r in rows if r["exists"]]
existing.sort(key=lambda x: (-x["score"], x["ckpt"]))
print(existing[0]["ckpt"])
PY
)"

echo "[BEST] ${best_ckpt}"

python - "${AUTO_EVAL_DIR}/${SERIES}__final_scores.json" "${best_ckpt}" <<'PY'
import json, os, shutil, sys
score_file, best_ckpt = sys.argv[1:3]
with open(score_file, 'r', encoding='utf-8') as f:
    rows = json.load(f)
for row in rows:
    ckpt = row["ckpt"]
    if not row["exists"] or ckpt == best_ckpt:
        continue
    output_path = f"/path/to/P2A-VLA/outputs/{ckpt}"
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
        print(f"[DEL] {output_path}")
PY

echo "[DONE] series cleanup complete"
