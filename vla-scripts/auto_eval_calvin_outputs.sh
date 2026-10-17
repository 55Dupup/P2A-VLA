#!/usr/bin/env bash

set -Eeuo pipefail

# ==============================================================================
# CALVIN 自动评估脚本
# ==============================================================================
#
# 推荐用法 1：评估某一个“系列前缀”下的全部 checkpoint
#   bash vla-scripts/auto_eval_calvin_outputs.sh \
#     /path/to/checkpoint
#
# 推荐用法 2：只评估某一个 checkpoint
#   TARGET_MODE=single bash vla-scripts/auto_eval_calvin_outputs.sh \
#     /path/to/checkpoint
#
# 推荐用法 3：评估某个目录下所有 *_chkpt 子目录
#   TARGET_MODE=dir bash vla-scripts/auto_eval_calvin_outputs.sh \
#     /path/to/P2A-VLA/outputs/Experiments-calvin-weights-old
#
# 可选：放到 tmux 后台跑，避免长时间评估被当前终端或 API session 影响
#   AUTO_TMUX=1 bash vla-scripts/auto_eval_calvin_outputs.sh /path/to/series_prefix
#
# 可选：只检查会评估哪些 checkpoint，不真正启动评估
#   DRY_RUN=1 bash vla-scripts/auto_eval_calvin_outputs.sh /path/to/series_prefix
#

# ------------------------------------------------------------------------------
# 路径和环境设置
# ------------------------------------------------------------------------------

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# CALVIN 评估代码需要从本地 calvin 仓库导入 calvin_agent 和 calvin_env。
# 这里保持之前能跑通的仿真环境路径，不修改仿真环境本身。
CALVIN_ROOT="${CALVIN_ROOT:-/path/to/calvin}"
export PYTHONPATH="${CALVIN_ROOT}/calvin_models:${CALVIN_ROOT}/calvin_env:${CALVIN_ROOT}:${PYTHONPATH:-}"

# 默认直接使用 vla-adapter 环境里的 python，避免 detached/background 时 conda run 不稳定。
# 如需覆盖：
#   PYTHON_BIN=/path/to/python bash ...
# 如果 PYTHON_BIN 不存在，会退回到 CONDA_ENV；再不设置则使用当前 shell 的 python。
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-vla-adapter}"

# summary/meta 只需要 Python 标准库，默认也走 vla-adapter 的 python，避免 tmux/PATH 不一致。
PY_HELPER_BIN="${PY_HELPER_BIN:-${PYTHON_BIN}}"
if [[ ! -x "${PY_HELPER_BIN}" ]]; then
  PY_HELPER_BIN="$(command -v python || true)"
fi
if [[ -z "${PY_HELPER_BIN}" ]]; then
  echo "No python executable found for summary/meta generation. Set PY_HELPER_BIN=/path/to/python." >&2
  exit 1
fi

# ------------------------------------------------------------------------------
# 输入目标设置
# ------------------------------------------------------------------------------

# 第一个位置参数仍然可用；也可以用 EVAL_TARGET 环境变量指定。
EVAL_TARGET="${1:-${EVAL_TARGET:-}}"

# TARGET_MODE:
#   auto   自动判断：*_chkpt 目录 => single；已存在目录 => dir；其他 => series
#   series 把目标当作系列前缀，展开为 “目标--*_chkpt”
#   single 把目标当作单个 checkpoint 目录
#   dir    扫描目标目录下所有 *_chkpt 子目录
TARGET_MODE="${TARGET_MODE:-auto}"

# 兼容旧变量：
#   CHECKPOINT_SERIES=/path/to/series_prefix
#   CHECKPOINT_PATH=/path/to/xxx--10000_chkpt
#   CHECKPOINT_DIR=/path/to/dir_with_checkpoints
#   CHECKPOINT_ROOT 不传目标时使用，默认扫描 outputs/calvin 下的 checkpoint
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/path/to/P2A-VLA/outputs/calvin}"
CHECKPOINT_SERIES="${CHECKPOINT_SERIES:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

# ------------------------------------------------------------------------------
# 输出、调度和评估参数
# ------------------------------------------------------------------------------

# launcher 日志：记录自动脚本每个 checkpoint 的启动、结束和 Python stdout/stderr。
LOG_ROOT="${LOG_ROOT:-/path/to/P2A-VLA/experiments/calvin_eval_auto_logs}"

# evaluate_calvin.py 的真实结果目录。每个 checkpoint 会生成 run_id-timestamp 子目录。
EVAL_LOG_DIR="${EVAL_LOG_DIR:-/path/to/P2A-VLA/experiments/calvin_eval_logs}"

# 可用 GPU 和并发数。CALVIN eval 很重，显存不够时把 MAX_CONCURRENT 改成 1。
GPU_LIST="${GPU_LIST:-6,7}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"

# 只选择显存占用 <= 该阈值 MiB 的 GPU。脚本还会使用 lock 防止多个自动脚本抢同一张 GPU。
MAX_USED_MB="${MAX_USED_MB:-}"
GPU_LOCK_DIR="${GPU_LOCK_DIR:-/tmp/calvin_eval_gpu_locks}"

# 没有 GPU 或并发满时的轮询间隔。
POLL_SECONDS="${POLL_SECONDS:-120}"

# 已经有 result.txt 的 checkpoint 默认跳过；需要重跑时设 FORCE=1。
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
FORCE="${FORCE:-0}"

# 只打印解析出的 checkpoint，不真正启动评估。
DRY_RUN="${DRY_RUN:-0}"

# 是否把整个脚本放入 tmux 后台运行。
AUTO_TMUX="${AUTO_TMUX:-0}"
TMUX_SESSION="${TMUX_SESSION:-calvin_auto_eval_$(date '+%Y%m%d_%H%M%S')}"

# 评估参数：保持与 evaluate_calvin.sh 一致。
SAVE_VIDEO="${SAVE_VIDEO:-False}"
USE_PRO_VERSION="${USE_PRO_VERSION:-True}"
USE_GATE="${USE_GATE:-True}"
USE_LATENT_OUTPUTS="${USE_LATENT_OUTPUTS:-True}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-Eval-CALVIN-ABC-auto}"

mkdir -p "${LOG_ROOT}" "${EVAL_LOG_DIR}" "${GPU_LOCK_DIR}"

SUMMARY_STAMP="$(date '+%Y%m%d_%H%M%S')"
SUMMARY_RUN_LIST="${LOG_ROOT}/calvin_eval_summary_${SUMMARY_STAMP}.runs.tsv"
SUMMARY_CSV="${LOG_ROOT}/calvin_eval_summary_${SUMMARY_STAMP}.csv"
SUMMARY_TXT="${LOG_ROOT}/calvin_eval_summary_${SUMMARY_STAMP}.txt"

# ------------------------------------------------------------------------------
# tmux 后台启动
# ------------------------------------------------------------------------------

if [[ "${AUTO_TMUX}" == "1" && -z "${TMUX:-}" && "${DRY_RUN}" != "1" ]]; then
  quoted_script="$(printf '%q' "${SCRIPT_PATH}")"
  quoted_project="$(printf '%q' "${PROJECT_ROOT}")"
  if [[ -n "${EVAL_TARGET}" ]]; then
    quoted_target="$(printf '%q' "${EVAL_TARGET}")"
    tmux new-session -d -s "${TMUX_SESSION}" "cd ${quoted_project} && bash ${quoted_script} ${quoted_target}"
  else
    tmux new-session -d -s "${TMUX_SESSION}" "cd ${quoted_project} && bash ${quoted_script}"
  fi
  echo "Started tmux session: ${TMUX_SESSION}"
  echo "Attach: tmux attach -t ${TMUX_SESSION}"
  echo "Launcher logs: ${LOG_ROOT}"
  echo "Eval logs: ${EVAL_LOG_DIR}"
  exit 0
fi

# ------------------------------------------------------------------------------
# checkpoint 解析
# ------------------------------------------------------------------------------

CHECKPOINTS=()
TARGET_DESC=""

sort_checkpoints() {
  sort -V
}

scan_series() {
  local series_prefix="$1"
  find "$(dirname "${series_prefix}")" -mindepth 1 -maxdepth 1 -type d \
    -name "$(basename "${series_prefix}")--*_chkpt" | sort_checkpoints
}

scan_dir() {
  local checkpoint_dir="$1"
  find "${checkpoint_dir}" -mindepth 1 -maxdepth 1 -type d -name '*_chkpt' | sort_checkpoints
}

set_single_checkpoint() {
  local checkpoint_path="$1"
  if [[ ! -d "${checkpoint_path}" ]]; then
    echo "Checkpoint directory does not exist: ${checkpoint_path}" >&2
    exit 1
  fi
  CHECKPOINTS=("${checkpoint_path}")
  TARGET_DESC="single:${checkpoint_path}"
}

set_series_checkpoints() {
  local series_prefix="$1"
  mapfile -t CHECKPOINTS < <(scan_series "${series_prefix}")
  TARGET_DESC="series:${series_prefix}"
}

set_dir_checkpoints() {
  local checkpoint_dir="$1"
  if [[ ! -d "${checkpoint_dir}" ]]; then
    echo "Checkpoint directory does not exist: ${checkpoint_dir}" >&2
    exit 1
  fi
  mapfile -t CHECKPOINTS < <(scan_dir "${checkpoint_dir}")
  TARGET_DESC="dir:${checkpoint_dir}"
}

set_dir_or_series_checkpoints() {
  local target="$1"
  if [[ -d "${target}" ]]; then
    mapfile -t CHECKPOINTS < <(scan_dir "${target}")
    if [[ "${#CHECKPOINTS[@]}" -gt 0 ]]; then
      TARGET_DESC="dir:${target}"
      return
    fi
  fi

  # 有些系列前缀本身也是一个目录，但真实 checkpoint 是 sibling 形式：前缀--*_chkpt。
  # 目录里没扫到 *_chkpt 时，继续按系列前缀展开，避免误报 No checkpoints found。
  set_series_checkpoints "${target}"
}

resolve_checkpoints() {
  local target="${EVAL_TARGET}"

  if [[ -n "${CHECKPOINT_PATH}" ]]; then
    set_single_checkpoint "${CHECKPOINT_PATH}"
    return
  fi

  if [[ -n "${CHECKPOINT_SERIES}" ]]; then
    set_series_checkpoints "${CHECKPOINT_SERIES}"
    return
  fi

  if [[ -n "${CHECKPOINT_DIR}" ]]; then
    set_dir_checkpoints "${CHECKPOINT_DIR}"
    return
  fi

  if [[ -n "${target}" ]]; then
    case "${TARGET_MODE}" in
      single)
        set_single_checkpoint "${target}"
        ;;
      series)
        set_series_checkpoints "${target}"
        ;;
      dir)
        set_dir_checkpoints "${target}"
        ;;
      auto)
        if [[ "${target}" == *_chkpt ]]; then
          set_single_checkpoint "${target}"
        elif [[ -d "${target}" ]]; then
          set_dir_or_series_checkpoints "${target}"
        else
          set_series_checkpoints "${target}"
        fi
        ;;
      *)
        echo "Unsupported TARGET_MODE=${TARGET_MODE}; expected auto|series|single|dir" >&2
        exit 1
        ;;
    esac
    return
  fi

  # 不传目标时：默认扫描 CHECKPOINT_ROOT。
  # 如果 CHECKPOINT_ROOT 是真实目录，按目录扫描；否则按系列前缀展开。
  if [[ -d "${CHECKPOINT_ROOT}" ]]; then
    set_dir_or_series_checkpoints "${CHECKPOINT_ROOT}"
  else
    set_series_checkpoints "${CHECKPOINT_ROOT}"
  fi
}

# 从 checkpoint 路径中解析系列名、step 和 run_id。
ckpt_meta() {
  local ckpt="$1"
  local name series step run_id

  name="$(basename "${ckpt}")"
  series="$(sed -n 's/--[0-9]\+_chkpt$//p' <<< "${name}")"
  step="$(sed -n 's/.*--\([0-9]\+\)_chkpt$/\1/p' <<< "${name}")"
  run_id="${RUN_ID_PREFIX}-${series:-unknown-series}-${step:-unknown}"
  printf '%s\t%s\t%s\n' "${series:-unknown-series}" "${step:-unknown}" "${run_id}"
}

latest_eval_dir() {
  local run_id="$1"
  find "${EVAL_LOG_DIR}" -mindepth 2 -maxdepth 2 -type f \
    -path "${EVAL_LOG_DIR}/${run_id}-*/result.txt" \
    -printf '%T@ %h\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-
}

has_completed_result() {
  local run_id="$1"
  local eval_dir
  eval_dir="$(latest_eval_dir "${run_id}")"
  [[ -n "${eval_dir}" && -f "${eval_dir}/result.txt" ]]
}

# ------------------------------------------------------------------------------
# Python 命令和 GPU lock
# ------------------------------------------------------------------------------

python_cmd_label() {
  if [[ -x "${PYTHON_BIN}" ]]; then
    echo "${PYTHON_BIN}"
  elif [[ -n "${CONDA_ENV}" ]]; then
    echo "conda run -n ${CONDA_ENV} python"
  else
    echo "python"
  fi
}

run_python_eval() {
  local gpu="$1"
  local ckpt="$2"
  local run_id="$3"

  if [[ -x "${PYTHON_BIN}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" vla-scripts/evaluate_calvin.py \
      --pretrained_checkpoint "${ckpt}" \
      --local_log_dir "${EVAL_LOG_DIR}" \
      --save_video "${SAVE_VIDEO}" \
      --use_pro_version "${USE_PRO_VERSION}" \
      --use_gate "${USE_GATE}" \
      --use_latent_outputs "${USE_LATENT_OUTPUTS}" \
      --run_id_note "${run_id}"
  elif [[ -n "${CONDA_ENV}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" conda run -n "${CONDA_ENV}" python vla-scripts/evaluate_calvin.py \
      --pretrained_checkpoint "${ckpt}" \
      --local_log_dir "${EVAL_LOG_DIR}" \
      --save_video "${SAVE_VIDEO}" \
      --use_pro_version "${USE_PRO_VERSION}" \
      --use_gate "${USE_GATE}" \
      --use_latent_outputs "${USE_LATENT_OUTPUTS}" \
      --run_id_note "${run_id}"
  else
    CUDA_VISIBLE_DEVICES="${gpu}" python vla-scripts/evaluate_calvin.py \
      --pretrained_checkpoint "${ckpt}" \
      --local_log_dir "${EVAL_LOG_DIR}" \
      --save_video "${SAVE_VIDEO}" \
      --use_pro_version "${USE_PRO_VERSION}" \
      --use_gate "${USE_GATE}" \
      --use_latent_outputs "${USE_LATENT_OUTPUTS}" \
      --run_id_note "${run_id}"
  fi
}

gpu_lock_path() {
  local gpu="$1"
  echo "${GPU_LOCK_DIR}/gpu_${gpu}.lock"
}

release_gpu_lock() {
  local gpu="$1"
  local lock_dir
  lock_dir="$(gpu_lock_path "${gpu}")"
  rm -f "${lock_dir}/owner" 2>/dev/null || true
  rmdir "${lock_dir}" 2>/dev/null || true
}

cleanup_stale_gpu_lock() {
  local gpu="$1"
  local lock_dir owner pid
  lock_dir="$(gpu_lock_path "${gpu}")"
  [[ -d "${lock_dir}" ]] || return 0

  owner="${lock_dir}/owner"
  pid="$(sed -n 's/^worker_pid=//p; s/^launcher_pid=//p' "${owner}" 2>/dev/null | head -n 1 || true)"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi

  release_gpu_lock "${gpu}"
}

mark_gpu_lock() {
  local gpu="$1"
  local pid="$2"
  local ckpt="$3"
  local lock_dir
  lock_dir="$(gpu_lock_path "${gpu}")"
  {
    echo "worker_pid=${pid}"
    echo "gpu=${gpu}"
    echo "checkpoint=${ckpt}"
    echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  } > "${lock_dir}/owner"
}

# 查询某块 GPU 当前已用显存，单位 MiB。
gpu_used_mb() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' '
}

# 从 GPU_LIST 中挑一块显存占用最低、低于 MAX_USED_MB、且未被 lock 的 GPU。
acquire_gpu() {
  local best_gpu=""
  local best_used=999999
  local gpu used lock_dir

  for gpu in ${GPU_LIST}; do
    cleanup_stale_gpu_lock "${gpu}"
    lock_dir="$(gpu_lock_path "${gpu}")"
    [[ -d "${lock_dir}" ]] && continue

    used="$(gpu_used_mb "${gpu}")"
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= MAX_USED_MB )) && (( used < best_used )); then
      best_gpu="${gpu}"
      best_used="${used}"
    fi
  done

  if [[ -n "${best_gpu}" ]]; then
    lock_dir="$(gpu_lock_path "${best_gpu}")"
    if mkdir "${lock_dir}" 2>/dev/null; then
      {
        echo "launcher_pid=$$"
        echo "gpu=${best_gpu}"
        echo "reserved_at=$(date '+%Y-%m-%d %H:%M:%S')"
      } > "${lock_dir}/owner"
      echo "${best_gpu}"
    fi
  fi
}

# ------------------------------------------------------------------------------
# 单个 checkpoint 评估和 summary
# ------------------------------------------------------------------------------

write_run_meta() {
  local ckpt="$1"
  local gpu="$2"
  local run_id="$3"
  local launcher_log="$4"
  local started_at="$5"
  local ended_at="$6"
  local status="$7"
  local eval_dir python_label

  eval_dir="$(latest_eval_dir "${run_id}")"
  [[ -n "${eval_dir}" ]] || return 0
  python_label="$(python_cmd_label)"

  "${PY_HELPER_BIN}" - "${eval_dir}/run_meta.json" "${ckpt}" "${gpu}" "${run_id}" "${launcher_log}" \
    "${started_at}" "${ended_at}" "${status}" "${EVAL_LOG_DIR}" "${SAVE_VIDEO}" \
    "${USE_PRO_VERSION}" "${USE_GATE}" "${USE_LATENT_OUTPUTS}" "${python_label}" "${CONDA_ENV}" <<'PY'
import json
import sys

(
    meta_path,
    checkpoint,
    gpu,
    run_id_note,
    launcher_log,
    started_at,
    ended_at,
    status,
    eval_log_dir,
    save_video,
    use_pro_version,
    use_gate,
    use_latent_outputs,
    python_label,
    conda_env,
) = sys.argv[1:]

meta = {
    "checkpoint": checkpoint,
    "gpu": gpu,
    "run_id_note": run_id_note,
    "launcher_log": launcher_log,
    "started_at": started_at,
    "ended_at": ended_at,
    "exit_status": int(status),
    "eval_log_dir": eval_log_dir,
    "save_video": save_video,
    "use_pro_version": use_pro_version,
    "use_gate": use_gate,
    "use_latent_outputs": use_latent_outputs,
    "python": python_label,
    "conda_env": conda_env,
}

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
}

run_eval() {
  local ckpt="$1"
  local gpu="$2"
  local series step run_id log_file started_at ended_at status

  IFS=$'\t' read -r series step run_id < <(ckpt_meta "${ckpt}")
  log_file="${LOG_ROOT}/${run_id}.${SUMMARY_STAMP}.launcher.log"

  trap 'release_gpu_lock "${gpu}"' EXIT

  started_at="$(date '+%Y-%m-%d %H:%M:%S')"
  {
    echo "START ${started_at} gpu=${gpu} ckpt=${ckpt}"
    echo "python=$(python_cmd_label)"
    echo "eval_log_dir=${EVAL_LOG_DIR}"
    echo "save_video=${SAVE_VIDEO} use_pro_version=${USE_PRO_VERSION} use_gate=${USE_GATE} use_latent_outputs=${USE_LATENT_OUTPUTS}"
    set +e
    run_python_eval "${gpu}" "${ckpt}" "${run_id}"
    status=$?
    set -e
    ended_at="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "END ${ended_at} status=${status} gpu=${gpu} ckpt=${ckpt}"
    write_run_meta "${ckpt}" "${gpu}" "${run_id}" "${log_file}" "${started_at}" "${ended_at}" "${status}"
    exit "${status}"
  } > "${log_file}" 2>&1
}

# 所有 checkpoint 完成后，汇总本轮 checkpoint 的最新 result.txt。
write_summary() {
  "${PY_HELPER_BIN}" - "${SUMMARY_RUN_LIST}" "${EVAL_LOG_DIR}" "${SUMMARY_CSV}" "${SUMMARY_TXT}" <<'PY'
import csv
import glob
import json
import os
import sys
from collections import defaultdict

run_list_path, eval_log_dir, csv_path, txt_path = sys.argv[1:]

rows = []
with open(run_list_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        series, step, run_id, ckpt = line.split("\t", 3)
        candidates = []
        for result_path in glob.glob(os.path.join(eval_log_dir, run_id + "-*", "result.txt")):
            try:
                candidates.append((os.path.getmtime(result_path), result_path))
            except OSError:
                continue

        status = "missing_result"
        eval_dir = ""
        avg_seq_len = ""
        chain_sr = {}
        if candidates:
            _, result_path = max(candidates)
            eval_dir = os.path.dirname(result_path)
            try:
                with open(result_path, "r", encoding="utf-8") as rf:
                    data = json.load(rf)
                # evaluate_calvin.py 当前用 epoch=None 写 JSON，key 通常是 "null"。
                payload = next(iter(data.values())) if data else {}
                avg_seq_len = payload.get("avg_seq_len", "")
                chain_sr = payload.get("chain_sr", {}) or {}
                status = "ok"
            except Exception as exc:
                status = f"parse_error:{type(exc).__name__}"

        def sr(idx):
            return chain_sr.get(str(idx), chain_sr.get(idx, ""))

        rows.append({
            "series": series,
            "step": step,
            "status": status,
            "avg_seq_len": avg_seq_len,
            "sr_1": sr(1),
            "sr_2": sr(2),
            "sr_3": sr(3),
            "sr_4": sr(4),
            "sr_5": sr(5),
            "checkpoint": ckpt,
            "eval_dir": eval_dir,
        })

def sort_key(row):
    try:
        step_value = int(row["step"])
    except ValueError:
        step_value = 10**18
    return (row["series"], step_value)

rows.sort(key=sort_key)

fieldnames = ["series", "step", "status", "avg_seq_len", "sr_1", "sr_2", "sr_3", "sr_4", "sr_5", "checkpoint", "eval_dir"]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

ok_rows = [r for r in rows if r["status"] == "ok" and r["avg_seq_len"] != ""]

def avg_value(row):
    return float(row["avg_seq_len"])

best_overall = max(ok_rows, key=avg_value) if ok_rows else None

best_by_series = {}
grouped = defaultdict(list)
for row in ok_rows:
    grouped[row["series"]].append(row)
for series, series_rows in grouped.items():
    best_by_series[series] = max(series_rows, key=avg_value)

with open(txt_path, "w", encoding="utf-8") as f:
    f.write("CALVIN auto eval summary\n")
    f.write(f"num_checkpoints={len(rows)}\n")
    f.write(f"num_finished={len(ok_rows)}\n")
    f.write(f"csv={csv_path}\n\n")

    if best_overall is not None:
        f.write("best_overall:\n")
        f.write(f"  series={best_overall['series']}\n")
        f.write(f"  step={best_overall['step']}\n")
        f.write(f"  avg_seq_len={best_overall['avg_seq_len']}\n")
        f.write(
            f"  sr_1={best_overall['sr_1']} sr_2={best_overall['sr_2']} "
            f"sr_3={best_overall['sr_3']} sr_4={best_overall['sr_4']} sr_5={best_overall['sr_5']}\n"
        )
        f.write(f"  checkpoint={best_overall['checkpoint']}\n")
        f.write(f"  eval_dir={best_overall['eval_dir']}\n\n")
    else:
        f.write("best_overall: none\n\n")

    f.write("best_by_series:\n")
    if best_by_series:
        for series in sorted(best_by_series):
            row = best_by_series[series]
            f.write(
                f"  series={series} step={row['step']} avg_seq_len={row['avg_seq_len']} "
                f"sr=[{row['sr_1']}, {row['sr_2']}, {row['sr_3']}, {row['sr_4']}, {row['sr_5']}] "
                f"ckpt={row['checkpoint']}\n"
            )
    else:
        f.write("  none\n")

    f.write("\nall_results:\n")
    for row in rows:
        f.write(
            f"  series={row['series']} step={row['step']} status={row['status']} "
            f"avg_seq_len={row['avg_seq_len']} "
            f"sr=[{row['sr_1']}, {row['sr_2']}, {row['sr_3']}, {row['sr_4']}, {row['sr_5']}] "
            f"ckpt={row['checkpoint']}\n"
        )

print(f"Summary CSV: {csv_path}")
print(f"Summary TXT: {txt_path}")
if best_overall is not None:
    print(
        "Best overall: "
        f"series={best_overall['series']} step={best_overall['step']} "
        f"avg_seq_len={best_overall['avg_seq_len']} ckpt={best_overall['checkpoint']}"
    )
else:
    print("Best overall: none")
PY
}

refresh_active_pids() {
  mapfile -t active_pids < <(jobs -pr)
}

cleanup_children() {
  local pid
  refresh_active_pids
  if (( ${#active_pids[@]} > 0 )); then
    echo "Stopping ${#active_pids[@]} active eval worker(s)..."
    for pid in "${active_pids[@]}"; do
      kill "${pid}" 2>/dev/null || true
    done
  fi
}

trap cleanup_children INT TERM

# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------

resolve_checkpoints

if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "No checkpoints found. target=${EVAL_TARGET:-<empty>} mode=${TARGET_MODE} checkpoint_root=${CHECKPOINT_ROOT}"
  exit 0
fi

echo "Auto CALVIN eval started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "TARGET_MODE=${TARGET_MODE}"
echo "TARGET=${EVAL_TARGET:-<empty>}"
echo "TARGET_DESC=${TARGET_DESC}"
echo "CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "EVAL_LOG_DIR=${EVAL_LOG_DIR}"
echo "PYTHON=$(python_cmd_label)"
echo "PY_HELPER_BIN=${PY_HELPER_BIN}"
echo "GPU_LIST=${GPU_LIST}"
echo "MAX_CONCURRENT=${MAX_CONCURRENT}"
echo "MAX_USED_MB=${MAX_USED_MB}"
echo "GPU_LOCK_DIR=${GPU_LOCK_DIR}"
echo "POLL_SECONDS=${POLL_SECONDS}"
echo "SKIP_COMPLETED=${SKIP_COMPLETED}"
echo "FORCE=${FORCE}"
echo "SAVE_VIDEO=${SAVE_VIDEO}"
echo "USE_PRO_VERSION=${USE_PRO_VERSION}"
echo "USE_GATE=${USE_GATE}"
echo "USE_LATENT_OUTPUTS=${USE_LATENT_OUTPUTS}"
echo "NUM_CHECKPOINTS=${#CHECKPOINTS[@]}"
echo "SUMMARY_CSV=${SUMMARY_CSV}"
echo "SUMMARY_TXT=${SUMMARY_TXT}"

: > "${SUMMARY_RUN_LIST}"
for ckpt in "${CHECKPOINTS[@]}"; do
  IFS=$'\t' read -r series step run_id < <(ckpt_meta "${ckpt}")
  printf '%s\t%s\t%s\t%s\n' "${series}" "${step}" "${run_id}" "${ckpt}" >> "${SUMMARY_RUN_LIST}"
done

echo "Checkpoint list:"
for ckpt in "${CHECKPOINTS[@]}"; do
  IFS=$'\t' read -r series step run_id < <(ckpt_meta "${ckpt}")
  echo "  step=${step} series=${series} ckpt=${ckpt}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1, no eval launched."
  write_summary
  exit 0
fi

active_pids=()

for ckpt in "${CHECKPOINTS[@]}"; do
  IFS=$'\t' read -r series step run_id < <(ckpt_meta "${ckpt}")

  if [[ "${FORCE}" != "1" && "${SKIP_COMPLETED}" == "1" ]] && has_completed_result "${run_id}"; then
    echo "Skipping completed checkpoint: step=${step} ckpt=${ckpt}"
    continue
  fi

  while true; do
    refresh_active_pids
    if (( ${#active_pids[@]} >= MAX_CONCURRENT )); then
      echo "Active workers=${#active_pids[@]} reaches MAX_CONCURRENT=${MAX_CONCURRENT}; waiting ${POLL_SECONDS}s"
      sleep "${POLL_SECONDS}"
      continue
    fi

    gpu="$(acquire_gpu || true)"
    if [[ -z "${gpu}" ]]; then
      echo "No unlocked GPU under ${MAX_USED_MB} MiB; waiting ${POLL_SECONDS}s"
      sleep "${POLL_SECONDS}"
      continue
    fi

    echo "Launching step=${step} on GPU ${gpu}: ${ckpt}"
    run_eval "${ckpt}" "${gpu}" &
    pid="$!"
    mark_gpu_lock "${gpu}" "${pid}" "${ckpt}"
    active_pids+=("${pid}")
    sleep 30
    break
  done
done

while true; do
  refresh_active_pids
  if (( ${#active_pids[@]} == 0 )); then
    break
  fi
  echo "Waiting for ${#active_pids[@]} active eval worker(s)..."
  sleep "${POLL_SECONDS}"
done

if ! wait; then
  echo "One or more eval workers exited with non-zero status. Check launcher logs under ${LOG_ROOT}."
fi

echo "Auto CALVIN eval completed at $(date '+%Y-%m-%d %H:%M:%S')"
write_summary
