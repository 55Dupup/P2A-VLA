#!/usr/bin/env bash
# ============================================================
# auto_eval_libero.sh
# 更安全的 LIBERO 自动评估脚本
#
# 设计目标:
# 1. 默认一次只评估 1 个模型，适合后台常驻
# 2. 仅在 GPU 满足条件时自动启动评估
# 3. 使用 manifest 记录每个 checkpoint 的评估状态，避免只靠旧日志误判
# 4. 评估时创建临时 staging checkpoint，避免评估过程改写原始 checkpoint
# 5. 自动维护每个系列的 best 标签目录，方便直接查找最佳权重
# 6. 日志文件名自动包含 success rate，便于肉眼检索
# 7. 默认不删除任何权重；删除行为需显式开启
#
# 使用前:
#   conda activate vla-eft
#   cd "$(dirname "${BASH_SOURCE[0]}")/.."
#
# 用法:
#   bash vla-scripts/auto_eval_libero.sh [选项]
#
# 选项:
#   -g <GPU列表>       指定允许使用的 GPU，逗号分隔，默认 "auto"（全部）
#   -m <显存阈值MB>    至少需要多少空闲显存才启动 eval，默认 10000
#   -u <利用率阈值>    GPU 利用率上限，默认 30
#   -p <轮询秒数>      检查 GPU 的间隔秒数，默认 45
#   -f <glob>          checkpoint 过滤规则，默认 "*libero*"
#   -o <输出目录>      权重目录，默认 "./outputs"
#   -l <日志目录>      日志目录，默认 "./experiments/libero_eval_logs/auto_eval"
#   -x <true|false>    是否允许使用完全空闲的卡，默认 false
#   -k <true|false>    评估完成后是否删除每个系列的非最佳权重，默认 false
#   -y <true|false>    删除前是否跳过确认，默认 false
#   -n <true|false>    dry-run，只打印将要执行的任务，默认 false
#   -h                 显示帮助
#
# 关键产物:
#   1. 单模型日志:
#      ${LOG_DIR}/${checkpoint_name}__sr-0.9860.log
#   2. 状态文件:
#      ${LOG_DIR}/status/${checkpoint_name}.state
#   3. 最佳权重标签目录:
#      ${OUTPUT_DIR}/_auto_eval_best/
# ============================================================

set -euo pipefail

GPUS=0 #"auto"
MIN_FREE_MB=10000
MAX_UTIL=30
POLL_SECONDS=45
FILTER_GLOB="*libero*"
OUTPUT_DIR="./outputs"
LOG_DIR="./experiments/libero_eval_logs/auto_eval"
ALLOW_EMPTY_GPU=false
PRUNE_LOSERS=false
ASSUME_YES=false
DRY_RUN=false
PYTHON_BIN="${PYTHON_BIN:-python}"
EMPTY_GPU_USED_MB=512

ROOT_DIR=""
STATE_DIR=""
INTERNAL_LOG_DIR=""
STAGING_ROOT=""
LABEL_DIR=""
LOCK_DIR=""
CURRENT_STAGING_DIR=""

usage() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//'
}

normalize_bool() {
    local value="${1:-}"
    value="${value,,}"
    case "$value" in
        true|1|yes|y) echo "true" ;;
        false|0|no|n) echo "false" ;;
        *)
            echo "[ERROR] 非法布尔值: ${1}" >&2
            exit 1
            ;;
    esac
}

trim_spaces() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

resolve_path() {
    local path="$1"
    if [[ "$path" == /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s\n' "${ROOT_DIR}/${path#./}"
    fi
}

validate_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] ${name} 必须是非负整数，当前值: ${value}" >&2
        exit 1
    fi
}

while getopts "g:m:u:p:f:o:l:x:k:y:n:h" opt; do
    case "$opt" in
        g) GPUS="$OPTARG" ;;
        m) MIN_FREE_MB="$OPTARG" ;;
        u) MAX_UTIL="$OPTARG" ;;
        p) POLL_SECONDS="$OPTARG" ;;
        f) FILTER_GLOB="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        l) LOG_DIR="$OPTARG" ;;
        x) ALLOW_EMPTY_GPU="$(normalize_bool "$OPTARG")" ;;
        k) PRUNE_LOSERS="$(normalize_bool "$OPTARG")" ;;
        y) ASSUME_YES="$(normalize_bool "$OPTARG")" ;;
        n) DRY_RUN="$(normalize_bool "$OPTARG")" ;;
        h)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] 未知选项: -${opt}" >&2
            exit 1
            ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="$(resolve_path "$OUTPUT_DIR")"
LOG_DIR="$(resolve_path "$LOG_DIR")"
STATE_DIR="${LOG_DIR}/status"
INTERNAL_LOG_DIR="${LOG_DIR}/internal"
STAGING_ROOT="${LOG_DIR}/staging"
LABEL_DIR="${OUTPUT_DIR}/_auto_eval_best"
LOCK_DIR="${LOG_DIR}/.scheduler.lock"

validate_integer "MIN_FREE_MB" "$MIN_FREE_MB"
validate_integer "MAX_UTIL" "$MAX_UTIL"
validate_integer "POLL_SECONDS" "$POLL_SECONDS"
validate_integer "EMPTY_GPU_USED_MB" "$EMPTY_GPU_USED_MB"

if (( MAX_UTIL > 100 )); then
    echo "[ERROR] MAX_UTIL 不应超过 100，当前值: ${MAX_UTIL}" >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$STATE_DIR" "$INTERNAL_LOG_DIR" "$STAGING_ROOT" "$LABEL_DIR"

cleanup_staging_dir() {
    if [[ -n "$CURRENT_STAGING_DIR" && -d "$CURRENT_STAGING_DIR" ]]; then
        rm -rf "$CURRENT_STAGING_DIR"
        CURRENT_STAGING_DIR=""
    fi
}

cleanup_on_exit() {
    cleanup_staging_dir
    if [[ -d "$LOCK_DIR" ]]; then
        rm -f "${LOCK_DIR}/pid"
        rmdir "$LOCK_DIR" 2>/dev/null || true
    fi
}

trap cleanup_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "[ERROR] 检测到已有一个 auto_eval_libero.sh 实例正在运行。锁目录: ${LOCK_DIR}" >&2
    exit 1
fi
printf '%s\n' "$$" > "${LOCK_DIR}/pid"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] 找不到 Python 可执行文件: ${PYTHON_BIN}" >&2
    exit 1
fi

if ! command -v find >/dev/null 2>&1 || ! command -v sort >/dev/null 2>&1; then
    echo "[ERROR] 需要 find 和 sort 命令。" >&2
    exit 1
fi

if [[ "$DRY_RUN" != "true" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] 需要 nvidia-smi 才能判断 GPU 空闲状态。" >&2
    exit 1
fi

gpu_allowed() {
    local gpu="$1"
    if [[ "$GPUS" == "auto" ]]; then
        return 0
    fi

    local allowed
    IFS=',' read -ra allowed <<< "$GPUS"
    local item
    for item in "${allowed[@]}"; do
        item="$(trim_spaces "$item")"
        if [[ "$item" == "$gpu" ]]; then
            return 0
        fi
    done
    return 1
}

series_name_from_checkpoint() {
    local checkpoint_name="$1"
    echo "${checkpoint_name%%--*_chkpt}"
}

should_skip_checkpoint() {
    local checkpoint_name="$1"
    local lower_name="${checkpoint_name,,}"

    if [[ "$lower_name" != *libero* ]]; then
        return 0
    fi

    if [[ "$lower_name" == best-* ]] || [[ "$lower_name" == old-* ]]; then
        return 0
    fi

    return 1
}

infer_task_suite() {
    local checkpoint_name="$1"
    local lower_name="${checkpoint_name,,}"

    if [[ "$lower_name" == *libero_spatial* ]]; then
        echo "libero_spatial"
    elif [[ "$lower_name" == *libero_object* ]]; then
        echo "libero_object"
    elif [[ "$lower_name" == *libero_goal* ]]; then
        echo "libero_goal"
    elif [[ "$lower_name" == *libero_10* ]] || [[ "$lower_name" == *libero-long* ]] || [[ "$lower_name" == *libero_long* ]]; then
        echo "libero_10"
    elif [[ "$lower_name" == *libero_90* ]]; then
        echo "libero_90"
    else
        return 1
    fi
}

infer_use_gate() {
    local checkpoint_name="$1"
    local lower_name="${checkpoint_name,,}"
    if [[ "$lower_name" =~ (^|[-_])gated?([_-]|$) ]]; then
        echo "True"
    else
        echo "False"
    fi
}

infer_use_latent_outputs() {
    local checkpoint_name="$1"
    local lower_name="${checkpoint_name,,}"
    if [[ "$lower_name" =~ (^|[-_])l(att)?ent([_-]|$) ]]; then
        echo "True"
    else
        echo "False"
    fi
}

infer_use_pro_version() {
    local checkpoint_name="$1"
    local lower_name="${checkpoint_name,,}"
    if [[ "$lower_name" =~ (^|[-_])pro([_-]|$) ]]; then
        echo "True"
    else
        echo "False"
    fi
}

extract_success_rate() {
    local log_file="$1"
    if [[ ! -f "$log_file" ]]; then
        return 1
    fi

    awk '/Overall success rate:/ {rate=$4} END {if (rate != "") print rate; else exit 1}' "$log_file"
}

extract_internal_log_path() {
    local log_file="$1"
    awk -F': ' '/Logging to local log file:/ {print $NF; exit}' "$log_file"
}

checkpoint_latest_mtime() {
    local checkpoint_dir="$1"
    local latest
    latest="$(
        find "$checkpoint_dir" -mindepth 1 -maxdepth 1 -printf '%T@\n' 2>/dev/null \
        | awk 'BEGIN{max=0} {if ($1 > max) max=$1} END {if (max > 0) printf "%.0f\n", max}'
    )"

    if [[ -n "$latest" ]]; then
        printf '%s\n' "$latest"
    else
        stat -c '%Y' "$checkpoint_dir"
    fi
}

checkpoint_signature() {
    local checkpoint_dir="$1"

    if command -v sha256sum >/dev/null 2>&1; then
        find "$checkpoint_dir" -mindepth 1 -maxdepth 1 -printf '%P\t%s\t%T@\t%l\n' 2>/dev/null \
        | LC_ALL=C sort \
        | sha256sum \
        | awk '{print $1}'
    else
        checkpoint_latest_mtime "$checkpoint_dir"
    fi
}

state_file_for_checkpoint() {
    local checkpoint_name="$1"
    printf '%s/%s.state\n' "$STATE_DIR" "$checkpoint_name"
}

write_state_file() {
    local state_file="$1"
    shift

    : > "$state_file"
    while (( $# >= 2 )); do
        printf '%s=%q\n' "$1" "$2" >> "$state_file"
        shift 2
    done
}

load_state_file() {
    local state_file="$1"
    [[ -f "$state_file" ]] || return 1
    unset -v status checkpoint_name checkpoint_dir checkpoint_signature checkpoint_latest_mtime
    unset -v task_suite use_gate use_latent use_pro rate final_log internal_log
    unset -v source gpu started_at finished_at
    # shellcheck disable=SC1090
    source "$state_file"
}

record_legacy_log_if_fresh() {
    local checkpoint_dir="$1"
    local checkpoint_name="$2"
    local task_suite="$3"
    local use_gate="$4"
    local use_latent="$5"
    local use_pro="$6"
    local state_file="$7"
    local signature="$8"
    local latest_mtime="$9"
    local legacy_log="${LOG_DIR}/${checkpoint_name}.log"

    if [[ ! -f "$legacy_log" ]]; then
        return 1
    fi

    local legacy_rate
    if ! legacy_rate="$(extract_success_rate "$legacy_log" 2>/dev/null)"; then
        return 1
    fi

    local legacy_log_mtime
    legacy_log_mtime="$(stat -c '%Y' "$legacy_log")"
    if (( legacy_log_mtime < latest_mtime )); then
        return 1
    fi

    write_state_file "$state_file" \
        status success \
        checkpoint_name "$checkpoint_name" \
        checkpoint_dir "$checkpoint_dir" \
        checkpoint_signature "$signature" \
        checkpoint_latest_mtime "$latest_mtime" \
        task_suite "$task_suite" \
        use_gate "$use_gate" \
        use_latent "$use_latent" \
        use_pro "$use_pro" \
        rate "$legacy_rate" \
        final_log "$legacy_log" \
        internal_log "" \
        source legacy_log \
        gpu unknown \
        started_at unknown \
        finished_at "$(date '+%F %T')"

    return 0
}

state_is_fresh_success() {
    local state_file="$1"
    local signature="$2"

    if ! load_state_file "$state_file"; then
        return 1
    fi

    [[ "${status:-}" == "success" ]] || return 1
    [[ "${checkpoint_signature:-}" == "$signature" ]] || return 1
    [[ -n "${rate:-}" ]] || return 1
    [[ -n "${final_log:-}" && -f "${final_log:-}" ]] || return 1
    return 0
}

print_gpu_snapshot() {
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits \
    | while IFS=',' read -r index used free util; do
        index="$(trim_spaces "$index")"
        used="$(trim_spaces "$used")"
        free="$(trim_spaces "$free")"
        util="$(trim_spaces "$util")"
        [[ "$util" == "[Not Supported]" ]] && util="0"
        if gpu_allowed "$index"; then
            echo "  GPU ${index}: used=${used}MB free=${free}MB util=${util}%"
        fi
    done
}

pick_gpu() {
    local best_busy_gpu=""
    local best_busy_free=-1
    local best_busy_util=999
    local best_idle_gpu=""
    local best_idle_free=-1
    local best_idle_util=999

    while IFS=',' read -r index _total used free util; do
        index="$(trim_spaces "$index")"
        used="$(trim_spaces "$used")"
        free="$(trim_spaces "$free")"
        util="$(trim_spaces "$util")"

        [[ "$util" == "[Not Supported]" ]] && util="0"

        gpu_allowed "$index" || continue
        (( free < MIN_FREE_MB )) && continue
        (( util > MAX_UTIL )) && continue

        if (( used > EMPTY_GPU_USED_MB )); then
            if (( free > best_busy_free )) || { (( free == best_busy_free )) && (( util < best_busy_util )); }; then
                best_busy_gpu="$index"
                best_busy_free="$free"
                best_busy_util="$util"
            fi
        else
            if (( free > best_idle_free )) || { (( free == best_idle_free )) && (( util < best_idle_util )); }; then
                best_idle_gpu="$index"
                best_idle_free="$free"
                best_idle_util="$util"
            fi
        fi
    done < <(nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits)

    if [[ -n "$best_busy_gpu" ]]; then
        echo "$best_busy_gpu"
        return 0
    fi

    if [[ "$ALLOW_EMPTY_GPU" == "true" ]] && [[ -n "$best_idle_gpu" ]]; then
        echo "$best_idle_gpu"
        return 0
    fi

    return 1
}

wait_for_gpu() {
    local gpu=""

    while true; do
        if gpu="$(pick_gpu)"; then
            echo "$gpu"
            return 0
        fi

        echo "[$(date '+%F %T')] 暂无满足条件的 GPU。要求: free>=${MIN_FREE_MB}MB, util<=${MAX_UTIL}%." >&2
        if [[ "$ALLOW_EMPTY_GPU" == "false" ]]; then
            echo "  当前策略: 不使用完全空闲的卡。" >&2
        fi
        print_gpu_snapshot >&2
        sleep "$POLL_SECONDS"
    done
}

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

run_eval() {
    local checkpoint_dir="$1"
    local gpu="$2"
    local checkpoint_name="$3"
    local task_suite="$4"
    local use_gate="$5"
    local use_latent="$6"
    local use_pro="$7"
    local log_file="$8"

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" ./experiments/robot/libero/run_libero_eval.py \
        --use_proprio True \
        --num_images_in_input 2 \
        --use_film False \
        --local_log_dir "${INTERNAL_LOG_DIR}" \
        --log_name "auto-${checkpoint_name}" \
        --save_video False \
        --save_version vla-adapter \
        --local_video_dir ./experiments/libero_eval_videos \
        --pretrained_checkpoint "${checkpoint_dir}" \
        --task_suite_name "${task_suite}" \
        --use_pro_version "${use_pro}" \
        --use_gate "${use_gate}" \
        --use_latent_outputs "${use_latent}" \
        > "${log_file}" 2>&1
}

update_best_label() {
    local series_name="$1"
    local checkpoint_name="$2"
    local rate="$3"
    local checkpoint_dir="${OUTPUT_DIR}/${checkpoint_name}"
    local stable_link="${LABEL_DIR}/${series_name}"
    local rate_link="${LABEL_DIR}/${series_name}__sr-${rate}"
    local info_file="${LABEL_DIR}/${series_name}.txt"

    shopt -s nullglob
    rm -f "${LABEL_DIR}/${series_name}__sr-"*
    shopt -u nullglob

    ln -sfn "../${checkpoint_name}" "$stable_link"
    ln -sfn "../${checkpoint_name}" "$rate_link"

    {
        echo "series=${series_name}"
        echo "checkpoint_name=${checkpoint_name}"
        echo "checkpoint_dir=${checkpoint_dir}"
        echo "rate=${rate}"
        echo "updated_at=$(date '+%F %T')"
    } > "$info_file"
}

echo "============================================="
echo "  更安全的 LIBERO 自动评估"
echo "  GPU 列表:        ${GPUS}"
echo "  显存阈值:        ${MIN_FREE_MB} MB"
echo "  利用率阈值:      ${MAX_UTIL}%"
echo "  轮询间隔:        ${POLL_SECONDS} s"
echo "  checkpoint 过滤: ${FILTER_GLOB}"
echo "  输出目录:        ${OUTPUT_DIR}"
echo "  日志目录:        ${LOG_DIR}"
echo "  状态目录:        ${STATE_DIR}"
echo "  标签目录:        ${LABEL_DIR}"
echo "  允许空卡:        ${ALLOW_EMPTY_GPU}"
echo "  删除非最佳权重:  ${PRUNE_LOSERS}"
echo "  Dry Run:         ${DRY_RUN}"
echo "============================================="

CHKPT_DIRS=()
while IFS= read -r -d '' dir; do
    checkpoint_name="$(basename "$dir")"
    if should_skip_checkpoint "$checkpoint_name"; then
        continue
    fi
    CHKPT_DIRS+=("$dir")
done < <(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "${FILTER_GLOB}--*_chkpt" -print0 | sort -zV)

if [[ ${#CHKPT_DIRS[@]} -eq 0 ]]; then
    echo "[ERROR] 没有找到待评估的 LIBERO checkpoint。"
    exit 1
fi

echo "待处理 checkpoint 数量: ${#CHKPT_DIRS[@]}"
for checkpoint_dir in "${CHKPT_DIRS[@]}"; do
    echo "  - $(basename "$checkpoint_dir")"
done
echo ""

for checkpoint_dir in "${CHKPT_DIRS[@]}"; do
    checkpoint_name="$(basename "$checkpoint_dir")"
    state_file="$(state_file_for_checkpoint "$checkpoint_name")"
    checkpoint_sig="$(checkpoint_signature "$checkpoint_dir")"
    checkpoint_mtime="$(checkpoint_latest_mtime "$checkpoint_dir")"

    if ! task_suite="$(infer_task_suite "$checkpoint_name")"; then
        echo "[SKIP] ${checkpoint_name} 无法从前缀推断 task_suite_name"
        continue
    fi

    use_gate="$(infer_use_gate "$checkpoint_name")"
    use_latent="$(infer_use_latent_outputs "$checkpoint_name")"
    use_pro="$(infer_use_pro_version "$checkpoint_name")"

    if ! state_is_fresh_success "$state_file" "$checkpoint_sig"; then
        record_legacy_log_if_fresh "$checkpoint_dir" "$checkpoint_name" "$task_suite" "$use_gate" "$use_latent" "$use_pro" "$state_file" "$checkpoint_sig" "$checkpoint_mtime" >/dev/null 2>&1 || true
    fi

    if state_is_fresh_success "$state_file" "$checkpoint_sig"; then
        echo "[SKIP] ${checkpoint_name} 已评估完成，success rate=${rate} | log=${final_log}"
        continue
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] ${checkpoint_name}"
        echo "  task_suite=${task_suite} use_gate=${use_gate} use_latent_outputs=${use_latent} use_pro_version=${use_pro}"
        echo "  state_file=${state_file}"
        echo "  result_log=${LOG_DIR}/${checkpoint_name}__sr-<rate>.log"
        continue
    fi

    shell_log_tmp="${LOG_DIR}/${checkpoint_name}.running.log"
    started_at="$(date '+%F %T')"

    write_state_file "$state_file" \
        status running \
        checkpoint_name "$checkpoint_name" \
        checkpoint_dir "$checkpoint_dir" \
        checkpoint_signature "$checkpoint_sig" \
        checkpoint_latest_mtime "$checkpoint_mtime" \
        task_suite "$task_suite" \
        use_gate "$use_gate" \
        use_latent "$use_latent" \
        use_pro "$use_pro" \
        rate "" \
        final_log "$shell_log_tmp" \
        internal_log "" \
        source staging_eval \
        gpu pending \
        started_at "$started_at" \
        finished_at ""

    gpu="$(wait_for_gpu)"
    CURRENT_STAGING_DIR="$(prepare_staging_checkpoint "$checkpoint_dir" "$checkpoint_name")"

    echo "[START] $(date '+%F %T') | GPU=${gpu} | ${checkpoint_name}"
    echo "  task_suite=${task_suite} use_gate=${use_gate} use_latent_outputs=${use_latent} use_pro_version=${use_pro}"
    echo "  staging_dir=${CURRENT_STAGING_DIR}"

    if run_eval "$CURRENT_STAGING_DIR" "$gpu" "$checkpoint_name" "$task_suite" "$use_gate" "$use_latent" "$use_pro" "$shell_log_tmp"; then
        internal_log="$(extract_internal_log_path "$shell_log_tmp" 2>/dev/null || true)"
        finished_at="$(date '+%F %T')"

        if rate="$(extract_success_rate "$shell_log_tmp" 2>/dev/null)"; then
            final_log="${LOG_DIR}/${checkpoint_name}__sr-${rate}.log"
            mv -f "$shell_log_tmp" "$final_log"

            write_state_file "$state_file" \
                status success \
                checkpoint_name "$checkpoint_name" \
                checkpoint_dir "$checkpoint_dir" \
                checkpoint_signature "$checkpoint_sig" \
                checkpoint_latest_mtime "$checkpoint_mtime" \
                task_suite "$task_suite" \
                use_gate "$use_gate" \
                use_latent "$use_latent" \
                use_pro "$use_pro" \
                rate "$rate" \
                final_log "$final_log" \
                internal_log "$internal_log" \
                source staging_eval \
                gpu "$gpu" \
                started_at "$started_at" \
                finished_at "$finished_at"

            echo "[DONE] ${checkpoint_name} | success rate=${rate} | log=${final_log}"
        else
            final_log="${LOG_DIR}/${checkpoint_name}__finished-no-rate-$(date +%Y%m%d_%H%M%S).log"
            mv -f "$shell_log_tmp" "$final_log"

            write_state_file "$state_file" \
                status finished_without_rate \
                checkpoint_name "$checkpoint_name" \
                checkpoint_dir "$checkpoint_dir" \
                checkpoint_signature "$checkpoint_sig" \
                checkpoint_latest_mtime "$checkpoint_mtime" \
                task_suite "$task_suite" \
                use_gate "$use_gate" \
                use_latent "$use_latent" \
                use_pro "$use_pro" \
                rate "" \
                final_log "$final_log" \
                internal_log "$internal_log" \
                source staging_eval \
                gpu "$gpu" \
                started_at "$started_at" \
                finished_at "$finished_at"

            echo "[WARN] ${checkpoint_name} 运行结束，但未在日志中找到 success rate。log=${final_log}"
        fi
    else
        internal_log="$(extract_internal_log_path "$shell_log_tmp" 2>/dev/null || true)"
        finished_at="$(date '+%F %T')"
        final_log="${LOG_DIR}/${checkpoint_name}__failed-$(date +%Y%m%d_%H%M%S).log"
        mv -f "$shell_log_tmp" "$final_log" || true

        write_state_file "$state_file" \
            status failed \
            checkpoint_name "$checkpoint_name" \
            checkpoint_dir "$checkpoint_dir" \
            checkpoint_signature "$checkpoint_sig" \
            checkpoint_latest_mtime "$checkpoint_mtime" \
            task_suite "$task_suite" \
            use_gate "$use_gate" \
            use_latent "$use_latent" \
            use_pro "$use_pro" \
            rate "" \
            final_log "$final_log" \
            internal_log "$internal_log" \
            source staging_eval \
            gpu "$gpu" \
            started_at "$started_at" \
            finished_at "$finished_at"

        echo "[FAIL] ${checkpoint_name} 评估失败，请检查日志: ${final_log}"
    fi

    cleanup_staging_dir
    echo ""
done

echo "============================================="
echo "  评估汇总"
echo "============================================="

declare -A BEST_RATE_BY_SERIES=()
declare -A BEST_CHKPT_BY_SERIES=()

for checkpoint_dir in "${CHKPT_DIRS[@]}"; do
    checkpoint_name="$(basename "$checkpoint_dir")"
    series_name="$(series_name_from_checkpoint "$checkpoint_name")"
    state_file="$(state_file_for_checkpoint "$checkpoint_name")"
    checkpoint_sig="$(checkpoint_signature "$checkpoint_dir")"

    if state_is_fresh_success "$state_file" "$checkpoint_sig"; then
        printf "  %-80s %s\n" "$checkpoint_name" "$rate"
        if [[ -z "${BEST_RATE_BY_SERIES[$series_name]:-}" ]] || awk "BEGIN{exit !($rate > ${BEST_RATE_BY_SERIES[$series_name]})}"; then
            BEST_RATE_BY_SERIES["$series_name"]="$rate"
            BEST_CHKPT_BY_SERIES["$series_name"]="$checkpoint_name"
        fi
    else
        printf "  %-80s %s\n" "$checkpoint_name" "N/A"
    fi
done

echo "---------------------------------------------"
echo "  每个系列的最佳权重"
echo "---------------------------------------------"

BEST_REPORT="${LOG_DIR}/best_by_series_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "Best checkpoint per series - $(date)"
    echo "============================================="
    for series_name in "${!BEST_CHKPT_BY_SERIES[@]}"; do
        echo "${series_name} -> ${BEST_CHKPT_BY_SERIES[$series_name]} (rate=${BEST_RATE_BY_SERIES[$series_name]})"
    done | sort
} > "$BEST_REPORT"

for series_name in "${!BEST_CHKPT_BY_SERIES[@]}"; do
    update_best_label "$series_name" "${BEST_CHKPT_BY_SERIES[$series_name]}" "${BEST_RATE_BY_SERIES[$series_name]}"
done

ln -sfn "$(basename "$BEST_REPORT")" "${LOG_DIR}/best_by_series_latest.txt"
cat "$BEST_REPORT"

if [[ "$PRUNE_LOSERS" == "true" ]]; then
    PRUNE_LIST=()
    for checkpoint_dir in "${CHKPT_DIRS[@]}"; do
        checkpoint_name="$(basename "$checkpoint_dir")"
        series_name="$(series_name_from_checkpoint "$checkpoint_name")"
        state_file="$(state_file_for_checkpoint "$checkpoint_name")"
        checkpoint_sig="$(checkpoint_signature "$checkpoint_dir")"

        if state_is_fresh_success "$state_file" "$checkpoint_sig"; then
            if [[ "${BEST_CHKPT_BY_SERIES[$series_name]:-}" != "$checkpoint_name" ]]; then
                PRUNE_LIST+=("$checkpoint_dir")
            fi
        fi
    done

    if [[ ${#PRUNE_LIST[@]} -eq 0 ]]; then
        echo "[INFO] 没有可删除的非最佳权重。"
        exit 0
    fi

    echo ""
    echo "以下非最佳权重将被删除:"
    for checkpoint_dir in "${PRUNE_LIST[@]}"; do
        echo "  - $(basename "$checkpoint_dir")"
    done

    if [[ "$ASSUME_YES" != "true" ]]; then
        echo ""
        read -r -p "确认删除以上权重？[y/N] " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "[SKIP] 已取消删除操作。"
            exit 0
        fi
    fi

    for checkpoint_dir in "${PRUNE_LIST[@]}"; do
        echo "[DEL] $(basename "$checkpoint_dir")"
        rm -rf "$checkpoint_dir"
    done
fi

echo ""
echo "最佳权重汇总已写入: ${BEST_REPORT}"
echo "最佳权重标签目录: ${LABEL_DIR}"
echo "完成。"
