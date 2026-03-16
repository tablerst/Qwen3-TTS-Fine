#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_STAGE_CONFIG="${SERVICE_ROOT}/configs/qwen3_tts.stage.yaml"
AGGRESSIVE_STAGE_CONFIG="${SERVICE_ROOT}/configs/qwen3_tts.aggressive.stage.yaml"
LOCAL_VENV_BIN="${SERVICE_ROOT}/.venv/bin"

TASK_TYPE="CustomVoice"
MODEL=""
STAGE_CONFIG=""
PROFILE="default"
HOST="0.0.0.0"
PORT="8091"
GPU_MEMORY_UTILIZATION="0.9"
LOG_LEVEL="info"
GLOBAL_ENFORCE_EAGER="1"
ENABLE_PREFIX_CACHING=""
PERFORMANCE_MODE=""
OPTIMIZATION_LEVEL=""
LOG_DIR="${SERVICE_ROOT}/logs"
LOG_PREFIX="vllm_omni_qwen3_tts"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CAPTURE_LOGS="1"
DRY_RUN="0"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --task-type)
            TASK_TYPE="$2"
            shift 2
            ;;
        --stage-config)
            STAGE_CONFIG="$2"
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --uvicorn-log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --global-enforce-eager)
            GLOBAL_ENFORCE_EAGER="1"
            shift
            ;;
        --no-global-enforce-eager)
            GLOBAL_ENFORCE_EAGER="0"
            shift
            ;;
        --enable-prefix-caching)
            ENABLE_PREFIX_CACHING="1"
            shift
            ;;
        --no-enable-prefix-caching)
            ENABLE_PREFIX_CACHING="0"
            shift
            ;;
        --performance-mode)
            PERFORMANCE_MODE="$2"
            shift 2
            ;;
        --optimization-level)
            OPTIMIZATION_LEVEL="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --log-prefix)
            LOG_PREFIX="$2"
            shift 2
            ;;
        --no-capture-logs)
            CAPTURE_LOGS="0"
            shift
            ;;
        --dry-run)
            DRY_RUN="1"
            shift
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

case "$PROFILE" in
    default)
        ;;
    aggressive)
        if [[ -z "$STAGE_CONFIG" && -f "$AGGRESSIVE_STAGE_CONFIG" ]]; then
            STAGE_CONFIG="$AGGRESSIVE_STAGE_CONFIG"
        fi
        GLOBAL_ENFORCE_EAGER="0"
        if [[ -z "$PERFORMANCE_MODE" ]]; then
            PERFORMANCE_MODE="interactivity"
        fi
        if [[ -z "$OPTIMIZATION_LEVEL" ]]; then
            OPTIMIZATION_LEVEL="3"
        fi
        if [[ -z "$ENABLE_PREFIX_CACHING" ]]; then
            ENABLE_PREFIX_CACHING="1"
        fi
        LOG_PREFIX="${LOG_PREFIX}_aggressive"
        ;;
    *)
        echo "Unsupported profile: $PROFILE" >&2
        echo "Expected one of: default, aggressive" >&2
        exit 1
        ;;
esac

if [[ -z "$STAGE_CONFIG" ]]; then
    if [[ -f "$LOCAL_STAGE_CONFIG" ]]; then
        STAGE_CONFIG="$LOCAL_STAGE_CONFIG"
    else
        STAGE_CONFIG="vllm_omni/model_executor/stage_configs/qwen3_tts.yaml"
    fi
fi

if [[ -z "$MODEL" ]]; then
    case "$TASK_TYPE" in
        CustomVoice)
            MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            ;;
        VoiceDesign)
            MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
            ;;
        Base)
            MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
            ;;
        *)
            echo "Unsupported task type: $TASK_TYPE" >&2
            echo "Expected one of: CustomVoice, VoiceDesign, Base" >&2
            exit 1
            ;;
    esac
fi

if [[ -x "${LOCAL_VENV_BIN}/vllm-omni" ]]; then
    CLI_BIN="${LOCAL_VENV_BIN}/vllm-omni"
elif [[ -x "${LOCAL_VENV_BIN}/vllm" ]]; then
    CLI_BIN="${LOCAL_VENV_BIN}/vllm"
elif command -v vllm-omni >/dev/null 2>&1; then
    CLI_BIN="vllm-omni"
elif command -v vllm >/dev/null 2>&1; then
    CLI_BIN="vllm"
else
    echo "Neither 'vllm-omni' nor 'vllm' was found in PATH." >&2
    echo "Also checked local service environment: ${LOCAL_VENV_BIN}" >&2
    echo "Install vllm-omni first, then re-run this script." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
STDOUT_LOG="${LOG_DIR}/${LOG_PREFIX}_${TIMESTAMP}.stdout.log"
STDERR_LOG="${LOG_DIR}/${LOG_PREFIX}_${TIMESTAMP}.stderr.log"
META_LOG="${LOG_DIR}/${LOG_PREFIX}_${TIMESTAMP}.meta.log"

CMD=(
    "$CLI_BIN" serve "$MODEL"
    --stage-configs-path "$STAGE_CONFIG"
    --host "$HOST"
    --port "$PORT"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --trust-remote-code
    --omni
    --uvicorn-log-level "$LOG_LEVEL"
)

if [[ "$GLOBAL_ENFORCE_EAGER" == "1" ]]; then
    CMD+=(--enforce-eager)
else
    CMD+=(--no-enforce-eager)
fi

if [[ -n "$ENABLE_PREFIX_CACHING" ]]; then
    if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
        CMD+=(--enable-prefix-caching)
    else
        CMD+=(--no-enable-prefix-caching)
    fi
fi

if [[ -n "$PERFORMANCE_MODE" ]]; then
    CMD+=(--performance-mode "$PERFORMANCE_MODE")
fi

if [[ -n "$OPTIMIZATION_LEVEL" ]]; then
    CMD+=(--optimization-level "$OPTIMIZATION_LEVEL")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Using CLI: $CLI_BIN"
echo "Model: $MODEL"
echo "Task type: $TASK_TYPE"
echo "Profile: $PROFILE"
echo "Stage config: $STAGE_CONFIG"
echo "Host: $HOST"
echo "Port: $PORT"
echo "GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo "Global enforce eager: $GLOBAL_ENFORCE_EAGER"
echo "Enable prefix caching: ${ENABLE_PREFIX_CACHING:-auto}"
echo "Performance mode: ${PERFORMANCE_MODE:-auto}"
echo "Optimization level: ${OPTIMIZATION_LEVEL:-auto}"
echo "Log capture: $CAPTURE_LOGS"
echo "Log dir: $LOG_DIR"

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

{
    echo "timestamp=$TIMESTAMP"
    echo "pwd=$(pwd)"
    echo "host=$HOST"
    echo "port=$PORT"
    echo "model=$MODEL"
    echo "task_type=$TASK_TYPE"
    echo "profile=$PROFILE"
    echo "stage_config=$STAGE_CONFIG"
    echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    echo "global_enforce_eager=$GLOBAL_ENFORCE_EAGER"
    echo "enable_prefix_caching=${ENABLE_PREFIX_CACHING:-auto}"
    echo "performance_mode=${PERFORMANCE_MODE:-auto}"
    echo "optimization_level=${OPTIMIZATION_LEVEL:-auto}"
    echo "cli_bin=$CLI_BIN"
    echo "stdout_log=$STDOUT_LOG"
    echo "stderr_log=$STDERR_LOG"
    echo "uname=$(uname -a)"
    echo "hostname=$(hostname)"
    echo "WSL_DISTRO_NAME=${WSL_DISTRO_NAME:-}"
    echo "http_proxy=${http_proxy:-${HTTP_PROXY:-}}"
    echo "https_proxy=${https_proxy:-${HTTPS_PROXY:-}}"
    echo "no_proxy=${no_proxy:-${NO_PROXY:-}}"
    printf 'command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
} > "$META_LOG"

if [[ "$DRY_RUN" == "1" ]]; then
    exit 0
fi

if [[ "$CAPTURE_LOGS" == "1" ]]; then
    exec > >(tee -a "$STDOUT_LOG") 2> >(tee -a "$STDERR_LOG" >&2)
    echo "[serve.sh] stdout log: $STDOUT_LOG"
    echo "[serve.sh] stderr log: $STDERR_LOG"
    echo "[serve.sh] meta log: $META_LOG"
fi

exec "${CMD[@]}"
