#!/usr/bin/env bash
# Shared evaluation settings and paths. Sourced by the four eval entry points.
# Edit defaults here or export the corresponding variables before launching.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

# One output directory per policy checkpoint and answer-model configuration.
# These can also be supplied with --checkpoint-path and --output-dir.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-$PROJECT_ROOT/data}"
RESOURCE_CONFIG="${RESOURCE_CONFIG:-$PROJECT_ROOT/configs/runtime_memory_tool.yaml}"

# Credentials and endpoints are inherited from the environment. Uncomment an
# export to set it here; script exports override the caller's values.
# export OPENROUTER_API_KEY="your_openrouter_api_key"
# export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
# Optional separate credentials/endpoints for answer replacement and judging:
# export ANSWER_API_KEY="your_answer_api_key"
# export ANSWER_BASE_URL="https://your-answer-provider.example/v1"
# export JUDGE_API_KEY="your_judge_api_key"
# export JUDGE_BASE_URL="https://your-judge-provider.example/v1"

# Policy inference. MODEL_PATH must match the model used to train the checkpoint.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
INFER_BACKEND="${INFER_BACKEND:-vllm}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_TURNS="${MAX_TURNS:-6}"
MAX_PARALLEL_CALLS="${MAX_PARALLEL_CALLS:-2}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-4096}"
DYNAMIC_RETRIEVAL_TOP_K="${DYNAMIC_RETRIEVAL_TOP_K:-5}"
BASE_MODEL_INPUT_PRICE_PER_MILLION_USD="${BASE_MODEL_INPUT_PRICE_PER_MILLION_USD:-0.05}"
BASE_MODEL_OUTPUT_PRICE_PER_MILLION_USD="${BASE_MODEL_OUTPUT_PRICE_PER_MILLION_USD:-0.25}"
BASE_MODEL_LATENCY_BASE_SECONDS="${BASE_MODEL_LATENCY_BASE_SECONDS:-0.332}"
BASE_MODEL_LATENCY_INPUT_SECONDS_PER_TOKEN="${BASE_MODEL_LATENCY_INPUT_SECONDS_PER_TOKEN:-0.000003}"
BASE_MODEL_LATENCY_OUTPUT_SECONDS_PER_TOKEN="${BASE_MODEL_LATENCY_OUTPUT_SECONDS_PER_TOKEN:-0.00664}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$PROJECT_ROOT/.runtime/vllm}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PROJECT_ROOT/.runtime/torchinductor}"

# Answer replacement. API mode uses ANSWER_API_KEY or the shared provider key.
ANSWER_BACKEND="${ANSWER_BACKEND:-vllm}"
ANSWER_MODEL="${ANSWER_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
ANSWER_CUDA_VISIBLE_DEVICES="${ANSWER_CUDA_VISIBLE_DEVICES:-0}"
ANSWER_BATCH_SIZE="${ANSWER_BATCH_SIZE:-32}"
ANSWER_MAX_NEW_TOKENS="${ANSWER_MAX_NEW_TOKENS:-512}"
ANSWER_MAX_MODEL_LEN="${ANSWER_MAX_MODEL_LEN:-16384}"
ANSWER_TENSOR_PARALLEL_SIZE="${ANSWER_TENSOR_PARALLEL_SIZE:-1}"
ANSWER_GPU_MEMORY_UTILIZATION="${ANSWER_GPU_MEMORY_UTILIZATION:-0.9}"
ANSWER_BASE_URL="${ANSWER_BASE_URL:-${OPENROUTER_BASE_URL:-${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}}}"
ANSWER_WORKERS="${ANSWER_WORKERS:-16}"
ANSWER_MAX_RETRIES="${ANSWER_MAX_RETRIES:-3}"
ANSWER_DISABLE_REASONING="${ANSWER_DISABLE_REASONING:-false}"

# Final metrics. The judge uses JUDGE_API_KEY or the shared provider key.
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-4o-mini}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-${OPENROUTER_BASE_URL:-${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}}}"
JUDGE_WORKERS="${JUDGE_WORKERS:-16}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-512}"
JUDGE_MAX_RETRIES="${JUDGE_MAX_RETRIES:-3}"

eval_error() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

eval_usage() {
    cat <<EOF
Usage: bash scripts/$(basename -- "$0") [options]

  --datasets NAME ...      Evaluate only these datasets (default: all five).
                           mem_gallery worldmemarena h2hmem memeye memlens
  --checkpoint-path DIR    verl global_step_* checkpoint; required for policy
                           inference and the full pipeline.
  --output-dir DIR         Shared evaluation directory; required for every stage.
  -h, --help               Show this help message.

CHECKPOINT_PATH and EVAL_OUTPUT_DIR can supply the two paths instead of flags.
Edit scripts/eval_common.sh or export variables to configure models and GPUs.
Use the same output directory and dataset selection for all three stages.
EOF
}

eval_absolute_path() {
    # Resolve CLI and environment paths before changing the working directory.
    realpath -m -- "$1"
}

eval_test_file() {
    # Override an individual *_TEST_FILE to use another prepared variant or
    # an external-memory adapter's output, without changing runtime code.
    local path
    case "$1" in
        mem_gallery) path="${MEM_GALLERY_TEST_FILE:-$EVAL_DATA_ROOT/mem_gallery/test.parquet}" ;;
        worldmemarena) path="${WORLDMEMARENA_TEST_FILE:-$EVAL_DATA_ROOT/worldmemarena_lifelong/test.parquet}" ;;
        h2hmem) path="${H2HMEM_TEST_FILE:-$EVAL_DATA_ROOT/h2hmem_dyadic/test.parquet}" ;;
        memeye) path="${MEMEYE_TEST_FILE:-$EVAL_DATA_ROOT/memeye/open/test.parquet}" ;;
        memlens) path="${MEMLENS_TEST_FILE:-$EVAL_DATA_ROOT/memlens_32k_agent/test.parquet}" ;;
    esac
    eval_absolute_path "$path"
}

eval_initialize() {
    local stage=$1 dataset value
    shift
    EVAL_DATASETS=()
    while (($#)); do
        case "$1" in
            --datasets)
                shift
                [[ $# -gt 0 && ${1:-} != --* ]] || eval_error '--datasets requires at least one name.'
                while (($#)) && [[ $1 != --* ]]; do
                    EVAL_DATASETS+=("$1")
                    shift
                done
                ;;
            --checkpoint-path|--output-dir)
                value=${2:-}
                [[ -n $value && $value != --* ]] || eval_error "$1 requires a directory."
                case "$1" in
                    --checkpoint-path) CHECKPOINT_PATH=$value ;;
                    --output-dir) EVAL_OUTPUT_DIR=$value ;;
                esac
                shift 2
                ;;
            -h|--help) eval_usage; exit 0 ;;
            *) eval_error "Unknown argument: $1 (use --help)." ;;
        esac
    done
    if ((${#EVAL_DATASETS[@]} == 0)) || [[ ${EVAL_DATASETS[*]} == all ]]; then
        EVAL_DATASETS=(mem_gallery worldmemarena h2hmem memeye memlens)
    fi
    local -A seen=()
    for dataset in "${EVAL_DATASETS[@]}"; do
        case "$dataset" in
            mem_gallery|worldmemarena|h2hmem|memeye|memlens) ;;
            *) eval_error "Unknown dataset: $dataset (use --help for supported names)." ;;
        esac
        [[ -z ${seen[$dataset]:-} ]] || eval_error "Dataset selected more than once: $dataset"
        seen[$dataset]=1
    done
    [[ -n $EVAL_OUTPUT_DIR ]] || eval_error 'Set EVAL_OUTPUT_DIR or pass --output-dir DIR.'
    EVAL_OUTPUT_DIR=$(eval_absolute_path "$EVAL_OUTPUT_DIR")
    EVAL_DATA_ROOT=$(eval_absolute_path "$EVAL_DATA_ROOT")
    RESOURCE_CONFIG=$(eval_absolute_path "$RESOURCE_CONFIG")
    VLLM_CACHE_ROOT=$(eval_absolute_path "$VLLM_CACHE_ROOT")
    TORCHINDUCTOR_CACHE_DIR=$(eval_absolute_path "$TORCHINDUCTOR_CACHE_DIR")
    [[ -f $RESOURCE_CONFIG ]] || eval_error "Missing resource/tool configuration: $RESOURCE_CONFIG"
    if [[ $stage == policy || $stage == all ]]; then
        [[ -n $CHECKPOINT_PATH ]] || eval_error 'Set CHECKPOINT_PATH or pass --checkpoint-path DIR.'
        CHECKPOINT_PATH=$(eval_absolute_path "$CHECKPOINT_PATH")
        [[ -d $CHECKPOINT_PATH ]] || eval_error "Missing policy checkpoint: $CHECKPOINT_PATH"
    fi
    # Normalize per-dataset overrides while paths still refer to the caller's cwd.
    declare -gA EVAL_TEST_FILES=()
    for dataset in "${EVAL_DATASETS[@]}"; do
        EVAL_TEST_FILES[$dataset]=$(eval_test_file "$dataset")
    done
    export VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR
    if [[ $stage != all ]]; then
        export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
        cd "$PROJECT_ROOT"
    fi
}

eval_dataset_paths() {
    DATASET_OUTPUT_DIR="$EVAL_OUTPUT_DIR/$1"
    POLICY_OUTPUT_DIR="$DATASET_OUTPUT_DIR/policy"
    ANSWER_OUTPUT_FILE="$DATASET_OUTPUT_DIR/answer_replacement/raw_responses.jsonl"
    METRICS_OUTPUT_DIR="$DATASET_OUTPUT_DIR/metrics"
    LOG_DIR="$DATASET_OUTPUT_DIR/logs"
}

eval_require_policy() {
    [[ -f $POLICY_OUTPUT_DIR/.complete ]] || eval_error \
        "Policy inference is incomplete for $1; run eval_policy.sh first."
    local files=("$POLICY_OUTPUT_DIR"/*.jsonl)
    [[ -s ${files[0]} ]] || eval_error "Missing policy responses for $1: $POLICY_OUTPUT_DIR"
}

eval_validate_answer_settings() {
    [[ $ANSWER_BACKEND == vllm || $ANSWER_BACKEND == openai ]] || \
        eval_error 'ANSWER_BACKEND must be vllm or openai.'
    [[ $ANSWER_DISABLE_REASONING == true || $ANSWER_DISABLE_REASONING == false ]] || \
        eval_error 'ANSWER_DISABLE_REASONING must be true or false.'
    [[ $ANSWER_BACKEND == openai || $ANSWER_DISABLE_REASONING == false ]] || \
        eval_error 'ANSWER_DISABLE_REASONING=true requires ANSWER_BACKEND=openai.'
}
