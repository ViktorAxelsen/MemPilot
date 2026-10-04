#!/usr/bin/env bash
# Generate answers from saved policy evidence for the selected datasets.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/eval_common.sh"
eval_initialize answers "$@"
eval_validate_answer_settings

backend_args=(--backend "$ANSWER_BACKEND")
case "$ANSWER_BACKEND" in
    vllm)
        export CUDA_VISIBLE_DEVICES="$ANSWER_CUDA_VISIBLE_DEVICES"
        backend_args+=(
            --max_model_len "$ANSWER_MAX_MODEL_LEN"
            --tensor_parallel_size "$ANSWER_TENSOR_PARALLEL_SIZE"
            --gpu_memory_utilization "$ANSWER_GPU_MEMORY_UTILIZATION"
            --disable_multimodal_inputs
        )
        ;;
    openai)
        backend_args+=(--base_url "$ANSWER_BASE_URL" --workers "$ANSWER_WORKERS" --max_retries "$ANSWER_MAX_RETRIES")
        if [[ $ANSWER_DISABLE_REASONING == true ]]; then
            backend_args+=(--disable_reasoning)
        fi
        ;;
esac

for dataset in "${EVAL_DATASETS[@]}"; do
    eval_dataset_paths "$dataset"
    eval_require_policy "$dataset"
done
for dataset in "${EVAL_DATASETS[@]}"; do
    eval_dataset_paths "$dataset"
    mkdir -p "$LOG_DIR"
    printf 'Answer replacement: %s\n' "$dataset"
    python3 -m evaluation.replace_saved_answers \
        --input "$POLICY_OUTPUT_DIR" \
        --output "$ANSWER_OUTPUT_FILE" \
        --model "$ANSWER_MODEL" \
        --batch_size "$ANSWER_BATCH_SIZE" \
        --max_new_tokens "$ANSWER_MAX_NEW_TOKENS" \
        --resource_config "$RESOURCE_CONFIG" \
        "${backend_args[@]}" 2>&1 | tee "$LOG_DIR/answer_replacement.log"
done
