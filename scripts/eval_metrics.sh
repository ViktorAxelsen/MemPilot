#!/usr/bin/env bash
# Score the replacement answers using benchmark metrics and the LLM judge.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/eval_common.sh"
eval_initialize metrics "$@"

for dataset in "${EVAL_DATASETS[@]}"; do
    eval_dataset_paths "$dataset"
    [[ -s $ANSWER_OUTPUT_FILE && -s $ANSWER_OUTPUT_FILE.manifest.json ]] || \
        eval_error "Missing completed answer replacement for $dataset; run eval_answer_replacement.sh first."
done
for dataset in "${EVAL_DATASETS[@]}"; do
    eval_dataset_paths "$dataset"
    mkdir -p "$LOG_DIR"
    printf 'Final metrics: %s\n' "$dataset"
    python3 -m evaluation.evaluate_saved_responses \
        --input "$ANSWER_OUTPUT_FILE" \
        --output_dir "$METRICS_OUTPUT_DIR" \
        --judge_model "$JUDGE_MODEL" \
        --judge_base_url "$JUDGE_BASE_URL" \
        --judge_workers "$JUDGE_WORKERS" \
        --judge_max_new_tokens "$JUDGE_MAX_NEW_TOKENS" \
        --judge_max_retries "$JUDGE_MAX_RETRIES" \
        --resource_config "$RESOURCE_CONFIG" 2>&1 | tee "$LOG_DIR/metrics.log"
done
