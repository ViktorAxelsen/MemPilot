#!/usr/bin/env bash
# Run policy inference, answer replacement, then final metrics.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/eval_common.sh"
eval_initialize all "$@"
eval_validate_answer_settings

common_args=(--output-dir "$EVAL_OUTPUT_DIR" --datasets "${EVAL_DATASETS[@]}")
bash "$SCRIPT_DIR/eval_policy.sh" --checkpoint-path "$CHECKPOINT_PATH" "${common_args[@]}"
bash "$SCRIPT_DIR/eval_answer_replacement.sh" "${common_args[@]}"
bash "$SCRIPT_DIR/eval_metrics.sh" "${common_args[@]}"
printf 'Evaluation complete: %s\n' "$EVAL_OUTPUT_DIR"
