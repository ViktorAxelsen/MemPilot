#!/usr/bin/env bash
# Run the trained policy and save trajectories; no answer replacement or judging.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/eval_common.sh"
eval_initialize policy "$@"

# Validate the entire selection before starting any GPU work.
for dataset in "${EVAL_DATASETS[@]}"; do
    [[ -f ${EVAL_TEST_FILES[$dataset]} ]] || eval_error "Missing $dataset test data: ${EVAL_TEST_FILES[$dataset]}"
    eval_dataset_paths "$dataset"
    [[ ! -e $DATASET_OUTPUT_DIR ]] || eval_error \
        "Output already exists: $DATASET_OUTPUT_DIR. Use a new --output-dir for policy inference, or run the remaining stages."
done
export CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES"
export DYNAMIC_RETRIEVAL_TOP_K MAX_PARALLEL_CALLS
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

for dataset in "${EVAL_DATASETS[@]}"; do
    eval_dataset_paths "$dataset"
    TEST_FILE=${EVAL_TEST_FILES[$dataset]}
    mkdir -p "$POLICY_OUTPUT_DIR" "$LOG_DIR"
    printf 'Policy inference: %s\nCheckpoint: %s\n' "$dataset" "$CHECKPOINT_PATH"

    # verl constructs both datasets in val-only mode; no training update occurs.
    PYTHONUNBUFFERED=1 python3 -m trainers.main_ppo_sync \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        +ray_kwargs.ray_init.runtime_env.env_vars.TRANSFER_QUEUE_ENABLE=1 \
        +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
        +ray_kwargs.ray_init.runtime_env.env_vars.TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" \
        +algorithm.gdpo.base_model_input_price_per_million_usd="$BASE_MODEL_INPUT_PRICE_PER_MILLION_USD" \
        +algorithm.gdpo.base_model_output_price_per_million_usd="$BASE_MODEL_OUTPUT_PRICE_PER_MILLION_USD" \
        +algorithm.gdpo.base_model_latency_base_seconds="$BASE_MODEL_LATENCY_BASE_SECONDS" \
        +algorithm.gdpo.base_model_latency_input_seconds_per_token="$BASE_MODEL_LATENCY_INPUT_SECONDS_PER_TOKEN" \
        +algorithm.gdpo.base_model_latency_output_seconds_per_token="$BASE_MODEL_LATENCY_OUTPUT_SECONDS_PER_TOKEN" \
        data.train_files="['$TEST_FILE']" \
        data.val_files="['$TEST_FILE']" \
        data.custom_cls.path="$PROJECT_ROOT/data/runtime_memory_dataset.py" \
        data.custom_cls.name=RuntimeMemoryDataset \
        data.return_raw_chat=True \
        data.return_multi_modal_inputs=False \
        data.validation_shuffle=false \
        data.train_batch_size="$EVAL_BATCH_SIZE" \
        data.val_batch_size="$EVAL_BATCH_SIZE" \
        data.max_prompt_length="$MAX_PROMPT_LENGTH" \
        data.max_response_length="$MAX_RESPONSE_LENGTH" \
        data.filter_overlong_prompts=False \
        data.truncation=error \
        reward.reward_manager.name=naive \
        reward.custom_reward_function.path="$PROJECT_ROOT/evaluation/capture_reward.py" \
        reward.custom_reward_function.name=compute_score \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=8 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_MODEL_LEN" \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$MAX_MODEL_LEN" \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.name="$INFER_BACKEND" \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
        actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
        actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.temperature=0.0 \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=false \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_TURNS" \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_TURNS" \
        actor_rollout_ref.rollout.multi_turn.max_parallel_calls="$MAX_PARALLEL_CALLS" \
        actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$MAX_TOOL_RESPONSE_LENGTH" \
        actor_rollout_ref.rollout.multi_turn.tool_config_path="$RESOURCE_CONFIG" \
        actor_rollout_ref.rollout.multi_turn.format=hermes \
        actor_rollout_ref.rollout.agent.default_agent_loop=rollout_ordered_tool_agent \
        actor_rollout_ref.rollout.agent.agent_loop_config_path="$PROJECT_ROOT/configs/rollout_ordered_agent_loop.yaml" \
        trainer.logger='["console"]' \
        trainer.validation_data_dir="$POLICY_OUTPUT_DIR" \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="$CHECKPOINT_PATH" \
        trainer.nnodes="$NNODES" \
        trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
        trainer.total_epochs=1 \
        trainer.val_before_train=true \
        trainer.val_only=true \
        trainer.test_freq=-1 \
        trainer.save_freq=-1 2>&1 | tee "$LOG_DIR/policy.log"

    response_files=("$POLICY_OUTPUT_DIR"/*.jsonl)
    [[ -s ${response_files[0]} ]] || eval_error "Policy inference produced no responses for $dataset."
    touch "$POLICY_OUTPUT_DIR/.complete"
    printf 'Saved policy responses: %s\n' "$POLICY_OUTPUT_DIR"
done
