#!/usr/bin/env bash
# Jointly train MemPilot on the prepared multimodal memory QA benchmarks.
# Edit the settings below; extra arguments are forwarded to verl/Hydra.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

# Account credentials are inherited from the environment.
# To set them in this script, uncomment and fill in the exports below.
# Script values override environment values with the same name.
# export HUGGING_FACE_HUB_TOKEN="your_hugging_face_token"
# export WANDB_API_KEY="your_wandb_api_key"
# export OPENROUTER_API_KEY="your_openrouter_api_key"
# export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"

# Mixed Mem-Gallery, WorldMemArena-Lifelong and H2HMem-Dyadic data.
# DATA_DIR must point to their merged output, including adapted memory banks.
# Choose the LLMLingua directory or the external adapter's output directory.
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/multimodal_unified}"
TRAIN_FILE="$DATA_DIR/train.parquet"
# Use the validation split for periodic evaluation and model selection.
VAL_FILE="$DATA_DIR/val.parquet"
MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507
BASE_MODEL_NAME="${MODEL_PATH##*/}"
TOOL_CONFIG="$PROJECT_ROOT/configs/runtime_memory_tool.yaml"
AGENT_LOOP_CONFIG="$PROJECT_ROOT/configs/rollout_ordered_agent_loop.yaml"
REWARD_PATH="$PROJECT_ROOT/rewards/unified_multimodal_reward.py"

# Hardware and rollout backend. Configure NCCL through the calling environment.
CUDA_VISIBLE_DEVICES=0,1,2,3
NNODES=1
NGPUS_PER_NODE=4
INFER_BACKEND=vllm
ROLLOUT_TP=1
ROLLOUT_GPU_MEMORY_UTILIZATION=0.6

# Training. Epoch intervals are resolved after overlong samples are filtered.
TRAIN_BATCH_SIZE=32
PPO_MINI_BATCH_SIZE=8
LEARNING_RATE=1e-6
KL_LOSS_COEF=0.001
TOTAL_EPOCHS=3
VALIDATION_INTERVAL_EPOCHS=0.5
CHECKPOINT_INTERVAL_EPOCHS=0.5

# Rollout and memory access.
ROLLOUT_N=4
ROLLOUT_TEMPERATURE=1.0
ROLLOUT_PRINT_PROBABILITY=0.05
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=4096
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_TURNS=6
# Shared by the rollout limit and the runtime-built system prompt.
MAX_PARALLEL_CALLS=2
MAX_TOOL_RESPONSE_LENGTH=4096
DYNAMIC_RETRIEVAL_TOP_K=5

# Each memory stage adds an answer probe, plus one probe before the first stage.
MARGINAL_UTILITY_WEIGHT=0.5
MARGINAL_UTILITY_MAX_ANSWER_TOKENS=256
# A zero resource weight removes that objective from GDPO fusion.
COST_ADVANTAGE_WEIGHT=0.0
LATENCY_ADVANTAGE_WEIGHT=0.0
# Raw cost multiplies USD-per-million rates by token counts without dividing by 1M.
BASE_MODEL_INPUT_PRICE_PER_MILLION_USD=0.05
BASE_MODEL_OUTPUT_PRICE_PER_MILLION_USD=0.25
# Latency uses the profiled Qwen3-VL-8B coefficients as a policy-model proxy.
BASE_MODEL_LATENCY_BASE_SECONDS=0.332
BASE_MODEL_LATENCY_INPUT_SECONDS_PER_TOKEN=0.000003
BASE_MODEL_LATENCY_OUTPUT_SECONDS_PER_TOKEN=0.00664

# Resolve resource overrides before deriving run names and paths. The last value wins.
# Emit each resource weight once; forward all other Hydra arguments unchanged.
TRAINER_ARGS=()
for argument in "$@"; do
    case "$argument" in
        algorithm.gdpo.cost_weight=*|+algorithm.gdpo.cost_weight=*|++algorithm.gdpo.cost_weight=*)
            COST_ADVANTAGE_WEIGHT="${argument#*=}"
            ;;
        algorithm.gdpo.latency_weight=*|+algorithm.gdpo.latency_weight=*|++algorithm.gdpo.latency_weight=*)
            LATENCY_ADVANTAGE_WEIGHT="${argument#*=}"
            ;;
        *) TRAINER_ARGS+=("$argument") ;;
    esac
done

# Logging and checkpoints. Use '["console"]' for local logging only.
LOGGER='["console","wandb"]'
WANDB_PROJECT=mempilot
WANDB_RUN_NAME="mempilot_${BASE_MODEL_NAME}_cost${COST_ADVANTAGE_WEIGHT}_lat${LATENCY_ADVANTAGE_WEIGHT}_$(date +%Y%m%d_%H%M%S)"
RUN_NAME_SAFE=$(printf '%s' "$WANDB_RUN_NAME" | tr -c 'A-Za-z0-9._-' '_')
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/$WANDB_PROJECT/$RUN_NAME_SAFE"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/${RUN_NAME_SAFE}.log"
MODEL_CALL_COUNT_LOG="$LOG_DIR/${RUN_NAME_SAFE}.pid${BASHPID}.model_calls.log"

mkdir -p "$LOG_DIR"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES DYNAMIC_RETRIEVAL_TOP_K MAX_PARALLEL_CALLS
exec > >(tee -a "$LOG_FILE") 2>&1
printf 'Run: %s\nLogs: %s\nCheckpoints: %s\n' "$WANDB_RUN_NAME" "$LOG_FILE" "$CHECKPOINT_DIR"

PYTHONUNBUFFERED=1 python3 -m trainers.main_ppo_sync \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    +algorithm.marginal_utility.weight="$MARGINAL_UTILITY_WEIGHT" \
    +algorithm.marginal_utility.max_answer_tokens="$MARGINAL_UTILITY_MAX_ANSWER_TOKENS" \
    +algorithm.gdpo.cost_weight="$COST_ADVANTAGE_WEIGHT" \
    +algorithm.gdpo.latency_weight="$LATENCY_ADVANTAGE_WEIGHT" \
    +algorithm.gdpo.base_model_input_price_per_million_usd="$BASE_MODEL_INPUT_PRICE_PER_MILLION_USD" \
    +algorithm.gdpo.base_model_output_price_per_million_usd="$BASE_MODEL_OUTPUT_PRICE_PER_MILLION_USD" \
    +algorithm.gdpo.base_model_latency_base_seconds="$BASE_MODEL_LATENCY_BASE_SECONDS" \
    +algorithm.gdpo.base_model_latency_input_seconds_per_token="$BASE_MODEL_LATENCY_INPUT_SECONDS_PER_TOKEN" \
    +algorithm.gdpo.base_model_latency_output_seconds_per_token="$BASE_MODEL_LATENCY_OUTPUT_SECONDS_PER_TOKEN" \
    data.train_files="['$TRAIN_FILE']" \
    data.val_files="['$VAL_FILE']" \
    data.custom_cls.path="$PROJECT_ROOT/data/runtime_memory_dataset.py" \
    data.custom_cls.name=RuntimeMemoryDataset \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs=False \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path="$REWARD_PATH" \
    reward.custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_MODEL_LEN \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_MODEL_LEN \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name="$INFER_BACKEND" \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enable_prefix_caching=true \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_TURNS" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_TURNS" \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls="$MAX_PARALLEL_CALLS" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$MAX_TOOL_RESPONSE_LENGTH" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.default_agent_loop=rollout_ordered_tool_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG" \
    trainer.logger="$LOGGER" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$WANDB_RUN_NAME" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    +trainer.model_call_count_log_path="$MODEL_CALL_COUNT_LOG" \
    trainer.resume_mode=disable \
    +trainer.rollout_print_probability="$ROLLOUT_PRINT_PROBABILITY" \
    trainer.nnodes="$NNODES" \
    trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.val_before_train=true \
    trainer.val_only=false \
    +trainer.validation_interval_epochs="$VALIDATION_INTERVAL_EPOCHS" \
    +trainer.checkpoint_interval_epochs="$CHECKPOINT_INTERVAL_EPOCHS" \
    trainer.test_freq=0 \
    trainer.save_freq=0 \
    +ray_kwargs.ray_init.runtime_env.env_vars.TRANSFER_QUEUE_ENABLE=1 \
    "${TRAINER_ARGS[@]}"
