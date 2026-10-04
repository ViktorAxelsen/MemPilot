"""Training and validation diagnostics for the runtime-memory verl trainer."""

from __future__ import annotations

from collections import Counter
from random import Random

from omegaconf import OmegaConf

from rollout_logging import format_rollout_sample, sample_rollout_indices
from runtime_metrics import (
    build_wandb_runtime_metrics,
    configured_runtime_memory_models,
    count_runtime_memory_calls,
    filter_validation_metrics,
    materialize_transfer_queue_value,
    write_model_call_count_log,
)
from verl.trainer import main_ppo_sync as verl_sync


class RuntimeDiagnosticsTrainerMixin:
    """Publish runtime-memory metrics, sampled rollouts, and route call counts."""

    def _initialize_runtime_diagnostics(self) -> None:
        self._runtime_memory_models = _load_runtime_memory_models(self.config)
        self._runtime_memory_routes = tuple(self._runtime_memory_models)
        self._runtime_memory_call_counts: Counter[str] = Counter()
        self._model_call_count_log_path = OmegaConf.select(
            self.config,
            "trainer.model_call_count_log_path",
            default=None,
        )
        self._model_call_count_log_error_reported = False
        self._rollout_print_probability = _load_rollout_print_probability(self.config)
        data_seed = int(OmegaConf.select(self.config, "data.seed", default=0) or 0)
        self._rollout_print_rng = Random(data_seed)
        self._checkpoint_model_call_count_log()

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)

        extra_data = verl_sync.tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["extra_fields"],
        )
        extra_fields = materialize_transfer_queue_value(extra_data["extra_fields"])
        if not isinstance(extra_fields, list):
            raise TypeError(
                "TransferQueue returned an unsupported extra_fields container: "
                f"{type(extra_fields).__name__}."
            )
        if len(extra_fields) != len(batch.tags):
            raise RuntimeError("TransferQueue returned misaligned runtime-memory diagnostics.")
        active_extra_fields = [
            field
            for field, tag in zip(extra_fields, batch.tags, strict=True)
            if not tag.get("is_padding", False)
        ]
        self._runtime_memory_call_counts.update(
            count_runtime_memory_calls(active_extra_fields)
        )
        self._checkpoint_model_call_count_log()
        metrics.update(
            build_wandb_runtime_metrics(
                active_extra_fields,
                configured_routes=self._runtime_memory_routes,
            )
        )
        self._print_sampled_rollouts(
            batch=batch,
            extra_fields=extra_fields,
            global_steps=global_steps,
            epoch=epoch,
        )

    def write_model_call_count_log(self):
        """Persist cumulative route usage after training, including zero-call routes."""
        if not self._model_call_count_log_path:
            return None
        path = write_model_call_count_log(
            self._model_call_count_log_path,
            dict(self._runtime_memory_call_counts),
            self._runtime_memory_models,
        )
        print(f"[MODEL CALL COUNTS] saved to {path}", flush=True)
        return path

    def _checkpoint_model_call_count_log(self) -> None:
        """Best-effort snapshot after each completed step without disrupting training."""
        if not self._model_call_count_log_path:
            return
        try:
            write_model_call_count_log(
                self._model_call_count_log_path,
                dict(self._runtime_memory_call_counts),
                self._runtime_memory_models,
            )
        except Exception as exc:
            if not self._model_call_count_log_error_reported:
                print(
                    f"[MODEL CALL COUNTS] failed to checkpoint log: {exc}",
                    flush=True,
                )
            self._model_call_count_log_error_reported = True
        else:
            self._model_call_count_log_error_reported = False

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        metrics = super()._val_metrics_update(
            data_sources, sample_uids, reward_extra_infos_dict, sample_turns
        )
        # Filter only the logged summary; retain raw reward fields and saved rows.
        return filter_validation_metrics(metrics)

    def _print_sampled_rollouts(
        self,
        *,
        batch,
        extra_fields,
        global_steps: int,
        epoch: int,
    ) -> None:
        selected_indices = sample_rollout_indices(
            batch.tags,
            self._rollout_print_probability,
            self._rollout_print_rng,
        )
        if not selected_indices:
            return

        selected_keys = [batch.keys[index] for index in selected_indices]
        rollout_data = verl_sync.tq.kv_batch_get(
            keys=selected_keys,
            partition_id=batch.partition_id,
            select_fields=["prompts", "responses", "reward_model"],
        )
        prompts = rollout_data["prompts"].to_padded_tensor(
            padding=self.tokenizer.pad_token_id
        )
        responses = rollout_data["responses"].to_padded_tensor(
            padding=self.tokenizer.pad_token_id
        )
        reward_models = materialize_transfer_queue_value(rollout_data["reward_model"])
        if not isinstance(reward_models, list) or len(reward_models) != len(
            selected_indices
        ):
            raise RuntimeError("TransferQueue returned misaligned rollout ground truths.")

        for local_index, batch_index in enumerate(selected_indices):
            extra_field = extra_fields[batch_index]
            reward_info = (
                extra_field.get("reward_extra_info", {})
                if isinstance(extra_field, dict)
                else {}
            )
            reward = reward_info.get("score") if isinstance(reward_info, dict) else None
            reward_model = reward_models[local_index]
            ground_truth = (
                reward_model.get("ground_truth")
                if isinstance(reward_model, dict)
                else None
            )
            print(
                format_rollout_sample(
                    key=str(batch.keys[batch_index]),
                    global_step=global_steps,
                    epoch=epoch + 1,
                    prompt=self.tokenizer.decode(
                        prompts[local_index],
                        skip_special_tokens=True,
                    ),
                    response=self.tokenizer.decode(
                        responses[local_index],
                        skip_special_tokens=True,
                    ),
                    reward=reward,
                    ground_truth=ground_truth,
                ),
                flush=True,
            )


def _load_runtime_memory_models(config) -> dict[str, str]:
    tool_config_path = OmegaConf.select(
        config,
        "actor_rollout_ref.rollout.multi_turn.tool_config_path",
    )
    if not tool_config_path:
        return {}

    tool_config = OmegaConf.to_container(
        OmegaConf.load(str(tool_config_path)),
        resolve=True,
    )
    return configured_runtime_memory_models(tool_config)


def _load_rollout_print_probability(config) -> float:
    probability = float(
        OmegaConf.select(
            config,
            "trainer.rollout_print_probability",
            default=0.0,
        )
    )
    if not 0.0 <= probability <= 1.0:
        raise ValueError("trainer.rollout_print_probability must be between 0 and 1")
    return probability
