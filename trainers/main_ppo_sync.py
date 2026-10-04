"""verl ``main_ppo_sync`` wrapper for runtime-memory training."""

from __future__ import annotations

import math
from pprint import pprint

import ray
from omegaconf import OmegaConf

from trainers.resource_aware_trainer import (
    ResourceAwareTrainerMixin,
    load_resource_optimization_settings,
)
from trainers.runtime_diagnostics import RuntimeDiagnosticsTrainerMixin
from verl.trainer import main_ppo_sync as verl_sync
from verl.trainer.main_ppo import TaskRunner as VerlTaskRunner
from verl.trainer.ppo.utils import need_critic


def epoch_interval_to_steps(steps_per_epoch: int, interval_epochs: float) -> int:
    """Convert a positive epoch fraction into a safe integer step interval."""

    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive.")
    interval_epochs = float(interval_epochs)
    if not math.isfinite(interval_epochs) or interval_epochs <= 0.0:
        raise ValueError("interval_epochs must be a positive finite number.")
    return max(math.ceil(steps_per_epoch * interval_epochs), 1)


def _install_critic_config_compatibility() -> None:
    """Bridge the critic model field mismatch in verl 0.8.0's sync trainer."""
    from verl.workers.config import CriticConfig

    if hasattr(CriticConfig, "model_config"):
        return
    if "model" not in getattr(CriticConfig, "__dataclass_fields__", {}):
        raise RuntimeError(
            "Unsupported verl CriticConfig: expected either 'model' or 'model_config'."
        )

    # verl 0.8.0 main_ppo_sync reads model_config although the dataclass field is model.
    setattr(CriticConfig, "model_config", property(lambda config: config.model))


def _reference_policy_is_in_actor(config) -> bool:
    """Match verl's rule for reusing the LoRA actor as its reference policy."""
    model_config = config.actor_rollout_ref.model
    lora_config = model_config.get("lora", {}) or {}
    lora_rank = int(lora_config.get("rank", 0) or 0)
    if lora_rank <= 0:
        lora_rank = int(model_config.get("lora_rank", 0) or 0)
    return lora_rank > 0 or model_config.get("lora_adapter_path") is not None


class RuntimeMetricsPPOTrainer(
    RuntimeDiagnosticsTrainerMixin,
    ResourceAwareTrainerMixin,
    verl_sync.PPOTrainer,
):
    """Compose runtime diagnostics and resource-aware optimization over verl."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._configure_periodic_frequencies()
        self._initialize_runtime_diagnostics()

    def _configure_periodic_frequencies(self) -> None:
        """Resolve epoch-relative validation/checkpoint intervals after filtering."""

        validation_interval = OmegaConf.select(
            self.config,
            "trainer.validation_interval_epochs",
            default=None,
        )
        checkpoint_interval = OmegaConf.select(
            self.config,
            "trainer.checkpoint_interval_epochs",
            default=None,
        )
        if validation_interval is None and checkpoint_interval is None:
            return

        steps_per_epoch = len(self.train_dataloader)
        if steps_per_epoch <= 0:
            raise RuntimeError("Cannot configure epoch-relative intervals with an empty train dataloader.")

        for config_key, raw_interval, target_key, label in (
            ("validation_interval_epochs", validation_interval, "test_freq", "Validation"),
            ("checkpoint_interval_epochs", checkpoint_interval, "save_freq", "Checkpoint"),
        ):
            if raw_interval is None:
                continue
            interval_epochs = float(raw_interval)
            try:
                interval_steps = epoch_interval_to_steps(steps_per_epoch, interval_epochs)
            except ValueError as exc:
                raise ValueError(f"Invalid trainer.{config_key}: {exc}") from exc
            self.config.trainer[target_key] = interval_steps
            verl_sync.logger.info(
                "%s frequency resolved to %s step(s) (%.3g epoch).",
                label,
                interval_steps,
                interval_epochs,
            )

    def init_workers(self):
        """Avoid verl 0.8.0's missing actor_rollout_ref lookup for LoRA."""
        reference_policy_enabled = self.use_reference_policy
        reference_is_in_actor = (
            reference_policy_enabled and _reference_policy_is_in_actor(self.config)
        )
        if not reference_is_in_actor:
            return super().init_workers()

        # In LoRA mode the worker is registered as ``actor_rollout`` and its
        # adapter-free base model is the reference policy. verl 0.8.0 detects
        # this correctly, but then unconditionally looks up a nonexistent
        # ``actor_rollout_ref`` worker. Skip only that independent-ref binding.
        self.use_reference_policy = False
        try:
            result = super().init_workers()
        finally:
            self.use_reference_policy = reference_policy_enabled

        if not self.ref_in_actor:
            raise RuntimeError("Expected the LoRA reference policy to be hosted by the actor.")
        self.ref_policy_wg = self.actor_rollout_wg
        return result

    def _compute_ref_log_prob(self, batch, metrics):
        """Keep the actor-only LoRA switch from leaking into the critic call."""
        try:
            return super()._compute_ref_log_prob(batch, metrics)
        finally:
            # verl 0.8.0 stores this one-call flag in the shared KVBatchMeta
            # extra_info but does not remove it after computing reference
            # log-probs. The following critic value inference would otherwise
            # try to disable a nonexistent adapter on the full-parameter critic.
            batch.extra_info.pop("no_lora_adapter", None)

    def _update_critic(self, batch, metrics):
        # verl 0.8.0 loses this metadata when advantage data is written back to TQ.
        batch.extra_info["temperature"] = (
            self.config.actor_rollout_ref.rollout.temperature
        )
        return super()._update_critic(batch, metrics)


class _RuntimeMetricsTaskRunner(VerlTaskRunner):
    """Use the metrics-aware trainer while preserving verl's sync setup."""

    def run(self, config):
        # Preserve main_ppo_sync's resolved-config startup output in the run log.
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        load_resource_optimization_settings(config)

        verl_sync.tq.init(config.transfer_queue)
        trainer = None
        try:
            self.add_actor_rollout_worker(config)
            if need_critic(config):
                _install_critic_config_compatibility()
                self.add_critic_worker(config)
            self.add_reward_model_resource_pool(config)
            self.add_teacher_model_resource_pool(config)
            resource_pool_manager = self.init_resource_pool_mgr(config)

            trainer = RuntimeMetricsPPOTrainer(
                config=config,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
            )
            trainer.init_workers()
            trainer.fit()
        finally:
            if trainer is not None:
                try:
                    trainer.write_model_call_count_log()
                except Exception as exc:
                    print(
                        f"[MODEL CALL COUNTS] failed to write log: {exc}",
                        flush=True,
                    )
                trainer.replay_buffer.close()
            verl_sync.tq.close()


RuntimeMetricsTaskRunner = ray.remote(_RuntimeMetricsTaskRunner)


def main() -> None:
    # Reuse verl's Hydra entry point and replace only its remote task runner.
    verl_sync.TaskRunner = RuntimeMetricsTaskRunner
    verl_sync.main()


if __name__ == "__main__":
    main()
