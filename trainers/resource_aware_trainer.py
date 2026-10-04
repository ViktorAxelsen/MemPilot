"""Resource-aware GDPO for MemPilot's quality-cost-latency preferences.

GDPO decouples task and resource advantages before preference-weighted fusion.
Prefix-based marginal-utility credits are applied afterward. Zero resource
weights retain the task-only GRPO computation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import numpy as np
import torch
from omegaconf import OmegaConf

from marginal_utility import (
    MARGINAL_UTILITY_INVALID_FORMAT,
    MARGINAL_UTILITY_KEY,
    MARGINAL_UTILITY_NO_STAGES,
    MARGINAL_UTILITY_OK,
    MARGINAL_UTILITY_PROBE_FAILED,
    add_marginal_utility_credit,
    load_marginal_utility_settings,
    parse_marginal_utility_credits,
)
from runtime_metrics import materialize_transfer_queue_value
from trainers.gdpo import (
    ResourceRewardBatch,
    build_resource_eligibility_mask,
    build_resource_rewards,
    compute_decoupled_advantages,
    final_session_layout,
)
from verl.trainer import main_ppo_sync as verl_sync


@dataclass(frozen=True)
class ResourceOptimizationSettings:
    cost_weight: float
    latency_weight: float
    marginal_utility_weight: float

    @property
    def active(self) -> bool:
        """Whether either resource objective affects optimization."""

        return self.cost_weight > 0.0 or self.latency_weight > 0.0


@dataclass(frozen=True)
class _TrajectoryResourceRewards:
    performance: np.ndarray
    cost: ResourceRewardBatch
    latency: ResourceRewardBatch


@dataclass(frozen=True)
class _FinalTrajectorySignals:
    final_indices: list[int]
    row_to_final: list[int]
    is_padding: np.ndarray
    format_valid: np.ndarray
    reward_infos: list[dict]
    performance: np.ndarray
    successful_calls: np.ndarray


def load_resource_optimization_settings(config) -> ResourceOptimizationSettings:
    """Validate GDPO preferences and its GRPO task-advantage backend."""

    estimator_name = _advantage_estimator_name(config)
    cost_weight = _nonnegative_setting(config, "algorithm.gdpo.cost_weight", 0.0)
    latency_weight = _nonnegative_setting(config, "algorithm.gdpo.latency_weight", 0.0)
    marginal_utility = load_marginal_utility_settings(config)
    if estimator_name != "grpo":
        raise ValueError(
            "MemPilot's GDPO optimization requires algorithm.adv_estimator=grpo "
            "for the task-advantage backend."
        )
    return ResourceOptimizationSettings(
        cost_weight=cost_weight,
        latency_weight=latency_weight,
        marginal_utility_weight=marginal_utility.weight,
    )


class ResourceAwareTrainerMixin:
    """Decouple task/resource advantages, then apply marginal stage credit."""

    def __init__(self, *args, **kwargs):
        settings = load_resource_optimization_settings(kwargs["config"]) if "config" in kwargs else None
        super().__init__(*args, **kwargs)
        settings = settings or load_resource_optimization_settings(self.config)

        self._resource_aware_enabled = settings.active
        self._gdpo_cost_weight = settings.cost_weight
        self._gdpo_latency_weight = settings.latency_weight
        self._marginal_utility_weight = settings.marginal_utility_weight

    def _compute_advantage(self, batch, metrics):
        if not self._resource_aware_enabled:
            batch = super()._compute_advantage(batch, metrics)
            return self._apply_marginal_utility_advantage(batch)

        extra_data = verl_sync.tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["extra_fields"],
        )
        extra_fields = self._materialize_reward_extra_fields(
            extra_data["extra_fields"],
            len(batch.keys),
        )
        signals = self._build_final_trajectory_signals(
            batch,
            extra_fields,
        )

        resources = self._build_resource_rewards(signals)

        batch = super()._compute_advantage(batch, metrics)
        batch = self._compute_resource_aware_gdpo_advantage(
            batch,
            metrics,
            final_indices=signals.final_indices,
            row_to_final=signals.row_to_final,
            is_padding=signals.is_padding,
            resources=resources,
            resource_weights={
                "cost": self._gdpo_cost_weight,
                "latency": self._gdpo_latency_weight,
            },
        )
        return self._apply_marginal_utility_advantage(batch)

    def _apply_marginal_utility_advantage(self, batch):
        """Apply prefix-based marginal utility after trajectory-level advantages."""

        extra_data = verl_sync.tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["extra_fields"],
        )
        extra_fields = self._materialize_reward_extra_fields(
            extra_data["extra_fields"],
            len(batch.keys),
        )
        final_indices, _ = final_session_layout(batch.keys)
        tensor_data = verl_sync.tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["response_mask", "advantages", "returns"],
        )
        nested_response_mask = tensor_data["response_mask"]
        padded = tensor_data.to_padded_tensor()
        response_mask = padded["response_mask"]
        advantages = padded["advantages"].clone()
        returns = padded["returns"].clone()

        for row_index in final_indices:
            if bool(batch.tags[row_index].get("is_padding", False)):
                continue
            response_length = int(batch.tags[row_index].get("response_len", -1))
            if response_length < 0 or response_length > response_mask.shape[1]:
                raise RuntimeError("Marginal-utility response length is missing or out of range.")

            extra_field = extra_fields[row_index]
            if not isinstance(extra_field, Mapping):
                raise RuntimeError("Marginal-utility shaping requires mapping-valued extra_fields.")
            metadata = extra_field.get(MARGINAL_UTILITY_KEY)
            if not isinstance(metadata, Mapping):
                raise RuntimeError(
                    "Marginal-utility metadata is missing; use the rollout_ordered_tool_agent loop."
                )
            status = str(metadata.get("status") or "")
            if status not in (
                MARGINAL_UTILITY_OK,
                MARGINAL_UTILITY_NO_STAGES,
                MARGINAL_UTILITY_INVALID_FORMAT,
                MARGINAL_UTILITY_PROBE_FAILED,
            ):
                raise RuntimeError(f"Unknown marginal-utility trajectory status: {status!r}.")
            for key in ("stage_count", "probe_count", "probe_output_tokens", "probe_seconds"):
                _nonnegative_metadata_number(metadata, key)
            if status != MARGINAL_UTILITY_OK:
                continue
            if not bool(_numeric_reward_field(_reward_info(extra_field), "format_valid")):
                raise RuntimeError("A format-invalid rollout contains successful marginal credits.")

            credits = parse_marginal_utility_credits(
                metadata,
                response_length=response_length,
            )
            shaped_advantages, shaped_returns = add_marginal_utility_credit(
                advantages=advantages[row_index],
                returns=returns[row_index],
                response_mask=response_mask[row_index],
                credits=credits,
                weight=self._marginal_utility_weight,
            )
            advantages[row_index] = shaped_advantages
            returns[row_index] = shaped_returns

        output = verl_sync.TensorDict(
            {
                "advantages": verl_sync.response_to_nested(
                    advantages,
                    nested_response_mask,
                ),
                "returns": verl_sync.response_to_nested(
                    returns,
                    nested_response_mask,
                ),
            },
            batch_size=len(batch),
        )
        return verl_sync.tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=output,
        )

    @staticmethod
    def _materialize_reward_extra_fields(value, expected_size: int) -> list:
        extra_fields = materialize_transfer_queue_value(value)
        if not isinstance(extra_fields, list) or len(extra_fields) != expected_size:
            raise RuntimeError("TransferQueue returned misaligned trajectory reward components.")
        return extra_fields

    def _build_final_trajectory_signals(
        self,
        batch,
        extra_fields: list,
    ) -> _FinalTrajectorySignals:
        final_indices, row_to_final = final_session_layout(batch.keys)
        reward_infos = [_reward_info(extra_fields[index]) for index in final_indices]
        is_padding = np.asarray(
            [bool(batch.tags[index].get("is_padding", False)) for index in final_indices],
            dtype=bool,
        )
        format_valid = np.asarray(
            [bool(_numeric_reward_field(info, "format_valid")) for info in reward_infos],
            dtype=bool,
        )
        successful_calls = np.asarray(
            [_numeric_reward_field(info, "num_successful_memory_calls") for info in reward_infos],
            dtype=np.float64,
        )
        performance = np.asarray(
            [_numeric_reward_field(info, "performance_reward") for info in reward_infos],
            dtype=np.float32,
        )
        return _FinalTrajectorySignals(
            final_indices=final_indices,
            row_to_final=row_to_final,
            is_padding=is_padding,
            format_valid=format_valid,
            reward_infos=reward_infos,
            performance=performance,
            successful_calls=successful_calls,
        )

    def _build_resource_rewards(
        self,
        signals: _FinalTrajectorySignals,
    ) -> _TrajectoryResourceRewards:
        reward_infos = signals.reward_infos
        successful_calls = signals.successful_calls
        is_padding = signals.is_padding
        failed_calls = np.asarray(
            [_numeric_reward_field(info, "num_failed_memory_calls") for info in reward_infos],
            dtype=np.float64,
        )

        def build(field: str):
            raw_values = np.asarray(
                [_numeric_reward_field(info, field) for info in reward_infos],
                dtype=np.float64,
            )
            eligible = build_resource_eligibility_mask(
                raw_values=raw_values,
                format_valid=signals.format_valid,
                successful_calls=successful_calls,
                failed_calls=failed_calls,
                is_padding=is_padding,
            )
            return build_resource_rewards(
                raw_values=raw_values,
                eligible=eligible,
            )

        return _TrajectoryResourceRewards(
            performance=signals.performance,
            cost=build("total_api_cost"),
            latency=build("total_estimated_latency"),
        )

    @staticmethod
    def _record_resource_metrics(
        metrics: dict,
        *,
        resources: _TrajectoryResourceRewards,
        is_padding: np.ndarray,
    ) -> None:
        active = ~is_padding
        if not np.any(active):
            return

        _add_metric_summary(metrics, "critic/performance_reward", resources.performance[active])
        for name, batch in (("cost", resources.cost), ("latency", resources.latency)):
            _add_metric_summary(metrics, f"critic/{name}_reward", batch.rewards[active])

    def _compute_resource_aware_gdpo_advantage(
        self,
        batch,
        metrics,
        *,
        final_indices: list[int],
        row_to_final: list[int],
        is_padding: np.ndarray,
        resources: _TrajectoryResourceRewards,
        resource_weights: dict[str, float],
    ):
        data = verl_sync.tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["uid", "response_mask", "advantages"],
        )
        nested_response_mask = data["response_mask"]
        data = verl_sync.DataProto(batch=data.to_padded_tensor())
        data.non_tensor_batch["uid"] = np.array(data.batch.pop("uid").tolist(), dtype=object)
        final_data = data.select_idxs(final_indices)

        final_response_mask = final_data.batch["response_mask"]
        response_positions = torch.arange(
            final_response_mask.shape[1],
            device=final_response_mask.device,
        ).expand_as(final_response_mask)
        last_response_indices = torch.where(
            final_response_mask.bool(),
            response_positions,
            -1,
        ).max(dim=1).values
        padding_rows = torch.as_tensor(
            is_padding,
            dtype=torch.bool,
            device=last_response_indices.device,
        )
        if torch.any((last_response_indices < 0) & ~padding_rows).item():
            raise RuntimeError("GDPO received an empty response trajectory.")

        row_indices = torch.arange(len(final_indices), device=last_response_indices.device)
        nonempty_rows = last_response_indices >= 0
        resource_batches = {"cost": resources.cost, "latency": resources.latency}
        token_rewards: dict[str, torch.Tensor] = {}
        sample_masks: dict[str, np.ndarray] = {}
        for name, weight in resource_weights.items():
            if weight <= 0.0:
                continue
            token_reward = torch.zeros_like(final_data.batch["advantages"], dtype=torch.float32)
            token_reward[row_indices[nonempty_rows], last_response_indices[nonempty_rows]] = torch.as_tensor(
                resource_batches[name].rewards,
                dtype=token_reward.dtype,
                device=token_reward.device,
            )[nonempty_rows]
            token_rewards[name] = token_reward
            sample_masks[name] = resource_batches[name].advantage_mask

        combined_advantages = compute_decoupled_advantages(
            task_advantages=final_data.batch["advantages"],
            resource_token_rewards=token_rewards,
            response_mask=final_response_mask,
            group_ids=final_data.non_tensor_batch["uid"],
            resource_weights=resource_weights,
            resource_sample_masks=sample_masks,
            normalize_by_group_std=bool(
                OmegaConf.select(
                    self.config,
                    "algorithm.norm_adv_by_std_in_grpo",
                    default=True,
                )
            ),
        )
        first_response_indices = final_response_mask.argmax(dim=1)
        final_scores = combined_advantages[row_indices, first_response_indices]
        scatter_indices = torch.as_tensor(row_to_final, dtype=torch.long, device=final_scores.device)
        scores = final_scores[scatter_indices].unsqueeze(-1) * data.batch["response_mask"]
        data.batch["advantages"] = scores
        data.batch["returns"] = scores

        self._record_resource_metrics(
            metrics,
            resources=resources,
            is_padding=is_padding,
        )
        output = {
            field: verl_sync.response_to_nested(data.batch[field], nested_response_mask)
            for field in ("advantages", "returns")
        }
        return verl_sync.tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=verl_sync.TensorDict(output, batch_size=len(batch)),
        )


def _nonnegative_setting(config, path: str, default: float) -> float:
    value = float(OmegaConf.select(config, path, default=default) or 0.0)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{path} must be a finite non-negative number.")
    return value


def _advantage_estimator_name(config) -> str:
    estimator = OmegaConf.select(config, "algorithm.adv_estimator", default="")
    return str(getattr(estimator, "value", estimator)).strip().lower()


def _reward_info(extra_field) -> dict:
    if not isinstance(extra_field, dict):
        raise RuntimeError("Trajectory shaping requires mapping-valued agent-loop extra_fields.")
    reward_info = extra_field.get("reward_extra_info")
    if not isinstance(reward_info, dict):
        raise RuntimeError("reward_extra_info is missing from an agent-loop output.")
    return reward_info


def _numeric_reward_field(reward_info: dict, key: str) -> float:
    value = reward_info.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeError(f"Trajectory reward component '{key}' must be numeric.")
    value = float(value)
    if not np.isfinite(value):
        raise RuntimeError(f"Trajectory reward component '{key}' must be finite.")
    return value


def _nonnegative_metadata_number(metadata: Mapping[str, object], key: str) -> float:
    value = metadata.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeError(f"Marginal-utility metadata field {key!r} must be numeric.")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise RuntimeError(
            f"Marginal-utility metadata field {key!r} must be finite and non-negative."
        )
    return value


def _add_metric_summary(metrics: dict, prefix: str, values: np.ndarray) -> None:
    metrics.update(
        {
            f"{prefix}/mean": float(values.mean()),
            f"{prefix}/max": float(values.max()),
            f"{prefix}/min": float(values.min()),
        }
    )
