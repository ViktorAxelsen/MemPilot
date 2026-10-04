"""Objective-wise advantage decoupling for resource-aware GRPO.

The paper's GDPO path normalizes answer quality, negative cost, and negative
proxy latency separately, then combines their advantages and batch-whitens.
Prefix-based marginal utility is applied separately by the trainer.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ResourceRewardBatch:
    """One trajectory-level resource objective."""

    rewards: np.ndarray
    advantage_mask: np.ndarray


def build_resource_eligibility_mask(
    *,
    raw_values: Iterable[float],
    format_valid: Iterable[bool],
    successful_calls: Iterable[float],
    failed_calls: Iterable[float],
    is_padding: Iterable[bool],
) -> np.ndarray:
    """Identify trajectories whose resource use is observable and optimizable."""

    values = np.asarray(tuple(raw_values), dtype=np.float64)
    valid = np.asarray(tuple(format_valid), dtype=bool)
    successes = np.asarray(tuple(successful_calls), dtype=np.float64)
    failures = np.asarray(tuple(failed_calls), dtype=np.float64)
    padding = np.asarray(tuple(is_padding), dtype=bool)
    if (
        values.ndim != 1
        or valid.shape != values.shape
        or successes.shape != values.shape
        or failures.shape != values.shape
        or padding.shape != values.shape
    ):
        raise ValueError("Resource-eligibility inputs must have identical one-dimensional shapes.")
    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isfinite(successes))
        or not np.all(np.isfinite(failures))
        or np.any(values < 0.0)
        or np.any(successes < 0.0)
        or np.any(failures < 0.0)
    ):
        raise ValueError("Resource and call-count inputs must be finite and non-negative.")

    observed = (failures == 0.0) | (successes > 0.0) | (values > 0.0)
    return valid & ~padding & observed


def build_resource_rewards(
    *,
    raw_values: Iterable[float],
    eligible: Iterable[bool],
) -> ResourceRewardBatch:
    """Use negative raw resource values, masking ineligible trajectories."""

    values = np.asarray(tuple(raw_values), dtype=np.float64)
    eligibility = np.asarray(tuple(eligible), dtype=bool)
    if values.ndim != 1 or eligibility.shape != values.shape:
        raise ValueError("Resource reward inputs must have identical one-dimensional shapes.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("raw_values must contain finite non-negative values.")

    rewards = -values.astype(np.float32)
    rewards *= eligibility
    return ResourceRewardBatch(
        rewards=rewards,
        advantage_mask=eligibility,
    )


def compute_decoupled_advantages(
    *,
    task_advantages: torch.Tensor,
    resource_token_rewards: Mapping[str, torch.Tensor],
    response_mask: torch.Tensor,
    group_ids: Sequence[Any],
    resource_weights: Mapping[str, float],
    resource_sample_masks: Mapping[str, Sequence[bool] | None] | None = None,
    epsilon: float = 1e-6,
    normalize_by_group_std: bool = True,
) -> torch.Tensor:
    """Fuse independently normalized quality and resource advantages.

    ``task_advantages`` are already group-normalized by verl. Normalize each
    resource objective separately, add its preference-weighted advantage to the
    task advantage (task weight 1), then whiten over masked batch tokens. These
    weights are training preferences, not additional inputs to the policy.
    """

    if task_advantages.shape != response_mask.shape or task_advantages.ndim != 2:
        raise ValueError("task_advantages and response_mask must share a 2-D shape.")
    if len(group_ids) != task_advantages.shape[0]:
        raise ValueError("group_ids must contain one value per rollout.")

    weights = {name: float(weight) for name, weight in resource_weights.items() if float(weight) > 0.0}
    if not weights:
        raise ValueError("At least one resource weight must be positive for GDPO fusion.")
    if any(float(weight) < 0.0 for weight in resource_weights.values()):
        raise ValueError("Resource weights must be non-negative.")

    sample_masks = resource_sample_masks or {}
    fused = task_advantages
    for name, weight in weights.items():
        if name not in resource_token_rewards:
            raise ValueError(f"Missing token rewards for resource {name!r}.")
        rewards = resource_token_rewards[name]
        if rewards.shape != task_advantages.shape:
            raise ValueError(f"Resource {name!r} rewards must match task advantage shape.")
        advantage = _group_normalized_advantage(
            rewards,
            response_mask,
            group_ids,
            sample_mask=sample_masks.get(name),
            epsilon=epsilon,
            normalize_by_std=normalize_by_group_std,
        )
        fused = fused + weight * advantage

    return _masked_whiten(fused, response_mask) * response_mask


def final_session_layout(batch_keys: Sequence[Any]) -> tuple[list[int], list[int]]:
    """Return final-output rows and a map from every row to its session final."""

    final_sessions: dict[str, tuple[int, int]] = {}
    row_session_keys = []
    for row_index, key in enumerate(batch_keys):
        uid, session_id, output_index = _parse_transfer_queue_key(key)
        session_key = f"{uid}_{session_id}"
        row_session_keys.append(session_key)
        if session_key not in final_sessions or final_sessions[session_key][0] < output_index:
            final_sessions[session_key] = (output_index, row_index)

    final_indices = [row_index for _, row_index in final_sessions.values()]
    session_to_final = {session_key: index for index, session_key in enumerate(final_sessions)}
    return final_indices, [session_to_final[session_key] for session_key in row_session_keys]


def _parse_transfer_queue_key(key: Any) -> tuple[str, str, int]:
    fields = str(key).rsplit("_", 2)
    if len(fields) != 3:
        raise RuntimeError(f"Unexpected TransferQueue key format: {key}")
    uid, session_id, output_index = fields
    try:
        return uid, session_id, int(output_index)
    except ValueError as exc:
        raise RuntimeError(f"Unexpected TransferQueue key format: {key}") from exc


def _group_normalized_advantage(
    token_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: Sequence[Any],
    *,
    sample_mask: Sequence[bool] | None,
    epsilon: float,
    normalize_by_std: bool,
) -> torch.Tensor:
    scores = token_rewards.sum(dim=-1)
    masked_mode = sample_mask is not None
    if sample_mask is None:
        included = [True] * scores.shape[0]
    else:
        included_tensor = torch.as_tensor(sample_mask, dtype=torch.bool)
        if included_tensor.shape != scores.shape:
            raise ValueError("sample_mask must contain one value per rollout.")
        included = included_tensor.tolist()

    grouped_indices: dict[Any, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        if included[index]:
            grouped_indices[group_id].append(index)

    normalized = torch.zeros_like(scores)
    with torch.no_grad():
        for indices in grouped_indices.values():
            index_tensor = torch.as_tensor(indices, device=scores.device)
            group_scores = scores[index_tensor]
            if len(indices) == 1:
                if masked_mode:
                    continue
                mean = torch.zeros((), dtype=scores.dtype, device=scores.device)
                std = torch.ones((), dtype=scores.dtype, device=scores.device)
            else:
                mean = group_scores.mean()
                std = group_scores.std()
            centered = group_scores - mean
            normalized[index_tensor] = centered / (std + epsilon) if normalize_by_std else centered
    return normalized.unsqueeze(-1) * response_mask


def _masked_whiten(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    count = mask.sum()
    if count.item() <= 1:
        raise ValueError("GDPO batch whitening requires at least two response tokens.")
    mean = (values * mask).sum() / (count + 1e-8)
    centered = values - mean
    variance = ((centered.square() * mask).sum() / (count + 1e-8)) * count / (count - 1)
    return centered * torch.rsqrt(variance + 1e-8)
