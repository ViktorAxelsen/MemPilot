"""Dataset-aware reward router for joint multimodal-memory training."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rewards.h2hmem_reward import compute_score as compute_h2hmem_score
from rewards.mem_gallery_reward import compute_score as compute_mem_gallery_score
from rewards.worldmemarena_reward import compute_score as compute_worldmemarena_score


RewardFunction = Callable[..., dict[str, Any]]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch each mixed-dataset row to its native deterministic reward."""

    reward_function = reward_function_for_source(data_source)
    return reward_function(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )


def reward_function_for_source(data_source: Any) -> RewardFunction:
    normalized = str(data_source or "").strip().casefold().replace("-", "_")
    if "worldmemarena" in normalized:
        return compute_worldmemarena_score
    if "h2hmem" in normalized:
        return compute_h2hmem_score
    if "mem_gallery" in normalized:
        return compute_mem_gallery_score
    raise ValueError(f"Unsupported unified multimodal data_source: {data_source!r}.")
