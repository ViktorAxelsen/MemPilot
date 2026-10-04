"""Sampling and formatting helpers for console rollout diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from random import Random
from typing import Any


def sample_rollout_indices(
    tags: Sequence[Any],
    probability: float,
    rng: Random,
) -> list[int]:
    """Independently sample non-padding rollout positions for console output."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("rollout print probability must be between 0 and 1")
    if probability == 0.0:
        return []

    return [
        index
        for index, tag in enumerate(tags)
        if not _is_padding(tag) and rng.random() < probability
    ]


def format_rollout_sample(
    *,
    key: str,
    global_step: int,
    epoch: int,
    prompt: str,
    response: str,
    reward: Any = None,
    ground_truth: Any = None,
) -> str:
    """Build one grep-friendly, multiline console record."""
    separator = "=" * 88
    reward_text = "n/a" if reward is None else str(reward)
    ground_truth_text = _format_ground_truth(ground_truth)
    return "\n".join(
        (
            separator,
            (
                "ROLLOUT SAMPLE | "
                f"step={global_step} | epoch={epoch} | key={key} | reward={reward_text}"
            ),
            "--- PROMPT ---",
            prompt,
            "--- RESPONSE ---",
            response,
            "--- GROUND TRUTH ---",
            ground_truth_text,
            separator,
        )
    )


def _is_padding(tag: Any) -> bool:
    return isinstance(tag, Mapping) and bool(tag.get("is_padding", False))


def _format_ground_truth(ground_truth: Any) -> str:
    if ground_truth is None:
        return "n/a"

    return str(ground_truth)
